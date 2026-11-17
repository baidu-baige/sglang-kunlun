"""Exact greedy fast path for the 0.5.22 Markov sampler on Kunlun."""

import torch


from sglang.srt.distributed.device_communicators.vocab_gather import LocalVocabGather

_LOCAL_CALL = LocalVocabGather.__call__
_LOCAL_STACKED = LocalVocabGather.gather_stacked


def _local_kept_logits_supported(kwargs):
    """Only the concrete identity gather over the complete valid vocabulary."""
    if (
        kwargs.get("has_sampled_rows") is not False
        or kwargs.get("step_dtype") != torch.float32
        or kwargs.get("vocab_start", 0) != 0
    ):
        return False
    local = kwargs.get("gather_logits")
    gather = kwargs.get("gather")
    if (
        type(local) is not LocalVocabGather
        or LocalVocabGather.__call__ is not _LOCAL_CALL
        or LocalVocabGather.gather_stacked is not _LOCAL_STACKED
        or getattr(gather, "__self__", None) is not local
        or getattr(gather, "__func__", None) is not _LOCAL_STACKED
    ):
        return False
    base, bias, out = (
        kwargs.get("base_logits"), kwargs.get("bias"), kwargs.get("logits_out")
    )
    if not all(isinstance(x, torch.Tensor) for x in (base, bias, out)):
        return False
    return (
        base.is_cuda
        and base.device == bias.device == out.device
        and base.ndim == bias.ndim == out.ndim == 2
        and base.shape[0] > 0
        and base.shape[0] == bias.shape[0] == out.shape[0]
        and 0 < bias.shape[1] == out.shape[1] <= base.shape[1]
        and out.shape[1] < 2147483647
        and base.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and bias.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and out.dtype == torch.float32
        and base.stride(1) == bias.stride(1) == out.stride(1) == 1
        and not any(x.requires_grad for x in (base, bias, out))
    )


def _local_kept_logits(cls, kwargs):
    """Preserve the FP32 recursive oracle, including its second add of +0."""
    base, bias, out = kwargs["base_logits"], kwargs["bias"], kwargs["logits_out"]
    cls._check(
        base, bias, kwargs["temperatures"], kwargs["offset"], kwargs["step_dtype"],
        kwargs.get("vocab_start", 0), kwargs["gather"], kwargs["gather_logits"], out,
    )
    # Crop as LocalVocabGather + the recursive oracle would. The temporary keeps
    # aliasing safe, and copy_ preserves the caller's [bs, gamma, vocab][:, step].
    value = base[:, : bias.shape[1]].to(torch.float32, copy=True)
    value.add_(bias.float())
    # Recursion adds zeros_like(full) after FP32 step rounding. Keep the add:
    # the tensor add is required: scalar add_(0.0) differs on signed zero on XPU.
    value.add_(torch.zeros_like(value))
    out.copy_(value)
    # Use the same device argmax as the oracle. With one complete shard its
    # local winner is also the final winner; no candidate transport is needed.
    return value.argmax(dim=1)


def markov_greedy_step(cls, **kwargs):
    """Keep rounding and the torch reference's [rows, 2] transport.

    The vendor partial adds fp16 base and bf16 bias in fp32 without rounding.
    This directly implements fp32 steps only for those exact input dtypes.
    For fp16 steps, rounding can create ties or infinities and change the
    lowest winning ID: supply already rounded logits and a zero bf16 bias.
    Collapse blocks locally before communicating one candidate per row.
    Both reductions use the vendor's NaN-aware, lowest-index ordering.
    """
    if _local_kept_logits_supported(kwargs):
        return _local_kept_logits(cls, kwargs)
    base = kwargs["base_logits"]
    bias = kwargs["bias"]
    step_dtype = kwargs["step_dtype"]
    direct_fp32 = (
        step_dtype == torch.float32
        and base.dtype == torch.float16
        and bias.dtype == torch.bfloat16
    )
    xsg = torch.ops.xspeedgate_ops
    if (
        kwargs.get("has_sampled_rows") is not False
        or kwargs.get("logits_out") is not None
        or not base.is_cuda
        or base.shape[0] == 0
        or (step_dtype != torch.float16 and not direct_fp32)
        or not hasattr(xsg, "sharded_greedy_partial")
        or not hasattr(xsg, "sharded_greedy_finish")
    ):
        return cls.torch(**kwargs)

    vocab_start = kwargs.get("vocab_start", 0)
    gather = kwargs.get("gather")
    cls._check(
        base, bias, kwargs["temperatures"], kwargs["offset"], step_dtype,
        vocab_start, gather, kwargs.get("gather_logits"), None,
    )
    rows, width = bias.shape
    if width:
        parts = (width + 4095) // 4096
        if parts > 256:
            return cls.torch(**kwargs)
        if direct_fp32:
            value, vendor_bias = base[:, :width], bias
        else:
            # fp16 steps must round before selecting, including newly tied IDs.
            value = (base[:, :width].float() + bias.float()).to(step_dtype)
            vendor_bias = torch.zeros(
                (1, width), device=base.device, dtype=torch.bfloat16
            ).expand(rows, width)
        blocks = torch.empty(
            (rows, parts, 2), device=base.device, dtype=torch.float32
        )
        xsg.sharded_greedy_partial(
            vendor_bias, value, blocks, width,
            vocab_start if gather is None else 0, parts, 4096,
        )
        index = torch.empty(rows, device=base.device, dtype=torch.int64)
        xsg.sharded_greedy_finish(blocks, index, rows, parts, 1)
        # The vendor leaves a sentinel on all--inf rows; they have a valid ID.
        index = torch.where(
            index == 2147483647, vocab_start if gather is None else 0, index
        )
        if gather is None:
            return index
        best = value.gather(1, index[:, None]).squeeze(1).float()
        if direct_fp32:
            best = best + bias.gather(1, index[:, None]).squeeze(1).float()
        index = (index + vocab_start).to(torch.int32)
    else:
        index = torch.full(
            (rows,), 2147483647, device=base.device, dtype=torch.int32
        )
        if gather is None:
            return index.to(torch.int64)
        best = torch.full(
            (rows,), -torch.inf, device=base.device, dtype=torch.float32
        )
    # The real sharded caller supplies gather_logits even for greedy-only steps.
    partial = torch.stack((best, index.view(torch.float32)), dim=1)
    gathered = gather(partial)
    # Preserve the collective's actual values, then map -inf below every
    # finite supported sum for the vendor's all--inf reduction limitation.
    # Max finite bf16 + max finite fp16 is strictly below max finite fp32.
    gathered = torch.stack(
        (gathered[:, 0].clamp_min(-torch.finfo(torch.float32).max), gathered[:, 1]),
        dim=1,
    )
    out = torch.empty(rows, device=base.device, dtype=torch.int64)
    xsg.sharded_greedy_finish(
        gathered.contiguous(), out, rows, 1, gathered.shape[0] // rows
    )
    return out
