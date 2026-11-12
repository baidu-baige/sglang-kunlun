"""Kunlun-backed implementations of layernorm ``sgl_kernel`` APIs."""

from __future__ import annotations

import torch


def _resolve_add_rmsnorm_inplace():
    """Return ``torch.ops.xspeedgate_ops.add_rmsnorm_inplace``, or None to use kunlun_ops."""
    return getattr(torch.ops.xspeedgate_ops, "add_rmsnorm_inplace", None)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm and return a new output tensor in the model dtype.

    For a half-precision weight the output carries the weight's dtype -- the
    dtype the model was loaded at -- not the dtype of whatever producer filled
    ``x``. An fp32 weight keeps ``x``'s dtype (native-precision paths).
    """

    import kunlun_ops

    # Measured table for `kunlun_ops.rmsnorm` on this image: weight must match
    # x's dtype or be fp32 (fp16 x + bf16 weight fails, and so does the
    # reverse). fp32 is the one value supported for every activation width, and
    # the weight is a per-channel vector, so pinning it costs nothing. The same
    # probe pins the *output*: `out` must carry `x`'s dtype exactly, since
    # `x=fp16 w=fp32 out=bf16` and `x=bf16 w=fp32 out=fp16` both fail with
    # `add_rmsnorm() not support x/weight/output dtype`. The buffer therefore
    # has to stay `empty_like(x)`, and the model dtype can only be restored by
    # casting the result.
    out = torch.empty_like(x)
    kunlun_ops.rmsnorm(
        x,
        weight.to(torch.float32),
        out,
        eps,
        False,
        True,
        None,
        None,
        None,
    )
    model_dtype = weight.dtype
    if out.dtype != model_dtype and model_dtype in (torch.float16, torch.bfloat16):
        # Why the cast, measured on this image: a dense linear requires x and
        # its weight to share a dtype (fp16 x + bf16 w and the reverse both fail
        # with `linear custom kernel call Failed`, left operand=4), and
        # `torch.cat` promotes a mixed fp16/bf16 pair to fp32. The EAGLE draft
        # builds its input from these norms and concatenates them:
        #   `fused_eh_norm` (`kernels/kernel_ops.py:7355`) ->
        #   `cat((rmsnorm(inputs_embeds, enorm_weight),
        #         rmsnorm(previous_hidden, hnorm_weight)), dim=-1)`
        # (`models/deepseek_nextn.py:204-218`; Kunlun takes the `_is_cuda`
        # branch, so the fused op above is the one that runs). `previous_hidden`
        # is the target's emitted hidden state -- bf16 on this platform, because
        # the vendor kernels emit bf16 while the model is loaded fp16 (measured
        # in `fused_add_rmsnorm`: `x=bf16 residual=fp16 weight=fp16`).
        # Inheriting that bf16 through `empty_like(x)` made the pair
        # fp16+bf16 -> fp32, and `eh_proj` (`deepseek_nextn.py:222`, fp16
        # weight) died with `ValueError: Check ret == 0 failed, left operand=4,
        # linear custom kernel call Failed`, killing decode on its first
        # draft-extend.
        # A norm's output is an activation the model's projections consume, so
        # it belongs to the model dtype. This is the only place that decision
        # can be expressed: upstream's in-place `fused_add_rmsnorm` writes back
        # into the caller's buffer and cannot retype it, and the vendor op
        # rejects an `out` of any other dtype. Casting at the consumer instead
        # (at `eh_proj`, or at its `hnorm`) would carry the producer's dtype
        # into a model whose weights are all one dtype, and the same mismatch
        # then reappears at every other dense projection the tensor reaches.
        # Restricted to half-precision weights: an fp32 weight belongs to a
        # native-precision path (e.g. `deepseek_v41_vit`'s vision norm, which
        # also sets `force_native` and never lands here), and upcasting its
        # activation to fp32 would break the very projections this guards.
        out = out.to(model_dtype)
    return out


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Apply fused residual add + RMSNorm in-place following sgl_kernel API.

    ``kunlun_ops.add_rmsnorm`` with an explicit ``output`` buffer is the op the
    0.5.8 line used for this API, at both of its seams and with this exact
    argument list:

      * ``sgl-kernel/python/sgl_kernel/elementwise.py:88-95`` --
        ``kunlun_ops.add_rmsnorm(input, residual, weight, out, eps, False, True,
        None, None, residual, None)`` then ``input.copy_(out)``;
      * ``sgl-kernel/python/sgl_kernel/patch/layers/layernorm.py:43-44`` --
        ``RMSNorm.forward_cuda``, the same call, returning ``(out, residual)``.

    It is the primary path because it is the only thing that emits the baseline
    kernels. ``kunlun_ops.add_rmsnorm`` -> ``xpu_flash_ops.add_rmsnorm``
    (``kunlun_ops/_xflashinfer_norm.py:128``) ->
    ``optimized_ops::add_rms_layer_norm<T, TW>``, and that host template in
    ``kunlun_ops/libapiinfer.so`` is the one function that launches
    ``xpuops3::infer_add_rms_layer_norm_with_output`` and
    ``..._stride_with_output``; the contiguous/strided choice is made inside the
    op, so a single call covers both. ``xspeedgate_ops.add_rmsnorm_inplace``
    writes both outputs in place and needs neither the scratch ``out`` nor the
    trailing ``x.copy_(out)``, but it belongs to another vendor library and
    traces as ``xinfer3::infer_add_rmsnorm_inplace`` -- the kernel the 0.5.23
    trace carries *instead of* the baseline pair -- so it is now the fallback.
    """

    import kunlun_ops

    # Measured table for `kunlun_ops.add_rmsnorm` on this image: x and residual
    # must agree, weight may be that dtype **or fp32**, and fp32 x/residual is
    # unsupported. fp32 weight is the one value that works for both activation
    # widths, so pin it rather than inheriting the stream's.
    #
    # The two producers disagree on this platform: the attention output arrives
    # bf16 from the vendor kernels while the residual stream is fp16 (measured
    # with a temporary probe: `x=torch.bfloat16 residual=torch.float16
    # weight=torch.float16`). Normalising the incoming activation to the
    # residual's dtype is the only direction that keeps the stream's dtype
    # stable; casting the residual instead would rewrite the KV/residual
    # buffers' dtype mid-layer. The write-back moves the *values*, not the
    # tensor: `x.copy_` writes into the caller's buffer, so `x` keeps its
    # original dtype and upstream's `result = (x, residual)` hands a bf16 tensor
    # on. A dtype that has already drifted is recovered by the out-of-place
    # `rmsnorm` above (which returns the weight's dtype), not here.
    x_typed = x if x.dtype == residual.dtype else x.to(residual.dtype)
    # `empty`, not `empty_like`: the caller's `x` can be a view, and `empty_like`
    # copies its strides, which makes the vendor op pick its *strided* epilogue
    # (`infer_add_rms_layer_norm_stride_with_output` + a `transpose_021_sdnn_bsp`
    # + a `clamp` per layer). 0.5.8 fed it a contiguous buffer and got
    # `infer_add_rms_layer_norm_with_output` (`elementwise.py:88-95`).
    out = torch.empty(x_typed.shape, dtype=x_typed.dtype, device=x_typed.device)
    # Pass the weight in the activation's dtype when it already is -- the module
    # weight is the model dtype, so in the steady state this is a no-op. Pinning
    # fp32 unconditionally (`weight.to(torch.float32)`) allocated and cast a full
    # [hidden] tensor on *every* layer of *every* step: that is the
    # `cluster_cast_kl3` count (14.3k -> 49.6k launches at matched load) and part
    # of the `memcpy_1d` growth in the decode trace. The measured dtype table
    # only requires weight ∈ {activation dtype, fp32}, not fp32 itself.
    # Match the activation's dtype (0.5.8 passed `self.weight` untouched, i.e.
    # the model dtype). An fp32 weight selects the `<T=half, TW=float>`
    # instantiation, which on this image only ships the *strided* epilogue: the
    # trace then shows `infer_add_rms_layer_norm_stride_with_output` plus a
    # `transpose_021_sdnn_bsp` and a `clamp` per layer (4860 launches each),
    # where 0.5.8 ran `infer_add_rms_layer_norm_with_output` 5735 times and
    # those two helpers 148/175 times in the whole capture.
    weight_for_op = (
        weight if weight.dtype == x_typed.dtype else weight.to(x_typed.dtype)
    )
    try:
        kunlun_ops.add_rmsnorm(
            x_typed,
            residual,
            weight_for_op,
            out,
            eps,
            False,
            True,
            None,
            None,
            residual,
            None,
        )
    except Exception:
        _fused_add_rmsnorm_inplace(x, residual, weight, eps)
        return
    x.copy_(out)


def _fused_add_rmsnorm_inplace(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """``xspeedgate_ops.add_rmsnorm_inplace`` form of :func:`fused_add_rmsnorm`.

    Reached only when ``kunlun_ops.add_rmsnorm`` is missing or rejects the dtype
    triple. Same dtype reasoning as the caller; the cost of taking it is that
    the step emits ``xinfer3::infer_add_rmsnorm_inplace`` rather than the
    ``xpuops3::infer_add_rms_layer_norm*_with_output`` pair the 0.5.8 trace has.
    """

    op = _resolve_add_rmsnorm_inplace()
    if op is not None:
        if x.dtype != residual.dtype:
            # The vendor op writes both buffers in place and requires one dtype,
            # but on this platform the two producers disagree: the attention
            # output arrives bf16 from the vendor kernels while the residual
            # stream is fp16 (measured with a temporary probe:
            # `x=torch.bfloat16 residual=torch.float16 weight=torch.float16`).
            # Normalising the incoming activation to the residual's dtype is the
            # only direction that keeps the stream's dtype stable; casting the
            # residual instead would rewrite the KV/residual buffers' dtype
            # mid-layer.
            # This moves the *values*, not the tensor: `x.copy_` writes into the
            # caller's buffer, so `x` keeps its original dtype and upstream's
            # `result = (x, residual)` hands a bf16 tensor on. A dtype that has
            # already drifted is therefore recovered by the out-of-place
            # `rmsnorm` above (which returns the weight's dtype), not here.
            x_typed = x.to(residual.dtype)
            op(x_typed, residual, weight.to(torch.float32), eps)
            x.copy_(x_typed)
            return
        op(x, residual, weight, eps)
        return

    raise RuntimeError(
        "Kunlun fused_add_rmsnorm has no usable vendor op: kunlun_ops.add_rmsnorm "
        "rejected the call and torch.ops.xspeedgate_ops.add_rmsnorm_inplace is "
        "not registered"
    )


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Gemma RMSNorm and return a new output tensor with the input shape."""

    import kunlun_ops

    input_for_norm = x.contiguous()
    normalized = torch.empty_like(input_for_norm)
    kunlun_ops.gemma_rmsnorm(
        input_for_norm,
        weight.contiguous(),
        normalized,
        eps,
    )
    return normalized


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Add residual and apply Gemma RMSNorm in-place following sgl_kernel API."""

    import kunlun_ops

    # MiniMax 的 hidden states 和 residual 在 prefill、decode 及graph replay中均为连续张量。
    # 如果上游布局变化破坏该约定，则直接报错；算子的原地别名不支持带stride的视图。
    if not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError(
            "Kunlun gemma_fused_add_rmsnorm requires contiguous x and residual, "
            f"got x.shape={tuple(x.shape)}, x.stride={x.stride()}, "
            f"residual.shape={tuple(residual.shape)}, "
            f"residual.stride={residual.stride()}"
        )

    kunlun_ops.gemma_add_rmsnorm(
        x,
        residual,
        weight.contiguous(),
        x,
        eps,
        residual_output=residual,
        store_output_before_norm=True,
    )
