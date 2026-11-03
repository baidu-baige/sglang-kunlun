"""Hooks for ``sglang.srt.layers.attention.dsv4.dsv41_sparse``.

After enabling deepep, `require_attn_tp_gather()` is true -> `require_mlp_sync()`
is true -> every forward (**including prefill**) ceil-aligns the token count to
`attn_tp_size` in `ForwardBatch.prepare_mlp_sync_batch`. When DSPARK PD forces
dp=1/cp=1, `attn_tp_size=8`, so a request with `extend_lens=[4]` comes in with 8
rows of q.

The extend branch of `token_req_indices` (`dsv41_sparse.py:73-75`) is:

    torch.repeat_interleave(req, extend_seq_lens, output_size=num_tokens)

`output_size` is declared as 8 while the sum of `repeats` is only 4 -> **the
last 4 elements are uninitialized device memory**. On CUDA this mostly just
reads garbage; on XPU the immediately following `torch.unique_consecutive(req)`
(`deepseek_v4_backend.py:3612`) directly raises

    RuntimeError: [UNKNOWN ERROR]: error code= -714
        (at .../eager_customized/unique_consecutive.cpp:95)

And even if it luckily does not crash, the garbage req is then used as an index
to look up `self.req_to_token[r, ...]`.

Here, when there is padding, we compute it ourselves: **do not pass
`output_size`** (let the output length equal the true sum of `repeats`), then
fill the tail padding rows with the req of the last real request. The attention /
indexer results of the padding rows are discarded by the upper layer based on
the real token count, so filling any value does not affect correctness; filling
a legal value is only to avoid going out of bounds. Without padding we go
straight through the original implementation, with byte-for-byte identical
behavior.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

try:  # soft dependency: when xdll_ops is missing everything below falls back to the torch / bmm implementation
    import xdll_ops  # noqa: F401  registers the torch.ops.xdll_ops namespace

    _FUSED_LOGITS_AVAILABLE = hasattr(
        torch.ops.xdll_ops, "dsv4_low_ratio_index_logits"
    )
except Exception:  # noqa: BLE001 - any import failure means it is unavailable
    _FUSED_LOGITS_AVAILABLE = False

logger = logging.getLogger(__name__)

_LOGGED_PAD = False
_LOGGED_FUSED_SCORES = False

# ``(t, n, device)`` -> the bs=1 cu_seqlens / k_offsets / out_offsets / visible_lens.
_LOGITS_OFFSETS: dict = {}


def _fused_logits_usable(q, k, weights) -> bool:
    """Shape/type gate for ``xdll_ops::dsv4_low_ratio_index_logits``.

    Hard constraints of the op: q ``[T, 32, 128]``, k ``[N, 128]``,
    weights ``[T, 32]``, bs(=1) <= 64. If any of these is not satisfied, fall
    through to the bmm implementation below.
    """
    if not _FUSED_LOGITS_AVAILABLE:
        return False
    if q.dim() != 3 or q.shape[1] != 32 or q.shape[2] != 128:
        return False
    if k.dim() != 2 or k.shape[1] != 128:
        return False
    if weights.dim() != 2 or tuple(weights.shape) != (q.shape[0], q.shape[1]):
        return False
    return q.shape[0] > 0 and k.shape[0] > 0


def _logits_offsets(t: int, n: int, device) -> tuple:
    """Fixed-length offset tensors cached by ``(t, n, device)``.

    ``scores`` sits inside the request x chunk double loop; building four int32
    tensors from scratch each time would produce hundreds to thousands of H2D
    copies -- that alone can eat up the benefit of the op itself.
    """
    key = (t, n, device.type, device.index)
    cached = _LOGITS_OFFSETS.get(key)
    if cached is None:
        if len(_LOGITS_OFFSETS) > 256:
            _LOGITS_OFFSETS.clear()
        cached = (
            torch.tensor([0, t], dtype=torch.int32, device=device),
            torch.tensor([0, n], dtype=torch.int32, device=device),
            torch.tensor([0, t * n], dtype=torch.int32, device=device),
            # Only one request is packed at a time, so the visible length is the
            # whole n; the out-of-range mask is handled by the caller's
            # immediately following `masked_fill(j >= lens_c, -inf)`, semantics
            # unchanged.
            torch.full((t,), n, dtype=torch.int32, device=device),
        )
        _LOGITS_OFFSETS[key] = cached
    return cached


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.token_req_indices",
    type=HookType.AROUND,
)
def token_req_indices_kunlun(original_fn, forward_batch, *, num_tokens=None):
    """Map token rows to requests, filling MLP-sync padding with a valid request.

    Args:
        original_fn: Upstream mapping used when no padding repair is needed.
        forward_batch: Batch containing request indices and extend lengths.
        num_tokens: Requested row count, including optional MLP-sync padding.

    Returns:
        Per-token request indices; repaired padding repeats the last request.
    """
    global _LOGGED_PAD

    mode = forward_batch.forward_mode
    if (
        num_tokens is not None
        and mode.is_extend()
        and not mode.is_target_verify()
        and forward_batch.extend_seq_lens is not None
    ):
        lens_cpu = forward_batch.extend_seq_lens_cpu
        total = int(sum(lens_cpu)) if lens_cpu else None
        if total is not None and 0 < total < int(num_tokens):
            req = forward_batch.req_pool_indices.to(torch.int64)
            out = torch.repeat_interleave(
                req, forward_batch.extend_seq_lens.to(torch.int64)
            )
            pad_rows = int(num_tokens) - out.shape[0]
            if pad_rows > 0 and out.numel():
                out = torch.cat([out, out[-1:].expand(pad_rows)])
            if not _LOGGED_PAD:
                _LOGGED_PAD = True
                logger.warning(
                    "[kunlun] token_req_indices: mlp-sync padding %d -> %d rows; "
                    "padding rows reuse the last request index (upstream would "
                    "leave them uninitialized via repeat_interleave output_size)",
                    total,
                    int(num_tokens),
                )
            return out
    return original_fn(forward_batch, num_tokens=num_tokens)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Indexer.scores",
    type=HookType.AROUND,
)
def scores_kunlun(original_fn, self, q, k, weights):
    """indexer scoring: one fewer large-tensor promotion + one fewer fp32 large-tensor materialization.

    Args:
        original_fn: the upstream scoring method called when the optimization is off.
        q: query tensor of shape [T, H, D].
        k: dequantized index keys of shape [N, D].
        weights: per-query, per-head weights of shape [T, H].

    Returns:
        float32 weighted ReLU dot-product scores of shape [T, N].

    Upstream implementation (``dsv41_sparse.py:260``)::

        s = torch.einsum("bhd,nd->bhn", q, k)
        s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
        return s.float()

    In the trace of a 16000-token prefill on the P side, these three lines are
    spread across three costs (rank0 / 3 steps):

    * ``q`` is fp16 and ``k`` is bf16 -> einsum promotes **the large one** q
      ([t*H, 128], 8x bigger than k) to bf16: ``cluster_cast_kl3`` 76 ms;
    * the einsum output is then copied bf16<-fp32 once more: ``aten::copy_``
      71 ms;
    * ``s.relu() * weights`` -- because weights is fp32, the whole [t*H, n]
      large tensor does an elementwise multiply in fp32: ``tensor_calc_n_nk``
      164 ms (effective bandwidth only ~34 GB/s).

    Two changes here, with the math form unchanged:

    1. Convert the **small** one (k, [n, 128]) to q's dtype, instead of letting
       torch promote q. The model itself is ``--dtype float16``; the bf16
       dequant output of index-K is the odd one out.
    2. First downcast the weights to s's dtype before the elementwise multiply
       (half the transfer volume), and accumulate the reduction in fp32 via
       ``dtype=torch.float32`` when summing -- no longer materializing the
       [t*H, n] fp32 tensor. (Not written as ``einsum("bhn,bh->bn")``: the
       backend might first transpose [t, H, n] to fit a GEMM, adding one more
       large-tensor copy.)

    On CPU, for random inputs of [512,32,128]x[4096,128] with weights both
    positive and negative, versus an fp32 full-precision reference: upstream
    max_abs=0.2074 / top64 hit 0.9960, here max_abs=0.0385 / top64 hit 0.9989
    -- the new form is not only faster but **closer to the fp32 reference than
    upstream** (fp16's 10 mantissa bits beat bf16's 7).

    Precision gate (**on by default**, `DSV4_INDEXER_SCORES_FP16=0` turns it off
    for comparison):

    * ifeval limit=300 (268 examples scored): the two versions' main metrics are
      character-for-character identical: `prompt_level_strict` both 89.9%,
      `inst_level_strict` both 92.8%. ifeval limit=100 under the same config ran
      92.1 / 91.0 / 89.9 over three runs, showing n=100 itself has roughly ±2pp
      run-to-run variance (batch 64 dynamic batching breaks greedy decoding
      determinism), so the n=100 delta cannot be taken as a conclusion.
    * **longbench_v2 short once exposed a real regression**: the initial form was
      ``(s.relu() * w.to(fp16)).sum(dtype=fp32)``, rounding each product to fp16
      before accumulating; with it off it ran 60.6% / 61.6% twice, with it on it
      ran 58.6% both times, a reproducible 2.5pp gap. After switching to the bmm
      form below (products accumulate directly in fp32 inside the MAC, no
      rounding) it returned to **61.6%**, matching the control's ceiling. The
      whole process is in ``文档/性能对标/过程记录.md``.

    Performance: 16k prefill step wall 1658.6→**1494.9 ms (−9.9%)**, effective
    throughput 9647→**10703 tok/s (+10.9%)**; 64k across four shards totals on
    the order of −17%.
    ⚠️ These numbers were measured on the baseline where
    `quantize_fp4_indexer_tensor` / `store_fp4_index_k_cache` were still torch
    fallbacks (before commit 0ad934e). After switching to the ops the baseline
    changed, so this item's relative gain needs to be re-measured on the new
    baseline -- it goes through the **read side** of index-K (scoring after
    dequant), not the same path as the **write side** (quantize + scatter) that
    those two ops handle, so the expectation should still hold.

    When ``xdll_ops::dsv4_low_ratio_index_logits`` is available, switch to the
    fused op (``DSV4_INDEXER_SCORES_FUSED=0`` turns it off for comparison):

    * It folds the whole "per-head dot product → ReLU → multiply head weight →
      sum across heads" into one kernel, accumulating in fp32, and no longer
      materializes ``[t, H, n]``; this implementation only writes ``[t, n]``
      fp32.
    * T=2048 / N=4096 measured 1.68 → **1.04 ms**; T=512 / N=4096 is 0.41 →
      0.29 ms.
    * **More accurate than the bmm version on the real distribution** (q has
      already passed fp4 fake-quant, and its values can be exactly represented
      in bf16): versus the fp32 full-precision reference, the bmm version
      max_abs=0.126 / top512 hit 0.9996, the fused version max_abs=6.1e-5 /
      top512 hit 1.0000. On random inputs it looks the opposite way, because the
      random values are not on the fp4 grid -- that is not the real input of
      this path.
    * Pass ``visible_lens = n``; the out-of-range mask is still handled by the
      caller's ``masked_fill`` line, so the semantics are bit-for-bit comparable
      to the bmm version above.
    """
    if os.environ.get("DSV4_INDEXER_SCORES_FP16", "1") != "1":
        return original_fn(self, q, k, weights)

    if os.environ.get("DSV4_INDEXER_SCORES_FUSED", "1") != "0" and _fused_logits_usable(
        q, k, weights
    ):
        global _LOGGED_FUSED_SCORES
        t, n = q.shape[0], k.shape[0]
        cu, ko, oo, vis = _logits_offsets(t, n, q.device)
        out = torch.empty(t * n, dtype=torch.float32, device=q.device)
        # If weights is fp16, `.float()` is an exact widening; the op only accepts bf16 / fp32.
        w_op = (
            weights
            if weights.dtype in (torch.bfloat16, torch.float32)
            else weights.float()
        )
        torch.ops.xdll_ops.dsv4_low_ratio_index_logits(
            q.contiguous().to(torch.bfloat16),
            k.contiguous().to(torch.bfloat16),
            w_op.contiguous(),
            cu,
            ko,
            vis,
            out,
            oo,
        )
        if not _LOGGED_FUSED_SCORES:
            _LOGGED_FUSED_SCORES = True
            logger.warning(
                "[kunlun] DeepseekV41Indexer.scores -> dsv4_low_ratio_index_logits "
                "(t=%d n=%d q=%s k=%s w=%s)",
                t,
                n,
                q.dtype,
                k.dtype,
                weights.dtype,
            )
        return out.view(t, n)

    # Weighted sum via bmm: [t,1,H] @ [t,H,n] -> [t,1,n]; s is already [t,H,n] so
    # no transpose is needed. GEMM products accumulate directly in fp32 inside
    # the MAC, unlike the elementwise version which rounds each product to fp16
    # first, so it is more accurate than the fp16 version of
    # `(s.relu() * w).sum()`.
    s = torch.einsum("bhd,nd->bhn", q, k.to(q.dtype))
    out = torch.bmm(weights.unsqueeze(1).to(s.dtype), s.relu())
    return out.squeeze(1).float()
