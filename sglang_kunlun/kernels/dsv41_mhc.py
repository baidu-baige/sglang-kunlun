"""Torch replacements for the DeepSeek-V4.1 mHC statistic/combine Triton kernels.

`hc_split_sinkhorn` is already handled by the plugin (`hooks/layers/mhc.py` via
the Kunlun op); these are the two remaining kernels on the manual HC path that
the runtime takes when neither TileLang nor DeepGEMM mHC prenorm is enabled
(`SGLANG_OPT_USE_TILELANG_MHC_PRE=0`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`):

  request -> DeepseekV4DecoderLayer._hc_mix_and_combine
          -> mhc.hc_mix_stats        [Triton split-K]        <-- here
          -> mhc.hc_split_sinkhorn   [already Kunlun]
          -> mhc.hc_combine          [Triton elementwise]    <-- here

The kernels exist upstream for batch invariance (a fixed K-split reduction
order); the torch bodies below use the same arithmetic as the repo's own torch
references (`_mhc_pre_torch`, `hc_combine`'s documented formula).

4.1 also added a fused entry point that collapses the first two stages into
`_hc_mix_stats_partial_kernel` + `_hc_mix_reduce_sinkhorn_kernel`:

  request -> DeepseekV4DecoderLayer._hc_mix_stats
          -> mhc.hc_mix_stats_sinkhorn                       <-- here

It sits behind an `is_blackwell or (is_sm90 and batch == 1)` bf16 gate, so XPU
does not reach it today, but it bypasses both hooks above when it is taken (the
sinkhorn never goes through `mhc.hc_split_sinkhorn`). The replacement below
composes the same two stages out of `hc_mix_stats_kunlun` and the XSpeedGate
`hc_split_sinkhorn` op, whose documented expression matches
`_hc_mix_reduce_sinkhorn_kernel`'s stabilized row softmax plus row/column
sinkhorn sweeps.
"""

from __future__ import annotations

import torch

from sglang_kunlun.kernels.kernel_ops import register_jit_op


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_mix_stats")
def hc_mix_stats_kunlun(
    x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float
) -> torch.Tensor:
    """``F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat**2) + eps)``, fp32.

    Same expression as `_mhc_pre_torch`'s statistics projection; fp32 GEMM and
    fp32 RMS reduce per row.
    """
    mix = hc_fn.shape[0]
    if x_flat.shape[0] == 0:
        return torch.empty((0, mix), dtype=torch.float32, device=x_flat.device)
    x = x_flat.float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return torch.nn.functional.linear(x, hc_fn) * rsqrt


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_mix_stats_sinkhorn")
def hc_mix_stats_sinkhorn_kunlun(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
):
    """`hc_mix_stats` followed by the XSpeedGate ``hc_split_sinkhorn`` op.

    Upstream fuses the stats reduction into the sinkhorn kernel, so the coupled
    `mhc.hc_split_sinkhorn` hook never runs on this path; the stats projection is
    reused from `hc_mix_stats_kunlun` to keep one definition of that arithmetic.
    Returns `pre[m, hc]`, `post[m, hc]`, `comb[m, hc, hc]` fp32, matching the
    Triton contract -- the op derives its leading dims from `mixes`, which is
    `[m, (2 + hc) * hc]` here. The op requires fp32 contiguous `mixes`, `hc_mult
    == 4` and `1 <= sinkhorn_iters <= 32`.
    """
    m = x_flat.shape[0]
    device = x_flat.device
    if m == 0:
        return (
            torch.empty((0, hc_mult), dtype=torch.float32, device=device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=device),
            torch.empty((0, hc_mult, hc_mult), dtype=torch.float32, device=device),
        )
    mixes = hc_mix_stats_kunlun(x_flat, hc_fn, rms_eps)
    return torch.ops.xspeedgate_ops.hc_split_sinkhorn(
        mixes.contiguous(),
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        hc_mult,
        sinkhorn_iters,
        hc_eps,
    )


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_combine")
def hc_combine_kunlun(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """``y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]``.

    `x_flat` is the flatten view of `[m, hc, H]`, so the reshape recovers the
    head axis without a copy for the contiguous case. fp32 accumulation, one
    cast at the end, matching the kernel's output dtype contract.
    """
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    if m == 0:
        return torch.empty((0, h), dtype=out_dtype, device=x_flat.device)
    x = x_flat.reshape(m, hc, h).float()
    y = (pre.reshape(m, hc).float().unsqueeze(-1) * x).sum(dim=1)
    return y.to(out_dtype)
