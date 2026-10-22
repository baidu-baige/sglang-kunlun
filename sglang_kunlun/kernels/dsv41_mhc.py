"""Torch replacements for the DeepSeek-V4.1 mHC statistic/combine Triton kernels.

`hc_split_sinkhorn` is already handled by the plugin (`hooks/layers/mhc.py` via
the Kunlun op); these are the two remaining kernels on the manual HC path that
the runtime takes when neither TileLang nor DeepGEMM mHC prenorm is enabled
(`SGLANG_OPT_USE_TILELANG_MHC_PRE=0`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`):

  request -> DeepseekV4DecoderLayer._hc_mix_and_combine
          -> mhc.hc_mix_stats        [Triton split-K]        <-- here
          -> mhc.hc_split_sinkhorn   [already Kunlun]
          -> mhc.hc_combine          [Triton elementwise]    <-- here

0.5.22 added one more on the same module: ``DeepseekV4Model.forward`` now builds
the initial mHC residual with ``mhc.hc_broadcast`` instead of a plain
``repeat``.

The kernels exist upstream for batch invariance (a fixed K-split reduction
order); the torch bodies below use the same arithmetic as the repo's own torch
references (`_mhc_pre_torch`, `hc_combine`'s documented formula).
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


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_broadcast")
def hc_broadcast_kunlun(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[T, H] -> [T, hc_mult, H]``, every copy equal to ``x``.

    New in 0.5.22: ``DeepseekV4Model.forward`` builds the initial mHC residual
    through this helper. It has an upstream non-CUDA fallback, but the gate is
    ``if not x.is_cuda`` -- true nowhere on Kunlun, because the platform reuses
    the CUDA device type -- so every prefill tried to compile
    ``_hc_broadcast_kernel`` and died with "0 compatible backends for target
    (cuda)". This is upstream's own fallback expression, verbatim.
    """
    return x.unsqueeze(1).repeat(1, hc_mult, 1)
