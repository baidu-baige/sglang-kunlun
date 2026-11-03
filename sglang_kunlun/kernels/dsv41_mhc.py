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

from sglang_kunlun.kernels.kernel_ops import (
    _dspark_warn_fallback,
    note_op_path,
    op_probe_shape,
    register_jit_op,
)


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_mix_stats")
def hc_mix_stats_kunlun(
    x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float
) -> torch.Tensor:
    """``F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat**2) + eps)``, fp32.

    The XSpeedGate op fuses the same two stages as upstream's Triton kernel (a
    split-K partial producing `[sumsq, dot...]` per k-block, then a per-token
    reduce applying the RMS scale), so it keeps the fp32 accumulation and the
    fp32 `[m, mix]` output. It reads `x_flat` in its own dtype, which drops the
    separate fp32 materialization the torch body needed. `mix` is 24 here, well
    inside the op's `mix <= 32`; `hc_fn` is already fp32, as the torch
    `F.linear` above also required.

    The enter/exit pair proves the op both ran and returned: an XPU kernel that
    faults inside takes the process down with no Python traceback, so a lone
    `enter` line localizes the crash to the op itself.
    """
    x_flat = x_flat.contiguous()
    hc_fn = hc_fn.contiguous()
    mixes = torch.ops.xspeedgate_ops.hc_mix_stats(x_flat, hc_fn, eps)
    return mixes


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "mhc_combine")
def hc_combine_kunlun(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """XSpeedGate ``hc_combine``: ``y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]``.

    The op carries no ``hc`` argument -- it reads the split count from ``pre``,
    which is ``[m, hc]`` here, and the head width from ``x``, so the hook's
    ``hc`` only has to go into the reshape of ``pre``. ``out_dtype`` maps onto
    the op's optional ``ScalarType?`` and selects the same output dtype the
    Triton kernel contracts for. Measured against the torch body below on
    ``m in {1, 8, 16, 64}``, ``hc=4``, fp16/bf16/fp32: identical for bf16 and
    fp32-in, max relative deviation 7.7e-4 for fp16-in, i.e. fp16 rounding of
    the same fp32 accumulation.

    Both tensors go in contiguous: ``x_flat`` is a flatten view of ``[m, hc, H]``
    and the op takes no strides, so a non-contiguous caller would otherwise be
    read as if it were packed. ``.contiguous()`` is free when it already is.

    Fall back to the torch body when the op is missing or rejects the call;
    ``_dspark_warn_fallback`` logs one line so the degradation is never silent.

    0.5.23 renamed the *upstream* target from ``hc_combine`` to ``mhc_combine``
    (same signature), which is what the ``register_jit_op`` above points at; the
    XSpeedGate op keeps its own name ``xspeedgate_ops.hc_combine``. Do not
    confuse the target with ``elementwise.hc_combine.hc_combine``, which is the
    HyperConnection gated-residual combine -- a different 7-parameter function
    that an earlier repoint pass wired here by mistake.
    """
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    if m == 0:
        return torch.empty((0, h), dtype=out_dtype, device=x_flat.device)

    op = "hc_combine"
    xsg = torch.ops.xspeedgate_ops
    if hasattr(xsg, "hc_combine"):
        try:
            y = xsg.hc_combine(
                x_flat.contiguous(),
                pre.reshape(m, hc).contiguous(),
                out_dtype,
            )
        except RuntimeError as exc:
            # The op's dtype/layout constraints are enforced by its own
            # TORCH_CHECKs, which throw before the launch and leave nothing
            # written, so continuing into the torch body is safe here.
            _dspark_warn_fallback(op, f"{type(exc).__name__}: {exc}")
        else:
            note_op_path(op, "xspeedgate")
            return y
    else:
        _dspark_warn_fallback(op, "xspeedgate_ops.hc_combine is unavailable")

    note_op_path(op, "torch")
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
