"""Kunlun replacement for the DeepSeek V4 MHC Sinkhorn kernel."""

from __future__ import annotations

import os

import torch
import kunlun_ops
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

# Escape hatch: force the upstream torch reference for both mHC stages.
_FORCE_TORCH = os.environ.get("SGLANG_KUNLUN_MHC_TORCH", "0") == "1"

_HALF_DTYPES = (torch.bfloat16, torch.float16)


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_split_sinkhorn",
    type=HookType.REPLACE,
)
def hc_split_sinkhorn_kunlun(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Run Sinkhorn through the Kunlun op while preserving the 0.5.14 contract."""
    batch, seq_len, _ = mixes.shape
    pre = mixes.new_empty(batch, seq_len, hc_mult)
    post = mixes.new_empty(batch, seq_len, hc_mult)
    comb = mixes.new_empty(batch, seq_len, hc_mult, hc_mult)
    if mixes.numel() == 0:
        return pre, post, comb

    flat_tokens = batch * seq_len
    flat_pre = pre.view(flat_tokens, hc_mult)
    flat_post = post.view(flat_tokens, hc_mult)
    flat_comb = comb.view(flat_tokens, hc_mult * hc_mult)
    kunlun_ops.mhc_split_sinkhorn(
        mixes.reshape(flat_tokens, -1),
        hc_scale,
        hc_base,
        flat_pre,
        flat_post,
        flat_comb,
        hc_mult,
        sinkhorn_iters,
        eps,
    )
    return pre, post, comb


@plugin_hook(
    "sglang.srt.models.deepseek_v4._get_mhc_ops",
    type=HookType.REPLACE,
)
def _get_mhc_ops_kunlun():
    """Resolve the MHC ops through the hooked module instead of ``sgl_kernel``.

    Upstream short-circuits to ``sgl_kernel.hc_split_sinkhorn`` on XPU, which
    returns fp16 ``post``/``comb``. Golden reaches the Kunlun replacement
    registered on the layernorm module, whose outputs follow the fp32 ``mixes``
    dtype, so route through that module to keep the mHC boundary aligned.
    """
    from sglang.srt.models.deepseek_v4 import MhcOps
    from sglang.kernels.ops.layernorm import mhc as mhc_module

    return MhcOps(mhc_module.hc_split_sinkhorn, None, None)


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc._mhc_pre_dispatch",
    type=HookType.AROUND,
)
def _mhc_pre_dispatch_kunlun(
    original_fn,
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
):
    """mHC pre through ``kunlun_ops.hc_pre_kunlun_impl``.

    P800 has no tilelang (the module is not even installed), so upstream falls
    back to ``_mhc_pre_torch`` - a fp32 GEMM plus a 20-iteration Sinkhorn loop
    of small kernels, run ~90 times per forward on GLM-5-Next (45 layers x
    attn/ffn sublayers). The Kunlun op fuses all of it.

    ``norm_weight`` is deliberately not folded: the op has no out-norm argument,
    so ``norm_fused=False`` is returned and the caller applies the layernorm
    itself, exactly like the torch path does.

    The op emits ``h_post``/``h_res`` in the activation dtype while the torch
    path keeps them fp32; they are widened back to fp32 here so the downstream
    mHC-post boundary stays fp32. Measured against the torch reference at
    (s, 4, 4096) bf16: y cos 0.999997, h_post cos 0.999999, h_res cos 0.999999.
    """
    s, n, hidden = residual.shape
    if (
        _FORCE_TORCH
        or n != 4
        or s == 0
        or residual.dtype not in _HALF_DTYPES
        or fn.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
    ):
        return original_fn(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )

    x = residual.reshape(s, n * hidden).contiguous()
    layer_input = torch.empty(s, hidden, device=x.device, dtype=x.dtype)
    h_post = torch.empty(s, n, device=x.device, dtype=x.dtype)
    h_res = torch.empty(s, n * n, device=x.device, dtype=x.dtype)
    kunlun_ops.hc_pre_kunlun_impl(
        x,
        fn.contiguous(),
        hc_base.contiguous(),
        hc_scale.contiguous(),
        layer_input,
        h_post,
        h_res,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        h_post.float().view(s, n, 1),
        h_res.float().view(s, n, n),
        layer_input,
        False,
    )


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc._mhc_post_dispatch",
    type=HookType.AROUND,
)
def _mhc_post_dispatch_kunlun(
    original_fn,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
):
    """mHC post through ``kunlun_ops.mhc_post_fusion``.

    Same formula as ``_mhc_post_torch``, but without materialising its
    ``(s, n, n, hidden)`` intermediate - that tensor is what OOMs the 8k-token
    prefill batches (~1.9 GiB at s=8192, n=4, hidden=4096).

    The op takes ``post_mix``/``comb_mix`` in fp32 (or in x's dtype) and both
    must agree, which the fp32 mHC-pre boundary already guarantees. Measured
    against the torch reference at (s, 4, 4096) bf16: cos 1.000000.
    """
    s, n, hidden = residual.shape
    if (
        _FORCE_TORCH
        or n != 4
        or s == 0
        or x.dtype not in _HALF_DTYPES
        or x.dtype != residual.dtype
        or post_layer_mix.dtype != comb_res_mix.dtype
    ):
        return original_fn(x, residual, post_layer_mix, comb_res_mix)

    return kunlun_ops.mhc_post_fusion(
        x.contiguous(),
        residual.contiguous(),
        post_layer_mix.reshape(s, n).contiguous(),
        comb_res_mix.contiguous(),
    )
