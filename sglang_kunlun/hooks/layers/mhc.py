"""Kunlun replacement for the DeepSeek V4 MHC Sinkhorn kernel."""

from __future__ import annotations

import torch
import kunlun_ops
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


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

    ``MhcOps`` gained ``mhc_fused_post_pre``, ``mhc_pre``, ``mhc_post`` and
    ``fused_hc_head`` in 4.1 (the V4-Flash model never resolved this NamedTuple).
    XPU takes the ``_is_xpu`` branches that call ``mhc_pre`` / ``mhc_post`` /
    ``fused_hc_head``, so each is bound to the repo's own torch reference; the
    TileLang/DeepGEMM kernels behind them are gated off by this deployment's
    ``SGLANG_OPT_USE_TILELANG_MHC_PRE`` / ``..._MHC_POST`` / ``DEEPGEMM_HC_PRENORM``.
    """
    from sglang.srt.models.deepseek_v4 import MhcOps
    from sglang.srt.models import deepseek_v4 as dsv4
    from sglang.kernels.ops.layernorm import mhc as mhc_module

    def mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=None,
        norm_eps=None,
    ):
        """Kernel-signature wrapper over `_mhc_pre_torch` (3-tuple, no norm).

        The norm variant is only produced by the TileLang kernel, which this
        deployment disables, so `norm_weight`/`norm_eps` are accepted and
        ignored to preserve the caller's contract.
        """
        return mhc_module._mhc_pre_torch(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    return MhcOps(
        hc_split_sinkhorn=mhc_module.hc_split_sinkhorn,
        mhc_fused_post_pre=None,
        npu_hc_pre=None,
        mhc_pre=mhc_pre_torch,
        mhc_post=mhc_module._mhc_post_torch,
        fused_hc_head=dsv4.hc_head_torch,
    )
