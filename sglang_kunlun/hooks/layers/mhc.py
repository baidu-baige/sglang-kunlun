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
    """
    from sglang.srt.models.deepseek_v4 import MhcOps
    from sglang.kernels.ops.layernorm import mhc as mhc_module

    return MhcOps(mhc_module.hc_split_sinkhorn, None, None)
