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
    "sglang.srt.models.deepseek_v4.get_mhc_ops",
    type=HookType.REPLACE,
)
def get_mhc_ops_kunlun():
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


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.mhc_pre",
    type=HookType.REPLACE,
)
def mhc_pre_kunlun(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    n_splits_pre=32,
    *,
    norm_weight=None,
    norm_eps=None,
    eps_outside_sqrt=False,
):
    """Torch-native replacement for the TileLang public ``mhc_pre``."""
    import torch.nn.functional as F

    from sglang.kernels.ops.layernorm import mhc as mhc_module

    post_mix, comb_mix, layer_input = mhc_module._mhc_pre_torch(
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
    if norm_weight is not None:
        # Same spelling as ``exact_rms_norm`` (F.rms_norm, eps inside the sqrt) so the
        # fused path is bit-for-bit the non-fused ``exact_rms_norm(layer_input, norm)``.
        layer_input = F.rms_norm(
            layer_input, (layer_input.shape[-1],), norm_weight, norm_eps
        )
    return post_mix, comb_mix, layer_input


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.mhc_post",
    type=HookType.REPLACE,
)
def mhc_post_kunlun(x, residual, post_layer_mix, comb_res_mix):
    """Fused mHC post: ``out[n] = post[n]*x + sum_k comb[k,n]*residual[k]``.
    """
    post = (
        post_layer_mix.squeeze(-1)
        if post_layer_mix.dim() == residual.dim()
        else post_layer_mix
    )
    if x.shape[0] == 0:
        return torch.empty_like(residual)

    batch, n, d = residual.shape
    if n == 4 and x.is_cuda:
        try:
            out = torch.empty((batch, n, d), dtype=x.dtype, device=x.device)
            kunlun_ops.hc_post_kunlun_impl(
                post.contiguous(),
                x.contiguous(),
                comb_res_mix.contiguous(),
                residual.contiguous(),
                out,
                batch,
                n,
                d,
            )
            return out
        except Exception:
            pass

    # Memory-light torch fallback: out[.,j,:] = sum_i comb[.,i,j]*res[.,i,:] via
    # bmm (comb^T @ res), fp32 accumulate like the kernel; no [T,hc,hc,H] tensor.
    mixed = torch.bmm(comb_res_mix.transpose(1, 2).float(), residual.float())
    out = post.unsqueeze(-1).float() * x.unsqueeze(1).float() + mixed
    return out.type_as(x)


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc_head.fused_hc_head",
    type=HookType.REPLACE,
)
def fused_hc_head_kunlun(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    eps_outside_sqrt=False,
):
    """Torch-native replacement for the Triton ``fused_hc_head``.

    ERNIE 5.5's ``Ernie5_5HcHead.forward`` (mHC stream collapse) calls the public
    ``fused_hc_head`` (``mhc_head.py``), a Triton kernel (``_hc_head_kernel[grid]``)
    that is a bare stub on Kunlun. The module docstring carries the exact torch
    reference; this reproduces it, honoring ``eps_outside_sqrt`` (ERNIE passes
    True, i.e. ``1/(sqrt(mean_sq)+eps)`` like ``mhc_pre``; the DSV4 inline
    ``hc_head`` uses eps-inside ``rsqrt``). ``x`` is ``[T, hc_mult, hidden]`` and
    the output is ``[T, hidden]`` in x's dtype.
    """
    import torch.nn.functional as F

    shape = x.shape  # [T, hc_mult, hidden]
    dtype = x.dtype
    if shape[0] == 0:
        return x.new_empty((0, shape[-1]))
    xf = x.flatten(1).float()
    mean_sq = xf.square().mean(-1, keepdim=True)
    if eps_outside_sqrt:
        norm = 1.0 / (torch.sqrt(mean_sq) + norm_eps)
    else:
        norm = torch.rsqrt(mean_sq + norm_eps)
    mixes = F.linear(xf, hc_fn) * norm
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=1)
    return y.to(dtype)
