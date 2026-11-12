"""Restore Kunlun MHC post at the module entry used by the 0.5.22 target model.

The old DecoderLayer.hc_post hook still handles the draft model. The target
model calls deepseek_v4_mhc.post directly. Keep the 0519 fused accumulation and
fp16 nan_to_num policy at this new entry; unsupported contracts use upstream.
"""
from __future__ import annotations

TARGET = "sglang.srt.models.deepseek_v4_mhc.post"
_vendor_post = None
_registered = False


def _supported(torch, cfg, x, residual, post_mix, comb):
    return (
        cfg.mult == 4
        and x.ndim == 2
        and x.shape[-1] > 0
        and x.dtype in (torch.float16, torch.bfloat16)
        and residual.dtype == x.dtype
        and post_mix.dtype == comb.dtype == torch.float32
        and residual.shape == (x.shape[0], 4, x.shape[-1])
        and post_mix.shape == (x.shape[0], 4)
        and comb.shape == (x.shape[0], 4, 4)
        and all(t.device == x.device for t in (residual, post_mix, comb))
        and all(t.layout == torch.strided for t in (x, residual, post_mix, comb))
        and not any(t.requires_grad for t in (x, residual, post_mix, comb))
    )


def mhc_post_kunlun(original_fn, cfg, x, residual, post_mix, comb):
    """Use the validated fused kernel, preserving the upstream fallback."""
    import torch

    if x.shape[0] == 0:
        return torch.empty(
            (0, cfg.mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )
    if (
        _vendor_post is None
        or x.device.type != "cuda"
        or not _supported(torch, cfg, x, residual, post_mix, comb)
    ):
        return original_fn(cfg, x, residual, post_mix, comb)

    out = torch.empty(
        (x.shape[0], cfg.mult, x.shape[-1]), dtype=x.dtype, device=x.device
    )
    _vendor_post(
        post_mix.contiguous(), x.contiguous(), comb.contiguous(),
        residual.contiguous(), out, x.shape[0], cfg.mult, x.shape[-1],
    )
    if out.dtype is torch.float16:
        out = torch.nan_to_num(out)
    return out


def register_hooks():
    """Resolve vendor at plugin startup, before model construction/capture."""
    global _vendor_post, _registered
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if _registered:
        return
    if TARGET in HookRegistry._patched:
        raise RuntimeError("Register before apply_hooks(); never hot-patch.")
    from kunlun_ops import hc_post_kunlun_impl

    _vendor_post = hc_post_kunlun_impl
    HookRegistry.register(TARGET, mhc_post_kunlun, HookType.AROUND)
    _registered = True
