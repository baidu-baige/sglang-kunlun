"""Kunlun method hook for the upstream DeepSeek V4 NextN model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ModelNextN.hc_head",
    type=HookType.REPLACE,
)
def hc_head_kunlun(
    self,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
):
    """Match the Golden explicit FP32 MHC-head sequence."""
    if x.numel() == 0:
        return x.new_empty((0, x.shape[-1]))
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
