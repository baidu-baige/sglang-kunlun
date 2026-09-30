"""Kunlun fallbacks for Kimi-K3 module-level GEMMs."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.models.kimi_k3._k3_bf16_gemm",
    type=HookType.REPLACE,
)
def k3_bf16_gemm_kunlun(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Keep the merged MoE front in bf16 on Kunlun."""
    if out is None:
        return torch.nn.functional.linear(x, weight)
    return torch.mm(x, weight.t(), out=out)
