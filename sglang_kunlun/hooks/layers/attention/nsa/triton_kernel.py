"""Function-level hooks for NSA Triton kernel helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.triton_kernel.act_quant",
    type=HookType.REPLACE,
)
def act_quant_kunlun(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations into FP8-compatible int8 blocks for Kunlun."""
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    x_blocks = x.view(-1, block_size)
    scale = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    fp8_tensor = (x_blocks / scale * 448.0).to(torch.int8).view(x.shape)
    return fp8_tensor, scale.view(x.shape[0], -1).to(torch.float32)
