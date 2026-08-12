"""Function-level hooks for NSA Triton kernel helpers."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from kunlun_ops import quant2d
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.kernels.ops.attention.dsa.triton_kernel.act_quant",
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

    x_shape = x.shape
    x_blocks = x.view(-1, block_size)
    quantized = torch.empty_like(x_blocks, dtype=torch.int8)
    scale = torch.empty(
        (x_blocks.shape[0], 1), device=x.device, dtype=torch.float32
    )
    quant2d(x, quantized, scale, force_sdnn=True)
    quantized = quantized.view(x_shape)
    scale = scale.view(x_shape[0], -1)
    return quantized, scale
