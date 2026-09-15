"""Kunlun MiniMax-VL model hooks."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.models.minimax_vl_common.MiniMaxVLVisionTransformer.forward",
    type=HookType.BEFORE,
)
def cast_minimax_vl_pixel_values_to_bfloat16(*args, **kwargs):
    """Cast MiniMax-VL pixel values to the dtype required by the vision tower."""
    if "pixel_values" in kwargs:
        pixel_values = kwargs["pixel_values"]
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            kwargs["pixel_values"] = pixel_values.to(dtype=torch.bfloat16)
        return args, kwargs

    if len(args) > 1:
        pixel_values = args[1]
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            args = list(args)
            args[1] = pixel_values.to(dtype=torch.bfloat16)
            args = tuple(args)

    return args, kwargs
