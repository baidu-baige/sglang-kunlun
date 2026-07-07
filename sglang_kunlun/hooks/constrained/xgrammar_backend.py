"""Hooks for ``sglang.srt.constrained.xgrammar_backend``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/constrained/xgrammar_backend.py

REPLACE ``XGrammarGrammar.apply_vocab_mask`` with a torch-compile fallback that
avoids the upstream Triton/CUDA kernel (unsupported on Kunlun).
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


# adapted from
# https://github.com/mlc-ai/xgrammar/blob/v0.1.30/python/xgrammar/kernels/apply_token_bitmask_inplace_torch_compile.py#L7
@torch.compile(dynamic=True)
def _apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
    logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int
) -> None:
    mask_expanded = torch.repeat_interleave(bitmask, 32, dim=-1)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(
        bitmask.shape[-1]
    )
    bit_masks = (mask_expanded >> bit_indices) & 1
    bit_masks = bit_masks[..., :vocab_size]
    logits[..., :vocab_size] = logits[..., :vocab_size].masked_fill_(
        bit_masks == 0, float("-inf")
    )


@plugin_hook(
    "sglang.srt.constrained.xgrammar_backend.XGrammarGrammar.apply_vocab_mask",
    type=HookType.REPLACE,
)
def apply_vocab_mask_kunlun(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Kunlun: torch.compile-based bitmask path."""
    vocab_size = min(logits.shape[-1], vocab_mask.shape[-1] * 32)
    return _apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
        logits, vocab_mask, vocab_size
    )
