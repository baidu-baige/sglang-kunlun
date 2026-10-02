"""Hooks for ``sglang.srt.constrained.xgrammar_backend``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/constrained/xgrammar_backend.py

REPLACE the bitmask application with a torch fallback that avoids the upstream
Triton/CUDA kernel (unsupported on Kunlun).

Both entry points need the hook, because they are reached by different features:

* ``XGrammarGrammar.apply_vocab_mask`` -- a per-request grammar object exists
  (JSON schema / regex / EBNF).
* ``XGrammarGrammarBackend.apply_vocab_mask`` -- a @staticmethod used when there
  is no inner grammar and only the token filter is active. ``ReasonerGrammarObject``
  takes exactly this path for ``--enable-strict-thinking`` (its ``apply_vocab_mask``
  property returns ``apply_vocab_mask_fn`` when ``self.grammar is None``), so
  without this second hook strict thinking dies in
  ``apply_token_bitmask_inplace_triton`` with
  "0 compatible backends for target (cuda)".
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


def _apply_vocab_mask_torch(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Shared body: keep one bitmask semantics for both entry points."""
    vocab_size = min(logits.shape[-1], vocab_mask.shape[-1] * 32)
    return _apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
        logits, vocab_mask, vocab_size
    )


@plugin_hook(
    "sglang.srt.constrained.xgrammar_backend.XGrammarGrammar.apply_vocab_mask",
    type=HookType.REPLACE,
)
def apply_vocab_mask_kunlun(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Kunlun: torch.compile-based bitmask path (per-request grammar object)."""
    return _apply_vocab_mask_torch(logits, vocab_mask)


@plugin_hook(
    "sglang.srt.constrained.xgrammar_backend.XGrammarGrammarBackend.apply_vocab_mask",
    type=HookType.REPLACE,
)
def apply_vocab_mask_backend_kunlun(
    logits: torch.Tensor, vocab_mask: torch.Tensor
) -> None:
    """Kunlun: same path for the backend @staticmethod (token-filter-only case).

    The registry restores the @staticmethod wrapper, so this takes no ``self``.
    """
    return _apply_vocab_mask_torch(logits, vocab_mask)
