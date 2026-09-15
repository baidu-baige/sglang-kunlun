"""Triton compatibility patches for the upstream MiniMax-M3 sparse modules.

Both patches must be applied *before* the upstream sparse modules are imported
(the Kunlun Triton runtime validates autotune keys while the decorator is
constructed, and upstream reads ``tl._experimental_make_tensor_descriptor`` at
import time), which is why ``sglang_kunlun.hooks.registry`` imports this package
before ``kernel_ops.install()`` and before the generic hook modules.
"""

from __future__ import annotations

from functools import wraps

_UPSTREAM_AUTOTUNE_KEY = [
    "BATCH_SIZE_BUCKET",
    "gqa_group_size",
    "head_dim",
    "block_size",
    "HAS_SINK",
]
_KUNLUN_AUTOTUNE_KEY = [
    "BATCH_SIZE_BUCKET",
    "gqa_group_size",
    "head_dim",
    "BLOCK_SIZE_N",
    "HAS_SINK",
]
_PATCHED_ATTR = "_sglang_kunlun_minimax_topk_sparse_autotune_patched"
_ORIGINAL_ATTR = "_sglang_kunlun_original_autotune"
_TMA_DESCRIPTOR_PATCHED_ATTR = "_sglang_kunlun_minimax_tma_descriptor_patched"


def patch_minimax_sparse_triton_compat() -> None:
    """Patch Triton language aliases before upstream MiniMax sparse modules import."""
    import triton.language as tl

    if getattr(tl, _TMA_DESCRIPTOR_PATCHED_ATTR, False):
        return

    if not hasattr(tl, "_experimental_make_tensor_descriptor"):
        tl._experimental_make_tensor_descriptor = getattr(
            tl, "make_tensor_descriptor", None
        )

    setattr(tl, _TMA_DESCRIPTOR_PATCHED_ATTR, True)


def patch_topk_sparse_autotune_key() -> None:
    """Patch Triton autotune before upstream MiniMax top-k sparse module imports.

    The current Kunlun Triton runtime validates autotune keys during decorator
    construction. Upstream uses the logical `block_size` key, but this kernel's
    constexpr argument is named `BLOCK_SIZE_N`, so importing the module fails
    before a post-import kernel patch can run. Keep the workaround narrow by
    rewriting only this exact key list.
    """
    import triton

    if getattr(triton.autotune, _PATCHED_ATTR, False):
        return

    original_autotune = triton.autotune

    @wraps(original_autotune)
    def kunlun_autotune(*args, **kwargs):
        if kwargs.get("key") == _UPSTREAM_AUTOTUNE_KEY:
            kwargs = dict(kwargs)
            kwargs["key"] = _KUNLUN_AUTOTUNE_KEY
        return original_autotune(*args, **kwargs)

    setattr(kunlun_autotune, _PATCHED_ATTR, True)
    setattr(kunlun_autotune, _ORIGINAL_ATTR, original_autotune)
    triton.autotune = kunlun_autotune


def apply_compat_patches() -> None:
    """Apply every import-order-sensitive MiniMax patch (idempotent)."""
    patch_minimax_sparse_triton_compat()
    patch_topk_sparse_autotune_key()
