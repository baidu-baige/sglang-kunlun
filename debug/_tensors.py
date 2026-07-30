"""Tensor description helpers shared by the DSV4 debug probes.

Every helper here is a verbatim move of a former in-tree helper; the payload
they produce must stay byte-identical so dumps captured before and after the
refactor remain comparable.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch


def gather_cache_rows(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather KV-cache rows for ``indices``, clamping out-of-range entries."""
    if indices.numel() == 0:
        return cache.new_empty((*indices.shape, cache.shape[-1]))
    safe_indices = indices.reshape(-1).clamp(min=0, max=cache.shape[0] - 1).long()
    return cache.index_select(0, safe_indices).reshape(*indices.shape, cache.shape[-1])


def tensor_stats(tensor: torch.Tensor) -> dict:
    """Summarize value ranges and sentinel counts of a captured tensor."""
    flat = tensor.reshape(-1)
    stats = {
        "numel": flat.numel(),
        "zero_count": int((flat == 0).sum().item()) if flat.numel() else 0,
        "minus_one_count": int((flat == -1).sum().item()) if flat.numel() else 0,
        "negative_count": int((flat < 0).sum().item()) if flat.numel() else 0,
    }
    if flat.numel():
        stats.update(min=float(flat.min().item()), max=float(flat.max().item()))
    if tensor.is_floating_point():
        stats.update(
            nan_count=int(torch.isnan(flat).sum().item()),
            inf_count=int(torch.isinf(flat).sum().item()),
        )
    return stats


def tensor_layout(tensor: torch.Tensor) -> dict:
    """Describe the memory layout of a captured tensor."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def eager_tensor_metadata(tensor: torch.Tensor) -> dict:
    """Describe a tensor for the eager MTP dump (no storage offset)."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def ifeval_tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[dict]:
    """Return the JSONL-friendly fingerprint used by the IFEval MTP diagnosis."""
    if not isinstance(tensor, torch.Tensor):
        return None
    flat = tensor.detach().reshape(-1)
    cpu = flat.contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
        "numel": flat.numel(),
        "head": cpu[:16].tolist(),
        "tail": cpu[-16:].tolist() if cpu.numel() else [],
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }


def metadata_tensor_summary(value):
    """Return the compact head/tail summary used by the metadata logs."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return repr(value)
    flat = value.detach().cpu().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "head": flat[:8].tolist(),
        "tail": flat[-8:].tolist() if flat.numel() else [],
    }


def current_rank() -> int:
    """Return the distributed rank, falling back to the ``RANK`` env var."""
    import os

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))
