"""Allocator probes moved out of ``sglang_kunlun/hooks/mem_cache/kunlun_allocator.py``.

Activation (``DSV4_ALLOC_EXTEND_PROBE_DIR``), payload keys and the
``0514-call<NNNN>.pt`` file naming are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ._dispatch import SiteRegistry

_CALL_INDEX = 0


def alloc_extend_begin(
    allocator,
    alloc_fn,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
):
    """Capture the alloc-extend inputs; returns a context or ``None``."""
    probe_dir = os.environ.get("DSV4_ALLOC_EXTEND_PROBE_DIR")
    if (
        probe_dir is None
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() != 0
    ):
        return None
    return {
        "probe_dir": probe_dir,
        "payload": {
            "version": "0514",
            "allocator_id": id(allocator),
            "alloc_fn": alloc_fn.__name__,
            "page_size": allocator.page_size,
            "extend_num_tokens": extend_num_tokens,
            "prefix_lens": prefix_lens.detach().cpu(),
            "prefix_lens_cpu": prefix_lens_cpu.detach().cpu(),
            "prefix_lens_dtype": str(prefix_lens.dtype),
            "prefix_lens_stride": tuple(prefix_lens.stride()),
            "seq_lens": seq_lens.detach().cpu(),
            "seq_lens_cpu": seq_lens_cpu.detach().cpu(),
            "seq_lens_dtype": str(seq_lens.dtype),
            "seq_lens_stride": tuple(seq_lens.stride()),
            "last_loc": last_loc.detach().cpu(),
            "last_loc_dtype": str(last_loc.dtype),
            "last_loc_shape": tuple(last_loc.shape),
            "last_loc_stride": tuple(last_loc.stride()),
            "last_loc_is_contiguous": last_loc.is_contiguous(),
            "free_pages_before": allocator.free_pages[:256].detach().cpu(),
        },
    }


def alloc_extend_end(
    context, out_indices: torch.Tensor, merged_value: int, num_new_pages: int
) -> None:
    """Persist the alloc-extend probe once the kernel has produced its output."""
    if context is None:
        return
    global _CALL_INDEX
    payload = context["payload"]
    payload.update(
        {
            "out_indices": out_indices.detach().cpu(),
            "merged_value": merged_value,
            "num_new_pages": num_new_pages,
        }
    )
    torch.save(payload, Path(context["probe_dir"]) / f"0514-call{_CALL_INDEX:04d}.pt")
    _CALL_INDEX += 1


_SITES = SiteRegistry()
capture = _SITES.capture


@_SITES.site("alloc_extend.begin")
def _site_alloc_extend_begin(scope) -> None:
    _SITES.set_context(
        "alloc_extend",
        alloc_extend_begin(
            scope["self"],
            scope["alloc_fn"],
            scope["prefix_lens"],
            scope["prefix_lens_cpu"],
            scope["seq_lens"],
            scope["seq_lens_cpu"],
            scope["last_loc"],
            scope["extend_num_tokens"],
        ),
    )


@_SITES.site("alloc_extend.end")
def _site_alloc_extend_end(scope) -> None:
    alloc_extend_end(
        _SITES.get_context("alloc_extend"),
        scope["out_indices"],
        scope["merged_value"],
        scope["num_new_pages"],
    )
