"""Hooks for ``sglang.srt.layers.attention.dsv4.metadata``.
"""

from __future__ import annotations

import torch
import xspeedgate_ops

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.metadata.expand_index_page_table",
    type=HookType.AROUND,
)
def expand_index_page_table_kunlun(
    original_fn,
    page_table,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
):
    """Fused ``[bs, n] -> [bs, n * blocks_per_page]`` index-K block table.

    ``xspeedgate_ops.expand_index_page_table`` does on device what the upstream
    torch path spends three kernels on (``int64`` widen, broadcast-add an
    ``arange``, reshape + ``int32`` cast). The op's output shape is fixed by the
    host-side ``(bs, n, blocks_per_page)`` with no D2H, so it is graph-capturable
    like the original.
    """
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, (
        f"{full_page_size = } / {compress_ratio = } must be a multiple of "
        f"{index_page_size = }"
    )
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table

    return torch.ops.xspeedgate_ops.expand_index_page_table(
        page_table, 
        full_page_size, 
        compress_ratio, 
        index_page_size
    )
