"""Kunlun-backed DeepSeek V4 HiCache host<->device transfer kernels.

The Kunlun runtime exposes only the layer-first, full-page MLA transfer
operators (``torch.ops.xspeedgate_ops.transfer_kv_*``). These wrappers make
them importable from the Kunlun kernel layer; the HiCache hooks in
``hooks/mem_cache/kunlun_hicache.py`` call through here instead of invoking the
raw ops.
"""

from __future__ import annotations

import torch

# Importing the extension registers the ``torch.ops.xspeedgate_ops`` namespace.
import xspeedgate_ops  # noqa: F401


def transfer_kv_all_layer_mla(
    src_layers,
    dst_layers,
    src_indices,
    dst_indices,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    """Back up all layers of a DSV4 MLA pool from device to host."""
    torch.ops.xspeedgate_ops.transfer_kv_all_layer_mla(
        src_layers=src_layers,
        dst_layers=dst_layers,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size,
        num_layers=num_layers,
        block_quota=block_quota,
        num_warps_per_block=num_warps_per_block,
    )


def transfer_kv_per_layer_mla(
    src,
    dst,
    src_indices,
    dst_indices,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    """Move one layer of a DSV4 MLA pool between host and device."""
    torch.ops.xspeedgate_ops.transfer_kv_per_layer_mla(
        src=src,
        dst=dst,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size,
        block_quota=block_quota,
        num_warps_per_block=num_warps_per_block,
    )
