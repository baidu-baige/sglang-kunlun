"""Kunlun-backed DeepSeek V4 HiCache host<->device transfer kernels.

The Kunlun runtime exposes only the layer-first, full-page MLA transfer
operators (``torch.ops.xspeedgate_ops.transfer_kv_*``). These wrappers make
them importable from the Kunlun kernel layer; the HiCache hooks in
``hooks/mem_cache/kunlun_hicache.py`` call through here instead of invoking the
raw ops.
"""

from __future__ import annotations

import torch


def _ops():
    # The pre-import shim binds these functions before runtime extensions load.
    import xspeedgate_ops  # noqa: F401

    return torch.ops.xspeedgate_ops


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
    _ops().transfer_kv_all_layer_mla(
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
    """Use the H2D kernel for loads and a one-layer D2H table for backups."""
    ops = _ops()
    if src.device.type != "cpu":
        src_layers = torch.tensor([src.data_ptr()], dtype=torch.uint64, device=src.device)
        dst_layers = torch.tensor([dst.data_ptr()], dtype=torch.uint64, device=src.device)
        ops.transfer_kv_all_layer_mla(
            src_layers=src_layers, dst_layers=dst_layers,
            src_indices=src_indices, dst_indices=dst_indices,
            item_size=item_size, num_layers=1,
            block_quota=block_quota, num_warps_per_block=num_warps_per_block,
        )
        return
    ops.transfer_kv_per_layer_mla(
        src=src,
        dst=dst,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size,
        block_quota=block_quota,
        num_warps_per_block=num_warps_per_block,
    )
