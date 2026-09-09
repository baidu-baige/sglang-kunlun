"""Kunlun DeepSeek V4 HiCache transfer hooks.

The Kunlun runtime currently exposes only the layer-first, full-page MLA
transfer operators.  Keep the upstream pool allocation and orchestration, but
replace the V4 transfer boundary so that it cannot fall back to the CUDA
token-granular ``transfer_cache_dsv4_mla`` implementation.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

# Importing the extension registers the ``torch.ops.xspeedgate_ops`` namespace.
import xspeedgate_ops  # noqa: F401


def _require_full_page_transfer(pool, host_indices, device_indices) -> None:
    if host_indices is None or device_indices is None:
        return

    host_numel = host_indices.numel()
    device_numel = device_indices.numel()
    if host_numel != device_numel:
        raise ValueError(
            f"{pool.pool_name} transfer index size mismatch: "
            f"host={host_numel}, device={device_numel}"
        )

    page_size = _pool_page_size(pool)
    if host_numel % page_size != 0 or device_numel % page_size != 0:
        raise ValueError(
            f"{pool.pool_name} DeepSeek V4 HiCache only supports full-page "
            f"transfer: host_numel={host_numel}, device_numel={device_numel}, "
            f"slot_page_size={page_size}"
        )


def _page_rows(indices: torch.Tensor, page_size: int) -> torch.Tensor:
    return indices.reshape(-1, page_size)[:, 0] // page_size


def _pool_page_size(pool) -> int:
    return pool.slot_page_size if hasattr(pool, "slot_page_size") else pool.swa_page_size


def _pool_item_bytes(pool) -> int:
    return pool.item_bytes if hasattr(pool, "item_bytes") else pool.state_page_bytes


def _require_supported_layout(pool, io_backend: str) -> None:
    if io_backend != "kernel" or pool.layout != "layer_first":
        raise ValueError(
            "Kunlun DeepSeek V4 HiCache only supports "
            f"layer_first/kernel, got {pool.layout}/{io_backend}"
        )


def _transfer_all_layer(pool, host_indices, device_indices, io_backend: str) -> None:
    _require_full_page_transfer(pool, host_indices, device_indices)
    if host_indices is None or host_indices.numel() == 0:
        return
    _require_supported_layout(pool, io_backend)
    page_size = _pool_page_size(pool)
    torch.ops.xspeedgate_ops.transfer_kv_all_layer_mla(
        src_layers=pool.device_ptrs,
        dst_layers=pool.data_ptrs,
        src_indices=_page_rows(device_indices, page_size),
        dst_indices=_page_rows(host_indices, page_size),
        item_size=_pool_item_bytes(pool),
        num_layers=pool.layer_num,
        block_quota=2,
        num_warps_per_block=32,
    )


def _transfer_per_layer(
    pool, host_indices, device_indices, layer_id, io_backend: str
) -> None:
    _require_full_page_transfer(pool, host_indices, device_indices)
    if host_indices is None or host_indices.numel() == 0:
        return
    _require_supported_layout(pool, io_backend)
    page_size = _pool_page_size(pool)
    torch.ops.xspeedgate_ops.transfer_kv_per_layer_mla(
        src=pool.data_refs[layer_id],
        dst=pool.device_buffers[layer_id]
        if hasattr(pool, "device_buffers")
        else pool.device_page_views[layer_id],
        src_indices=_page_rows(host_indices, page_size),
        dst_indices=_page_rows(device_indices, page_size),
        item_size=_pool_item_bytes(pool),
        block_quota=2,
        num_warps_per_block=32,
    )


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.DeepSeekV4PagedHostPool."
    "backup_from_device_all_layer",
    type=HookType.AROUND,
)
def dsv4_paged_backup_kunlun(
    original_fn, self, device_pool, host_indices, device_indices, io_backend
):
    """Back up all layers of the DSV4 paged host pool with the Kunlun kernel."""
    _transfer_all_layer(self, host_indices, device_indices, io_backend)


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.DeepSeekV4PagedHostPool."
    "load_to_device_per_layer",
    type=HookType.AROUND,
)
def dsv4_paged_load_kunlun(
    original_fn, self, device_pool, host_indices, device_indices, layer_id, io_backend
):
    """Load one layer of the DSV4 paged host pool with the Kunlun kernel."""
    _transfer_per_layer(self, host_indices, device_indices, layer_id, io_backend)


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.DeepSeekV4StateHostPool."
    "backup_from_device_all_layer",
    type=HookType.AROUND,
)
def dsv4_state_backup_kunlun(
    original_fn, self, device_pool, host_indices, device_indices, io_backend
):
    """Back up all layers of the DSV4 state host pool with the Kunlun kernel."""
    _transfer_all_layer(self, host_indices, device_indices, io_backend)


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.DeepSeekV4StateHostPool."
    "load_to_device_per_layer",
    type=HookType.AROUND,
)
def dsv4_state_load_kunlun(
    original_fn, self, device_pool, host_indices, device_indices, layer_id, io_backend
):
    """Load one layer of the DSV4 state host pool with the Kunlun kernel."""
    _transfer_per_layer(self, host_indices, device_indices, layer_id, io_backend)
