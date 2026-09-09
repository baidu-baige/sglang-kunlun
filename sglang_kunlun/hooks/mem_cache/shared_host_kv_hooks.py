"""Hook registration for shared DeepSeek V4 host KV storage."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_host_register,
)

from . import shared_host_kv


def _configure_dsv4(params, kvcache) -> None:
    if not shared_host_kv.is_enabled():
        return
    if "DeepSeekV4" not in type(kvcache).__name__:
        raise RuntimeError(
            "HICACHE_SHARED_HOST_KV is only supported for DeepSeek V4"
        )
    group = params.tp_cache_group
    rank = torch.distributed.get_rank(group=group) if group is not None else 0
    shared_host_kv.configure(rank, group)


@plugin_hook(
    "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
    "build_deepseek_v4_hicache_stack",
    type=HookType.AROUND,
)
def configure_dsv4_hicache(original_fn, *args, **kwargs):
    """Configure shared DSV4 host KV before the HiCache pool stack is built."""
    params = kwargs.get("params")
    kvcache = kwargs.get("kvcache")
    if params is None or kvcache is None:
        raise RuntimeError(
            "Unable to configure shared DSV4 host KV: missing params or kvcache"
        )
    _configure_dsv4(params, kvcache)
    return original_fn(*args, **kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.hiradix_cache.HiRadixCache.__init__",
    type=HookType.AROUND,
)
def configure_dsv4_hiradix(original_fn, self, *args, **kwargs):
    """Configure shared DSV4 host KV lazily when HiRadixCache is constructed."""
    params = kwargs.get("params", args[0] if args else None)
    if shared_host_kv.is_enabled() and not shared_host_kv.active():
        if params is None:
            raise RuntimeError(
                "Unable to configure shared DSV4 host KV: missing params"
            )
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if "DeepSeekV4" in type(kvcache).__name__:
            _configure_dsv4(params, kvcache)
    return original_fn(self, *args, **kwargs)


_ORIGINAL_ALLOC_WITH_HOST_REGISTER = alloc_with_host_register


def alloc_with_shared_host_kv(dims, *, dtype, device, pin_memory, allocator):
    """Allocate from shared host KV when active, else fall back to the original."""
    shared = shared_host_kv.maybe_shared_alloc(dims, dtype, device)
    if shared is not None:
        return shared
    return _ORIGINAL_ALLOC_WITH_HOST_REGISTER(
        dims,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        allocator=allocator,
    )


# Host pool modules use the defaultdict's factory rather than looking up the
# function by name at call time.  Replacing the function symbol alone would
# therefore leave the already-imported allocator path unchanged.
ALLOC_MEMORY_FUNCS.default_factory = lambda: alloc_with_shared_host_kv


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.HostPoolGroup."
    "backup_from_device_all_layer",
    type=HookType.AROUND,
)
def dsv4_shared_backup(original_fn, self, *args, **kwargs):
    """Skip host writes on non-leader TP ranks sharing the same host KV."""
    if shared_host_kv.skip_write():
        return None
    return original_fn(self, *args, **kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller."
    "HybridCacheController.start_writing",
    type=HookType.AROUND,
)
def verify_shared_indices(original_fn, self, *args, **kwargs):
    """Verify queued host indices agree across TP ranks before writing."""
    if shared_host_kv.active() and shared_host_kv.VERIFY:
        for operation in self.write_queue:
            shared_host_kv.verify_indices(operation.host_indices)
    return original_fn(self, *args, **kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.HostPoolGroup.destroy",
    type=HookType.AROUND,
)
def shutdown_shared_host_kv(original_fn, self, *args, **kwargs):
    """Release shared host KV mappings when the host pool group is destroyed."""
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        shared_host_kv.shutdown()
