"""Kunlun KV pools for the MiniMax-M3 sparse attention path.

``KunlunMiniMaxSparseKVPool`` mirrors ``MiniMaxSparseKVPool`` but delegates to
Kunlun paged MHA sub-pools (``KunlunMHATokenToKVPool`` from the platform
factory). 0.5.19 made the sub-pool classes injectable constructor arguments
(the NPU line uses the same seam via ``NPUMiniMaxSparseKVPool``), so unlike the
0.5.13 plugin this needs no monkey-patching of the upstream class.

The K-only index sub-pool gets a synchronous host-copy API: upstream
``MHATokenToKOnlyPool`` stores only K and exposes no CPU-copy, while PD-decode
retraction offloads/reloads every sub-pool uniformly through the wrapper.
"""

from __future__ import annotations

from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MiniMaxSparseKVPool,
)
from sglang.srt.platforms import current_platform

from sglang_kunlun.hooks.mem_cache.kunlun_pools import KunlunMHATokenToKVPool


class KunlunMHATokenToKOnlyPool(MHATokenToKOnlyPool):
    """K-only index sub-pool (MiniMax block selector) with CPU offload.

    Copies are blocking (not ``non_blocking``) so the host tensors are fully
    materialized before the caller frees these slots — it does not rely on XPU
    ``synchronize()`` ordering an async copy stream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Not set by upstream MHATokenToKOnlyPool; the offload path reads it.
        self.cpu_offloading_chunk_size = 8192

    def get_cpu_copy(self, indices, mamba_indices=None):
        """Gather K-only slot rows at ``indices`` to CPU tensors, chunked."""
        current_platform.synchronize()
        k_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            k_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                k_cache_cpu[-1].append(k_cpu)
        current_platform.synchronize()
        return k_cache_cpu

    def load_cpu_copy(self, k_cache_cpu, indices, mamba_indices=None):
        """Scatter CPU K-only chunks back into slot rows at ``indices``."""
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = k_cache_cpu[layer_id][i // chunk_size]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[layer_id].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
        current_platform.synchronize()


class KunlunMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse KV pool using Kunlun paged MHA sub-pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=KunlunMHATokenToKVPool,
            index_kv_pool_cls=KunlunMHATokenToKVPool,
            index_k_pool_cls=KunlunMHATokenToKOnlyPool,
            **kwargs,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        """Offload KV of the given token slots to host memory.

        Used by PD-decode retraction (``Req.offload_kv_cache``). Fans out to
        every sub-pool (main K/V, index K/V, and the K-only block-selector
        index) and returns their host copies keyed by sub-pool. All per-token
        state must be offloaded: retract frees these slots and other requests
        immediately reuse them, so anything left behind would be restored as
        foreign data on resume, corrupting the context. MiniMax sparse
        attention has no mamba state, so ``mamba_indices`` is unused.
        """
        return {
            "main": self.main_pool.get_cpu_copy(indices),
            "index_kv": (
                self.index_kv_pool.get_cpu_copy(indices)
                if self.index_kv_pool is not None
                else None
            ),
            "index_k": (
                self.index_k_pool.get_cpu_copy(indices)
                if self.index_k_pool is not None
                else None
            ),
        }

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """Reload KV previously produced by ``get_cpu_copy`` back onto device."""
        self.main_pool.load_cpu_copy(kv_cache_cpu["main"], indices)
        if self.index_kv_pool is not None:
            self.index_kv_pool.load_cpu_copy(kv_cache_cpu["index_kv"], indices)
        if self.index_k_pool is not None:
            self.index_k_pool.load_cpu_copy(kv_cache_cpu["index_k"], indices)
