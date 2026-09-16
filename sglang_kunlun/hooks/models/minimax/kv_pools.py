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


class KunlunMiniMaxMHATokenToKVPool(KunlunMHATokenToKVPool):
    """MHA sub-pool for the MiniMax sparse KV path (main K/V + index K/V).

    Only this subclass knows how the MiniMax sparse path stores KV; the shared
    ``KunlunMHATokenToKVPool`` keeps its previous behaviour so no other model can
    be affected. The two differences:

    * the MiniMax sparse sub-pools are 3D NHD slot-row buffers at page_size > 1
      and the XPU ``reshape_and_cache`` only accepts the 4D paged layout, so the
      NHD store is a plain slot scatter instead of a kernel launch;
    * the write target can arrive as ``KVWriteLoc`` (0.5.19 backend contract),
      which the shared writer now unwraps too; unwrapped here so the layout
      branch below sees the plain tensor.

    Every other layout (padded 4D / HND) falls back to the shared writer.
    """

    #: Marker for the dense backend's paged-KV view gate (duck-typed, so the
    #: shared backend does not import this package).
    is_minimax_sparse_pool = True

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        """Store K/V, scattering NHD slot-row buffers instead of using the kernel."""
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

        loc, _, _ = unwrap_write_loc(loc_info)
        if self.head_dim == self.v_head_dim and not self.use_hnd:
            # NHD slot-row (MiniMax sparse sub-pools). dtype/scale handling is
            # identical to the shared writer; only the final store differs.
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k.div_(k_scale)
                if v_scale is not None:
                    cache_v.div_(v_scale)
                if not self.enable_int8_kv_cache:
                    cache_k = cache_k.to(self.dtype)
                    cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            if cache_k.dim() == 3 and cache_k.shape[0] == loc.numel():
                k_bhd = cache_k.contiguous()
                v_bhd = cache_v.contiguous()
            elif cache_k.dim() == 3 and cache_k.shape[1] == loc.numel():
                k_bhd = cache_k.permute(1, 0, 2).contiguous()
                v_bhd = cache_v.permute(1, 0, 2).contiguous()
            else:
                raise RuntimeError(
                    f"Unexpected K/V shape {tuple(cache_k.shape)} vs loc "
                    f"{tuple(loc.shape)}"
                )

            if layer_id_override is not None:
                layer_id = layer_id_override
            else:
                layer_id = layer.layer_id
            self.k_buffer[layer_id - self.start_layer][loc] = k_bhd
            self.v_buffer[layer_id - self.start_layer][loc] = v_bhd
            return

        return super().set_kv_buffer(
            layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override
        )


class KunlunMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse KV pool using Kunlun paged MHA sub-pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=KunlunMiniMaxMHATokenToKVPool,
            index_kv_pool_cls=KunlunMiniMaxMHATokenToKVPool,
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
