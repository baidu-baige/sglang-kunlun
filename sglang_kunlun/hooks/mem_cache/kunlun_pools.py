"""Kunlun-specific KV pool subclasses.

These subclasses override methods that would otherwise need REPLACE hooks,
allowing the platform factory methods to return them directly so the main
sglang code never needs patching for these classes.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)

from .common import _alloc_2m_aligned
from sglang_kunlun.kernels.mixed_int8_kv import (
    mixed_int8_bytes_per_token,
    unpack_mixed_int8,
    write_mixed_int8_rows,
)


class KunlunMHATokenToKVPool(MHATokenToKVPool):
    """Kunlun MHA KV pool with custom buffer creation and reshape-and-cache support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Upstream's KV-copy kernels are CUDA-target Triton kernels and XPU has no
        # backend for them ("0 compatible backends for target (cuda)"), so move KV
        # rows through the pool's torch path. `_init_kv_copy_and_warmup` already ran
        # inside super().__init__; it only fills `_kv_copy_config`, which the native
        # branch of `_move_kv_cache_impl` does not read.
        self.use_native_move_kv_cache = True

    def _init_kv_copy_and_warmup(self):
        """Skip the tiled-copy warmup -- it JITs the CUDA-only Triton kernel."""
        self._kv_copy_config = None

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.head_dim == self.v_head_dim:
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.v_head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    num_pages = self.size // self.page_size
                    num_blocks_total = num_pages + 1
                    shape = (num_blocks_total, self.head_num, self.page_size, self.head_dim)
                    self.k_buffer = [
                        torch.zeros(shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer] + [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

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
        """Set KV cache buffer using the kunlun_ops reshape_and_cache kernel.

        Signature alignment only (behaviour is unchanged for every previous
        caller): 0.5.19 hands the write target down as ``KVWriteLoc`` at the
        backend boundary, so unwrap it first — inert for callers that pass a
        plain tensor, which are the only ones that worked.
        """
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

        loc, _, _ = unwrap_write_loc(loc_info)
        if dcp_kv_mask is not None:
            # Kunlun does not run decode-context-parallel; a masked write here
            # would silently drop tokens, so fail loudly if it ever appears.
            raise NotImplementedError(
                "KunlunMHATokenToKVPool.set_kv_buffer does not support dcp_kv_mask"
            )

        from kunlun_ops import reshape_and_cache

        if self.head_dim != self.v_head_dim:
            cache_v = F.pad(cache_v, pad=(0, 64), mode="constant", value=0)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
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
                f"Unexpected K/V shape {tuple(cache_k.shape)} vs loc {tuple(loc.shape)}"
            )

        k4 = self.k_buffer[layer_id - self.start_layer]
        v4 = self.v_buffer[layer_id - self.start_layer]
        reshape_and_cache(k_bhd, v_bhd, k4, v4, loc.to(torch.int32), None, None, 0)

    def get_contiguous_buf_infos(self):
        """Return contiguous buffer pointers, byte lengths, and item sizes.

        The PD transfer addresses the pool by *page* (upstream builds the same
        triple from ``_kv_buffer_descs`` via ``item_len_bytes(page_size)``), so
        ``item_lens`` must be the bytes of one page. The two layouts created in
        :meth:`_create_buffers` differ in what ``buffer[0]`` is:

        * 3D NHD slot-row (``head_dim == v_head_dim``, the MiniMax sparse path):
          ``buffer[0]`` is a single *token row*, so one page is ``page_size`` rows.
        * 4D paged (``head_dim != v_head_dim``): ``buffer[0]`` is already a page.

        Reporting the raw ``buffer[0].nbytes`` for the 3D layout made every PD
        transfer advance one token row per page index, so the decode side only
        received 1/``page_size`` of each page and generated garbage after the
        prefill-produced first token.
        """
        page_rows = 1 if self.head_dim != self.v_head_dim else self.page_size
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * page_rows
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * page_rows
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Copy KV cache pages from GPU to CPU in chunks.

        Signature alignment: 0.5.22's ``KVCache.get_cpu_copy`` takes
        ``(indices, mamba_indices, req_pool_index)`` and every allocator
        (``allocator/token.py``, ``allocator/paged.py``) forwards both extras
        **by keyword**, so an override missing them raises TypeError on the
        first retraction. Neither is this pool's business, exactly as in
        upstream ``MHATokenToKVPool.get_cpu_copy``: ``mamba_indices`` addresses
        the separate mamba state pool (no mamba state here) and
        ``req_pool_index`` is only read by the DSA compress-tail pool
        (``_get_compress_tail_cpu_copy``), which this pool does not have.
        """
        torch.cuda.synchronize()
        is_4d = self.head_dim != self.v_head_dim
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            k_buf = self.k_buffer[layer_id]
            v_buf = self.v_buffer[layer_id]
            if is_4d:
                k_flat = k_buf.reshape(-1, self.head_num, self.head_dim)
                v_flat = v_buf.reshape(-1, self.head_num, self.head_dim)
            else:
                k_flat = k_buf
                v_flat = v_buf
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = k_flat[chunk_indices].to("cpu", non_blocking=True)
                v_cpu = v_flat[chunk_indices].to("cpu", non_blocking=True)
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        """Load KV cache pages from CPU back to GPU in chunks.

        Same 0.5.22 signature alignment as :meth:`get_cpu_copy`; both extras are
        keyword-forwarded by the allocators and are inert for this pool.
        """
        torch.cuda.synchronize()
        is_4d = self.head_dim != self.v_head_dim
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            k_buf = self.k_buffer[layer_id]
            v_buf = self.v_buffer[layer_id]
            if is_4d:
                k_flat = k_buf.reshape(-1, self.head_num, self.head_dim)
                v_flat = v_buf.reshape(-1, self.head_num, self.head_dim)
            else:
                k_flat = k_buf
                v_flat = v_buf
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(k_flat.device, non_blocking=True)
                v_chunk = v_cpu.to(v_flat.device, non_blocking=True)
                k_flat[chunk_indices] = k_chunk
                v_flat[chunk_indices] = v_chunk
        torch.cuda.synchronize()


class KunlunMLATokenToKVPool(MLATokenToKVPool):
    """Kunlun MLA KV pool with custom KV buffer set/get operations."""

    def set_mla_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k_nope,
        cache_k_rope,
        layer_id_override: Optional[int] = None,
    ):
        """Set MLA KV buffer using the kunlun_ops triton kernel.

        0.5.22 renamed the write target ``loc`` -> ``loc_info`` and changed its
        type: the backends now build a ``KVWriteLoc`` dataclass
        (``KVWriteLoc.for_batch`` / ``for_layer`` in flashattention_backend,
        flashinfer_mla_backend, trtllm_mla_backend, ...), so unwrap it before
        touching the kernel. Inert for the callers that still hand over a plain
        tensor (``dsa_backend`` and our own NSA backend pass ``cache_loc``).

        ``layer_id_override`` is mirrored too. It is not new in 0.5.22 (0.5.19
        already had it) -- this override simply never carried it: SWA's hybrid
        pool calls the sub-pool with ``layer=None`` and
        ``layer_id_override=<pool-local layer id>``, so ``layer.layer_id`` may
        only be read when no override was given.
        """
        from kunlun_ops import set_mla_kv_buffer_triton

        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id],
            loc,
            cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
            cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
        )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        """Get MLA KV buffer using the xspeedgate_ops kernel."""
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        torch.ops.xspeedgate_ops.get_mla_kv_buffer(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        return cache_k_nope, cache_k_rope


class KunlunIndexKeyCache(IndexKeyCache):
    """``IndexKeyCache`` whose per-layer buffers start on a 2MiB boundary.

    Upstream allocates each layer's index-K(+scale) buffer with a plain
    ``torch.zeros``, so the base addresses are only 2MiB-aligned by accident
    (measured on DS V3.2 @ tp8: alternating 0 / +532992 across layers). The
    Kunlun peermem/XDR path maps device memory at 2MiB granularity, so an
    unaligned base makes ``ibv_reg_mr`` fail with EINVAL(22) once the buffer is
    registered for PD KV transfer -- and the disaggregation hook only rounds the
    registered *lengths* up, not the pointers.
    """

    def __init__(self, pool, index_buf_size: int):
        self.pool = pool
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        dtype = pool.index_k_with_scale_buffer_dtype
        itemsize = torch.tensor([], dtype=dtype).element_size()
        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.custom_mem_pool
            else nullcontext()
        ):
            self.buffer = []
            for layer_idx in range(pool.layer_num):
                shape = self._buffer_shape(self._layer_num_pages(layer_idx, num_pages))
                total_bytes = shape[0] * shape[1] * itemsize
                if total_bytes == 0:
                    # Layers that reuse the previous layer's top-k keep upstream's
                    # 0-row placeholder.
                    self.buffer.append(
                        torch.zeros(shape, dtype=dtype, device=pool.device)
                    )
                    continue
                self.buffer.append(
                    _alloc_2m_aligned(total_bytes, dtype, pool.device).reshape(shape)
                )


class KunlunNSATokenToKVPool(DSATokenToKVPool):
    """Kunlun DSA/NSA KV pool: int8 index buffer dtype, 2MiB-aligned buffers."""

    index_k_with_scale_buffer_dtype = torch.int8

    @property
    def mixed_int8_kv(self) -> bool:
        """Whether the latent rows are the packed mixed INT8 layout.

        True exactly when ``--kv-cache-dtype int8`` reached the pool (the value
        `configure_kv_cache_dtype_kunlun` maps it to). It is the same test the
        writer below and the sparse backend dispatch on, so there is one answer
        to "is this row a truncating cast or a quantized row".
        """
        return self.store_dtype is torch.int8

    def _create_buffers(self):
        """Allocate the MLA KV cache 2MiB-aligned (see KunlunIndexKeyCache).

        The int8 branch also fixes the row width, because upstream does not:
        ``MLATokenToKVPool.__init__`` only honours ``override_kv_cache_dim``
        for fp8 (``memory_pool.py:4793-4797``, gated on
        ``dsa_kv_cache_store_fp8``), so an int8 pool is built 576 bytes/token
        wide no matter what the width hook reports. 576 is exactly what keeps
        int8 KV unreadable: the row has no room for the per-group fp32 scales,
        so ``_write_mla_kv_buffer`` had nothing to quantize with and fell
        through to ``.to(self.dtype)`` -- a truncating cast to int8.

        656 B/token is `mixed_int8_bytes_per_token`, the same number the width
        hook hands to every sizing site and the same stride the writer packs
        against, so pool, sizer and row agree by construction.
        """
        if self.mixed_int8_kv:
            self.kv_cache_dim = mixed_int8_bytes_per_token(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.quant_block_size,
            )
            # One dense dequantization scratch per dtype, reused by every layer
            # (see `dequant_mixed_kv_table`). Allocated here rather than on
            # first use so that a capture never has to allocate it.
            self._mixed_kv_scratch = {}
        shape = (self.size + self.page_size, 1, self.kv_cache_dim)
        itemsize = torch.tensor([], dtype=self.store_dtype).element_size()
        total_bytes = shape[0] * shape[1] * shape[2] * itemsize
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # Slot 0 stays the dummy sink for padded tokens, as upstream.
                self.kv_buffer = [
                    _alloc_2m_aligned(
                        total_bytes, self.store_dtype, self.device
                    ).reshape(shape)
                    for _ in range(self.layer_num)
                ]

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        """Store one MLA write as packed mixed INT8 rows.

        0.5.17 reached this layout by hooking the DSV4 single-KV pool's
        accessor from outside (`nsa/index_buf_accessor_v4.py`); the DSA pool is
        ours (`platform/srt.py:get_dsa_kv_pool_cls`), so the same seam is the
        subclass method every ``set_mla_kv_buffer`` funnels through
        (``memory_pool.py:5026-5043``). One seam, both paths: every other dtype
        still gets upstream's fp8/half handling through ``super()``.

        No decode-context-parallel variant. Upstream resolves the DCP owner
        rule inside ``_scatter_mla_rows`` and refuses a resolved loc for DSA
        pools, but this override replaces the whole method, so that guard has to
        live here: under ``--dcp-size > 1`` one logical id spans
        ``attn_dcp_size`` rows while the quantizer writes exactly one, storing
        1/n of every token. Fail loudly instead.

        A pool that is *not* the mixed INT8 one takes
        :meth:`_write_mla_kv_buffer_concat`, which is 0.5.8's write for that
        case; see its docstring for why the two layouts cannot share a kernel.
        """
        if not self.mixed_int8_kv:
            if self.dsa_kv_cache_store_fp8 or self._write_loc_dcp_span != 1:
                # fp8 rows carry their own per-tile scales and DCP needs the
                # owner rule; both are upstream's to resolve.
                return super()._write_mla_kv_buffer(
                    dst_buffer, loc, cache_k_nope, cache_k_rope
                )
            return self._write_mla_kv_buffer_concat(
                dst_buffer, loc, cache_k_nope, cache_k_rope
            )
        if self._write_loc_dcp_span != 1:
            raise NotImplementedError(
                "KunlunNSATokenToKVPool._write_mla_kv_buffer has no DCP-aware "
                "variant: under --dcp-size > 1 the mixed INT8 quantizer would "
                "write one row per logical id instead of the replicated span."
            )
        write_mixed_int8_rows(
            cache_k_nope,
            cache_k_rope,
            dst_buffer.reshape(-1, self.kv_cache_dim),
            loc,
            self.quant_block_size,
        )

    def _write_mla_kv_buffer_concat(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        """Concat noPE+RoPE and scatter them with the vendor paged-write kernel.

        This is the 0.5.8 write for a non-INT8 MLA pool, argument for argument:
        ``sgl-kernel/python/sgl_kernel/patch/mem_cache/memory_pool.py:157-169``
        casts to ``self.dtype``, re-views to ``self.store_dtype``, flattens both
        halves to ``[N, dim]`` and calls ``kunlun_ops.set_mla_kv_buffer_triton
        (kv_buffer, loc, nope, rope)``. That wrapper
        (``kunlun_ops/_xflashinfer_cache.py:833-852``) is a thin alias for
        ``xpu_flash_ops.concat_and_cache_mla``, whose ``impl_concat_and_cache_mla
        <T_c, T_pe, TID, T_out>`` picks ``optimized_ops::concat_and_cache_mla_v2``
        -- i.e. ``xpuops3::infer_concat_and_cache_mla_v2`` -- for every dtype
        combination except the ones involving fp32 (disassembled: the branch is
        ``kv_c.dtype == float32 || kv_cache.dtype == float32`` -> the non-v2
        wrapper, else v2), and that host template is the *only* caller of the v2
        kernel in ``kunlun_ops/libapiinfer.so``.

        It replaces upstream's ``_scatter_mla_rows``
        (``memory_pool.py:4928-4940``), which routes to the Triton
        ``set_mla_kv_buffer_triton`` and therefore to
        ``mla_buffer.py:154 set_mla_kv_buffer_naive`` on a platform with no
        Triton -- the frame that asked for a 7.57 GiB transient during
        decode-graph capture and OOMed when the draft pool was still fp16
        (``hooks/mem_cache/common.py:60-78``).

        It is deliberately *not* reachable for the mixed INT8 row. This kernel
        concatenates and scatters; it neither quantizes nor emits the per-tile
        fp32 scales, and its int8 instantiation
        (``concat_and_cache_mla_v2<signed char, signed char, signed char, TID>``)
        wants both inputs already int8 and writes ``kv_lora_rank +
        qk_rope_head_dim`` = 576 bytes, not the 656-byte
        ``[int8 nope 512 | fp32 scales 16 | bf16 rope 128]`` row this pool is
        built with (``kernels/mixed_int8_kv.py``). The INT8 row's writer is
        ``quantize_mla_kv_cache_split``, which is what 0.5.8 used for it too
        (``patch/mem_cache/memory_pool.py:141-148``).
        """
        from kunlun_ops import set_mla_kv_buffer_triton

        nope, rope = cache_k_nope, cache_k_rope
        if nope.dtype != self.dtype:
            nope = nope.to(self.dtype)
            rope = rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            nope = nope.view(self.store_dtype)
            rope = rope.view(self.store_dtype)

        try:
            set_mla_kv_buffer_triton(
                dst_buffer,
                loc,
                nope.contiguous().view(nope.shape[0], nope.shape[-1]),
                rope.contiguous().view(rope.shape[0], rope.shape[-1]),
            )
        except Exception:
            # Upstream's scatter is the documented fallback: slower and a
            # different kernel, but it writes the same rows. It is handed the
            # *original* tensors, because it redoes the dtype coercion above and
            # a second `.view(store_dtype)` would reinterpret the bytes.
            super()._write_mla_kv_buffer(
                dst_buffer, loc, cache_k_nope, cache_k_rope
            )

    def mixed_int8_kv_rows(self, layer_id: int) -> torch.Tensor:
        """One layer's packed rows as ``[slots, kv_cache_dim]`` int8.

        Single definition of "the packed table", shared by the dequantizing
        reader below and by the native mixed-cache attention in
        `kunlun_nsa_backend.py`: the vendor op takes exactly this tensor as its
        ``kv_cache`` (``{kv_token_num, per_token_bytes}``, int8 -- see
        ``kunlun_ops.dsa_prefill_with_mixed_cache``'s own docstring), so the two
        readers cannot drift on the stride or on which accessor yields the
        slot-ordered table.
        """
        return self.get_key_buffer(layer_id).reshape(-1, self.kv_cache_dim)

    def dequant_mixed_kv_table(
        self, layer_id: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Dense ``[slots, kv_lora_rank + qk_rope_head_dim]`` view of one layer.

        `xspeedgate_ops.sparse_attn_fwd` has no scale argument and no int8
        instantiation (schema ``(q, kv, indices, sm_scale, d_v, topk_length,
        attn_sink=None)``, requires ``kv.dtype == q.dtype``), so the packed rows
        have to be dequantized before the call -- unlike 0.5.8/0.5.17, whose
        int8 attention took the row natively (`sparse_mixed_attn_int8` /
        `hybrid_attention_with_mixed_cache`) with the scales read from inside
        it. This is the *fallback* reader now: the sparse backend calls that
        native op first (`kunlun_nsa_backend.py`, `_native_mixed_attn`) and only
        lands here when it is unavailable or rejects the call.

        Cost per layer per forward, at 77k slots: 656 B read + 1 KiB written
        per slot. The fp16 pool path already hands that op the whole table, so
        this is the dense row's own traffic again rather than a new order of
        it -- and it is what buys the mixed row's half-size pool.

        The returned tensor is the pool's scratch, reused by the next layer, so
        it is only valid until the next call. That is deliberate: the table is
        a full copy of one layer's pool (77k slots x 576 halves = 88 MB), and a
        fresh allocation per call would be captured once per CUDA-graph bucket
        -- ~10-20 copies, which would cost more than the int8 row saves. One
        shared buffer is also what 0.5.17 did for its DSV4 mixed-cache scratch
        (`_mixed_cache_l3_buffer`, "shared per (device, stream)").
        """
        rows = self.mixed_int8_kv_rows(layer_id)
        num_slots = rows.shape[0]
        scratch = self._mixed_kv_scratch.get(dtype)
        if scratch is None or scratch.shape[0] != num_slots:
            scratch = torch.empty(
                (num_slots, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=dtype,
                device=rows.device,
            )
            self._mixed_kv_scratch[dtype] = scratch
        return unpack_mixed_int8(
            rows,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.quant_block_size,
            dtype,
            scratch,
        )

    def _create_index_key_cache(self) -> IndexKeyCache:
        return KunlunIndexKeyCache(self, self.index_buf_size)
