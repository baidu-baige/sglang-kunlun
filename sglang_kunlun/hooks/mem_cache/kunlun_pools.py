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
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)


class KunlunMHATokenToKVPool(MHATokenToKVPool):
    """Kunlun MHA KV pool with custom buffer creation and reshape-and-cache support."""

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
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Set KV cache buffer using the kunlun_ops reshape_and_cache kernel."""
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
        """Return contiguous buffer pointers, byte lengths, and item sizes."""
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
            self._get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices):
        """Copy KV cache pages from GPU to CPU in chunks."""
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

    def load_cpu_copy(self, kv_cache_cpu, indices):
        """Load KV cache pages from CPU back to GPU in chunks."""
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

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        """Set MLA KV buffer using the kunlun_ops triton kernel."""
        from kunlun_ops import set_mla_kv_buffer_triton

        layer_id = layer.layer_id
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


class KunlunNSATokenToKVPool(DSATokenToKVPool):
    """Kunlun DSA/NSA KV pool with int8 index buffer dtype."""
    index_k_with_scale_buffer_dtype = torch.int8
