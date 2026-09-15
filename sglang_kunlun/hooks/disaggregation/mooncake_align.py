# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun: 2MB-aligned re-allocation of DSV4 KV / indexer / compress-state buffers.

Kunlun RDMA (ibv_reg_mr) requires BOTH the MR start address AND length to be
2MB-aligned, else registration fails ("Invalid argument [22]"), breaking mooncake
KV transfer (PD disaggregation). Ported from image aiak_sglang patch
mem_cache/deepseekv4_memory_pool.py.

Approach: let ORIGINAL create/_alloc run (correct shape/dtype for every pool
variant), then swap each freshly-zeroed buffer for a 2MB-aligned buffer of the
SAME shape/dtype/device. CRITICAL: free the original buffer BEFORE allocating its
aligned replacement (and empty the XPU cache) so peak memory stays at +1 buffer,
not 2x the whole pool (that OOMs at mem-fraction 0.8+).
"""
from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

ALIGNMENT_2M = 2 * 1024 * 1024


def _is_kunlun() -> bool:
    """Return True when running on the Kunlun (XPU) platform."""
    return (
        os.environ.get("SGLANG_PLATFORM", "").lower() == "kunlun"
        or os.environ.get("SGLANG_USE_XPU") == "1"
    )


def _empty_cache():
    """Release cached device memory back to the allocator (XPU/CUDA)."""
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _alloc_2m_aligned(shape, dtype, device) -> torch.Tensor:
    """Allocate a zero buffer whose start address and length are 2MB-aligned."""
    numel = 1
    for s in shape:
        numel *= int(s)
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = numel * element_size
    aligned_bytes = (total_bytes + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M
    flat = torch.zeros(aligned_bytes + ALIGNMENT_2M, dtype=torch.uint8, device=device)
    offset = (ALIGNMENT_2M - flat.data_ptr() % ALIGNMENT_2M) % ALIGNMENT_2M
    return flat[offset : offset + total_bytes].view(dtype).reshape(tuple(shape))


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.create_buffer",
    type=HookType.AROUND,
)
def single_kv_create_buffer_kunlun(original_fn, self, *, num_pages):
    """2MB-align the DeepSeekV4SingleKVPool buffer for Kunlun RDMA."""
    buf = original_fn(self, num_pages=num_pages)
    if not _is_kunlun() or not isinstance(buf, torch.Tensor):
        return buf
    shape, dtype, device = tuple(buf.shape), buf.dtype, buf.device
    del buf  # original not stored anywhere yet -> freed here
    _empty_cache()
    return _alloc_2m_aligned(shape, dtype, device)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool._create_buffer",
    type=HookType.AROUND,
)
def indexer_create_buffer_kunlun(original_fn, self):
    """2MB-align the DeepSeekV4IndexerPool layer buffers for Kunlun RDMA."""
    original_fn(self)
    if not _is_kunlun():
        return
    bufs = getattr(self, "index_k_with_scale_buffer", None)
    if not (isinstance(bufs, list) and bufs and isinstance(bufs[0], torch.Tensor)):
        return
    for i in range(len(bufs)):
        b = bufs[i]
        shape, dtype, device = tuple(b.shape), b.dtype, b.device
        bufs[i] = None  # free original layer buffer before reallocating
        del b
        _empty_cache()
        bufs[i] = _alloc_2m_aligned(shape, dtype, device)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool._alloc_kv_score_buffer",
    type=HookType.AROUND,
)
def compress_state_alloc_kunlun(original_fn, self, **kwargs):
    """2MB-align the CompressStatePool kv_score buffer for Kunlun RDMA."""
    original_fn(self, **kwargs)
    if not _is_kunlun():
        return
    from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore

    ks = getattr(self, "kv_score_buffer", None)
    if ks is None or not isinstance(getattr(ks, "kv_score", None), torch.Tensor):
        return
    t = ks.kv_score
    shape, dtype, device = tuple(t.shape), t.dtype, t.device
    self.kv_score_buffer = None  # free original before reallocating
    del ks, t
    _empty_cache()
    self.kv_score_buffer = KVAndScore(_alloc_2m_aligned(shape, dtype, device))



@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4UnifiedKVPool.__init__",
    type=HookType.AROUND,
)
def unified_kv_init_kunlun(original_fn, self, *args, **kwargs):
    """2MB-align the DeepSeekV4UnifiedKVPool layer buffers for Kunlun RDMA."""
    original_fn(self, *args, **kwargs)
    if not _is_kunlun():
        return
    bufs = getattr(self, "kv_buffer", None)
    if not (isinstance(bufs, list) and bufs and isinstance(bufs[0], torch.Tensor)):
        return
    for i in range(len(bufs)):
        b = bufs[i]
        shape, dtype, device = tuple(b.shape), b.dtype, b.device
        bufs[i] = None
        del b
        _empty_cache()
        bufs[i] = _alloc_2m_aligned(shape, dtype, device)

logger.info(
    "sglang-kunlun: 2MB-aligned KV/indexer/compress-state realloc hooks registered"
)
