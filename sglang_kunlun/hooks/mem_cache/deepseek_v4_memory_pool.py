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
"""Hooks for ``sglang.srt.mem_cache.deepseek_v4_memory_pool``.

Three groups, all AROUND:

1. bf16 KV pool layout. Upstream ``DeepSeekV4SingleKVPool`` lays the KV cache out as
   fp8-584. Aligning to the AIAK float16 reference, ``get_bytes_per_token`` /
   ``create_buffer`` switch it to a bf16-512 layout (1024 bytes/token: nope 448 +
   rope 64, no fp8 scale) so the decode store can use the native,
   cuda-graph-capturable ``torch.ops.xspeedgate_ops.set_k_and_s_v4``. Only activates
   when ``store_dtype`` is bf16/fp16 (i.e. launched with ``--kv-cache-dtype
   bfloat16``); the fp8 path falls through to the original.

2. 2MB-aligned allocation of the KV / indexer / unified-KV buffers for Kunlun RDMA
   (see ``_align`` for why). For the pool variants whose layout we don't override, let
   ORIGINAL create/init run (correct shape/dtype for every variant), then swap each
   freshly-zeroed buffer for a 2MB-aligned buffer of the SAME shape/dtype/device.
   CRITICAL: free the original buffer BEFORE allocating its aligned replacement (and
   empty the XPU cache) so peak memory stays at +1 buffer, not 2x the whole pool (that
   OOMs at mem-fraction 0.8+). ``create_buffer_kunlun`` folds groups 1 and 2 into one
   hook so the bf16 path allocates the aligned buffer directly instead of allocating a
   pool-sized buffer only to free it again.

3. ``SGLANG_KUNLUN_KV_REG_DIAG=1`` buffer-info size histograms, used to debug
   mooncake MR registration failures.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from ._align import alloc_2m_aligned, empty_cache

logger = logging.getLogger(__name__)



def _is_bf16(self) -> bool:
    return getattr(self, "store_dtype", None) in (torch.bfloat16, torch.float16)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.get_bytes_per_token",
    type=HookType.AROUND,
)
def get_bytes_per_token_kunlun(original_fn, self):
    """bf16-512 layout: 512 bf16 elements per token (nope 448 + rope 64), no scale."""
    if _is_bf16(self):
        return 512
    return original_fn(self)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.create_buffer",
    type=HookType.AROUND,
)
def create_buffer_kunlun(original_fn, self, *, num_pages):
    """Allocate the KV pool 2MB-aligned, in the bf16-512 layout when store_dtype is bf16."""
    if _is_bf16(self):
        # Build the layout here rather than calling the original: its fp8-584 buffer
        # would be discarded anyway, and it is pool-sized.
        self.kv_cache_total_dim = 512
        self.bytes_per_page_padded = self.page_size * 512  # elements/page
        logger.info(
            "sglang-kunlun: DSV4 bf16 KV pool create_buffer page_size=%d "
            "elems_per_page=%d store_dtype=%s num_pages=%d",
            self.page_size, self.page_size * 512, self.store_dtype, num_pages,
        )
        return alloc_2m_aligned(
            (num_pages, self.page_size * 512), self.store_dtype, self.device
        )
    buf = original_fn(self, num_pages=num_pages)
    if not isinstance(buf, torch.Tensor):
        return buf
    shape, dtype, device = tuple(buf.shape), buf.dtype, buf.device
    del buf  # original not stored anywhere yet -> freed here
    empty_cache()
    return alloc_2m_aligned(shape, dtype, device)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool._create_buffer",
    type=HookType.AROUND,
)
def indexer_create_buffer_kunlun(original_fn, self):
    """2MB-align the DeepSeekV4IndexerPool layer buffers for Kunlun RDMA."""
    original_fn(self)
    bufs = getattr(self, "index_k_with_scale_buffer", None)
    if not (isinstance(bufs, list) and bufs and isinstance(bufs[0], torch.Tensor)):
        return
    for i in range(len(bufs)):
        b = bufs[i]
        shape, dtype, device = tuple(b.shape), b.dtype, b.device
        bufs[i] = None  # free original layer buffer before reallocating
        del b
        empty_cache()
        bufs[i] = alloc_2m_aligned(shape, dtype, device)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4UnifiedKVPool.__init__",
    type=HookType.AROUND,
)
def unified_kv_init_kunlun(original_fn, self, *args, **kwargs):
    """2MB-align the DeepSeekV4UnifiedKVPool layer buffers for Kunlun RDMA."""
    original_fn(self, *args, **kwargs)
    bufs = getattr(self, "kv_buffer", None)
    if not (isinstance(bufs, list) and bufs and isinstance(bufs[0], torch.Tensor)):
        return
    for i in range(len(bufs)):
        b = bufs[i]
        shape, dtype, device = tuple(b.shape), b.dtype, b.device
        bufs[i] = None
        del b
        empty_cache()
        bufs[i] = alloc_2m_aligned(shape, dtype, device)


def _log_infos(tag, lens):
    """Log a size histogram of the buffer lengths returned by a buf_infos call."""
    from collections import Counter

    logger.warning(
        "sglang-kunlun[bufinfo:%s]: n=%d sizes=%s", tag, len(lens), dict(Counter(lens)))


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool."
    "get_contiguous_buf_infos",
    type=HookType.AROUND,
)
def get_contiguous_buf_infos_kunlun(original_fn, self):
    """Diagnostic only (SGLANG_KUNLUN_KV_REG_DIAG=1): log KV buffer size histogram."""
    r = original_fn(self)
    if os.environ.get("SGLANG_KUNLUN_KV_REG_DIAG") == "1":
        _log_infos("contiguous", r[1])
    return r


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool."
    "get_state_buf_infos",
    type=HookType.AROUND,
)
def get_state_buf_infos_kunlun(original_fn, self):
    """Diagnostic only (SGLANG_KUNLUN_KV_REG_DIAG=1): log state buffer size histogram."""
    r = original_fn(self)
    if os.environ.get("SGLANG_KUNLUN_KV_REG_DIAG") == "1":
        _log_infos("state", r[1])
    return r
