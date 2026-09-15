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
"""Hooks for ``sglang.srt.mem_cache.deepseek_v4_compress_state``.

2MB-aligned re-allocation of the CompressStatePool kv_score buffer for Kunlun RDMA
(see ``_align`` for why). Approach: let the ORIGINAL ``_alloc_kv_score_buffer`` run
(correct shape/dtype for every pool variant), then swap the freshly-zeroed buffer for
a 2MB-aligned buffer of the SAME shape/dtype/device. CRITICAL: free the original
buffer BEFORE allocating its aligned replacement (and empty the XPU cache) so peak
memory stays at +1 buffer, not 2x the whole pool (that OOMs at mem-fraction 0.8+).
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from ._align import alloc_2m_aligned, empty_cache, is_kunlun

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool."
    "_alloc_kv_score_buffer",
    type=HookType.AROUND,
)
def compress_state_alloc_kunlun(original_fn, self, **kwargs):
    """2MB-align the CompressStatePool kv_score buffer for Kunlun RDMA."""
    original_fn(self, **kwargs)
    if not is_kunlun():
        return
    from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore

    ks = getattr(self, "kv_score_buffer", None)
    if ks is None or not isinstance(getattr(ks, "kv_score", None), torch.Tensor):
        return
    t = ks.kv_score
    shape, dtype, device = tuple(t.shape), t.dtype, t.device
    self.kv_score_buffer = None  # free original before reallocating
    del ks, t
    empty_cache()
    self.kv_score_buffer = KVAndScore(alloc_2m_aligned(shape, dtype, device))
