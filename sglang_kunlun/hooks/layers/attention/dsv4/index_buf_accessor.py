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
"""Hooks for ``sglang.srt.layers.attention.dsv4.index_buf_accessor``.

Two REPLACE hooks that make the SWA-KV pack/store path work with a bf16/fp16 KV
cache on Kunlun:

1. ``NopeFp8RopeBf16Pack`` (class REPLACE). The upstream dataclass has no
   ``kv_cache_dtype`` field and its ``__post_init__`` / consumers hard-assume real
   ``k_rope_bf16`` / ``scale_k_nope_ue8m0`` tensors (no None support). Under a
   bf16/fp16 cache the Kunlun pack (see ``quant_k_cache.py``) skips fp8
   quantization entirely and leaves both None, so the dataclass gains
   ``kv_cache_dtype`` plus dtype-conditional asserts. Mirrors aiak_sglang's
   ``index_buf_accessor_v4.NopeFp8RopeBf16Pack``.

2. ``_set_k_and_s_triton`` (function REPLACE). The triton store splits
   nope/rope/scale; with the half-precision pack there is nothing to split, so
   write the 512-wide bf16 tensor straight through the native, cuda-graph
   capturable ``torch.ops.xspeedgate_ops.set_k_and_s_v4``.

Class REPLACE preserves type identity (the registry substitutes the class
object directly) and ``_propagate_patch`` fixes the stale
``from ... import NopeFp8RopeBf16Pack`` bindings in ``quant_k_cache`` /
``deepseek_v4_memory_pool``.
"""

from __future__ import annotations

import dataclasses
import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.index_buf_accessor.NopeFp8RopeBf16Pack",
    type=HookType.REPLACE,
)
@dataclasses.dataclass
class NopeFp8RopeBf16PackKunlun:
    """Pack that also supports a bf16/fp16 KV cache (no rope split, no scale)."""

    k_nope_fp8: torch.Tensor
    k_rope_bf16: torch.Tensor
    scale_k_nope_ue8m0: torch.Tensor
    kv_cache_dtype: torch.dtype = None

    def __post_init__(self):
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = self.k_nope_fp8.dtype
        if self.kv_cache_dtype in (torch.bfloat16, torch.float16):
            assert self.k_nope_fp8.shape[-1] == 448 + 64
            assert self.k_rope_bf16 is None
            assert self.scale_k_nope_ue8m0 is None
            assert self.k_nope_fp8.dtype is self.kv_cache_dtype
        else:
            assert self.k_nope_fp8.shape[-1] == 448
            assert self.k_rope_bf16.shape[-1] == 64
            assert self.scale_k_nope_ue8m0.shape[-1] == 7

    def slice_pack(self, _slice):
        """Return a new pack containing only the rows selected by ``_slice``."""
        is_half_cache = self.kv_cache_dtype in (torch.bfloat16, torch.float16)
        return NopeFp8RopeBf16PackKunlun(
            k_nope_fp8=self.k_nope_fp8[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice] if not is_half_cache else None,
            scale_k_nope_ue8m0=(
                self.scale_k_nope_ue8m0[_slice] if not is_half_cache else None
            ),
            kv_cache_dtype=self.kv_cache_dtype,
        )


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.index_buf_accessor._set_k_and_s_triton",
    type=HookType.REPLACE,
)
def set_k_and_s_triton_kunlun(buf, loc, nope_fp8_rope_bf16_pack, page_size):
    """Store the half-precision pack via the kunlun fused ``set_k_and_s_v4`` op."""
    k_val = nope_fp8_rope_bf16_pack.k_nope_fp8
    assert k_val.dtype in (torch.bfloat16, torch.float16), \
        f"in this kernel, kv dtype is bf16/fp16, got {k_val.dtype}"
    # TEMP FIX: the pack doesn't receive the real kv_cache_dtype (aiak_sglang's
    # quant_to_nope_fp8_rope_bf16_pack_triton gets it from
    # token_to_kv_pool.store_dtype; our pack infers it from the input tensor's
    # own dtype, which is wrong when the input is bf16 but the cache is fp16).
    # Force-cast here to match buf's dtype until kv_cache_dtype is plumbed
    # through the pack.
    if k_val.dtype != buf.dtype:
        k_val = k_val.to(buf.dtype)
    num_pages = buf.shape[0]
    max_valid_loc = num_pages * page_size - 1
    loc_safe = loc.clamp(min=0, max=max_valid_loc) if loc.numel() > 0 else loc

    torch.ops.xspeedgate_ops.set_k_and_s_v4(buf, loc_safe, k_val, page_size)
