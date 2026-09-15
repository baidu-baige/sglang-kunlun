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
"""Hooks for ``sglang.jit_kernel.dsv4.attn``.

One AROUND hook on ``fused_store_cache``. With
``SGLANG_OPT_USE_FUSED_STORE_CACHE``, ``store_cache`` routes
``set_swa_key_buffer_radix_fused`` -> ``SingleKVPool.set_key_buffer_fused`` ->
``sglang.jit_kernel.dsv4.fused_store_cache``, a tvm/tilelang JIT kernel with no
XPU backend. Under a bf16/fp16 KV cache, replace it with the graph-safe kunlun
fused store ``torch.ops.xspeedgate_ops.set_k_and_s_v4`` (the same op the non-CP
path uses), which avoids the fp8 pack entirely. Any other cache dtype falls
through to the original JIT kernel.

The target is the defining module (``...dsv4.attn``); the registry's
``_propagate_patch`` fixes the ``sglang.jit_kernel.dsv4`` package re-export and
the ``from sglang.jit_kernel.dsv4 import fused_store_cache`` binding in
``sglang.srt.mem_cache.deepseek_v4_memory_pool``.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.jit_kernel.dsv4.attn.fused_store_cache",
    type=HookType.AROUND,
)
def fused_store_cache_kunlun(
    original_fn, input, cache, indices, page_size, type="flashmla"
):
    """bf16/fp16 SWA fused store via ``set_k_and_s_v4``; else the JIT kernel."""
    if cache.dtype in (torch.bfloat16, torch.float16):
        kv_out = input.to(cache.dtype).contiguous()
        npages = cache.shape[0]
        loc = indices.to(torch.int64).clamp_(0, npages * page_size - 1)
        torch.ops.xspeedgate_ops.set_k_and_s_v4(cache, loc, kv_out, page_size)
        return
    return original_fn(
        input=input, cache=cache, indices=indices, page_size=page_size, type=type
    )
