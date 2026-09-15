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
"""Hooks for ``sglang.srt.layers.attention.dsv4.quant_k_cache``.

One REPLACE hook on the CP-path SWA-KV fp8 pack kernel. The stock triton
implementation has no XPU backend ("0 compatible backends for target cuda"), so
mirror AIAK's ``nsa/quant_k_cache_v4.quant_to_nope_fp8_rope_bf16_pack_triton``:
when the KV cache dtype is bf16/fp16, skip fp8 quantization entirely and just
cast the input to the cache dtype (no separate rope split, no scale).

The pack it returns is the Kunlun ``NopeFp8RopeBf16Pack`` replacement from
``index_buf_accessor.py`` (the upstream dataclass rejects the None
rope/scale fields).

Registering here is enough for every consumer: ``apply_hooks`` setattr's the
defining module (so later ``from ... import
quant_to_nope_fp8_rope_bf16_pack_triton`` in ``compressor_v2`` picks it up) and
``_propagate_patch`` rewrites the already-bound module-level names in
``deepseek_v4_backend``, ``deepseek_v4_backend_hip_radix`` and ``compressor``.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .index_buf_accessor import NopeFp8RopeBf16PackKunlun

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.quant_k_cache."
    "quant_to_nope_fp8_rope_bf16_pack_triton",
    type=HookType.REPLACE,
)
def quant_to_nope_fp8_rope_bf16_pack_kunlun(k_bf16):
    """bf16/fp16 pass-through pack (no fp8 quant, no rope/scale split)."""
    _n, hidden = k_bf16.shape
    assert hidden == 512
    kv_cache_dtype = k_bf16.dtype
    if kv_cache_dtype in (torch.bfloat16, torch.float16):
        return NopeFp8RopeBf16PackKunlun(
            k_nope_fp8=k_bf16.to(kv_cache_dtype),
            k_rope_bf16=None,
            scale_k_nope_ue8m0=None,
        )
    raise NotImplementedError("Only support bf16/fp16 by now.")
