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
"""Shared 2MB-alignment helpers for the DSV4 KV / indexer / compress-state pools.

Kunlun RDMA (``ibv_reg_mr``) requires BOTH the MR start address AND length to be
2MB-aligned, else registration fails ("Invalid argument [22]"), breaking mooncake
KV transfer (PD disaggregation). Ported from image aiak_sglang patch
mem_cache/deepseekv4_memory_pool.py.

Not a mirror of an upstream module (hence the leading underscore); the hooks that
use these helpers live in the modules mirroring their patch targets:
``deepseek_v4_memory_pool.py`` and ``deepseek_v4_compress_state.py``.
"""

from __future__ import annotations

import os

import torch

ALIGNMENT_2M = 2 * 1024 * 1024


def is_kunlun() -> bool:
    """Return True when running on the Kunlun (XPU) platform."""
    return (
        os.environ.get("SGLANG_PLATFORM", "").lower() == "kunlun"
        or os.environ.get("SGLANG_USE_XPU") == "1"
    )


def empty_cache():
    """Release cached device memory back to the allocator (XPU/CUDA)."""
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def alloc_2m_aligned(shape, dtype, device) -> torch.Tensor:
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
