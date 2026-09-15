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
"""Function-level hooks for NSA index buffer accessors."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.index_buf_accessor._get_k_triton",
    type=HookType.REPLACE,
)
def _get_k_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
) -> torch.Tensor:
    _, buf_numel_per_page = buf.shape
    return torch.ops.xspeedgate_ops.get_k_kernel(
        buf.contiguous(),
        page_indices.contiguous(),
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
    )


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.index_buf_accessor._get_s_triton",
    type=HookType.REPLACE,
)
def _get_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
) -> torch.Tensor:
    _, buf_numel_per_page = buf.shape
    return torch.ops.xspeedgate_ops.get_s_kernel(
        buf.contiguous(),
        page_indices.contiguous(),
        seq_len,
        page_size,
        buf_numel_per_page,
        page_size * index_head_dim,
    )


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.index_buf_accessor.SetKAndS.triton",
    type=HookType.REPLACE,
)
def set_k_and_s_triton(cls, pool, buf, loc, index_k, index_k_scale) -> None:
    """Set NSA K and S index buffers with Kunlun kernels."""
    import kunlun_ops

    kunlun_ops.set_k_and_s_triton(
        buf=buf.contiguous(),
        loc=loc.to(torch.int64).contiguous(),
        index_k=index_k.contiguous(),
        index_k_scale=index_k_scale.contiguous(),
        page_size=pool.page_size,
    )
