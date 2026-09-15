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
"""rotary_embedding"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward_cuda",
    type=HookType.REPLACE,
)
def rotary_embedding_forward_kunlun(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    fused_set_kv_buffer_arg=None,
):
    """Kunlun rotary embedding via ``xspeedgate_ops.flashinfer_rotary_embedding``."""
    assert (
        fused_set_kv_buffer_arg is None
    ), "fused_set_kv_buffer_arg is not supported on Kunlun"

    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)

    query, key = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions,
        rotary_dim=self.rotary_dim,
        head_size=self.head_size,
        cos_sin_cache=self.cos_sin_cache,
        is_neox_style=self.is_neox_style,
        query=query,
        key=key,
        offsets=offsets,
    )
    query = query.reshape(query_shape)
    key = key.reshape(key_shape)
    return query, key
