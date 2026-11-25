# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""W8A8-INT8 engram host table for DeepSeek-V4.1 on Kunlun/XPU.

Upstream ``EngramEmbedding._init_host_table`` sizes the host mapping for the FP8
release -- an ``float8_e4m3`` payload plus one ``float8_e8m0`` block scale per
32-element block. The W8A8-INT8 release stores

    layers.N.engram.embed.weight        int8 [num_embeddings, dim]
    layers.N.engram.embed.weight_scale  bf16 [num_embeddings, 1]

i.e. compressed-tensors channel-wise int8, one scale per row, so the payload
dtype and the scale shape both differ. Everything downstream is already int8
aware and needs no change: the plugin's ``load_weights`` AROUND hook renames
``engram.embed.weight_scale`` onto the module's ``scale`` param, and
``engram_owned_rows_kunlun`` dequantizes int8 * per-row bf16 through
``xspeedgate_ops.engram_gather``. Only the host mapping has to be re-derived here
so those two see int8 [rows, dim] + bf16 [rows, 1] instead of the fp8 block
layout -- whose fp8 payload then fails every lookup on XPU.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers import engram as engram_mod
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_parallel


@plugin_hook(
    target="sglang.srt.layers.engram.EngramEmbedding._init_host_table",
    type=HookType.REPLACE,
)
def init_host_table_int8(self, num_embeddings: int, dim: int, layer_id: int):
    """Host-table layout for the channel-wise int8 engram (one bf16 scale/row)."""
    layout = engram_mod.envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
    n = num_embeddings if layout == "shared" else self.rows
    w_bytes = n * dim  # int8 payload
    s_bytes = n * 2  # one bf16 scale per row
    self.host_table = engram_mod._HostTable(
        layout,
        max(1, w_bytes + s_bytes),  # mmap requires storage even for an empty shard
        f"sglang_engram_{layer_id}",
        get_parallel().tp_group,
    )
    raw = self.host_table.bytes[: w_bytes + s_bytes]
    weight = raw[:w_bytes].view(torch.int8).view(n, dim)
    scale = raw[w_bytes : w_bytes + s_bytes].view(torch.bfloat16).view(n, 1)
    self.weight = nn.Parameter(weight, requires_grad=False)
    self.scale = nn.Parameter(scale, requires_grad=False)
