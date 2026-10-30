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
"""Hooks for ``sglang.srt.models.deepseek_v2`` (MoEGate).

Mirrors the aiak_sglang 5.8 patch location (``models/deepseek_v2.py``): for
DeepSeek V4 the router gate is a plain ``hidden_states @ self.weight.T`` matmul
instead of the upstream ``linear_bf16_fp32`` / ``F.linear`` path.
"""
from __future__ import annotations

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.srt.models.deepseek_v2.MoEGate.forward",
    type=HookType.REPLACE,
)
def gate_v4_forward_kunlun(
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    forward_batch=None,
):
    """Kunlun MoEGate forward falling back."""
    if hidden_states.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
        logits = hidden_states @ self.weight.T
    elif hidden_states.dtype == torch.float16 and self.weight.dtype == torch.float16:
        logits = hidden_states @ self.weight.T

    return logits