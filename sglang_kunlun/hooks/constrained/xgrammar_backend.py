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
"""Hooks for ``sglang.srt.constrained.xgrammar_backend``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/constrained/xgrammar_backend.py

REPLACE ``XGrammarGrammar.apply_vocab_mask`` with a torch-compile fallback that
avoids the upstream Triton/CUDA kernel (unsupported on Kunlun).
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


# adapted from
# https://github.com/mlc-ai/xgrammar/blob/v0.1.30/python/xgrammar/kernels/apply_token_bitmask_inplace_torch_compile.py#L7
@torch.compile(dynamic=True)
def _apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
    logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int
) -> None:
    mask_expanded = torch.repeat_interleave(bitmask, 32, dim=-1)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(
        bitmask.shape[-1]
    )
    bit_masks = (mask_expanded >> bit_indices) & 1
    bit_masks = bit_masks[..., :vocab_size]
    logits[..., :vocab_size] = logits[..., :vocab_size].masked_fill_(
        bit_masks == 0, float("-inf")
    )


@plugin_hook(
    "sglang.srt.constrained.xgrammar_backend.XGrammarGrammar.apply_vocab_mask",
    type=HookType.REPLACE,
)
def apply_vocab_mask_kunlun(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Kunlun: torch.compile-based bitmask path."""
    vocab_size = min(logits.shape[-1], vocab_mask.shape[-1] * 32)
    return _apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
        logits, vocab_mask, vocab_size
    )
