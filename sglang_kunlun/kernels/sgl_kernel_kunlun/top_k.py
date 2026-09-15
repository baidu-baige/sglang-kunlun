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
"""Kunlun implementations of sgl_kernel top-k operations.

Only ``fast_topk`` is implemented here — it is a pure-Python fallback
identical to the mimo-branch reference (no XPU-specific kernel needed).

``fast_topk_v2``, ``fast_topk_transform_fused``, and
``fast_topk_transform_ragged_fused`` rely on CUDA kernels
(``torch.ops.sgl_kernel.*``) that are only available in the CUDA
sgl_kernel build and are part of the NSA (Native Sparse Attention) path.
Those functions are intentionally left as stubs in ``_sgl_kernel_stub.py``
and will raise ``NotImplementedError`` if reached.
"""

import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    else:
        return torch.topk(values, topk, dim=dim)


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    """Kunlun implementation of ``sgl_kernel.moe_fused_gate``."""

    import kunlun_ops

    if num_fused_shared_experts != 0:
        raise NotImplementedError("Kunlun moe_fused_gate does not support fused shared experts")

    num_tokens, num_experts = input_tensor.shape
    block_statistic = torch.empty(
        12, num_experts, dtype=torch.int32, device=input_tensor.device
    )
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=input_tensor.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=input_tensor.device
    )
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=input_tensor,
        topk_index=topk_ids,
        norm_score=topk_weights,
        block_static=block_statistic,
        bias=bias.float(),
        scale=(routed_scaling_factor if apply_routed_scaling_factor_on_output else 1.0),
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_weights, topk_ids
