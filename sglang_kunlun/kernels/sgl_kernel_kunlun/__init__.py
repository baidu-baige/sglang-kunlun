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
"""Kunlun-backed implementations of sgl_kernel APIs."""

from .layernorm import fused_add_rmsnorm, rmsnorm
from .sampling import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)
from .speculative import build_tree_kernel_efficient, verify_tree_greedy
from .top_k import fast_topk, moe_fused_gate

__all__ = [
    "build_tree_kernel_efficient",
    "fast_topk",
    "fused_add_rmsnorm",
    "moe_fused_gate",
    "rmsnorm",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
    "tree_speculative_sampling_target_only",
    "verify_tree_greedy",
]
