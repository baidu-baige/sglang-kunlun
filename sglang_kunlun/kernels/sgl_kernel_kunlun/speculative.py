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
"""Kunlun-backed implementations of speculative ``sgl_kernel`` APIs."""

from __future__ import annotations

from typing import Any

import torch


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    seq_lens: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: Any = None,
) -> None:
    """Build Eagle tree metadata using Kunlun's fused implementation."""

    from kunlun_ops import build_tree_efficient

    batch_size = seq_lens.numel()
    seq_lens_sum = -1
    parent_list_for_kernel = parent_list.to(torch.long)
    if parent_list_for_kernel.numel() == 0 and spec_steps == 1 and topk > 1:
        parent_list_for_kernel = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device=parent_list.device,
        )
    seq_lens_for_kernel = seq_lens.to(torch.int32)
    retrive_index_for_kernel = retrive_index.to(torch.long)
    retrive_next_token_for_kernel = retrive_next_token.to(torch.long)
    retrive_next_sibling_for_kernel = retrive_next_sibling.to(torch.long)

    build_tree_efficient(
        parent_list_for_kernel,
        top_scores_index,
        seq_lens_for_kernel,
        tree_mask,
        positions,
        retrive_index_for_kernel,
        retrive_next_token_for_kernel,
        retrive_next_sibling_for_kernel,
        topk,
        spec_steps,
        num_verify_tokens,
        batch_size,
        seq_lens_sum,
    )


def verify_tree_greedy(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Verify an Eagle tree greedily in-place."""

    batch_size = candidates.size(0)
    num_speculative_tokens = accept_index.size(1)

    if target_predict.dim() > 1:
        target_predict = target_predict.flatten()
    for batch_idx in range(batch_size):
        last_accepted_retrive_idx = retrive_index[batch_idx, 0].item()
        accept_index[batch_idx, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for _ in range(1, num_speculative_tokens):
            cur_index = retrive_next_token[batch_idx, cur_index].item()

            while cur_index != -1:
                draft_index = retrive_index[batch_idx, cur_index].item()
                draft_token_id = candidates[batch_idx, cur_index].item()
                target_token_id = target_predict[last_accepted_retrive_idx].item()

                if draft_token_id == target_token_id:
                    predicts[last_accepted_retrive_idx] = target_token_id
                    num_accepted += 1
                    accept_index[batch_idx, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                cur_index = retrive_next_sibling[batch_idx, cur_index].item()

            if cur_index == -1:
                break

        accept_token_num[batch_idx] = num_accepted
        predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx]
