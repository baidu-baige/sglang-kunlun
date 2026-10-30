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
    """Verify an Eagle tree greedily in-place.

    Fast path mirrors the 5.8 baseline (``sgl_kernel/speculative.py``): dispatch
    the whole traversal to the fused on-device ``kunlun_ops.verify_tree_greedy``
    kernel in a single call, instead of copying every tensor to host and looping
    in Python. Kernel contract (see 5.8 ``sgl-kernel/csrc/speculative/eagle_utils.cc``):
        predicts            : [tot_num_draft_tokens]   int32  (mutable)
        accept_index        : [bs, num_spec_step]      int32  (mutable)
        accept_token_num    : [bs]                     int32  (mutable)
        candidates          : [bs, num_draft_tokens]   int64
        retrive_index       : [bs, num_draft_tokens]   int64
        retrive_next_token  : [bs, num_draft_tokens]   int64
        retrive_next_sibling: [bs, num_draft_tokens]   int64
        target_predict      : [bs, num_draft_tokens]   int64

    Falls back to the host implementation only if the fused kernel is not
    exposed by this ``kunlun_ops`` build.
    """

    try:
        from kunlun_ops import verify_tree_greedy as verify_tree_greedy_kunlun
    except (ImportError, AttributeError):
        verify_tree_greedy_kunlun = None

    if verify_tree_greedy_kunlun is not None:
        batch_size = candidates.size(0)
        num_draft_tokens = candidates.size(1)
        # Kernel reads target_predict as a [bs, num_draft_tokens] int64 tensor
        # (flat data_ptr indexed by the global retrive_index), so keep it 2D
        # rather than flattening as the host fallback did.
        if target_predict.dim() == 1:
            target_predict = target_predict.reshape(batch_size, num_draft_tokens)
        # Mutable outputs (predicts/accept_index/accept_token_num) are written
        # in place by the kernel and must be passed through untouched; only the
        # read-only inputs are coerced to the int64 layout the kernel expects.
        verify_tree_greedy_kunlun(
            predicts,
            accept_index,
            accept_token_num,
            candidates.to(torch.int64),
            retrive_index.to(torch.int64),
            retrive_next_token.to(torch.int64),
            retrive_next_sibling.to(torch.int64),
            target_predict.to(torch.int64),
        )
        return

    _verify_tree_greedy_host(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


def _verify_tree_greedy_host(
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
    """Host-side greedy tree verification (fallback when no fused kernel)."""

    batch_size = candidates.size(0)
    num_speculative_tokens = accept_index.size(1)

    if target_predict.dim() > 1:
        target_predict = target_predict.flatten()

    # Pull every tensor the traversal reads to host in ONE bulk D2H each,
    # instead of thousands of per-scalar ``.item()`` device syncs. The
    # verification logic below is byte-for-byte identical (integer compares,
    # no floating point), so acceptance decisions / accept length are
    # unchanged; we only collapse the sync count from O(bs * tree) to O(1).
    retrive_index_cpu = retrive_index.tolist()
    retrive_next_token_cpu = retrive_next_token.tolist()
    retrive_next_sibling_cpu = retrive_next_sibling.tolist()
    candidates_cpu = candidates.tolist()
    target_predict_cpu = target_predict.tolist()

    # Stage outputs on host, write back in one shot per tensor at the end.
    accept_index_cpu = [[-1] * num_speculative_tokens for _ in range(batch_size)]
    accept_token_num_cpu = [0] * batch_size
    predict_positions: list[int] = []
    predict_values: list[int] = []

    for batch_idx in range(batch_size):
        last_accepted_retrive_idx = retrive_index_cpu[batch_idx][0]
        accept_index_cpu[batch_idx][0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for _ in range(1, num_speculative_tokens):
            cur_index = retrive_next_token_cpu[batch_idx][cur_index]

            while cur_index != -1:
                draft_index = retrive_index_cpu[batch_idx][cur_index]
                draft_token_id = candidates_cpu[batch_idx][cur_index]
                target_token_id = target_predict_cpu[last_accepted_retrive_idx]

                if draft_token_id == target_token_id:
                    predict_positions.append(last_accepted_retrive_idx)
                    predict_values.append(target_token_id)
                    num_accepted += 1
                    accept_index_cpu[batch_idx][num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                cur_index = retrive_next_sibling_cpu[batch_idx][cur_index]

            if cur_index == -1:
                break

        accept_token_num_cpu[batch_idx] = num_accepted
        predict_positions.append(last_accepted_retrive_idx)
        predict_values.append(target_predict_cpu[last_accepted_retrive_idx])

    # Single bulk write-back per output tensor (no per-element device writes).
    accept_index.copy_(
        torch.tensor(
            accept_index_cpu, dtype=accept_index.dtype, device=accept_index.device
        )
    )
    accept_token_num.copy_(
        torch.tensor(
            accept_token_num_cpu,
            dtype=accept_token_num.dtype,
            device=accept_token_num.device,
        )
    )
    if predict_positions:
        pos = torch.tensor(
            predict_positions, dtype=torch.long, device=predicts.device
        )
        vals = torch.tensor(
            predict_values, dtype=predicts.dtype, device=predicts.device
        )
        predicts[pos] = vals
