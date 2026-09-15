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
"""Kunlun paged KV pool allocator selected by platform factory.

Main-line ``SWATokenToKVPoolAllocator`` now uses
``current_platform.get_paged_allocator_cls()`` for OOT platforms, so allocator
customization can be expressed as a subclass instead of REPLACE hooks.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2


def _alloc_decode_torch_native(
    page_size,
    free_pages,
    seq_lens,
    last_loc,
    bs,
):
    """Pure-torch alloc_decode, ported from the pre-refactor allocator patch.

    Kept as an A/B reference for the xspeedgate_ops kernel: slicing free_pages
    cannot read out of bounds, so a crash here would rule the kernel out.
    """
    pre_lens = seq_lens - 1
    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    mask = (num_pages_after - num_pages_before) > 0
    sum_new_pages = mask.sum()

    out_indices = last_loc + 1
    if sum_new_pages > 0:
        out_indices[mask] = free_pages[:sum_new_pages] * page_size
    return out_indices.to(torch.int64)


def _alloc_extend_kunlun_xdnn(
    page_size,
    free_pages,
    prefix_lens,
    seq_lens,
    last_loc,
    extend_num_tokens,
):
    """xdnn fast path."""
    from torch_xmlir.nn.alloc_extend import Alloc_extend

    op = Alloc_extend()
    if last_loc.dtype != torch.int64:
        last_loc = last_loc.to(torch.int64)
    if prefix_lens.dtype != torch.int64:
        prefix_lens = prefix_lens.to(torch.int64)
    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    out_indices, origin_num_new_pages = op(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        page_size,
        extend_num_tokens,
    )
    out_indices = out_indices.to(torch.int32)
    return out_indices, origin_num_new_pages


def _alloc_extend_kunlun_kernel(
    page_size,
    free_pages,
    prefix_lens,
    seq_lens,
    last_loc,
    extend_num_tokens,
):
    """xspeedgate_ops kernel path."""
    bs = prefix_lens.shape[0]
    if prefix_lens.dtype != torch.int64:
        prefix_lens = prefix_lens.to(torch.int64)
    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    if last_loc.dtype != torch.int64:
        last_loc = last_loc.to(torch.int64)
    out_indices = torch.zeros(extend_num_tokens, dtype=torch.int64, device="cuda")
    ret_value = torch.zeros(1, dtype=torch.int64, device="cuda")
    torch.ops.xspeedgate_ops.alloc_extend(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        bs,
        page_size,
        extend_num_tokens,
        out_indices,
        ret_value,
    )
    return out_indices, ret_value


def _select_alloc_extend_func():
    if os.environ.get("USE_FAST_ALLOC_EXTEND_KUNLUN", "1") != "0":
        return _alloc_extend_kunlun_xdnn
    return _alloc_extend_kunlun_kernel


class KunlunPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Kunlun allocator using xspeedgate_ops/xdnn-backed allocation kernels."""

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate pages for extend tokens using the selected backend kernel."""
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if extend_num_tokens == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        alloc_fn = _select_alloc_extend_func()
        out_indices, origin_num_new_pages = alloc_fn(
            self.page_size,
            self.free_pages,
            prefix_lens,
            seq_lens,
            last_loc,
            extend_num_tokens,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        merged_value = origin_num_new_pages.item()
        num_new_pages = merged_value >> 32
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        """Allocate pages for decode tokens using the xspeedgate_ops kernel."""
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if num_new_pages == 0:
            out_indices.copy_(last_loc + 1)
            return out_indices

        # The kernel indexes free_pages over the padded range [0, bs_upper), so it
        # needs bs_upper entries available even when num_new_pages is smaller.
        bs_upper = next_power_of_2(bs)
        if len(self.free_pages) < bs_upper:
            self.merge_and_sort_free()
        if len(self.free_pages) < bs_upper:
            return None

        if get_bool_env_var("USE_TORCH_ALLOC_DECODE_KUNLUN", "false"):
            out_indices = _alloc_decode_torch_native(
                self.page_size,
                self.free_pages,
                seq_lens,
                last_loc,
                bs,
            )
        else:
            torch.ops.xspeedgate_ops.alloc_decode_kernel(
                seq_lens,
                last_loc.contiguous(),
                self.free_pages,
                out_indices.contiguous(),
                bs_upper,
                self.page_size,
                bs,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
