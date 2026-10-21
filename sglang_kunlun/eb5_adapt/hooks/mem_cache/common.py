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
"""Kunlun memory-cache allocator hooks."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_num_new_pages, next_power_of_2


@plugin_hook(
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_extend",
    type=HookType.REPLACE,
)
def alloc_extend_kunlun(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int = None,
):
    """Avoid launching XDNN alloc_extend when the KV page pool is exhausted."""
    if self.debug_mode:
        assert torch.all(
            (last_loc + 1) % self.page_size == prefix_lens % self.page_size
        )

    bs = len(prefix_lens)
    if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
        self.free_pages
    ):
        self.merge_and_sort_free()

    # Kunlun-specific delta from the upstream alloc_extend implementation:
    # check KV page capacity before launching alloc_extend_kernel.  The XDNN
    # custom op aborts with ret=1 when the pool is full instead of returning a
    # recoverable allocation failure; the caller already handles None.
    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
    if num_new_pages > len(self.free_pages):
        return None

    out_indices = torch.empty(
        (extend_num_tokens,), dtype=torch.int64, device=self.device
    )
    # Call the Kunlun implementation directly instead of importing the upstream
    # symbol through the ``sglang.srt.mem_cache.triton_ops.allocator`` compat
    # alias: that alias is re-bound at runtime and the module object the caller
    # sees may not be the one ``kernel_ops.install()`` patched (verified: the
    # alias path yields a *duplicate* module object holding the raw Triton
    # JITFunction), so the replacement is lost and the raw kernel is launched
    # (fails on XPU with ``'_CudaTarget' object has no attribute 'arch'``).
    from sglang_kunlun.eb5_adapt.kernels.kernel_ops import alloc_extend_kernel

    alloc_extend_kernel(
        prefix_lens,
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
    )

    if self.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices


# ---------------------------------------------------------------------------
# Community main added an explicit guard that forbids out-of-tree platforms
# from reaching the in-tree DeepSeek-V4 KV pool.  That pool is ordinary Python
# (torch) code and the Kunlun hooks in this package already replace the
# device-specific pieces, so allow it through.
# ---------------------------------------------------------------------------
from sglang.srt.platforms import interface as _platform_interface

_ORIG_REJECT_OUT_OF_TREE_PATH = _platform_interface.reject_out_of_tree_path


@plugin_hook(
    "sglang.srt.platforms.interface.reject_out_of_tree_path",
    type=HookType.REPLACE,
)
def reject_out_of_tree_path_kunlun(platform, *, subsystem: str):
    if subsystem == "the DeepSeek-V4 KV pool":
        return None
    return _ORIG_REJECT_OUT_OF_TREE_PATH(platform, subsystem=subsystem)
