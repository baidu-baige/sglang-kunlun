"""Kunlun paged KV pool allocator selected by platform factory.

Main-line ``SWATokenToKVPoolAllocator`` now uses
``current_platform.get_paged_allocator_cls()`` for OOT platforms, so allocator
customization can be expressed as a subclass instead of REPLACE hooks.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2
from sglang_kunlun.debug_bridge import DEBUG_ENABLED as _DEBUG
from sglang_kunlun.debug_bridge import allocator as debug_allocator


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
    if get_bool_env_var("USE_FAST_ALLOC_EXTEND_KUNLUN", "true"):
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
        if _DEBUG:
            debug_allocator.capture("alloc_extend.begin", locals())
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
        if _DEBUG:
            debug_allocator.capture("alloc_extend.end", locals())
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

        torch.ops.xspeedgate_ops.alloc_decode_kernel(
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
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
