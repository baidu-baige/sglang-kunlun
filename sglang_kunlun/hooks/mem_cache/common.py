"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_num_new_pages, next_power_of_2


logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
    type=HookType.AROUND,
)
def configure_kv_cache_dtype_kunlun(original_fn, **kwargs):
    """把 `--kv-cache-dtype fp16` 解析成 `torch.float16`。

    昆仑的 DSV4 池子是 fp16/bf16 的"半 cache"（见下面 `dsv4_create_buffer_kunlun`），
    所以本平台跑 fp16 KV cache。上游的解析表没有 fp16 分支，会落到
    `raise ValueError(f"Unsupported kv_cache_dtype: ...")`，因此在这里拦掉。字符串槽返回
    None —— 上游用 None 表示未量化池（auto 分支就不设这个标记，backend 靠它判断是否 descale）。

    配套改动：`sitecustomize.py` 把 fp16 加回 argparse choices；`platform/srt.py` 在上游
    DeepSeek-V4 hook 断言这个值时先换成 bfloat16、跑完再声明回来。
    """
    if kwargs.get("server_args_kv_cache_dtype") == "fp16":
        return None, torch.float16
    return original_fn(**kwargs)


@plugin_hook(
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_decode",
    type=HookType.REPLACE,
)
def paged_alloc_decode_kunlun(self, seq_lens, seq_lens_cpu, last_loc):
    """Decode allocation with a CPU-side page-count gate and zero-page fast path."""
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
        out_indices.copy_(last_loc.to(out_indices.dtype).contiguous() + 1)
        return out_indices

    torch.ops.xspeedgate_ops.alloc_decode_kernel(
        seq_lens,
        last_loc.to(torch.int32).contiguous(),
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
        bs,
    )

    if self.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.get_bytes_per_token",
    type=HookType.AROUND,
)
def dsv4_get_bytes_per_token_kunlun(original_fn, self):
    """Return the Kunlun half-cache byte footprint for one token."""
    if self.store_dtype in (torch.bfloat16, torch.float16):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.store_dtype.itemsize
    return original_fn(self)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool.create_buffer",
    type=HookType.AROUND,
)
def dsv4_create_buffer_kunlun(original_fn, self, *, num_pages: int):
    """Allocate the Kunlun half-precision KV cache buffer."""
    if self.store_dtype not in (torch.bfloat16, torch.float16):
        return original_fn(self, num_pages=num_pages)

    dim_per_token = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.kv_cache_total_dim = dim_per_token
    self.bytes_per_page_padded = (
        self.page_size * dim_per_token * self.store_dtype.itemsize
    )
    if (
        os.environ.get("DSV4_MTP_PROBE") == "1"
        and os.environ.get("RANK", "0") == "0"
        and not getattr(dsv4_create_buffer_kunlun, "_probe_logged", False)
    ):
        logger.warning(
            "[DSV4_CALLSTACK] half-cache pool buffer store_dtype=%s "
            "shape=(%d, %d) page_size=%d dim_per_token=%d",
            self.store_dtype,
            num_pages,
            self.page_size * dim_per_token,
            self.page_size,
            dim_per_token,
        )
        dsv4_create_buffer_kunlun._probe_logged = True
    dim_per_page = self.page_size * dim_per_token
    total_bytes = num_pages * dim_per_page * self.store_dtype.itemsize
    return _alloc_2m_aligned(total_bytes, self.store_dtype, self.device).reshape(
        num_pages, dim_per_page
    )


_ALIGNMENT_2M = 2 * 1024 * 1024


def _alloc_2m_aligned(total_bytes: int, dtype: torch.dtype, device: str):
    """Allocate a tensor whose ``data_ptr()`` is 2MB-aligned with 2MB-aligned
    backing pages.

    The Kunlun peermem/XDR path maps device memory at 2MB granularity, so
    ``ibv_reg_mr`` on a host VA that is not 2MB-aligned fails with EINVAL(22).
    Allocating ``ALIGN_2M(total_bytes) + 2MB`` guarantees both that the returned
    view starts on a 2MB boundary and that physical pages cover the full
    ``ALIGN_2M(total_bytes)`` MR registration range from that boundary. The view
    itself exposes exactly ``total_bytes`` so callers can reshape as expected.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    assert total_bytes % element_size == 0
    aligned_bytes = (total_bytes + _ALIGNMENT_2M - 1) // _ALIGNMENT_2M * _ALIGNMENT_2M
    flat = torch.zeros(
        aligned_bytes + _ALIGNMENT_2M,
        dtype=torch.uint8,
        device=device,
    )
    offset = (_ALIGNMENT_2M - flat.data_ptr() % _ALIGNMENT_2M) % _ALIGNMENT_2M
    return flat[offset : offset + total_bytes].view(dtype)


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4IndexerPool._create_buffer",
    type=HookType.AROUND,
)
def dsv4_indexer_pool_create_buffer_kunlun(original_fn, self):
    """Allocate the Kunlun indexer pool with RDMA-safe 2MB backing pages."""
    from sglang_kunlun.hooks.utils.common import _is_kunlun

    if not _is_kunlun():
        return original_fn(self)

    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

    page_bytes = self.page_size * self.get_bytes_per_token()
    num_pages = (self.size + self.page_size + 1) // self.page_size
    dtype = self.index_k_with_scale_buffer_dtype
    total_bytes = num_pages * page_bytes * torch.tensor([], dtype=dtype).element_size()
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                _alloc_2m_aligned(total_bytes, dtype, self.device).reshape(
                    num_pages, page_bytes
                )
                for _ in range(self.layer_num)
            ]


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool."
    "_alloc_kv_score_buffer",
    type=HookType.AROUND,
)
def compress_state_alloc_kv_score_buffer_kunlun(
    original_fn, self, *, dtype, device, enable_memory_saver
):
    """Allocate the compress-state kv+score buffer with RDMA-safe 2MB pages.

    ``kv_score_buffer.kv_score`` is published to the PD peer through
    ``DeepSeekV4TokenToKVPool.get_state_buf_infos()``, so it has to satisfy the
    same 2MB alignment constraint as the KV and indexer pools.
    """
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
    from sglang.srt.mem_cache.memory_pool import (
        TorchMemorySaverAdapter,
        maybe_init_custom_mem_pool,
    )
    from sglang_kunlun.hooks.utils.common import _is_kunlun

    if not _is_kunlun():
        return original_fn(
            self,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

    self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=enable_memory_saver
    )
    self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
        maybe_init_custom_mem_pool(device=device)
    )
    total_bytes = (
        self._size * self.last_dim * torch.tensor([], dtype=dtype).element_size()
    )
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.kv_score_buffer = KVAndScore(
                _alloc_2m_aligned(total_bytes, dtype, device).reshape(
                    self._size, self.last_dim
                )
            )
    self.kv_score_buffer.clear()


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool.MambaPool._should_fuse_slot_ops",
    type=HookType.AROUND,
)
def disable_fused_mamba_slot_ops(original_fn, self) -> bool:
    """Disable fused slot operations only on the Kunlun platform.

    Returns:
        ``False`` on Kunlun so the upstream Torch clear/copy fallback runs;
        ``True`` elsewhere so native fused operations remain unchanged.
    """
    from sglang.srt.platforms import current_platform

    if getattr(current_platform, "device_name", None) == "kunlun":
        return False
    return original_fn(self)
