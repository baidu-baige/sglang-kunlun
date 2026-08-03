"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


logger = logging.getLogger(__name__)


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
    return torch.zeros(
        num_pages,
        self.page_size * dim_per_token,
        dtype=self.store_dtype,
        device=self.device,
    )


_ALIGNMENT_2M = 2 * 1024 * 1024


def _alloc_2m_aligned(total_bytes: int, dtype: torch.dtype, device: str):
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


# from __future__ import annotations

# import torch

# from sglang.srt.plugins.hook_registry import HookType, plugin_hook


# def _get_last_loc_kunlun(
#     req_to_token: torch.Tensor,
#     req_pool_indices_tensor: torch.Tensor,
#     prefix_lens_tensor: torch.Tensor,
# ) -> torch.Tensor:
#     if prefix_lens_tensor.dtype != torch.int64:
#         prefix_lens_tensor = prefix_lens_tensor.to(torch.int64)
#     return torch.ops.xspeedgate_ops.get_last_loc(
#         req_to_token, req_pool_indices_tensor, prefix_lens_tensor
#     )


# @plugin_hook(
#     "sglang.srt.speculative.eagle_info_v2.get_last_loc",
#     type=HookType.REPLACE,
# )
# def get_last_loc_eagle_info_v2(*args, **kwargs):
#     """Kunlun replacement for eagle_info_v2 get_last_loc."""
#     return _get_last_loc_kunlun(*args, **kwargs)


# @plugin_hook(
#     "sglang.srt.speculative.spec_utils.get_last_loc",
#     type=HookType.REPLACE,
# )
# def get_last_loc_spec_utils(*args, **kwargs):
#     """Kunlun replacement for spec_utils get_last_loc."""
#     return _get_last_loc_kunlun(*args, **kwargs)


# @plugin_hook(
#     "sglang.srt.mem_cache.common.write_cache_indices",
#     type=HookType.REPLACE,
# )
# def write_cache_indices_kunlun(
#     out_cache_loc: torch.Tensor,
#     req_pool_indices_tensor: torch.Tensor,
#     req_pool_indices_cpu: torch.Tensor,
#     prefix_lens_tensor: torch.Tensor,
#     prefix_lens_cpu: torch.Tensor,
#     seq_lens_tensor: torch.Tensor,
#     seq_lens_cpu: torch.Tensor,
#     extend_lens_tensor: torch.Tensor,
#     extend_lens_cpu: torch.Tensor,
#     prefix_tensors,
#     req_to_token_pool,
# ):
#     """Kunlun replacement for write_cache_indices using xspeedgate_ops."""
#     prefix_pointers = torch.tensor(
#         [t.data_ptr() for t in prefix_tensors],
#         device=req_to_token_pool.device,
#         dtype=torch.uint64,
#     )
#     torch.ops.xspeedgate_ops.write_req_to_token_pool(
#         req_to_token_pool.req_to_token,
#         req_pool_indices_tensor.to(torch.int32),
#         prefix_pointers,
#         prefix_lens_tensor,
#         seq_lens_tensor,
#         extend_lens_tensor,
#         out_cache_loc.to(torch.int64),
#     )
