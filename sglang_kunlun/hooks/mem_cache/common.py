"""Kunlun memory-cache hooks."""

from __future__ import annotations

import logging
import os

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
