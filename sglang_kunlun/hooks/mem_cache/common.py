"""Hooks for ``sglang.srt.mem_cache.common``.

Patches:
  - get_last_loc → xspeedgate_ops.get_last_loc (both call-site local bindings)
  - write_cache_indices → xspeedgate_ops.write_req_to_token_pool
    (called from within common.py, so the module-level attribute is patched)
"""

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
