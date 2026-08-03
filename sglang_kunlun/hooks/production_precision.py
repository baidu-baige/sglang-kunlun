"""Kunlun runtime correctness hooks shared by scheduling and memory-cache code."""

from __future__ import annotations

import inspect

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.managers.schedule_batch.ScheduleBatch._evict_swa",
    type=HookType.REPLACE,
)
def evict_swa_with_page_margin_kunlun(self, req, pre_len):
    """Keep the full one-page SWA eviction margin for chunk-cache batches."""
    from sglang.srt.mem_cache.common import free_swa_out_of_window_slots

    assert self.tree_cache.supports_swa(), "prefix cache must support swa"
    free_swa_out_of_window_slots(
        req,
        pre_len,
        sliding_window_size=self.tree_cache.sliding_window_size,
        page_size=self.tree_cache.page_size,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        drop_page_margin=False,
    )


@plugin_hook(
    "sglang.srt.mem_cache.common.write_cache_indices",
    type=HookType.REPLACE,
)
def write_cache_indices_kunlun(
    out_cache_loc,
    req_pool_indices_tensor,
    req_pool_indices_cpu,
    prefix_lens_tensor,
    prefix_lens_cpu,
    seq_lens_tensor,
    seq_lens_cpu,
    extend_lens_tensor,
    extend_lens_cpu,
    prefix_tensors,
    req_to_token_pool,
):
    """Build prefix pointer tables directly on the consuming device."""
    import sglang.srt.mem_cache.common as upstream

    if upstream.support_triton(upstream.get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [tensor.data_ptr() for tensor in prefix_tensors],
            dtype=torch.uint64,
            device=req_to_token_pool.device,
        )
        upstream.write_req_to_token_pool_triton[
            (req_pool_indices_tensor.shape[0],)
        ](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
        return

    offset = 0
    for index in range(req_pool_indices_cpu.shape[0]):
        req_idx = req_pool_indices_cpu[index].item()
        prefix_len = prefix_lens_cpu[index].item()
        seq_len = seq_lens_cpu[index].item()
        extend_len = extend_lens_cpu[index].item()
        req_to_token_pool.write(
            (req_idx, slice(0, prefix_len)), prefix_tensors[index]
        )
        req_to_token_pool.write(
            (req_idx, slice(prefix_len, seq_len)),
            out_cache_loc[offset : offset + extend_len],
        )
        offset += extend_len


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool.__init__",
    type=HookType.AFTER,
)
def initialize_non_online_compress_state_kunlun(
    result,
    self,
    size,
    ring_size,
    overlap,
    head_dim,
    dtype,
    device,
    enable_memory_saver,
    ratio,
    online=False,
    swa_page_size=0,
    online_mtp_max_draft_tokens=0,
):
    """Initialize every non-online KV row and score sentinel, not only the last."""
    if not online:
        wrapped_init = inspect.unwrap(type(self).__init__)
        try:
            upstream_has_full_clear = (
                "self.kv_score_buffer.clear()" in inspect.getsource(wrapped_init)
            )
        except (OSError, TypeError):
            upstream_has_full_clear = False
        if not upstream_has_full_clear:
            self.kv_score_buffer.clear()
    return result
