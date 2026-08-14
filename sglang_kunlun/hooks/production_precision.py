"""Kunlun runtime correctness hooks shared by scheduling and memory-cache code."""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


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
        is_chunk_cache=False,
    )


@plugin_hook(
    "sglang.srt.mem_cache.allocation.write_cache_indices",
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
    from sglang_kunlun.kernels.kernel_ops import write_req_to_token_pool_triton

    prefix_pointers = torch.tensor(
        [tensor.data_ptr() for tensor in prefix_tensors],
        dtype=torch.uint64,
        device=req_to_token_pool.device,
    )
    write_req_to_token_pool_triton(
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
        self.kv_score_buffer.clear()
    return result


_DSV4_VERIFY_MASK_DISABLED_LOGGED = False


@plugin_hook(
    "sglang.srt.layers.attention.verify_mask.maybe_create_verify_mask",
    type=HookType.AROUND,
)
def _dsv4_disable_verify_mask(original_fn, *args, **kwargs):
    """Kunlun verify attention reads a FULL_MASK tree mask, so never hand it one.

    Upstream 0.5.17 lets the target backend own a preallocated verify mask
    (`VerifyMask`, created only when graph runners exist) and, for DeepSeek-V4,
    declares it write-only: mode=QLEN_ONLY, is_read=False. `build_eagle_verify_input`
    then keys three decisions off that object, so the tree mask handed to verify is a
    qlen-only buffer that nobody fills the prefix of. Upstream can afford this because
    its DSV4 kernels never read the mask -- but the Kunlun verify path does, with
    FULL_MASK indexing (`kunlun_backend.py:703`), and `generate_attn_arg_prefill`
    pads the short buffer with True (`eagle_info.py:123`). Every draft row therefore
    attends to its non-ancestors, and acceptance collapses at EAGLE 3/1/4 while the
    score barely moves.

    Returning None restores the 0.5.14 contract the Kunlun kernels were written
    against: the caller allocates and fills a FULL_MASK tree mask each iteration.
    Set DSV4_KUNLUN_VERIFY_MASK=1 to fall back to the upstream behaviour.
    """
    import os

    if os.environ.get("DSV4_KUNLUN_VERIFY_MASK") == "1":
        return original_fn(*args, **kwargs)
    global _DSV4_VERIFY_MASK_DISABLED_LOGGED
    if not _DSV4_VERIFY_MASK_DISABLED_LOGGED:
        _DSV4_VERIFY_MASK_DISABLED_LOGGED = True
        logger.warning(
            "[DSV4_VERIFY_MASK] backend-owned verify mask disabled on Kunlun; "
            "the caller fills a FULL_MASK tree mask each iteration "
            "(set DSV4_KUNLUN_VERIFY_MASK=1 to restore upstream behaviour)"
        )
    return None
