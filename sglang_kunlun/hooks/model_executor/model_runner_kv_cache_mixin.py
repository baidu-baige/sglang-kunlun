"""Hooks for ``sglang.srt.model_executor.model_runner_kv_cache_mixin``.

Kunlun out-of-tree platform path in ``_init_pools`` creates non-SWA pools even
for hybrid-SWA models (it ignores ``is_hybrid_swa``). An AFTER hook still fixes
that gap by rebuilding the SWA wrappers, but the underlying base pool class now
comes from ``current_platform.get_mha_kv_pool_cls()`` and allocator factory paths.

Python reference counting handles cleanup of the old pool objects;
no explicit ``empty_cache()`` is needed because:
  - full_max_total_num_tokens + swa_max_total_num_tokens ≤ max_total_num_tokens
  - XPU caching allocator recycles freed memory for the new allocations
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.maybe_init_hisparse_coordinator",
    type=HookType.AROUND,
)
def _skip_hisparse_coordinator_on_draft(original_fn, self):
    """Do not build a HiSparse coordinator for the DSPARK draft worker.

    The draft's DeepSeek-V4 pool is sized with
    ``COMPRESS_RATIO_NEXTN_LAYER == 0`` for every effective layer, so
    ``DeepSeekV4TokenToKVPool`` creates the C4 sub-pool with zero layers and
    never calls ``create_buffer`` -- the call that defines
    ``bytes_per_page_padded``.  ``HiSparseCoordinator.__init__`` then dies on
    that missing attribute.  The draft has no compressed layer at all, so there
    is nothing for the coordinator to manage (and the hisparse swap-in only
    runs on a plain decode forward, never on the draft's target-verify one).
    """
    if getattr(self, "is_draft_worker", False):
        from sglang.srt.mem_cache.allocator.hisparse import (
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )

        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if isinstance(allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator) and not hasattr(
            allocator.hisparse_kvcache, "bytes_per_page_padded"
        ):
            logger.info(
                "sglang-kunlun: draft worker has no C4 layers; skipping the "
                "HiSparse coordinator."
            )
            return None
    return original_fn(self)


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.configure_kv_cache_dtype",
    type=HookType.AROUND,
)
def _configure_fp16_kv_cache(original_fn, self):
    if self.server_args.kv_cache_dtype == "fp16":
        import torch

        self.kv_cache_dtype = torch.float16
        self.kv_cache_dtype_str = "fp16"
        return None
    return original_fn(self)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator."
    "_build_token_to_kv_pool_allocator",
    type=HookType.AFTER,
)
def _fix_dsv4_swa_allocator(
    result,
    self,
    *,
    sizes,
    token_to_kv_pool,
    is_dsv4_model,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    """Restore the hybrid-SWA allocator contract bypassed by the OOT branch.

    The OOT branch hands back the platform's paged allocator for every worker, while the
    pool it is paired with is only sometimes an ``SWAKVPool``:
      * the DSV4 main worker and MiMo-V2-Flash's EAGLE/MTP draft worker get a real
        ``SWAKVPool`` (the draft builds one for the full/swa split), so upstream's
        ``isinstance(allocator, SWATokenToKVPoolAllocator)`` assert on the draft path
        fails and ``full_to_swa_index_mapping`` -- asserted at use time -- is never
        registered;
      * the MiMo main worker gets a non-SWA pool (`_init_mha_pools` checks
        ``is_out_of_tree()`` before ``is_hybrid_swa``), where the paged allocator is
        self-consistent and must be left alone.
    Rebuild what upstream would have built, mirroring its branch on
    ``full_max_total_num_tokens == 0``.

    DeepSeek V4 + ``--enable-hisparse`` arrives here already wrapped in
    ``DeepSeekV4HiSparseTokenToKVPoolAllocator`` (upstream wraps the OOT paged
    allocator at the end of the original function).  Rebuilding the SWA
    allocator would silently drop that wrapper -- the coordinator then gets a
    plain SWA allocator and dies on ``allocator.compress_ratio`` -- so re-apply
    it on top of the rebuilt SWA allocator.
    """
    from sglang.srt.mem_cache.allocator.hisparse import (
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator.swa import (
        PureSWATokenToKVPoolAllocator,
        SWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    hisparse_wrapped = isinstance(result, DeepSeekV4HiSparseTokenToKVPoolAllocator)
    if (
        not self.is_hybrid_swa
        or not isinstance(token_to_kv_pool, BaseSWAKVPool)
        or (isinstance(result, SWATokenToKVPoolAllocator) and not hisparse_wrapped)
    ):
        return result

    need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
    if sizes.full_max_total_num_tokens == 0:
        allocator = PureSWATokenToKVPoolAllocator(
            sizes.swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=token_to_kv_pool,
            need_sort=need_sort,
        )
    else:
        allocator = SWATokenToKVPoolAllocator(
            sizes.full_max_total_num_tokens,
            sizes.swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=token_to_kv_pool,
            need_sort=need_sort,
        )
    token_to_kv_pool.register_mapping(allocator.full_to_swa_index_mapping)
    if hisparse_wrapped:
        allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(allocator)
    if hasattr(req_to_token_pool, "register_dsv4_allocator"):
        req_to_token_pool.register_dsv4_allocator(allocator)
    return allocator
