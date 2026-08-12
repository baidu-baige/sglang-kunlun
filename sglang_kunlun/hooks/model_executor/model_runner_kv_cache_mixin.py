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

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


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
    """Restore the hybrid-SWA allocator contract bypassed by the OOT branch."""
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    if (
        self.is_draft_worker
        or not self.is_hybrid_swa
        or not is_dsv4_model
        or isinstance(result, SWATokenToKVPoolAllocator)
    ):
        return result

    need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
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
    if hasattr(req_to_token_pool, "register_dsv4_allocator"):
        req_to_token_pool.register_dsv4_allocator(allocator)
    return allocator
