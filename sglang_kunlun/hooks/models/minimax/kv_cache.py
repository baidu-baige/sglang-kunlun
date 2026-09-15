"""Kunlun KV-cache hook for the MiniMax-M3 sparse path (0.5.19 layout).

0.5.19 builds KV pools in ``KVCacheConfigurator``. Its out-of-tree branch
(``_build_oot_mha_kv_pool``) returns a plain ``current_platform.get_mha_kv_pool_cls()``
pool for every non-MLA model, which is wrong for MiniMax-M3: the sparse backend
calls ``get_index_k_buffer`` / ``set_fused_kv_index_buffer`` on the pool, so the
model needs ``MiniMaxSparseKVPool``.

The hook reuses upstream's own ``_build_minimax_sparse_kv_pool`` (so its kwargs
track upstream drift) and only swaps the class it instantiates for the Kunlun
subclass that injects the paged sub-pool classes. The allocator needs no fixup:
``_build_token_to_kv_pool_allocator`` already takes
``current_platform.get_paged_allocator_cls()`` for OOT platforms.
"""

from __future__ import annotations

from sglang.srt.configs.model_config import is_minimax_sparse
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.models.minimax.kv_pools import KunlunMiniMaxSparseKVPool


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_oot_mha_kv_pool",
    type=HookType.AROUND,
)
def build_oot_mha_kv_pool_minimax_kunlun(original_fn, self, *, max_total_num_tokens: int):
    """Build the Kunlun MiniMax sparse pool instead of a plain MHA pool."""
    if not is_minimax_sparse(self.model_config.hf_config):
        return original_fn(self, max_total_num_tokens=max_total_num_tokens)

    import sglang.srt.mem_cache.kv_cache_configurator as kv_cache_configurator

    original_pool_cls = kv_cache_configurator.MiniMaxSparseKVPool
    kv_cache_configurator.MiniMaxSparseKVPool = KunlunMiniMaxSparseKVPool
    try:
        return self._build_minimax_sparse_kv_pool(
            max_total_num_tokens=max_total_num_tokens
        )
    finally:
        kv_cache_configurator.MiniMaxSparseKVPool = original_pool_cls
