"""Kunlun KV-cache hook for the MiniMax-M3 sparse path.

The MiniMax-M3 sparse backend calls ``get_index_k_buffer`` /
``set_fused_kv_index_buffer`` on the KV pool, so the model needs
``MiniMaxSparseKVPool`` rather than a plain MHA pool.

0.5.19 reached that pool through an out-of-tree branch
(``KVCacheConfigurator._build_oot_mha_kv_pool``), which the hook had to divert.
0.5.22 deleted that branch: every platform now takes the ``is_minimax_sparse``
branch of ``_build_token_to_kv_pool`` directly, so the hook moves onto
upstream's own ``_build_minimax_sparse_kv_pool`` and only swaps the class it
instantiates for the Kunlun subclass that injects the paged sub-pool classes
(its kwargs keep tracking upstream drift). The allocator needs no fixup:
``_build_token_to_kv_pool_allocator`` already takes
``current_platform.get_paged_allocator_cls()`` for OOT platforms.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.models.minimax.kv_pools import KunlunMiniMaxSparseKVPool


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator."
    "_build_minimax_sparse_kv_pool",
    type=HookType.AROUND,
)
def build_minimax_sparse_kv_pool_kunlun(
    original_fn, self, *, max_total_num_tokens: int
):
    """Instantiate the Kunlun MiniMax sparse pool subclass."""
    import sglang.srt.mem_cache.kv_cache_configurator as kv_cache_configurator

    original_pool_cls = kv_cache_configurator.MiniMaxSparseKVPool
    kv_cache_configurator.MiniMaxSparseKVPool = KunlunMiniMaxSparseKVPool
    try:
        return original_fn(self, max_total_num_tokens=max_total_num_tokens)
    finally:
        kv_cache_configurator.MiniMaxSparseKVPool = original_pool_cls
