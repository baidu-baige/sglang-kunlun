# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator.configure",
    type=HookType.AFTER,
)
def _ensure_full_to_swa_mapping(result, self, **kwargs):
    """Make sure the DSV4 KV pool exposes ``full_to_swa_index_mapping``.

    The internal fork builds the pools in ``KVCacheConfigurator.configure`` and
    returns them in a ``KVCacheConfigResult``.  Only ``_init_pools``'s
    ``elif not uses_unified_virtual_ids`` branch calls
    ``token_to_kv_pool.register_mapping(...)``; under speculative decoding
    (unified virtual ids) the *target* pool can be left with the attribute
    still ``None``, and the DSV4 verify path then dies with
    ``TypeError: 'NoneType' object is not subscriptable`` in
    ``build_causal_swa_page_indices``.  Re-register it from the allocator that
    owns the mapping.
    """
    if result is None:
        return None
    pool = getattr(result, "token_to_kv_pool", None)
    alloc = getattr(result, "token_to_kv_pool_allocator", None)
    if pool is None or alloc is None:
        return None
    if not hasattr(pool, "register_mapping"):
        return None
    if getattr(pool, "full_to_swa_index_mapping", None) is not None:
        return None
    mapping = getattr(alloc, "full_to_swa_index_mapping", None)
    if mapping is None:
        return None
    pool.register_mapping(mapping)
    logger.info(
        "sglang-kunlun: registered full_to_swa_index_mapping on %s (from %s)",
        type(pool).__name__,
        type(alloc).__name__,
    )
    return None


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._init_pools",
    type=HookType.AFTER,
)
def _fix_swa_pools(result, self, **kwargs):
    """... (see below) ...

    ``**kwargs`` absorbs the keywords newer sglang passes to ``_init_pools``
    (``sizes`` / ``req_to_token_pool`` / ``token_to_kv_pool_allocator``).

    On sglang releases where ``_init_pools`` still lives on
    ``ModelRunnerKVCacheMixin``, ``self`` is the model runner and the body
    below applies.  Where it moved to ``KVCacheConfigurator`` (internal fork),
    ``self`` is a configurator that owns none of these attributes and the
    pools come back as the return value instead, so bail out and let the
    in-tree implementation handle it.
    """
    if not hasattr(self, "token_to_kv_pool") and not hasattr(
        self, "full_max_total_num_tokens"
    ):
        return
    """Replace non-SWA pools with SWA-aware versions when is_hybrid_swa=True.

    AFTER signature: hook(result, self) — result is None (original returns None).

    Two cases handled:
      - Target worker: both token_to_kv_pool and token_to_kv_pool_allocator are
        plain MHA types → replace both.
      - Draft worker: token_to_kv_pool_allocator is already SWATokenToKVPoolAllocator
        (shared from target worker via get_memory_pool()), but token_to_kv_pool is a
        fresh MHATokenToKVPool → replace pool only, keep shared allocator.
    """
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    if os.environ.get("KUNLUN_DEBUG_SWA_POOLS"):
        import sys as _sys

        _pool = getattr(self, "token_to_kv_pool", None)
        _alloc = getattr(self, "token_to_kv_pool_allocator", None)
        print(
            "[kdebug] _fix_swa_pools "
            f"is_hybrid_swa={getattr(self, 'is_hybrid_swa', None)} "
            f"is_hybrid_swa_compress={getattr(self, 'is_hybrid_swa_compress', None)} "
            f"is_draft_worker={getattr(self, 'is_draft_worker', None)} "
            f"draft_swa_full_capacity={getattr(self, 'draft_swa_full_capacity', None)} "
            f"pool={type(_pool).__name__} "
            f"pool.full_to_swa={'set' if getattr(_pool, 'full_to_swa_index_mapping', None) is not None else 'None'} "
            f"alloc={type(_alloc).__name__}",
            file=_sys.stderr,
            flush=True,
        )

    if not getattr(self, "is_hybrid_swa", False):
        return

    # DeepSeek V4 is hybrid-SWA but uses its own DeepSeekV4TokenToKVPool (a
    # dedicated c4/c128 ring-buffer pool), NOT SWAKVPool. On the CUDA path
    # mainline wraps that pool with a SWATokenToKVPoolAllocator (see
    # model_runner_kv_cache_mixin._init_pools final else-branch), which calls
    # ``pool.register_mapping()`` and installs ``full_to_swa_index_mapping``.
    # The Kunlun OOT path short-circuits to ``get_paged_allocator_cls()``
    # (KunlunPagedTokenToKVPoolAllocator), which never registers that mapping,
    # so the dsv4 attention backend later hits
    # ``AttributeError: ... has no attribute 'full_to_swa_index_mapping'``.
    # Fix: give DSV4 the SWA allocator (keeping its own DeepSeekV4 pool).
    is_dsv4 = False
    try:
        from sglang.srt.configs.model_config import is_deepseek_v4

        is_dsv4 = is_deepseek_v4(self.model_config.hf_config)
    except Exception:
        is_dsv4 = getattr(self.model_config, "is_deepseek_v4_arch", False)

    if is_dsv4:
        if isinstance(
            getattr(self, "token_to_kv_pool_allocator", None),
            SWATokenToKVPoolAllocator,
        ):
            return  # already correct
        if self.is_draft_worker:
            return  # draft shares target's allocator; nothing to rebuild
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
            self.full_max_total_num_tokens,
            self.swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
        )
        # SWATokenToKVPoolAllocator.__init__ already called
        # self.token_to_kv_pool.register_mapping(full_to_swa_index_mapping).
        return



    pool_ok = isinstance(getattr(self, "token_to_kv_pool", None), SWAKVPool)
    alloc_ok = isinstance(
        getattr(self, "token_to_kv_pool_allocator", None), SWATokenToKVPoolAllocator
    )

    if pool_ok and alloc_ok:
        return  # both already correct — nothing to do

    from sglang.srt.layers.dp_attention import get_attention_tp_size
    from sglang.srt.platforms import current_platform

    kwargs = {}
    if getattr(self, "is_hybrid_swa_compress", False):
        kwargs = {
            "swa_head_num": max(
                1,
                self.model_config.hf_text_config.swa_num_key_value_heads
                // get_attention_tp_size(),
            ),
            "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
            "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
            "v_head_dim": self.model_config.hf_text_config.v_head_dim,
        }

    if not pool_ok:
        # Drop old non-SWA pool reference (Python GC recycles XPU memory)
        self.token_to_kv_pool = None
        PoolCls = current_platform.get_mha_kv_pool_cls()
        self.token_to_kv_pool = SWAKVPool(
            size=self.full_max_total_num_tokens,
            size_swa=self.swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
            full_attention_layer_ids=self.model_config.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
            token_to_kv_pool_class=PoolCls,
            **kwargs,
        )

    if not alloc_ok:
        self.token_to_kv_pool_allocator = None
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
            self.full_max_total_num_tokens,
            self.swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
        )

    self.token_to_kv_pool.full_to_swa_index_mapping = (
        self.token_to_kv_pool_allocator.full_to_swa_index_mapping
    )


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool_allocator",
    type=HookType.AROUND,
)
def _build_token_to_kv_pool_allocator_kunlun(
    original_fn,
    self,
    *,
    sizes,
    token_to_kv_pool,
    is_dsv4_model,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    """Give hybrid-SWA models the SWA allocator on Kunlun.

    The in-tree builder short-circuits to ``current_platform.get_paged_allocator_cls()``
    whenever the platform is out-of-tree and *ignores* ``is_hybrid_swa``; a later
    ``assert isinstance(swa_allocator, SWATokenToKVPoolAllocator)`` then fails
    because the Kunlun allocator is a sibling of ``SWATokenToKVPoolAllocator``,
    not a subclass.  Build the SWA allocator ourselves for that combination
    (mirroring the NPU branch in the same function).
    """
    from sglang.srt.platforms import current_platform

    if not (
        current_platform.is_out_of_tree() and getattr(self, "is_hybrid_swa", False)
    ):
        return original_fn(
            self,
            sizes=sizes,
            token_to_kv_pool=token_to_kv_pool,
            is_dsv4_model=is_dsv4_model,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    if token_to_kv_pool_allocator is not None:
        return token_to_kv_pool_allocator

    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.runtime_context import get_disagg, get_schedule

    need_sort = get_disagg().disaggregation_mode in ("decode", "prefill")
    return SWATokenToKVPoolAllocator(
        sizes.full_max_total_num_tokens,
        sizes.swa_max_total_num_tokens,
        page_size=get_schedule().page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=token_to_kv_pool,
        need_sort=need_sort,
    )
