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

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._init_pools",
    type=HookType.AFTER,
)
def _fix_swa_pools(result, self):
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
