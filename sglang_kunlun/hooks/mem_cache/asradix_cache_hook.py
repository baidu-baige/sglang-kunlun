# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Hooks for the ``mem_cache`` wiring that selects the AttentionStore cache.

Three injection points from aiak_sglang 1d855da5 (luno-5195):

* ``KVCacheConfigurator._build_dsv4_kv_pool`` -- upstream threads
  ``max_context_length=self.model_config.context_len`` into the DSV4 pool so the
  chunk-tail staging pool can be sized when chunked prefill is off. A plugin
  cannot add that keyword, so the value is stashed for the pool's own hook.
* ``registry.default_radix_cache_factory`` -- return the AttentionStore cache
  adapter (SPA for V4, SPAG for V4.1) when the cache is on.
* ``release_kv_cache`` -- free the chunk-tail SWA slots this request still owns.

The ``release_kv_cache`` hook targets the name as bound in
``disaggregation.prefill``, not ``mem_cache.common`` where it is defined.
``prefill.py`` does ``from sglang.srt.mem_cache.common import release_kv_cache``
at import time, and plugin hooks are applied after that module is loaded, so
patching the definition site would leave the caller holding the original. The
prefill module is the only binding that matters here: AttentionStore is
rejected outright on decode instances.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import asradix_enabled

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_dsv4_kv_pool",
    type=HookType.AROUND,
)
def build_dsv4_kv_pool_kunlun(original_fn, self, *args, **kwargs):
    """Stash the pool-sizing context, then build the pool as usual."""
    from sglang_kunlun.hooks.mem_cache.asradix_pool_hook import stash_pool_context

    context_len = getattr(self.model_config, "context_len", None)
    stash_pool_context(context_len, bool(getattr(self, "is_draft_worker", False)))
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        # Do not let a stale value leak into an unrelated later construction.
        stash_pool_context(None, False)


@plugin_hook(
    "sglang.srt.mem_cache.registry.default_radix_cache_factory",
    type=HookType.AROUND,
)
def default_radix_cache_factory_kunlun(original_fn, ctx):
    """Select the AttentionStore cache adapter when the cache is on.

    Upstream inserts this branch after the C++ radix tree check and before the
    unified-cache external linker, so run it ahead of the stock chain: both of
    those are mutually exclusive with AttentionStore anyway, and the arg-group
    guard has already rejected the combinations that are not.
    """
    if not asradix_enabled():
        return original_fn(ctx)

    from sglang_kunlun.mem_cache.asradix.cache import create_attention_store_cache
    from sglang_kunlun.mem_cache.asradix.plans import resolve_attention_store_plan

    params = ctx.params
    plan = resolve_attention_store_plan(
        ctx.model_config,
        params.token_to_kv_pool_allocator.get_kvcache(),
    )
    logger.info("[AS] radix cache backend: AttentionStore plan=%s", plan.name)
    return create_attention_store_cache(
        plan,
        params=params,
        server_args=ctx.server_args,
    )


@plugin_hook(
    "sglang.srt.disaggregation.prefill.release_kv_cache",
    type=HookType.AROUND,
)
def release_kv_cache_kunlun(original_fn, req, tree_cache, is_insert: bool = True):
    """Free the chunk-tail SWA slots this request still owns.

    Ownership is split at ``as_written_count``:

      ``[0, as_written_count)``   already handed to an AS write task, which holds
                                  them until it retires; freeing here would be a
                                  double-free. ``retire_ascache_writes()`` does it.
      ``[as_written_count, ...)`` allocated but never submitted (copied-not-
                                  written, pool-full skips, or the request ended
                                  first). Nothing else will free these.

    Upstream places this inside ``release_kv_cache`` after ``cache_finished_req``
    so a write submitted there has already advanced ``as_written_count``; running
    after the whole function preserves that ordering.
    """
    result = original_fn(req, tree_cache, is_insert)

    if not asradix_enabled() or not getattr(req, "as_swa_kv_slots", None):
        return result

    from sglang_kunlun.mem_cache.chunk_tail_swa_pool import free_chunk_tail_swa_slots

    kvcache = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    free_chunk_tail_swa_slots(kvcache, req.as_swa_kv_slots[req.as_written_count :])
    req.as_swa_kv_slots = []
    return result
