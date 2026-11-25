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
"""Hooks for ``sglang.srt.disaggregation.prefill`` -- AttentionStore SWA rescue.

The upstream reference implementation calls
``self.detect_and_may_copy_as_swa(req)`` from two places inside
``SchedulerDisaggregationPrefillMixin``: once in
``process_batch_result_disagg_prefill`` right after the final chunk finishes,
and once in the chunked path immediately before
``maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)``.

Both sites sit mid-method, which an AROUND hook cannot reach. They do however
share an anchor: every one of them is immediately followed by a
``maybe_cache_unfinished_req`` call on the same request (prefill.py:852 for the
final chunk, :1209 for the chunked path), so hooking that callee reproduces the
placement. The hook targets the name as bound in ``disaggregation.prefill``,
because that module imports it by name at load time and plugin hooks are
applied later.

One behavioural difference: upstream deliberately copies *before* the
early-continues in ``process_batch_result_disagg_prefill``, so an aborted or
force-retried request still has its window rescued. Anchoring on
``maybe_cache_unfinished_req`` runs after those continues, so such requests skip
the copy. That is harmless -- they never reach an AttentionStore write, and
``release_kv_cache`` frees their staging slots -- but it is a difference, and if
a future abort path ever needs the checkpoint the anchor has to move.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import asradix_enabled

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req",
    type=HookType.AROUND,
)
def maybe_cache_unfinished_req_kunlun(original_fn, req, tree_cache, *args, **kwargs):
    """Rescue this request's committed SWA checkpoints before the cache insert.

    Strictly before the original: ``cache_unfinished_req`` evicts SWA beyond
    ``pre_len - max(sliding_window, page_size)``, which is exactly the window
    these checkpoints need, and it is also what runs the AttentionStore write
    path that submits them.
    """
    if asradix_enabled():
        from sglang_kunlun.hooks.managers.asradix_scheduler_hook import (
            current_scheduler,
        )

        scheduler = current_scheduler()
        if scheduler is not None and getattr(
            scheduler, "enable_asradix_cache", False
        ):
            try:
                scheduler.detect_and_may_copy_as_swa(req)
            except Exception:
                logger.exception(
                    "[AS] detect_and_may_copy_as_swa failed for rid=%s; "
                    "this request's checkpoints are lost",
                    getattr(req, "rid", None),
                )
    return original_fn(req, tree_cache, *args, **kwargs)
