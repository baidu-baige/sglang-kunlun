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
"""Hooks for ``sglang.srt.mem_cache.allocation`` -- AttentionStore staging slots.

Ports ``_alloc_chunk_tail_slots_for_batch`` plus its call at the end of
``alloc_for_extend`` from the reference implementation.

The hook targets the name as bound in ``managers.schedule_batch``, not
``mem_cache.allocation`` where ``alloc_for_extend`` is defined.
``schedule_batch.py`` does ``from sglang.srt.mem_cache.allocation import
alloc_for_extend`` at import time and plugin hooks are applied after that
module is loaded, so patching the definition site would leave the one real
caller holding the original function. ``schedule_batch.py:2751`` is that caller.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import asradix_enabled

logger = logging.getLogger(__name__)


def _alloc_chunk_tail_slots_for_batch(batch) -> None:
    """Reserve chunk_tail_swa_pool slots for the swa_step checkpoints this extend
    crosses, so ``detect_and_may_copy_as_swa`` has somewhere to rescue the tail
    SWA window to before eviction drops it.

    One slot per checkpoint, appended to ``req.as_swa_kv_slots``. A ``-1`` entry
    is the "no SWA data for this checkpoint" sentinel and must be skipped by the
    copy, write and free paths alike.
    """
    kvcache = batch.tree_cache.token_to_kv_pool_allocator.get_kvcache()
    pool = getattr(kvcache, "chunk_tail_swa_pool", None)
    if pool is None:
        return
    policy = kvcache.swa_policy
    new_lens = batch.seq_lens_cpu.tolist()
    for i, (req, new_len) in enumerate(zip(batch.reqs, new_lens)):
        extend_len = batch.extend_lens[i]
        prev_checkpoints = policy.num_checkpoints(new_len - extend_len)
        new_checkpoints = policy.num_checkpoints(new_len)

        # First extend: placeholder the whole hit prefix with -1 and never
        # back-fill it. SWA is a sliding window, so only the last
        # `sliding_window` prefix tokens survive in HBM -- out-of-window prefix
        # checkpoints have no SWA left to copy. An AS-hit prefix is already in
        # AttentionStore; an HBM-hit prefix was written when it was first
        # prefilled. Continuity still holds: the first new checkpoint's window
        # covers the retained tail tokens.
        if not req.as_swa_kv_slots and prev_checkpoints > 0:
            req.as_swa_kv_slots.extend([-1] * prev_checkpoints)
            # Lift both water-marks past the un-writable prefix so the
            # incremental write slice [as_written_count:as_swa_copy_count)
            # starts at the first NEW checkpoint. Otherwise an HBM hit leaves
            # as_written_count=0, the slice still contains the leading -1s, and
            # the writer's sentinel guard drops the whole write including the
            # valid new checkpoints. max() also covers the mixed HBM+AS case,
            # where init_load_back only advanced as_written_count over the
            # (shorter) AS segment.
            req.as_written_count = max(req.as_written_count, prev_checkpoints)
            req.as_swa_copy_count = max(req.as_swa_copy_count, prev_checkpoints)

        n_new = new_checkpoints - prev_checkpoints
        if n_new <= 0:
            continue
        # Key the fail-fast off the explicit flag, never off as_swa_kv_slots[-1]:
        # a hit prefix ends in -1 placeholders, which are not alloc failures.
        if req.as_swa_alloc_failed:
            req.as_swa_kv_slots.extend([-1] * n_new)
            continue
        slots = pool.alloc(n_new)
        if slots is None:
            req.as_swa_alloc_failed = True
            req.as_swa_kv_slots.extend([-1] * n_new)
            logger.warning(
                "chunk_tail_swa_pool full for req=%s, need=%d slots, avail=%d, "
                "num_pages=%d (consider raising "
                "SGLANG_KUNLUN_AS_SWA_BUFFER_COEF)",
                req.rid,
                n_new,
                pool.available_size(),
                pool.num_pages,
            )
        else:
            req.as_swa_kv_slots.extend(slots)


@plugin_hook(
    "sglang.srt.managers.schedule_batch.alloc_for_extend",
    type=HookType.AROUND,
)
def alloc_for_extend_kunlun(original_fn, batch):
    """Allocate staging slots for the checkpoints this extend just crossed.

    Upstream calls this on the last line of ``alloc_for_extend`` before the
    return, which is where an AROUND hook lands anyway.
    """
    result = original_fn(batch)
    if asradix_enabled():
        _alloc_chunk_tail_slots_for_batch(batch)
    return result
