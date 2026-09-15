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
"""Kunlun: register mooncake KV/state buffers ONE AT A TIME with 2MB-rounded length.

Batched registration (``batch_register`` of the whole kv_data list in one call) is
all-or-nothing in mooncake: if ANY buffer fails ``ibv_reg_mr`` the whole batch is
rejected and NONE of the KV segments get published to the metadata server, so the
prefill peer later cannot resolve decode's KV address
(``transfer_metadata.cpp: cannot get mmapped addr`` -> ``Failed to send kv chunk``).
Registering per-buffer isolates failures and guarantees the good KV/state buffers
are registered and synced. Lengths are rounded up to 2MB (Kunlun RDMA requires
2MB-aligned MR start+length; buffers are 2MB-aligned and over-allocated by
``hooks/mem_cache/_align``).
"""
from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

ALIGNMENT_2M = 2 * 1024 * 1024


def _reg_one(engine, ptr, length):
    """Register a single buffer with 2MB-rounded length; return 0 on success."""
    l2 = (int(length) + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M
    try:
        rc = engine.batch_register([int(ptr)], [l2])
        return 0 if (rc == 0 or rc is None) else 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("kunlun mooncake: register 0x%x len=%d failed: %s", int(ptr), l2, exc)
        return 1


@plugin_hook(
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.register_buffer_to_engine",
    type=HookType.REPLACE,
)
def register_buffer_to_engine_kunlun(self):
    """Register KV/aux/state buffers one at a time (isolates ibv_reg_mr failures)."""
    ka = self.kv_args
    ok = fail = 0
    groups = []
    if ka.kv_data_ptrs and ka.kv_data_lens:
        groups.append((list(ka.kv_data_ptrs), list(ka.kv_data_lens)))
    if ka.aux_data_ptrs and ka.aux_data_lens:
        groups.append((list(ka.aux_data_ptrs), list(ka.aux_data_lens)))
    for ptrs, lens in zip(ka.state_data_ptrs or [], ka.state_data_lens or []):
        if ptrs and lens:
            groups.append((list(ptrs), list(lens)))
    for ptrs, lens in groups:
        for p, l in zip(ptrs, lens):
            if _reg_one(self.engine, p, l) == 0:
                ok += 1
            else:
                fail += 1
    logger.info("kunlun mooncake: per-buffer registration ok=%d fail=%d", ok, fail)
