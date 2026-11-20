# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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

Registration status is read from the single-buffer engine API: ``batch_register``
returns 0 even when ``ibv_reg_mr`` fails inside the C++ layer (only
``rdma_context.cpp:246`` logs the ENOMEM), so a buffer that never reached the RDMA
metadata still counted as registered and the peer later failed with ``Failed to get
segment descriptor``. Every failure is now logged with its group/index/ptr and the
peer-visible consequence is spelled out in one place.
"""
from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

ALIGNMENT_2M = 2 * 1024 * 1024


def _round_2m(length) -> int:
    """Round a buffer length up to the 2MB MR granularity Kunlun RDMA requires."""
    return (int(length) + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M


def _reg_one(engine, ptr, length):
    """Register a single buffer with 2MB-rounded length; return ``(ok, detail)``.

    ``batch_register`` swallows per-buffer ``ibv_reg_mr`` failures (it still
    returns 0), so prefer the single-buffer engine call whose return code reports
    them. The batch path stays as a fallback for engines that do not expose it.
    """
    ptr = int(ptr)
    l2 = _round_2m(length)
    inner = getattr(engine, "engine", None)
    single = getattr(inner, "register_memory", None)
    try:
        if single is not None:
            rc = single(ptr, l2)
        else:
            rc = engine.batch_register([ptr], [l2])
    except Exception as exc:  # noqa: BLE001
        return False, f"exception: {exc!r}"
    if rc is None:
        return True, "rc=None"
    return int(rc) == 0, f"rc={rc}"


@plugin_hook(
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.register_buffer_to_engine",
    type=HookType.REPLACE,
)
def register_buffer_to_engine_kunlun(self):
    """Register KV/aux/state buffers one at a time (isolates ibv_reg_mr failures)."""
    ka = self.kv_args
    groups = []
    if ka.kv_data_ptrs and ka.kv_data_lens:
        groups.append(("kv", list(ka.kv_data_ptrs), list(ka.kv_data_lens)))
    if ka.aux_data_ptrs and ka.aux_data_lens:
        groups.append(("aux", list(ka.aux_data_ptrs), list(ka.aux_data_lens)))
    for idx, (ptrs, lens) in enumerate(
        zip(ka.state_data_ptrs or [], ka.state_data_lens or [])
    ):
        if ptrs and lens:
            groups.append((f"state{idx}", list(ptrs), list(lens)))
    ok = fail = 0
    failed = []
    for name, ptrs, lens in groups:
        for i, (p, l) in enumerate(zip(ptrs, lens)):
            good, detail = _reg_one(self.engine, p, l)
            if good:
                ok += 1
                continue
            fail += 1
            tag = f"{name}[{i}]"
            failed.append(f"{tag} ptr=0x{int(p):x} len={_round_2m(l)}")
            logger.warning(
                "kunlun mooncake: register %s ptr=0x%x len=%d failed (%s)",
                tag,
                int(p),
                _round_2m(l),
                detail,
            )
    logger.info("kunlun mooncake: per-buffer registration ok=%d fail=%d", ok, fail)
    if failed:
        logger.warning(
            "kunlun mooncake: %d buffer(s) missing from RDMA metadata: %s; "
            "prefill peers writing them fail with 'Failed to get segment descriptor'",
            len(failed),
            ", ".join(failed),
        )
