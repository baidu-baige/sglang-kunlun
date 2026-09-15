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
"""Disaggregation patches for Kunlun (P800).

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/disaggregation/__init__.py

The community decode/prefill code sets ``kv_args.kv_data_lens`` /
``state_data_lens`` directly from the kv-pool buffer infos. On Kunlun, RDMA
registration requires 2MiB-aligned lengths, so we round each entry up before
``KVManager`` is built. This wraps ``get_contiguous_buf_infos`` /
``get_state_buf_infos`` on the kv-pool instances right before
``_init_kv_manager`` is invoked.

Implementation note: this patch wraps an instance-level method bound at
runtime (after ``self.token_to_kv_pool`` exists), not a class-level method.
The plugin framework's REPLACE works on class targets only, so we install
this with an AROUND hook on the queue's ``_init_kv_manager`` class method.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

ALIGNMENT_2M = 2 * 1024 * 1024


def _align(values):
    return [(x + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M for x in values]


def _wrap_buf_infos(orig):
    def _wrapped(*args, **kwargs):
        ptrs, lens, items = orig(*args, **kwargs)
        return ptrs, _align(lens), items

    return _wrapped


def _patch_pool(pool):
    if pool is None or getattr(pool, "_kunlun_2m_aligned", False):
        return
    if hasattr(pool, "get_contiguous_buf_infos"):
        pool.get_contiguous_buf_infos = _wrap_buf_infos(pool.get_contiguous_buf_infos)
    if hasattr(pool, "get_state_buf_infos"):
        pool.get_state_buf_infos = _wrap_buf_infos(pool.get_state_buf_infos)
    pool._kunlun_2m_aligned = True


def _kunlun_init_kv_manager_around(original_fn, self):
    """AROUND wrapper: align kv-pool buf-infos to 2MiB before init."""
    _patch_pool(getattr(self, "token_to_kv_pool", None))
    _patch_pool(getattr(self, "draft_token_to_kv_pool", None))
    if getattr(self, "scheduler", None) is not None and getattr(
        self.scheduler, "enable_hisparse", False
    ):
        _patch_pool(self.scheduler.hisparse_coordinator.mem_pool_host)
    return original_fn(self)


# Register AROUND on both decode and prefill queues
plugin_hook(
    "sglang.srt.disaggregation.decode.DecodePreallocQueue._init_kv_manager",
    type=HookType.AROUND,
)(_kunlun_init_kv_manager_around)

plugin_hook(
    "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue._init_kv_manager",
    type=HookType.AROUND,
)(_kunlun_init_kv_manager_around)

# Kunlun P800: per-buffer RDMA registration.
from . import mooncake_register  # noqa: F401,E402
