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
"""Config adapter for the AttentionStore (asradix) secondary KV cache.

The reference implementation puts these knobs on the ``memory`` namespace, so its code
reads them off the object ``get_memory()`` returns. A plugin cannot add fields
to that ``msgspec.Struct`` once the class is built, so the knobs live in
``sglang_kunlun.environ`` instead and this module presents them under the
upstream field names. The vendored asradix code therefore stays verbatim: it
just receives this object where it used to receive ``get_memory()``.

``enable_as_swa_pool`` is derived rather than configured, matching upstream's
``declare_resolution(..., enable_as_swa_pool=bool(cfg.enable_asradix_cache))``
in ``arg_groups/kv_cache_hook.py`` -- the staging pool exists only to feed
AttentionStore writes, so it is not an operator knob.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_kunlun.environ import envs


@dataclass(frozen=True)
class AsCacheConfig:
    """The ``memory``-namespace fields the asradix backend reads."""

    enable_asradix_cache: bool
    per_node_gpu_num: int
    swa_step: int
    enable_dynamic_swa_step: bool
    swa_step_tiers: str
    as_swa_buffer_coef: float

    @property
    def enable_as_swa_pool(self) -> bool:
        """Whether to allocate the chunk-tail SWA staging pool.

        Derived, not configured: the pool exists only to feed AttentionStore
        writes, so it follows ``enable_asradix_cache`` rather than being an
        operator knob. Upstream gets the same effect through
        ``declare_resolution(..., enable_as_swa_pool=bool(cfg.enable_asradix_cache))``
        in ``arg_groups/kv_cache_hook.py``.
        """
        return self.enable_asradix_cache


def as_cfg() -> AsCacheConfig:
    """Read the AttentionStore knobs out of the environment.

    Not cached: ``EnvField.override()`` is how the tests flip these, and the
    scheduler reads the config once at init anyway.
    """
    return AsCacheConfig(
        enable_asradix_cache=envs.SGLANG_KUNLUN_ENABLE_ASRADIX_CACHE.get(),
        per_node_gpu_num=envs.SGLANG_KUNLUN_PER_NODE_GPU_NUM.get(),
        swa_step=envs.SGLANG_KUNLUN_AS_SWA_STEP.get(),
        enable_dynamic_swa_step=envs.SGLANG_KUNLUN_ENABLE_DYNAMIC_SWA_STEP.get(),
        swa_step_tiers=envs.SGLANG_KUNLUN_SWA_STEP_TIERS.get(),
        as_swa_buffer_coef=envs.SGLANG_KUNLUN_AS_SWA_BUFFER_COEF.get(),
    )


def asradix_enabled() -> bool:
    """Whether the AttentionStore secondary KV cache is turned on."""
    return bool(envs.SGLANG_KUNLUN_ENABLE_ASRADIX_CACHE.get())
