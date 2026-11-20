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
"""Hooks for ``sglang.srt.arg_groups.kv_cache_hook`` -- AttentionStore guards.

Ports the ``--enable-asradix-cache`` validation block that the upstream reference
adds to ``handle_cache_compatibility``. The knobs themselves are env vars here
(a plugin cannot extend the ``memory`` msgspec namespace), so the messages name
the env var rather than a CLI flag, but the four constraints are unchanged:

* the AttentionStore cache is layered on the radix cache and cannot run without it
* it conflicts with HiCache -- both claim the secondary tier
* a PD decode instance never writes KV it did not prefill, so it is P-only
* the KV read/write path addresses whole pages, so ``page_size`` must exceed 1

``enable_as_swa_pool`` needs no ``declare_resolution`` counterpart: upstream
derives it from ``enable_asradix_cache`` through the resolver because it lives
on ``server_args``, whereas here ``AsCacheConfig.enable_as_swa_pool`` is a
property over the same source.
"""

from __future__ import annotations

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import asradix_enabled

_ENV = "SGLANG_KUNLUN_ENABLE_ASRADIX_CACHE"


@plugin_hook(
    "sglang.srt.arg_groups.kv_cache_hook.handle_cache_compatibility",
    type=HookType.AROUND,
)
def handle_cache_compatibility_kunlun(original_fn, server_args):
    """Run the stock compatibility pass, then the AttentionStore guards."""
    original_fn(server_args)

    if not asradix_enabled():
        return

    cfg = resolving_view(server_args)

    if cfg.disable_radix_cache:
        raise ValueError(
            "AttentionStore (asradix) requires the radix cache to be enabled. "
            f"Unset --disable-radix-cache or turn off {_ENV}."
        )
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "HiCache and AttentionStore cannot be enabled simultaneously. "
            f"Use only one of --enable-hierarchical-cache and {_ENV}."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "AttentionStore (asradix) is not supported on a PD decode instance. "
            f"Unset {_ENV} for decode (D) servers; set it only on prefill (P)."
        )
    if cfg.page_size <= 1:
        raise ValueError(
            "AttentionStore KV read/write requires --page-size > 1, "
            f"got {cfg.page_size}."
        )
