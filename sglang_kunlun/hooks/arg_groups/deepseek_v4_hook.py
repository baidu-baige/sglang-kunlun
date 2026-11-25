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
"""Hooks for ``sglang.srt.arg_groups.deepseek_v4_hook``.

Part of the float16 KV cache support for DeepSeek V4 on Kunlun (see
``sglang_kunlun/hooks/server_args.py`` for the full picture).

The upgraded arg-group resolver seals ``server_args`` during resolution: raw
``server_args.<field> = ...`` writes now raise (they would clobber the
operator's input), and a resolution decision must go through
``declare_resolution(server_args, source, **fields)``. So instead of masking
``kv_cache_dtype`` as bfloat16 around the whole ``apply_deepseek_v4_defaults``
run and restoring it with raw writes, tolerate float16 directly in the single
pass that rejects it (``_deepseek_v4_kv_cache_dtype``), and pin the Kunlun DSV4
attention backend through ``declare_resolution``.
"""

from __future__ import annotations

import logging
import os

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_FLOAT16_ALIASES = ("float16", "half")
_ALLOW_CP_DP_ENV = "SGLANG_KUNLUN_ALLOW_DSPARK_CP_DP"
logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
    type=HookType.AROUND,
)
def validate_deepseek_v41_features_kunlun(original_fn, server_args):
    """Let an explicit opt-in start DSpark with CP/DP greater than one.

    The stock V4.1 check rejects DSpark PD whenever dp_size, DP attention or
    attn_cp_size leaves 1. That clause is a policy guard rather than a shape or
    dtype check, and the mat-036 phase exists to find the real runtime boundary.
    With ``SGLANG_KUNLUN_ALLOW_DSPARK_CP_DP=1`` the guard is skipped and the
    run proceeds; without it the upstream error is re-raised unchanged. Every
    other V4.1 check still runs first, so an unsupported feature or a non-static
    verify mode still fails.

    The only code after this clause is the prefill-graph max_seq_len
    declaration, which does not apply while the prefill graph is disabled (the
    P role here). Re-enable this guard before capturing a prefill graph.
    """
    try:
        return original_fn(server_args)
    except ValueError as exc:
        if (os.environ.get(_ALLOW_CP_DP_ENV) == "1"
                and "DSpark PD requires static verify" in str(exc)):
            logger.warning(
                "sglang-kunlun: %s=1; continuing past the upstream CP/DP guard: %s",
                _ALLOW_CP_DP_ENV, exc,
            )
            return None
        raise


@plugin_hook(
    "sglang.srt.arg_groups.deepseek_v4_hook._deepseek_v4_kv_cache_dtype",
    type=HookType.AROUND,
)
def _deepseek_v4_kv_cache_dtype_kunlun(original_fn, view):
    """Keep a float16 KV cache instead of asserting on the fp8/bf16 whitelist.

    The stock pass only defaults ``auto`` and then validates the result; for an
    explicit float16/half there is nothing to default, so returning ``{}``
    leaves the operator's input intact and skips the whitelist assert."""
    if getattr(view, "kv_cache_dtype", None) in _FLOAT16_ALIASES:
        return {}
    return original_fn(view)


@plugin_hook(
    "sglang.srt.arg_groups.deepseek_v4_hook.apply_deepseek_v4_defaults",
    type=HookType.AROUND,
)
def apply_deepseek_v4_defaults_kunlun(original_fn, server_args, model_arch):
    """Run the stock V4 defaults, then pin the Kunlun DSV4 attention backend."""
    original_fn(server_args, model_arch)
    declare_resolution(
        server_args,
        "sglang_kunlun.deepseek_v4",
        attention_backend="kunlun_dsv4",
    )
