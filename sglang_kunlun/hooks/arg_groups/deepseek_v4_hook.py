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

``apply_deepseek_v4_defaults`` asserts ``kv_cache_dtype in ("fp8_e4m3",
"bfloat16")`` (deepseek_v4_hook.py:37-40), but it cannot simply be skipped for
float16: it also pins ``attention_backend="dsv4"``, ``page_size``,
``max_running_requests``, ``swa_full_tokens_ratio``, the EAGLE constraints and the
NVFP4 MoE runner default. So run the original with the dtype temporarily set to
``bfloat16`` to satisfy the assert, then restore float16 in ``finally``.

On success the dtype is normalized to ``"float16"``; if the original raised, the
user's original spelling (``float16`` or ``half``) is restored so the error text
reflects what was actually requested.

``server_args.py:3513`` imports this function inside the function body, so the
call always resolves through the module attribute — nothing to propagate.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.arg_groups.deepseek_v4_hook.apply_deepseek_v4_defaults",
    type=HookType.AROUND,
)
def apply_deepseek_v4_defaults_kunlun(original_fn, server_args, model_arch):
    """Run the stock V4 defaults with float16 masked as bfloat16."""
    original_dtype = server_args.kv_cache_dtype
    forced_float16 = original_dtype in ("float16", "half")
    if forced_float16:
        server_args.kv_cache_dtype = "bfloat16"
    succeeded = False
    try:
        original_fn(server_args, model_arch)
        succeeded = True
        server_args.attention_backend = "kunlun_dsv4"
    finally:
        if forced_float16:
            server_args.kv_cache_dtype = (
                "float16" if succeeded else original_dtype
            )
