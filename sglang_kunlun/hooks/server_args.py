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
"""Hooks for ``sglang.srt.server_args``.

Two of the four AROUND hooks that enable a **float16 KV cache** for DeepSeek V4 on
the Kunlun torch stack. Upstream only ever allows ``fp8_e4m3`` / ``bfloat16`` and
rejects ``float16`` in four places; the other two live in
``sglang_kunlun/hooks/model_executor/model_runner.py`` and
``sglang_kunlun/hooks/arg_groups/deepseek_v4_hook.py``. All four are needed
together — with only some applied, a ``--kv-cache-dtype float16`` launch fails on
whichever check is still stock (loudly, with upstream's own assert message).

- ``ServerArgs.add_cli_args`` — append ``float16`` / ``half`` to the
  ``--kv-cache-dtype`` argparse choices, otherwise the value is rejected during
  CLI parsing before any of the other hooks get a chance to run. The target is a
  ``staticmethod``; ``_apply_target`` unwraps ``__func__`` before wrapping and
  re-applies ``staticmethod`` afterwards, so the hook sees the plain
  ``(parser)`` signature.
- ``ServerArgs._set_default_dsa_kv_cache_dtype`` — upstream asserts the dtype is
  ``bfloat16`` or ``fp8_e4m3`` (server_args.py:3426). Normalize ``half`` to
  ``float16`` and return early; every other value still goes through the original.

``HookRegistry._patched`` replaces the legacy ``_kunlun_float16_kv_patched`` flag
and its lock, since ``apply_hooks()`` runs once per process and skips
already-patched targets.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_FLOAT16_ALIASES = ("float16", "half")


@plugin_hook(
    "sglang.srt.server_args.ServerArgs.add_cli_args",
    type=HookType.AROUND,
)
def add_cli_args_kunlun(original_fn, parser):
    """Widen the ``--kv-cache-dtype`` choices with ``float16`` / ``half``."""
    result = original_fn(parser)
    for action in getattr(parser, "_actions", []):
        if getattr(action, "dest", None) == "kv_cache_dtype" and action.choices:
            choices = list(action.choices)
            for dtype in _FLOAT16_ALIASES:
                if dtype not in choices:
                    choices.append(dtype)
            action.choices = choices
    return result


@plugin_hook(
    "sglang.srt.server_args.ServerArgs._set_default_dsa_kv_cache_dtype",
    type=HookType.AROUND,
)
def _set_default_dsa_kv_cache_dtype_kunlun(original_fn, self, major, quantization):
    """Accept float16 instead of asserting on the bf16/fp8_e4m3 whitelist."""
    if self.kv_cache_dtype in _FLOAT16_ALIASES:
        self.kv_cache_dtype = "float16"
        return
    return original_fn(self, major, quantization)
