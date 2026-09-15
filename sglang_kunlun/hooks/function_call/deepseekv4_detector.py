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
"""Hooks for ``sglang.srt.function_call.deepseekv4_detector``.

Two AROUND hooks on ``DeepSeekV4Detector`` that loosen non-stream tool-call
parsing, because the "image-aligned" grammar-force approach does NOT work for the
P800 w8a8_int8 checkpoint (verified):

- ``tool_choice=auto``: sglang's structural tag only constrains AFTER the model
  emits the ``<｜DSML｜tool_calls>`` trigger, but this checkpoint never emits that
  trigger voluntarily (it outputs bare JSON), so auto stays unconstrained and
  yields no tool_calls under the stock wrapper-only parser;
- ``tool_choice=required`` emits a bare ``<｜DSML｜invoke ...>`` (invoke-only
  structural tag, no wrapper), which the stock wrapper-only parser also drops.

So ``detect_and_parse`` additionally accepts (1) wrapper-less invoke blocks and
(2) a bare / preamble-embedded JSON object whose function name matches an
available tool; ``has_tool_call`` widens accordingly. Streaming already scans
invoke blocks directly and is left alone.

Gated on ``SGLANG_KUNLUN_ENABLE_LOOSE_TOOLCALL=1``. The legacy shim checked the
env once at install time; the hook checks per call and delegates to
``original_fn`` when off, so the stock parser is bit-identical when disabled.

``detect_and_parse`` / ``has_tool_call`` are defined on the parent
``DeepSeekV32Detector``; ``getattr``/``setattr`` in ``_apply_target`` therefore
lands the wrapper in ``DeepSeekV4Detector.__dict__`` only, leaving V3.2 alone.
"""

from __future__ import annotations

import json
import logging
import os
import re

from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _loose_enabled() -> bool:
    return os.environ.get("SGLANG_KUNLUN_ENABLE_LOOSE_TOOLCALL") == "1"


@plugin_hook(
    "sglang.srt.function_call.deepseekv4_detector.DeepSeekV4Detector.detect_and_parse",
    type=HookType.AROUND,
)
def detect_and_parse_kunlun(original_fn, self, text, tools):
    """Accept wrapper-less invoke blocks and bare/embedded JSON tool calls."""
    if not _loose_enabled():
        return original_fn(self, text, tools)

    # (1) DSML invoke blocks, wrapper optional
    try:
        invoke_iter = list(re.finditer(self.invoke_regex, text, re.DOTALL))
    except Exception:  # noqa: BLE001
        invoke_iter = []
    if invoke_iter:
        head = text[: invoke_iter[0].start()].replace(self.bot_token, "")
        normal_text = head.rstrip().removesuffix("\n\n")
        calls = []
        for m in invoke_iter:
            try:
                name, body, _complete = self._unpack_invoke_match(m)
                args = self._parse_parameters_from_xml(body)
                match_result = {"name": name, "parameters": json.loads(args)}
                calls.extend(self.parse_base_json(match_result, tools))
            except Exception as e:  # noqa: BLE001
                logger.warning("sglang-kunlun[toolcall]: invoke parse skip (%s)", e)
        if calls:
            return StreamingParseResult(normal_text=normal_text, calls=calls)

    # (2) bare / preamble-embedded JSON: {"function"|"name": <tool>,
    # "arguments"|"parameters": {...}}. Extract the FIRST balanced {...} span
    # (depth-matched) so a leading preamble or a stray trailing brace is tolerated.
    lb = text.find("{")
    if lb != -1:
        depth = 0
        end = -1
        for i in range(lb, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            cand = text[lb : end + 1]
            try:
                obj = json.loads(cand)
            except Exception:  # noqa: BLE001
                obj = None
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments")
                if args is None:
                    args = obj.get("parameters")
                tool_names = {t.function.name for t in tools} if tools else set()
                if name in tool_names:
                    match_result = {"name": name, "parameters": args or {}}
                    calls = self.parse_base_json(match_result, tools)
                    if calls:
                        normal_text = text[:lb].rstrip().removesuffix("\n\n")
                        return StreamingParseResult(
                            normal_text=normal_text, calls=calls
                        )

    # (3) stock behaviour (full wrapper form, or genuinely no tool call)
    return original_fn(self, text, tools)


@plugin_hook(
    "sglang.srt.function_call.deepseekv4_detector.DeepSeekV4Detector.has_tool_call",
    type=HookType.AROUND,
)
def has_tool_call_kunlun(original_fn, self, text):
    """Also report True for a bare JSON function-call object."""
    if original_fn(self, text):
        return True
    if not _loose_enabled():
        return False
    return (
        "{" in text
        and ('"function"' in text or '"name"' in text)
        and ('"arguments"' in text or '"parameters"' in text)
    )
