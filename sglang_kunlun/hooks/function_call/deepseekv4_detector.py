"""Hooks for ``sglang.srt.function_call.deepseekv4_detector``.
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