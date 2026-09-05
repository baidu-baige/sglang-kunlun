"""Hooks for ``sglang.srt.parser.reasoning_parser``.

One AROUND hook that makes the DeepSeek-V4 *reasoning* parser tolerate this
checkpoint's habit of emitting a reasoning preamble that ends with a bare
``</think>`` but has NO opening ``<think>`` (notably on tool-result follow-up
turns).

Stock ``_DeepSeekV3Detector`` only enters reasoning mode when it sees ``<think>``
(``force_reasoning=False``), so a stray ``</think>`` leaks into ``content``. We
add a "lazy close" rule to the one-shot path: if the text has ``</think>`` but no
``<think>`` and we're not already in reasoning, split on ``</think>`` (prefix ->
reasoning_content, suffix -> clean content). When no ``</think>`` is present the
text is left as normal content unchanged, so ordinary answers are unaffected.

Target is the class the ``ReasoningParser.DetectorMap["deepseek-v4"]`` entry
points at, i.e. ``_DeepSeekV3Detector`` — which is also the ``"deepseek-v3"``
entry, so both share the relaxed behaviour (same as the legacy shim, which
resolved the class through the map). ``detect_and_parse`` itself is defined on
``BaseReasoningFormatDetector``, so ``_apply_target``'s ``getattr``/``setattr``
lands the wrapper in ``_DeepSeekV3Detector.__dict__`` only; every other detector
keeps the stock implementation.
"""

from __future__ import annotations

from sglang.srt.parser.reasoning_parser import StreamingParseResult
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.parser.reasoning_parser._DeepSeekV3Detector.detect_and_parse",
    type=HookType.AROUND,
)
def detect_and_parse_kunlun(original_fn, self, text):
    """Treat a bare trailing ``</think>`` as closing an implicit reasoning block."""
    if (
        not getattr(self, "_in_reasoning", False)
        and self.think_start_token not in text
        and self.think_end_token in text
    ):
        reasoning_text, _sep, normal_text = text.partition(self.think_end_token)
        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )
    return original_fn(self, text)