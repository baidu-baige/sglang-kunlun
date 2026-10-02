"""Hooks for ``sglang.srt.layers.logits_processor``.

``--enable-fp32-lm-head`` crashes during decode graph capture on Kunlun:

    NotImplementedError: Could not run 'aten::mm.dtype' with arguments from the
    'CPU' backend

Upstream picks the fused ``torch.mm(..., out_dtype=torch.float32)`` overload
whenever the input looks like a same-dtype CUDA fp16/bf16 tensor
(``logits_processor.py:872-882``). Kunlun tensors report ``is_cuda=True``, so the
gate opens, but the ``aten::mm.dtype`` overload has no implementation for this
platform's dispatch key and falls through to CPU.

Upstream already carries the portable path immediately below it
(``logits_processor.py:883-887``: cast both operands to fp32, then ``matmul``),
and that gate is ``hidden_states.dtype == lm_head.weight.dtype``. So the fix is
to hand the original an fp32 ``hidden_states``: the fused gate closes on the
dtype mismatch, upstream takes its own explicit-cast branch, and none of the
surrounding logic (LoRA, quant-method, AMX, GGUF branches) is duplicated here.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
    type=HookType.AROUND,
)
def compute_lm_head_kunlun(original_fn, self, hidden_states, lm_head, *args, **kwargs):
    """Route the fp32 LM head through upstream's explicit-cast branch."""
    if not getattr(self, "use_fp32_lm_head", False):
        return original_fn(self, hidden_states, lm_head, *args, **kwargs)

    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    quant_method = getattr(lm_head, "quant_method", None)
    plain_weight_path = (
        not (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        and not should_apply_lm_head_quant_method(lm_head, quant_method)
        and hasattr(lm_head, "weight")
    )
    if plain_weight_path and hidden_states.dtype != torch.float32:
        # Only the dtype changes; the cast is what closes the mm.dtype gate.
        hidden_states = hidden_states.to(torch.float32)
    return original_fn(self, hidden_states, lm_head, *args, **kwargs)
