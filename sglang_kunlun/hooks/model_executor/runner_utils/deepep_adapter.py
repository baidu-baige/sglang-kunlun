"""Unify the DeepEP buffer used by the base CUDA-graph adapter with Kunlun's.

Mirrors ``sglang.srt.model_executor.runner_utils.deepep_adapter``.

Kunlun ships its own ``DeepEPBuffer`` (``layers.moe.token_dispatcher.deepep``)
that the real MoE dispatch uses, but the base ``DeepEPCudaGraphRunnerAdapter``
here (driving the EAGLE draft / target CUDA graph runners) references the *base*
``DeepEPBuffer`` via ``from ...deepep import DeepEPBuffer``. On Kunlun that base
buffer is never populated, so under EP + full CUDA graph + speculative decoding
the draft-graph replay calls base ``clean_buffer()`` -> ``state.buffer`` is
``None`` -> ``AttributeError: 'NoneType' object has no attribute
'low_latency_mode'``.

Rebind the base module's ``DeepEPBuffer`` name (and any already-imported alias)
to the Kunlun class so the adapter drives the buffer the dispatch actually uses;
Kunlun's ``clean_buffer`` iterates its own populated cache and is None-safe.
"""

from __future__ import annotations

import sys

import sglang.srt.layers.moe.token_dispatcher.deepep as _base_deepep

from sglang_kunlun.hooks.layers.moe.token_dispatcher.deepep import DeepEPBuffer

_base_deepep.DeepEPBuffer = DeepEPBuffer

_adapter = sys.modules.get("sglang.srt.model_executor.runner_utils.deepep_adapter")
if _adapter is not None:
    _adapter.DeepEPBuffer = DeepEPBuffer
