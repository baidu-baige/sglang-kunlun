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
"""Hook for ``sglang.srt.speculative.draft_utils.DraftBackendFactory``.

0.5.14's ``DraftBackendFactory._create_backend`` resolves the draft attention
backend through a fixed ``backend_map`` keyed by backend name (flashinfer /
triton / fa3 / dsa / ...). It has **no out-of-tree platform branch**, so when
the active attention backend is ``"kunlun"`` it raises::

    ValueError: EAGLE is not supported in attention backend kunlun

The Kunlun platform already exposes the draft backend classes via
``KunlunSRTPlatform.get_draft_prefill_attention_backend_cls()`` /
``get_draft_decode_attention_backend_cls()``. We add an AROUND hook on
``_create_backend`` that intercepts the ``"kunlun"`` backend type and builds the
Kunlun draft backend from those platform factory methods, delegating every
other backend to the original implementation.

Mirrors the defensive registration style of ``attention_registry.py``.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.speculative.draft_utils.DraftBackendFactory._create_backend",
    type=HookType.AROUND,
)
def _create_backend_kunlun(original_fn, self, backend_name, backend_map, error_template):
    """Route the ``kunlun`` draft attention backend through the platform.

    ``backend_name`` is the ServerArgs attribute consulted for the backend
    type: ``"decode_attention_backend"`` for the multi-step decode backend and
    ``"prefill_attention_backend"`` (or decode, per speculative_attention_mode)
    for the draft-extend backend. We replicate the upstream resolution order to
    decide whether the effective backend is ``"kunlun"``.
    """
    backend_type = (
        self.draft_attn_backend
        if self.draft_attn_backend
        else getattr(self.server_args, backend_name)
    )
    if backend_type is None:
        backend_type = self.server_args.attention_backend

    if backend_type == "kunlun_dsv4":
        from sglang_kunlun.hooks.layers.attention.deepseek_v4_backend import (
            KunlunDeepseekV4AttnBackend,
            KunlunDeepseekV4MultiStepBackend,
        )

        if backend_name == "decode_attention_backend":
            return KunlunDeepseekV4MultiStepBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        return KunlunDeepseekV4AttnBackend(
            self.draft_model_runner, skip_prefill=False
        )

    if backend_type != "kunlun":
        return original_fn(self, backend_name, backend_map, error_template)

    from sglang.srt.platforms import current_platform

    # decode multi-step backend uses the "decode_attention_backend" slot;
    # draft-extend (prefill) uses the prefill/decode slot per attention mode.
    is_decode_multistep = backend_name == "decode_attention_backend" and (
        self.server_args.speculative_attention_mode != "decode"
    )

    if is_decode_multistep:
        cls = current_platform.get_draft_decode_attention_backend_cls()
        return cls(self.draft_model_runner, self.topk, self.speculative_num_steps)

    # draft-extend / prefill path: a single Kunlun attention backend that
    # also handles prefill (skip_prefill=False).
    cls = current_platform.get_draft_prefill_attention_backend_cls()
    return cls(self.draft_model_runner, skip_prefill=False)
