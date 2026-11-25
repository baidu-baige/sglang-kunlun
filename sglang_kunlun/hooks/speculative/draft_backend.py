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
"""Keep the DSpark draft attention backend aligned with DP1.

With DP1 the draft worker gets no backend override, so it inherits the target's
backend (``kunlun_dsv4`` on this stack). The DP-attention path instead calls
``_resolve_draft_attention_backend_fallback``, which only accepts a fixed list
and otherwise falls back to ``flashinfer``. FlashInfer is not installed on
Kunlun (``is_flashinfer_available()`` is false), so the fallback cannot even be
constructed (``BatchPrefillWithRaggedKVCacheWrapper`` is undefined).

This hook mirrors the DP1 behaviour: an explicit
``--speculative-draft-attention-backend`` still wins; otherwise use the process's
prefill backend (the target backend) instead of the CUDA-only fallback.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_announced = False


@plugin_hook(
    "sglang.srt.speculative.draft_worker_common."
    "_resolve_draft_attention_backend_fallback",
    type=HookType.AROUND,
)
def resolve_draft_backend_kunlun(original_fn, *, algo_label):
    """Prefer the target backend over the CUDA-only fallback."""
    global _announced
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.utils import attention_backends

    requested = get_spec().speculative_draft_attention_backend
    backend = requested
    if backend is None:
        backend, _ = attention_backends()
    if not backend:
        return original_fn(algo_label=algo_label)

    if requested is None and not _announced:
        _announced = True
        logger.warning(
            "sglang-kunlun: DSpark draft attention inherits the target backend %r "
            "(DP1 parity); pass --speculative-draft-attention-backend to override",
            backend,
        )
    return backend
