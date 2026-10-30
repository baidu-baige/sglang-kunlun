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
"""Hook for the DeepSeek-V4 attention-metadata kernel dispatch.

Upstream builds DSV4 attention metadata with Triton kernels and picks the
implementation from where the inputs live::

    def _inputs_on_cuda(*args, **kwargs) -> bool:
        for value in (*args, *kwargs.values()):
            if isinstance(value, torch.Tensor):
                return value.is_cuda
        raise AssertionError("kernel dispatch requires at least one tensor argument")

    class ExpandPrefillCausally:
        @classmethod
        def execute(cls, *args, **kwargs):
            if _inputs_on_cuda(*args, **kwargs):
                return cls.triton(*args, **kwargs)
            return cls.torch(*args, **kwargs)

Its own docstring states the intent: "CUDA inputs take the fused triton
kernel; CPU inputs take the torch reference implementation (triton is
CUDA-only)". That is a CUDA-versus-CPU split, and Kunlun sits outside it: our
tensors report ``is_cuda=True``, so the predicate is always true and every call
goes to Triton -- but the Kunlun Triton backend registers the target ``xpu``,
there is no ``cuda`` backend in the image, and decode graph capture aborts::

    File ".../dsv4/attn_metadata_kernels.py", in build_causal_swa_page_indices_triton
        _causal_swa_page_indices_kernel[(num_qo_tokens,)](...)
    RuntimeError: 0 compatible backends for target (cuda) ([]). There should
    only be one.

So wrap the predicate instead of the three ``execute`` methods that consume it.
One hook then covers ``ExpandPrefillCausally``, ``BuildPageTablePositions`` and
``BuildCausalSwaPageIndices``, and any op later added to that module with the
same dispatch shape. The original is still called so that its "at least one
tensor argument" assertion keeps firing; only the verdict is replaced.

This takes the reference path, which is upstream's own code, so the numerics
are not ours to get wrong. The proper fix is to register Kunlun
implementations of the three kernels in ``sglang_kunlun/kernels/kernel_ops.py``
alongside the ops already registered there; this hook is the smaller step that
makes the model runnable without new kernels.

The module moved between releases, so resolve the path instead of hard-coding
one: a hard-coded miss would make ``apply_hooks`` log a full traceback on every
start, which reads like a defect.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# Newest layout first. 0.5.17 shipped the flat name; it later moved under
# ``dsv4/``.
_CANDIDATE_MODULES = (
    "sglang.kernels.ops.attention.dsv4.attn_metadata_kernels",
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels",
)


def force_torch_metadata_dispatch(original_fn, *args, **kwargs) -> bool:
    """Report "not CUDA" so the ops take upstream's torch reference path."""
    original_fn(*args, **kwargs)
    return False


def _resolve_module() -> str | None:
    """Name of whichever candidate module this SGLang build actually has."""
    for name in _CANDIDATE_MODULES:
        if name in sys.modules:
            return name
    for name in _CANDIDATE_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (AttributeError, ImportError, ValueError):
            continue
    return None


_module_name = _resolve_module()

if _module_name is None:
    logger.debug(
        "sglang-kunlun: no DSV4 attention-metadata kernel module found "
        "(looked for %s); nothing to hook",
        ", ".join(_CANDIDATE_MODULES),
    )
else:
    plugin_hook(f"{_module_name}._inputs_on_cuda", type=HookType.AROUND)(
        force_torch_metadata_dispatch
    )
    logger.debug(
        "sglang-kunlun: routing DSV4 attention metadata in %s to the torch path",
        _module_name,
    )
