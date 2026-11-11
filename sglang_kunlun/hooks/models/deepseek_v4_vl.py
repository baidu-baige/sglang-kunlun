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
"""VL router branch for the DeepEP MoE forward (DSV4.1-Flash).

``DeepseekV2MoE.forward_deepep`` calls ``self.topk(...)`` unconditionally,
unlike ``forward_normal`` / ``forward_normal_dual_stream``, which branch to
``vision_topk`` when ``gate.e_score_correction_bias_vl is not None``. A VL model
(DSV4.1-Flash with ``vision_n_layers > 0``) carries that parameter, and VL
routing must pick ``e_score_correction_bias`` vs ``e_score_correction_bias_vl``
per token by whether the token is an image token. Without the branch, image
tokens are routed on the text bias and land on the wrong experts, so the
all-to-all answer diverges from the all-to-all-off answer.

This module supplies the missing branch for the a2a path; a non-VL model
(incl. vision disabled) passes straight through. It is a separate module from
``deepseek_v4`` so the vision fix does not have to be interleaved with the
decoder-layer hooks.

It also lifts the same guard's **CP** clause (``attn_cp_size != 1``) so a VL
model can be built under prefill context parallelism. The lift is line-scoped:
only the two reads inside the guard's own condition are answered as if CP were
off, so the decoder layers / indexer still see the real ``attn_cp_size`` and
build the CP path. The rest of the CP+vision data path is in
``hooks/model_executor/cp_mm.py`` (the CP extend runner bypasses
``DeepseekV4ForCausalLM.forward``, so the guard lift alone is not enough).
"""
from __future__ import annotations

import inspect
import logging
import sys

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


class _VisionTopkProxy:
    """Stand-in for ``DeepseekV2MoE.topk`` that routes through ``vision_topk``.

    Only the call is intercepted; every other attribute (``topk_config``,
    ``empty_topk_output``, ...) is delegated to the real ``TopK`` so
    ``forward_deepep``'s empty-batch branch and ``vision_topk``'s own
    ``moe.topk.topk_config`` read behave exactly as before.
    """

    def __init__(self, moe, real_topk, input_ids_global):
        self._moe = moe
        self._real_topk = real_topk
        self._input_ids_global = input_ids_global

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
        **kwargs,
    ):
        # forward_deepep's only router call passes exactly these keywords.
        # ``input_ids`` in ``kwargs`` (the is_hash path) is irrelevant to vision
        # routing, which keys off ``input_ids_global``.
        from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

        return vision_topk(
            self._moe,
            router_logits,
            self._input_ids_global,
            num_token_non_padded=num_token_non_padded,
        )

    def __getattr__(self, name):
        return getattr(self._real_topk, name)


@plugin_hook(
    target="sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_deepep",
    type=HookType.AROUND,
)
def forward_deepep_vision_kunlun(
    original_fn, self, hidden_states, forward_batch, input_ids_global=None
):
    """Add the missing VL router branch to the DeepEP MoE forward on Kunlun.

    On Kunlun ``vl_routing.is_cuda`` is forced to the torch branch by
    ``vl_routing_force_torch``, so swapping ``self.topk`` for the proxy makes the
    single router call inside ``forward_deepep`` match the non-DeepEP path
    exactly.
    """
    if getattr(self.gate, "e_score_correction_bias_vl", None) is None:
        return original_fn(self, hidden_states, forward_batch, input_ids_global)

    real_topk = self.topk
    # object.__setattr__ bypasses nn.Module.__setattr__: the proxy lands in the
    # instance __dict__, so plain attribute lookup returns it without touching
    # _modules; popping it afterwards falls back to the real module via
    # __getattr__. This keeps _modules (and therefore state_dict) untouched.
    object.__setattr__(
        self, "topk", _VisionTopkProxy(self, real_topk, input_ids_global)
    )
    try:
        return original_fn(self, hidden_states, forward_batch, input_ids_global)
    finally:
        self.__dict__.pop("topk", None)


_LOGGED_GUARD: set[int] = set()


def _vision_guard_span(fn):
    """Locate the CP/PP/MoE-A2A guard condition inside ``fn``.

    Anchored on the ``ValueError`` text rather than on line numbers: the engine
    is a vendored upstream file whose line numbers shift between images. Returns
    ``(fn.__code__, frozenset(line_numbers))`` covering the ``if (...)`` block,
    or ``None`` when the anchor is gone.

    ``None`` must fail closed: without a span the CP clause stays in force and a
    vision+CP boot raises the upstream ``ValueError`` (an explicit, visible
    failure) instead of building a model that silently skips the CP shard.
    """
    try:
        src, start = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return None

    anchor = next(
        (i for i, line in enumerate(src) if "V4.1 vision currently supports" in line),
        None,
    )
    if anchor is None:
        return None
    if_start = next(
        (i for i in range(anchor - 1, -1, -1) if src[i].lstrip().startswith("if (")),
        None,
    )
    if if_start is None:
        return None
    if_end = next(
        (i for i in range(if_start + 1, anchor) if src[i].strip() == "):"),
        None,
    )
    if if_end is None:
        return None
    return fn.__code__, frozenset(range(start + if_start, start + if_end + 1))


class _GuardParallel:
    """``get_parallel()`` view for the vision guard: CP looks unsharded, PP real.

    Only ``attn_cp_size`` is lifted. ``pp_group`` (and everything else) is
    delegated, so the PP clause of the same guard still rejects a PP build, and
    any read outside the guard's condition gets the real context.
    """

    __slots__ = ("_real",)

    def __init__(self, real):
        self._real = real

    @property
    def attn_cp_size(self):
        """Report an unsharded CP size so the vision guard sees CP=1."""
        return 1

    def __getattr__(self, name):
        return getattr(self._real, name)


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.__init__",
    type=HookType.AROUND,
)
def causal_lm_init_vision_a2a_kunlun(original_fn, self, config, *args, **kwargs):
    """Lift the upstream "V4.1 vision has no MoE A2A / CP" guard on Kunlun.

    ``DeepseekV4ForCausalLM.__init__`` refuses to build a DSV4.1 vision model
    (``vision_n_layers > 0``) whenever a non-none MoE all-to-all backend is
    selected, with ``ValueError("V4.1 vision currently supports TP/EP/DP without
    CP, PP or MoE A2A")``. On Kunlun both clauses are handled elsewhere:

    * the a2a path is made vision-correct by the router hook above;
    * the CP path is made vision-correct by ``hooks/model_executor/cp_mm.py``.

    This hook lifts the guard's own reads only — it installs temporary module
    symbols on ``sglang.srt.models.deepseek_v4`` while ``__init__`` runs and
    answers just the reads made inside the guard's ``if (...)`` block:

    * ``get_moe_a2a_backend()`` -> ``NONE``;
    * ``get_parallel()`` -> a view whose ``attn_cp_size`` is 1 (``pp_group`` and
      every other attribute stay real, so PP still raises).

    The scope is the guard's *code object + source lines*, not its frame type:
    the same ``__init__`` assigns ``self.pp_group`` from ``get_parallel()`` a few
    lines below, and ``DeepseekV4Model`` / ``DeepseekV4DecoderLayer`` /
    ``DsaIndexer`` read the real ``attn_cp_size`` during construction to build
    the CP path — a frame- or context-wide shim would silently disable it.

    When the guard span cannot be located (image rebuild, reworded message) the
    CP clause is *not* lifted: a vision+CP boot then raises the upstream error,
    which is the correct fail-closed behaviour.
    """
    if getattr(config, "vision_n_layers", 0) <= 0:
        return original_fn(self, config, *args, **kwargs)

    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.models import deepseek_v4 as _v4

    real_a2a = _v4.get_moe_a2a_backend
    real_parallel = _v4.get_parallel
    code, guard_lines = _vision_guard_span(original_fn) or (None, frozenset())

    if code is not None:
        if id(code) not in _LOGGED_GUARD:
            _LOGGED_GUARD.add(id(code))
            logger.info(
                "[mm-cp] vision guard at lines %d-%d: CP clause lifted, PP clause kept",
                min(guard_lines),
                max(guard_lines),
            )
    elif id(original_fn) not in _LOGGED_GUARD:
        _LOGGED_GUARD.add(id(original_fn))
        logger.warning(
            "[mm-cp] vision guard not located; CP/PP clauses stay in force "
            "(only the MoE-A2A clause is lifted)"
        )

    def _in_span(frame) -> bool:
        return (
            code is not None
            and frame.f_code is code
            and frame.f_lineno in guard_lines
        )

    def _guard_lifting_getter():
        frame = sys._getframe(1)
        if _in_span(frame):
            return MoeA2ABackend.NONE
        if code is None:
            # Legacy discriminator, kept only for the unlocated-span fallback:
            # the guard is the one direct read made from the LM's own __init__.
            owner = frame.f_locals.get("self")
            if owner is not None and "DeepseekV4ForCausalLM" in {
                cls.__name__ for cls in type(owner).__mro__
            }:
                return MoeA2ABackend.NONE
        return real_a2a()

    def _guard_parallel_getter():
        frame = sys._getframe(1)
        real = real_parallel()
        return _GuardParallel(real) if _in_span(frame) else real

    _v4.get_moe_a2a_backend = _guard_lifting_getter
    _v4.get_parallel = _guard_parallel_getter
    try:
        return original_fn(self, config, *args, **kwargs)
    finally:
        _v4.get_moe_a2a_backend = real_a2a
        _v4.get_parallel = real_parallel
