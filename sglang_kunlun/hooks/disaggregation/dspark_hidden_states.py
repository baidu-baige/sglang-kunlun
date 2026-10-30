"""Carry the DSPARK draft seed across the PD boundary, without touching upstream.

Upstream ships the target's last hidden state to the decode node only for
EAGLE: ``disaggregation/prefill.py`` gates both the D2H copy and the
``req.hidden_states_tensor`` assignment on ``spec_algorithm.is_eagle()``, and
``dspark_disaggregation.build_dspark_disagg_draft_input`` builds the decode-side
draft input without a hidden state at all.

DSPARK's draft head consumes that hidden state exactly like EAGLE's does, so on
PD the draft started from an empty tensor and its proposals were almost never
accepted -- accept length collapsed to ~1.0 out of 6 while the target itself
stayed correct. The fix used to live as two upstream commits; upstream must stay
untouched, so both halves are hooks here.

Ordering matters on the decode side. ``build_dspark_disagg_draft_input`` calls
``make_next_draft_input`` first and then, under overlap, stashes the result via
``RelayPayload.from_draft_input``, which *captures* ``hidden_states``. Setting
the field on the object returned by ``build_...`` would therefore be too late, so
the injection hooks ``make_next_draft_input`` and the batch is handed to it
through a module global that the outer hook sets for the duration of the call.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

#: Set by the ``build_dspark_disagg_draft_input`` hook for the duration of that
#: call so the inner ``make_next_draft_input`` hook can see the batch. One
#: scheduler builds one draft input at a time, so a module global is enough.
_CURRENT_BATCH: Optional[Any] = None


def _stacked_hidden_states(batch) -> Optional[torch.Tensor]:
    """Stack the per-request hidden states the prefill node shipped."""
    reqs = getattr(batch, "reqs", None)
    if not reqs:
        return None
    tensors = [getattr(r, "hidden_states_tensor", None) for r in reqs]
    if any(t is None for t in tensors):
        return None
    return torch.stack(tensors, dim=0).to(batch.device)


@plugin_hook(
    "sglang.srt.speculative.dspark_disaggregation.build_dspark_disagg_draft_input",
    type=HookType.AROUND,
)
def _expose_batch_to_draft_builder(original_fn, batch, *args, **kwargs):
    """Publish the batch so the inner ``make_next_draft_input`` hook can read it."""
    global _CURRENT_BATCH
    prev = _CURRENT_BATCH
    _CURRENT_BATCH = batch
    try:
        return original_fn(batch, *args, **kwargs)
    finally:
        _CURRENT_BATCH = prev


@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_draft.make_next_draft_input",
    type=HookType.AROUND,
)
def _seed_draft_input_with_hidden_states(original_fn, *args, **kwargs):
    """Attach the shipped hidden state before the caller stashes the draft input."""
    spec = original_fn(*args, **kwargs)
    batch = _CURRENT_BATCH
    if batch is None:
        return spec
    try:
        hidden = _stacked_hidden_states(batch)
    except Exception:  # noqa: BLE001 - never break draft construction
        logger.warning("sglang-kunlun: could not stack DSPARK draft hidden states",
                       exc_info=True)
        return spec
    if hidden is not None:
        spec.hidden_states = hidden
    return spec


@plugin_hook(
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
    "process_batch_result_disagg_prefill",
    type=HookType.AROUND,
)
def _ship_dspark_hidden_states(original_fn, self, batch, result, *args, **kwargs):
    """Fill ``req.hidden_states_tensor`` for DSPARK, which upstream skips.

    Runs after the original so the requests are already queued for transfer;
    only the field upstream left unset is written, and only for requests this
    batch actually handed to the inflight queue.
    """
    out = original_fn(self, batch, result, *args, **kwargs)
    spec = getattr(self, "spec_algorithm", None)
    if spec is None or not spec.is_dspark():
        return out
    draft_input = getattr(result, "next_draft_input", None)
    hidden = getattr(draft_input, "hidden_states", None)
    if hidden is None:
        return out
    try:
        host = hidden.to("cpu", non_blocking=False)
        queued = {id(r) for r in getattr(self, "disagg_prefill_inflight_queue", [])}
        for i, req in enumerate(getattr(batch, "reqs", []) or []):
            if i >= host.shape[0] or id(req) not in queued:
                continue
            if getattr(req, "hidden_states_tensor", None) is None:
                req.hidden_states_tensor = host[i].clone()
    except Exception:  # noqa: BLE001 - a missed seed costs accept length, not correctness
        logger.warning("sglang-kunlun: could not ship DSPARK draft hidden states",
                       exc_info=True)
    return out
