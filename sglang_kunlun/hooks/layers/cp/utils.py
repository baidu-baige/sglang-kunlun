"""Kunlun hooks for ``sglang.srt.layers.cp.utils``.

Adapts ERNIE 5.5's model output to the prefill-CP runner's DSV4 contract.
`EagerRunner._execute_extend_cp` expects DeepSeek V4's two-value return
`(hidden_states, hidden_states_before_norm)`: it maps `cp_gather_after_forward`
over the tuple treating each element as a tensor or None, then unpacks exactly
two values. `Ernie5_5Model.forward` returns three, the last being a capture list
that is empty unless a DSpark draft asked for captures -- so prefill CP died on
`'list' object has no attribute 'shape'`, and then on
`too many values to unpack (expected 2)`.

This hook sits on the gather, the last call before the unpack, and (a) descends
into list/tuple elements, since each capture is a `[tokens, hidden]` tensor on
the same CP token split, and (b) drops a trailing *empty* capture list, exactly
what `Ernie5_5_MoeForCausalLM.forward` does on the non-CP path.

A non-empty capture list is left alone and still fails the unpack: routing
captures through CP needs a real runner change, and CP + DSpark is rejected at
config time anyway.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_logged = False


def _gather_item(original_fn, item, forward_batch, stream):
    if item is None:
        return None
    if isinstance(item, (list, tuple)):
        return type(item)(
            _gather_item(original_fn, sub, forward_batch, stream) for sub in item
        )
    return original_fn(item, forward_batch, stream)


@plugin_hook(
    "sglang.srt.layers.cp.utils.cp_gather_after_forward",
    type=HookType.AROUND,
)
def cp_gather_after_forward_kunlun(original_fn, x, forward_batch, stream=None):
    """Gather nested sequences, and shed ERNIE 5.5's empty capture list."""
    global _logged

    if not isinstance(x, (list, tuple)):
        return original_fn(x, forward_batch, stream)

    items = list(x)
    # ERNIE 5.5: (hidden_states, pre_hc_head, aux_hidden_states=[])
    shed = len(items) == 3 and isinstance(items[-1], (list, tuple)) and not items[-1]
    if shed:
        items = items[:-1]

    gathered = tuple(
        _gather_item(original_fn, item, forward_batch, stream) for item in items
    )
    if not _logged:
        _logged = True
        logger.info(
            "cp gather: %d element(s) [%s]%s",
            len(x),
            ", ".join(
                "Tensor" if isinstance(i, torch.Tensor) else type(i).__name__
                for i in x
            ),
            "; dropped the empty aux-capture list" if shed else "",
        )
    return gathered
