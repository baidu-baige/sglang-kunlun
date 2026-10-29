"""Keep ``input_ids`` aligned with the CP-sharded hidden states.

With prefill context parallel the model shards the embeddings and the position
ids down to ``tokens / attn_cp_size`` rows per rank, but DeepSeek-V4's hash-topk
MoE gate still receives the *full* ``input_ids`` and asserts on the row
mismatch:

    AssertionError: input_ids.shape=[8192] hidden_states.shape=[1024, 4096]

0.5.19 retired the old ``cp_round_robin_input_ids`` hook assuming the native
path covered this; 0.5.22 does not, so the per-rank shard is rebuilt here with
upstream's own ``cp_interleave_input_ids`` (the same mapping the positions use)
and narrowed to this rank. The hook only fires when the row counts disagree and
CP is active, so eager and non-CP paths are untouched.

``DSV4_CP_SHARD_INPUT_IDS=0`` disables the hook.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.moe.hash_topk.HashTopK.forward",
    type=HookType.AROUND,
)
def hash_topk_cp_shard_input_ids_kunlun(
    original, self, hidden_states, router_logits, input_ids, *args, **kwargs
):
    if os.environ.get("DSV4_CP_SHARD_INPUT_IDS", "1") != "0" and (
        input_ids is not None
        and hidden_states is not None
        and input_ids.dim() == 1
        and input_ids.shape[0] != hidden_states.shape[0]
    ):
        sharded = _shard_input_ids(input_ids, hidden_states.shape[0])
        if sharded is not None:
            input_ids = sharded
    return original(self, hidden_states, router_logits, input_ids, *args, **kwargs)


def _shard_input_ids(input_ids: torch.Tensor, rows: int):
    try:
        from sglang.srt.layers.cp.utils import cp_interleave_input_ids, is_cp_active
        from sglang.srt.model_executor.runner_utils.forward_batch import (
            get_forward_batch,
        )
        from sglang.srt.runtime_context import get_parallel
    except Exception:  # noqa: BLE001 - a tree without the CP helpers
        return None

    try:
        forward_batch = get_forward_batch()
    except Exception:  # noqa: BLE001 - outside a model runner
        return None
    if forward_batch is None or not is_cp_active(forward_batch):
        return None

    sharded = cp_interleave_input_ids(input_ids, forward_batch)
    if sharded.shape[0] == rows:
        # The helper already returns this rank's shard when the MoE a2a backend
        # is not "none" (it routes through cp_shard_hidden_states).
        return sharded
    if sharded.shape[0] > rows and sharded.shape[0] % rows == 0:
        rank = get_parallel().attn_cp_rank
        return sharded.narrow(0, rank * rows, rows)
    if sharded.shape[0] > rows:
        return sharded[:rows]
    return torch.cat([sharded, sharded.new_zeros(rows - sharded.shape[0])])
