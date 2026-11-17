"""Default-enabled V4.1 CP combine candidate; device validation pending.

Import BEFORE cp_bounded_replay: later AROUND hooks are outermost.
No token-order permutation: the current gather is padded rank-major.
Set SGLANG_KUNLUN_DSV41_CP_REDUCE_SCATTER=0 to use the original path.
"""
from __future__ import annotations

import os

from sglang.srt.plugins.hook_registry import HookRegistry, HookType, plugin_hook

TARGET = "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync"
ENABLE_ENV = "SGLANG_KUNLUN_DSV41_CP_REDUCE_SCATTER"


def _same_ordered_group(parallel):
    """Check whether the CP and TP groups are the same ordered rank group."""
    cp, tp = parallel.attn_cp_group, parallel.tp_group
    n, r = parallel.attn_cp_size, parallel.attn_cp_rank
    return (
        n > 1
        and parallel.tp_size == n
        and parallel.tp_rank == r
        and cp is not None and tp is not None
        and cp.world_size == tp.world_size == n
        and cp.rank_in_group == tp.rank_in_group == r
        and len(cp.ranks) == len(tp.ranks) == n
        and tuple(cp.ranks) == tuple(tp.ranks)
        and len(set(cp.ranks)) == n
        and 0 <= r < n
        and cp.rank == tp.rank == cp.ranks[r]
        and callable(getattr(tp, "reduce_scatter_tensor", None))
    )


def _physical_rows(forward_batch):
    # Match the existing tail wrapper's active-metadata condition.
    from sglang_kunlun.hooks.model_executor.cp_bounded_replay import _active_cp_tail
    from sglang.srt.model_executor.forward_context import get_attn_backend

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    tail = _active_cp_tail(forward_batch)
    if tail is not None:
        backend = get_attn_backend()
        if backend.forward_metadata is backend.tail_forward_metadata:
            metadata = tail.cp_metadata
    return getattr(metadata, "max_rank_len", None)


def _equal_nonempty_rows(lengths, size, rows):
    # CPU metadata only. Logical lengths may be unequal despite equal padding.
    return (
        isinstance(lengths, (list, tuple))
        and len(lengths) == size
        and rows > 0
        and all(type(n) is int and n == rows for n in lengths)
    )


def _combine(partial, parallel, expected_rows):
    """Post-gather fallback finishes AR+split, never repeats gather or MLP."""
    import torch

    n = parallel.attn_cp_size
    if (
        partial.ndim != 2
        or partial.shape[0] != expected_rows
        or expected_rows <= 0
        or expected_rows % n
        or partial.shape[1] == 0
        or partial.dtype not in (torch.float16, torch.bfloat16)
    ):
        reduced = parallel.tp_group.all_reduce(partial)
        return reduced.tensor_split(n)[parallel.attn_cp_rank].contiguous()
    output = partial.new_empty((expected_rows // n, partial.shape[1]))
    parallel.tp_group.reduce_scatter_tensor(output, partial.contiguous())
    return output


def cp_moe_reduce_scatter(
    original_fn, self, hidden_states, forward_batch, *,
    input_ids, input_ids_global, return_moe_output=False, **kwargs,
):
    """Replace AR+split with reduce-scatter when CP and TP groups coincide."""
    def fallback():
        return original_fn(
            self, hidden_states, forward_batch, input_ids=input_ids,
            input_ids_global=input_ids_global,
            return_moe_output=return_moe_output, **kwargs,
        )

    if os.environ.get(ENABLE_ENV, "1") != "1" or kwargs:
        return fallback()
    import torch
    from sglang.srt.models import deepseek_v4 as model

    if (
        getattr(self.config, "model_type", None) != "deepseek_v41"
        or not model.is_cp_active(forward_batch)
        or not model.get_moe_a2a_backend().is_none()
    ):
        return fallback()
    parallel = model.get_parallel()
    if (
        not _same_ordered_group(parallel)
        or hidden_states.ndim != 2
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or hidden_states.shape[1] == 0
        or not _equal_nonempty_rows(
            _physical_rows(forward_batch), parallel.attn_cp_size,
            hidden_states.shape[0],
        )
    ):
        return fallback()

    gathered = model.attn_cp_interleave_gather(hidden_states)
    with model.get_forward().scoped(mlp_reduce_scatter=True):
        with model._every_row_routed(forward_batch, gathered.shape[0]):
            partial = self.mlp(
                gathered, forward_batch, input_ids=input_ids,
                input_ids_global=input_ids_global, skip_shared_experts=False,
                return_moe_output=False,
            )
    return _combine(
        partial, parallel, hidden_states.shape[0] * parallel.attn_cp_size,
    )


def register():
    """Register the CP reduce-scatter hook on the MoE FFN sync target."""
    # Fail closed rather than silently bypassing a previously registered wrapper.
    if TARGET in HookRegistry._patched or HookRegistry._hooks.get(TARGET):
        raise RuntimeError(
            "CP reduce-scatter must register before all hooks on " + TARGET
            + "; import it immediately before cp_bounded_replay"
        )
    plugin_hook(TARGET, type=HookType.AROUND)(cp_moe_reduce_scatter)


register()
