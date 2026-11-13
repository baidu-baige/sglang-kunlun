"""V4.1 CP combine: reduce-scatter instead of all-reduce + slice.

0.5.23 re-ported. On 0.5.22 the whole MoE/DP step was one method
(`DeepseekV4DecoderLayer._run_moe_ffn_dp_sync`) and this hook had to
re-implement gather -> mlp -> combine around it. 0.5.23 split that method into
`MoeRowTransport` (`__init__` gathers, `scope()` runs the MLP, `return_output`
combines) and deleted the old name, so only the **combine** needs replacing and
the gather and the MLP stay upstream's.

Why it is a legal substitution rather than a heuristic: upstream takes the
`all_reduce` then `tensor_split(cp)[rank]` branch whenever
`cp_return_contiguous` is set, which for `deepseek_v41` is unconditional
(`deepseek_v4_platform_stage.py:342`, `deepseek_v4_boundary.py:513`).
Reduce-scatter over the same ordered group computes exactly that slice, so the
result is identical, with one collective instead of a full all-reduce plus a
copy. `_same_ordered_group` is what makes the identity hold -- it refuses
unless the CP and TP groups are the same ranks in the same order -- and any
shape the collective cannot serve falls back to upstream's own branch.

Set SGLANG_KUNLUN_DSV41_CP_REDUCE_SCATTER=0 to use the original path.
"""
from __future__ import annotations

import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

TARGET = (
    "sglang.srt.models.deepseek_common.moe_transport.MoeRowTransport.return_output"
)
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


def cp_moe_reduce_scatter(original_fn, self, hidden_states, *args, **kwargs):
    """Finish the CP combine with reduce-scatter when it is exactly equivalent.

    Replaces upstream's all-reduce plus a slice of the local shard with
    ``reduce_scatter_tensor``, which moves the same bytes in one pass.

    Args:
        original_fn: The wrapped method; called unchanged on every path where
            the substitution is not provably equivalent.
        self: The MoE layer the method was bound to.
        hidden_states: Combine output to reduce.
        *args: Passed through to ``original_fn``; any value forces a fallback.
        **kwargs: Passed through to ``original_fn``; any key forces a fallback.

    Returns:
        The reduced hidden states, from either path.

    Raises:
        Nothing of its own: every disqualifying condition (extra arguments,
        another MoE a2a backend, no TP group, missing ``reduce_scatter_tensor``)
        falls back to ``original_fn`` rather than raising.
    """
    def fallback():
        return original_fn(self, hidden_states, *args, **kwargs)

    if os.environ.get(ENABLE_ENV, "1") != "1" or args or kwargs:
        return fallback()
    import torch
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_parallel

    # Only the branch upstream would finish with all_reduce + slice.
    if not (
        getattr(self, "_use_cp", False)
        and getattr(self, "cp_return_contiguous", False)
        and get_moe_a2a_backend().is_none()
    ):
        return fallback()
    parallel = get_parallel()
    n = parallel.attn_cp_size
    if (
        not _same_ordered_group(parallel)
        or hidden_states.ndim != 2
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or hidden_states.shape[1] == 0
        or hidden_states.shape[0] <= 0
        or hidden_states.shape[0] % n
    ):
        return fallback()
    output = hidden_states.new_empty(
        (hidden_states.shape[0] // n, hidden_states.shape[1])
    )
    parallel.tp_group.reduce_scatter_tensor(output, hidden_states.contiguous())
    return output


plugin_hook(TARGET, type=HookType.AROUND)(cp_moe_reduce_scatter)
