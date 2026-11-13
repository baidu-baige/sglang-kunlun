"""V4.1 CP combine: reduce-scatter instead of all-reduce + slice.

Upstream takes the `all_reduce` + `tensor_split(cp)[rank]` branch whenever
`cp_return_contiguous` is set, which for `deepseek_v41` is unconditional.
Reduce-scatter over the same ordered group computes that slice with one
collective instead of two, and `_same_ordered_group` is what makes the identity
hold. Anything the collective cannot serve falls back to upstream's branch.

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


@plugin_hook(TARGET, type=HookType.AROUND)
def cp_moe_reduce_scatter(original_fn, self, hidden_states, *args, **kwargs):
    """Finish the CP combine with reduce-scatter when it is exactly equivalent.

    Args:
        original_fn: Wrapped ``MoeRowTransport.return_output``.
        self: The transport the method was bound to.
        hidden_states: Combine output to reduce.
        *args: Any value falls back to ``original_fn``.
        **kwargs: Any key falls back to ``original_fn``.

    Returns:
        The reduced hidden states, from either path.
    """
    def fallback():
        """Fallback to the original method."""
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

