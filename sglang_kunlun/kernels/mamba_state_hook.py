"""Kunlun replacements for the Mamba/KDA cuda-graph state-index and
state-copy Triton kernels.

These live under ``sglang.kernels.ops.mamba`` and are hit on the decode
cuda-graph capture/replay path (``HybridLinearAttnBackend._replay_metadata``),
the prefix-cache track path (``_track_mamba_state_decode`` /
``_track_mamba_state_extend``), and the MTP verify path
(``scatter_mamba_states_after_mtp_verify``). All are Triton, which cannot
compile on P800 (see ``kpool_hook.py`` docstring for why). Two of the four
have a direct ``kunlun_ops`` op with matching semantics
(``track_mamba_state``, ``fused_mamba_state_scatter_with_mask``); the other
two (index gather with a padding sentinel, and the non-contiguous
conv-window scatter) have no kunlun_ops counterpart and are ported to plain
torch instead.
"""

from __future__ import annotations

import torch

import kunlun_ops
import xpu_flash_ops

from sglang_kunlun.kernels.kernel_ops import register_jit_op

_STATE_INDICES_MODULE = "sglang.kernels.ops.mamba.mamba_state_indices_triton"
_STATE_SCATTER_MODULE = "sglang.kernels.ops.mamba.mamba_state_scatter_triton"


def _i64(t: torch.Tensor) -> torch.Tensor:
    """Index tensors as int64, which the kunlun_ops scatter ops require."""
    return t if t.dtype == torch.int64 else t.to(torch.int64)


def _track_mamba_state_graph_safe(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
) -> None:
    """One layer's track copy, callable inside a decode cuda-graph capture.

    ``kunlun_ops.track_mamba_state`` cannot be: its python wrapper starts with
    ``if not mamba_track_mask.any(): return 0``, i.e. a device->host read. Under
    capture that read hits a recorded-but-unexecuted kernel and puts the XPU
    runtime in an error state, so the very next launch dies with
    ``track_mamba_state ... failed ret={}2``. (The Triton kernel this replaces
    is pure device code, which is why the reference path captures fine.) So the
    underlying op is called directly, and the early-out the wrapper wanted is
    left to the op's own per-row mask test.

    Bypassing the wrapper also means bypassing nothing else: the mask rows are
    sanitized here instead, on device (no sync). A padded decode row carries
    ``cache_indices == -1`` (the replay-prep sentinel) and a stale/zero track
    dest; masking those rows off and clamping the indices keeps the op from
    ever addressing row -1.
    """

    valid = mamba_track_mask & (cache_indices >= 0) & (mamba_track_indices >= 0)
    xpu_flash_ops.track_mamba_state(
        conv_states,
        ssm_states,
        cache_indices.clamp_min(0),
        valid,
        mamba_track_indices.clamp_min(0),
    )


@register_jit_op(_STATE_INDICES_MODULE, "fused_replay_state_indices")
def fused_replay_state_indices_kunlun(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> torch.Tensor:
    """Torch port of the fused replay-prep gather.

    No kunlun_ops equivalent: this is a tiny (total_bs,) index gather plus a
    padding sentinel, not worth a dedicated kernel. Same three effects as the
    reference chain / Triton kernel:
      1. out_state_indices[:valid_bs] = mamba_index_mapping[req_pool_indices[:valid_bs]]
      2. out_state_indices[valid_bs:total_bs] = -1 (padding sentinel)
      3. req_pool_indices[valid_bs:total_bs] = 0 (in place, reference side effect)
    """

    valid = req_pool_indices[:valid_bs]
    gathered = mamba_index_mapping[valid]
    out_state_indices[:valid_bs] = gathered.to(out_state_indices.dtype)
    if valid_bs < total_bs:
        out_state_indices[valid_bs:total_bs] = -1
        req_pool_indices[valid_bs:total_bs] = 0
    return out_state_indices[:total_bs]


@register_jit_op(_STATE_SCATTER_MODULE, "track_mamba_states_if_needed")
def track_mamba_states_if_needed_kunlun(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
) -> None:
    """Single-layer conv/ssm track-copy: maps directly onto kunlun_ops'
    ``track_mamba_state``, which has the identical per-batch masked-copy
    contract (conv_states[dst] <- conv_states[src] where mask is True).

    ``check_freed_slots`` (skip src/dst < 0, used only by the unified memory
    pool) is not exposed by the kunlun_ops signature; this build only runs
    the static hybrid pool (no -1 tombstones), so it is asserted off rather
    than silently ignored.
    """

    assert not check_freed_slots, (
        "track_mamba_states_if_needed_kunlun: check_freed_slots=True needs the "
        "unified memory pool's -1 tombstone semantics, which kunlun_ops."
        "track_mamba_state does not implement"
    )
    _track_mamba_state_graph_safe(
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
    )


@register_jit_op(_STATE_SCATTER_MODULE, "track_mamba_states_all_layers")
def track_mamba_states_all_layers_kunlun(
    conv_states_pool: torch.Tensor,
    ssm_states_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
) -> None:
    """All-layers track-copy. kunlun_ops.track_mamba_state takes one layer's
    [pool_size, ...] state at a time, so this loops it over the layer axis
    of the [num_layers, pool_size, ...] pools -- equivalent to the reference
    per-layer launches the Triton kernel replaced, just without their single
    fused launch. The mask/index sanitizing (see
    ``_track_mamba_state_graph_safe``) is hoisted out of the loop: all layers
    share the same rows.
    """

    assert not check_freed_slots, (
        "track_mamba_states_all_layers_kunlun: check_freed_slots=True needs the "
        "unified memory pool's -1 tombstone semantics, which kunlun_ops."
        "track_mamba_state does not implement"
    )
    valid = mamba_track_mask & (cache_indices >= 0) & (mamba_track_indices >= 0)
    src = cache_indices.clamp_min(0)
    dst = mamba_track_indices.clamp_min(0)
    num_layers = conv_states_pool.shape[0]
    for layer_idx in range(num_layers):
        xpu_flash_ops.track_mamba_state(
            conv_states_pool[layer_idx],
            ssm_states_pool[layer_idx],
            src,
            valid,
            dst,
        )


@register_jit_op(_STATE_SCATTER_MODULE, "fused_mamba_state_scatter_with_mask")
def fused_mamba_state_scatter_with_mask_kunlun(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    """MTP-verify state scatter: kunlun_ops has a same-named, same-signature
    op with the identical masked gather-scatter contract, so this is a direct
    pass-through apart from the index dtype.

    Upstream hands both index tensors in as int32 -- ``last_correct_step_indices``
    and ``mamba_steps_to_track`` are allocated int32 by
    ``nextn_mamba_commit_prologue``, and the eager fallback inherits int32 from
    ``accept_index`` -- which the Triton kernel accepts but kunlun_ops rejects
    ("Expected int64"). Cast rather than change the producers, so the upstream
    contract stays intact. ``to`` is a plain kernel, so this stays cuda-graph
    capturable.
    """

    kunlun_ops.fused_mamba_state_scatter_with_mask(
        dst, src, _i64(dst_indices_raw), _i64(step_indices_raw)
    )


@register_jit_op(_STATE_SCATTER_MODULE, "fused_conv_window_scatter_with_mask")
def fused_conv_window_scatter_with_mask_kunlun(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    """MTP-verify conv-window scatter. ``src`` is an intentionally
    non-contiguous ``as_strided`` view (overlapping sliding windows over a
    shared buffer), so there is no matching kunlun_ops op -- gather/scatter
    kernels there assume contiguous rows. Ported to torch advanced indexing:
    same asymptotic cost as the Triton kernel's per-request loop, just
    without the fused launch.

    Like the KDA decode recurrence, this can run inside a cuda-graph capture
    (MTP target-verify graph), so it must be shape-static and free of
    device->host syncs: ``bool(valid.any())`` would read back a recorded but
    unexecuted kernel, and ``x[valid]`` would turn that garbage into a
    data-dependent shape. Invalid rows are therefore aliased onto the first
    valid row -- both index and payload -- so the duplicate writes are
    byte-identical and ``index_copy_``'s undefined ordering stops mattering.
    """

    valid = step_indices_raw >= 0
    n_req = valid.shape[0]
    first_valid = torch.argmax(valid.to(torch.uint8)).reshape(1)

    req_idx = torch.arange(n_req, device=src.device)
    step_idx = step_indices_raw.clamp_min(0).long()
    # src: [num_layers, spec_size, draft_tokens, dim, K-1]; index (req, step)
    # per layer via advanced indexing on dims 1-2, batched over dim 0.
    windows = src[:, req_idx, step_idx]  # [num_layers, n_req, dim, K-1]

    dst_idx = dst_indices_raw.clamp_min(0).long()
    idx = torch.where(
        valid, dst_idx, dst_idx.index_select(0, first_valid).expand_as(dst_idx)
    )
    # Broadcast the row mask over the layer axis and the trailing state dims.
    mask = valid.reshape(1, n_req, *([1] * (windows.dim() - 2)))
    val = torch.where(
        mask, windows, windows.index_select(1, first_valid).expand_as(windows)
    )
    # Gather with the *redirected* indices so an all-invalid batch rewrites
    # each slot with its own current contents (and duplicates stay identical).
    existing = dst.index_select(1, idx)
    val = torch.where(valid.any(), val, existing)
    dst.index_copy_(1, idx, val.to(dst.dtype))


@register_jit_op(_STATE_SCATTER_MODULE, "fused_conv_window_scatter_multi")
def fused_conv_window_scatter_multi_kunlun(
    pairs,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
    dst_indices2_raw: torch.Tensor = None,
    step_indices2_raw: torch.Tensor = None,
) -> None:
    """Single-launch multi-pair conv-window scatter, unfused.

    The upstream fused kernel packs every ``(dst, src)`` conv-type pair and both
    request-index sets into one Triton launch over a device-side meta table,
    which does not build on P800. This is exactly the shape of the upstream
    fallback loop taken when ``_conv_multi_eligible`` says no, so it delegates
    to the already-ported per-pair scatter instead of duplicating the math.
    """

    for conv_states, intermediate_conv_window_cache in pairs:
        fused_conv_window_scatter_with_mask_kunlun(
            conv_states,
            intermediate_conv_window_cache,
            dst_indices_raw,
            step_indices_raw,
        )
    if step_indices2_raw is None:
        return
    for conv_states, intermediate_conv_window_cache in pairs:
        fused_conv_window_scatter_with_mask_kunlun(
            conv_states,
            intermediate_conv_window_cache,
            dst_indices2_raw,
            step_indices2_raw,
        )
