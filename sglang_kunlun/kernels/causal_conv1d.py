"""Kimi-K3 depthwise causal-convolution implementations for Kunlun."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch


logger = logging.getLogger(__name__)
_CACHE_AUDIT_EVENTS = 0


def _cache_audit_enabled() -> bool:
    if os.getenv("SGLANG_KUNLUN_K3_CACHE_AUDIT") != "1":
        return False
    limit = int(os.getenv("SGLANG_KUNLUN_K3_CACHE_AUDIT_LIMIT", "32"))
    if _CACHE_AUDIT_EVENTS >= limit:
        return False
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _audit_conv_state(mode, state, indices, before) -> None:
    global _CACHE_AUDIT_EVENTS
    if not _cache_audit_enabled():
        return
    limit = int(os.getenv("SGLANG_KUNLUN_K3_CACHE_AUDIT_LIMIT", "32"))
    if _CACHE_AUDIT_EVENTS >= limit:
        return
    valid = indices[indices >= 0].to(device=state.device, dtype=torch.long)
    if valid.numel() == 0:
        return
    after = state.index_select(0, valid).float().abs().sum().item()
    logger.warning(
        "K3_CACHE_AUDIT component=conv mode=%s slots=%s before_abs_sum=%.9g after_abs_sum=%.9g changed=%s",
        mode,
        valid.detach().cpu().tolist(),
        before,
        after,
        before != after,
    )
    _CACHE_AUDIT_EVENTS += 1


def _validate_indices(indices: torch.Tensor, batch: int, state_size: int, pad_slot_id: int) -> None:
    if indices.dim() != 1 or indices.numel() != batch:
        raise ValueError("conv_state_indices must have shape [batch]")
    invalid = (indices != pad_slot_id) & ((indices < 0) | (indices >= state_size))
    if bool(invalid.any().item()):
        raise ValueError("conv_state_indices contains an invalid cache slot")


def causal_conv1d_update_kunlun(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    *,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    metadata=None,
    validate_data: bool = False,
):
    """Run the Kunlun causal-conv decode op and update state in place.

    K3 stores the state pool as ``[slots, state_len, dim]`` and passes its
    ``[slots, dim, state_len]`` transpose here. The XPU op requires its state
    tensor to be contiguous, so prefer the original NWC backing layout when
    available instead of copying the entire state pool.

    Args:
        x: Input activations with shape ``[batch, dim]`` or ``[batch, dim, steps]``.
        conv_state: Mutable cache with shape ``[slots, dim, state_width]``.
        weight: Depthwise convolution weights.
        bias: Optional per-channel bias.
        silu_activation: Whether to apply SiLU to the convolution output.
        cache_seqlens: Unsupported circular-buffer metadata.
        conv_state_indices: Cache slot for each batch row.
        pad_slot_id: Sentinel slot whose output is zero and state is untouched.
        num_accept_tokens: Unsupported speculative metadata.
        intermediate_conv_window: Unsupported speculative metadata.
        intermediate_state_indices: Unsupported speculative metadata.
        retrieve_next_token: Unsupported tree-retrieve metadata.
        retrieve_next_sibling: Unsupported tree-retrieve metadata.
        retrieve_parent_token: Unsupported tree-retrieve metadata.
        metadata: Upstream compatibility metadata.
        validate_data: Enable expensive runtime validation for debugging.
    """
    import kunlun_ops

    del metadata
    if any(
        value is not None
        for value in (
            cache_seqlens,
            num_accept_tokens,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
        )
    ):
        raise NotImplementedError("advanced causal-conv decode metadata is unsupported")

    if x.dim() not in (2, 3):
        raise ValueError(f"expected x rank 2 or 3, got {x.shape}")
    if conv_state.dim() != 3:
        raise ValueError(
            f"expected conv_state [slots, dim, state_len], got {conv_state.shape}"
        )
    if weight.dim() != 2:
        raise ValueError(f"expected weight [dim, width], got {weight.shape}")

    batch, dim = x.shape[:2]
    if weight.shape[0] != dim or conv_state.shape[1] != dim:
        raise ValueError("causal convolution dimension mismatch")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != dim):
        raise ValueError(f"expected bias [dim] with dim={dim}, got {bias.shape}")
    width = weight.shape[1]
    if conv_state.shape[-1] != width - 1:
        raise ValueError(
            "causal convolution state length must equal weight width - 1: "
            f"state={conv_state.shape[-1]}, width={width}"
        )

    indices = (
        None
        if conv_state_indices is None
        else conv_state_indices.to(
            device=x.device,
            dtype=torch.int32,
        ).contiguous()
    )
    if validate_data:
        validation_indices = (
            torch.arange(batch, device=x.device, dtype=torch.int32)
            if indices is None
            else indices
        )
        _validate_indices(
            validation_indices,
            batch,
            conv_state.shape[0],
            pad_slot_id,
        )

    audit_enabled = _cache_audit_enabled()
    if audit_enabled:
        audit_indices = (
            torch.arange(batch, device=x.device, dtype=torch.int32)
            if indices is None
            else indices
        )
        valid = audit_indices[
            (audit_indices != pad_slot_id) & (audit_indices >= 0)
        ]
        audit_before = (
            conv_state.index_select(0, valid.to(torch.long))
            .float()
            .abs()
            .sum()
            .item()
            if valid.numel()
            else 0.0
        )

    state_nwc = conv_state.transpose(-1, -2)
    if state_nwc.is_contiguous():
        output = (
            x.unsqueeze(1).clone(memory_format=torch.contiguous_format)
            if x.dim() == 2
            else x.transpose(1, 2).contiguous()
        )
        state_for_op = state_nwc
        is_ncw = False
    elif conv_state.is_contiguous():
        output = (
            x.unsqueeze(-1).clone(memory_format=torch.contiguous_format)
            if x.dim() == 2
            else x.clone(memory_format=torch.contiguous_format)
        )
        state_for_op = conv_state
        is_ncw = True
    else:
        raise ValueError(
            "causal-conv state must be contiguous in NCW or NWC backing layout"
        )

    status = kunlun_ops.causal_conv1d_update(
        x=output,
        conv_state=state_for_op,
        weight=weight.contiguous(),
        bias=None if bias is None else bias.contiguous(),
        silu_activation=silu_activation,
        cache_seqlens=None,
        conv_state_indices=indices,
        intermediate_conv_window=None,
        is_ncw=is_ncw,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=None,
    )
    if status != 0:
        raise RuntimeError(
            f"kunlun_ops.causal_conv1d_update failed with status={status}"
        )

    if audit_enabled:
        _audit_conv_state(
            "decode",
            conv_state,
            audit_indices,
            audit_before,
        )

    if x.dim() == 2:
        return output[:, :, 0] if is_ncw else output[:, 0, :]
    return output if is_ncw else output.transpose(1, 2).contiguous()


def kda_causal_conv1d_update_kunlun(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation=None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    metadata=None,
    validate_data: bool = False,
):
    """Adapt KDA's direct Triton decode signature to the torch kernel.

    Args:
        x: KDA input activations.
        conv_state: Mutable convolution state cache.
        weight: Depthwise convolution weights.
        bias: Optional per-channel bias.
        activation: ``None``, ``False``, ``True``, ``"silu"`` or ``"swish"``.
        cache_seqlens: Unsupported circular-buffer metadata.
        conv_state_indices: Cache slot for each batch row.
        num_accept_tokens: Unsupported speculative metadata.
        intermediate_conv_window: Unsupported speculative metadata.
        intermediate_state_indices: Unsupported speculative metadata.
        retrieve_next_token: Unsupported tree-retrieve metadata.
        retrieve_next_sibling: Unsupported tree-retrieve metadata.
        retrieve_parent_token: Unsupported tree-retrieve metadata.
        pad_slot_id: Sentinel slot for padded rows.
        metadata: Upstream compatibility metadata.
        validate_data: Upstream validation flag.
    """

    if activation not in (None, False, "silu", "swish", True):
        raise ValueError(f"unsupported KDA causal-conv activation: {activation}")
    return causal_conv1d_update_kunlun(
        x,
        conv_state,
        weight,
        bias,
        activation in ("silu", "swish", True),
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
        num_accept_tokens=num_accept_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
        metadata=metadata,
        validate_data=validate_data,
    )


def kda_causal_conv1d_prefill_kunlun(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    seq_lens_cpu,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation="silu",
    pad_slot_id: int = -1,
    validate_data: bool = False,
    **kwargs,
):
    """Adapt KDA's direct Triton call signature to the torch prefill kernel.

    Args:
        x: Flattened activations with shape ``[dim, tokens]``.
        weight: Depthwise convolution weights.
        bias: Optional per-channel bias.
        conv_states: Optional per-request convolution state cache.
        query_start_loc: Cumulative token boundaries with shape ``[batch + 1]``.
        seq_lens_cpu: Upstream CPU sequence lengths retained for compatibility.
        cache_indices: Cache slot for each request.
        has_initial_state: Whether each request reuses its cached state.
        activation: ``None``, ``False``, ``True``, ``"silu"`` or ``"swish"``.
        pad_slot_id: Sentinel slot for padded requests.
        validate_data: Upstream validation flag.
        kwargs: Additional arguments are rejected as unsupported.
    """

    if kwargs:
        raise NotImplementedError(
            "unsupported KDA causal-conv prefill arguments: "
            f"{sorted(kwargs)}"
        )
    if activation not in (None, False, "silu", "swish", True):
        raise ValueError(f"unsupported KDA causal-conv activation: {activation}")
    query_start_loc_cpu = [0]
    for seq_len in seq_lens_cpu:
        query_start_loc_cpu.append(query_start_loc_cpu[-1] + int(seq_len))
    return causal_conv1d_fwd_kunlun(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ("silu", "swish", True),
        pad_slot_id,
        query_start_loc_cpu=query_start_loc_cpu,
        validate_data=validate_data,
    )


def causal_conv1d_fwd_kunlun(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
    *,
    query_start_loc_cpu=None,
    validate_data: bool = False,
):
    """Run the Kunlun varlen prefill op and update selected states in place.

    Args:
        x: Flattened activations with shape ``[dim, tokens]``.
        weight: Depthwise convolution weights.
        bias: Optional per-channel bias.
        conv_states: Optional cache with one state row per slot.
        query_start_loc: Cumulative token boundaries for each request.
        cache_indices: Cache slot for each request.
        has_initial_state: Selects cached versus zero-initialized state per request.
        silu_activation: Whether to apply SiLU to the convolution output.
        pad_slot_id: Sentinel slot for padded requests.
        query_start_loc_cpu: CPU copy of the cumulative token boundaries.
    """
    if x.dim() != 2 or query_start_loc is None:
        raise ValueError("K3 prefill expects x [dim, tokens] and query_start_loc")
    import kunlun_ops

    dim, total_tokens = x.shape
    if query_start_loc.dim() != 1 or query_start_loc.numel() == 0:
        raise ValueError("query_start_loc must be a non-empty 1-D tensor")
    batch = query_start_loc.numel() - 1
    if weight.dim() != 2 or weight.shape[0] != dim:
        raise ValueError(
            f"expected weight [dim, width] with dim={dim}, got {weight.shape}"
        )
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != dim):
        raise ValueError(f"expected bias [dim] with dim={dim}, got {bias.shape}")
    if cache_indices is not None and (
        cache_indices.dim() != 1 or cache_indices.numel() != batch
    ):
        raise ValueError("cache_indices must have shape [batch]")
    if conv_states is not None:
        if conv_states.dim() != 3 or conv_states.shape[1] != dim:
            raise ValueError(
                "expected KDA conv_states [slots, dim, state_len], "
                f"got {conv_states.shape}"
            )
        if conv_states.shape[-1] != weight.shape[1] - 1:
            raise ValueError(
                "conv state length must equal weight width - 1: "
                f"state={conv_states.shape[-1]}, width={weight.shape[1]}"
            )
    indices = (
        torch.arange(batch, device=x.device, dtype=torch.int32)
        if cache_indices is None
        else cache_indices.to(device=x.device, dtype=torch.int32).contiguous()
    )
    initial_state_mask = (
        torch.zeros(batch, device=x.device, dtype=torch.bool)
        if has_initial_state is None
        else has_initial_state.to(device=x.device, dtype=torch.bool).contiguous()
    )
    if initial_state_mask.dim() != 1 or initial_state_mask.numel() != batch:
        raise ValueError("has_initial_state must have shape [batch]")
    if conv_states is None and has_initial_state is not None:
        raise ValueError("has_initial_state requires conv_states")
    if query_start_loc_cpu is None:
        query_start_loc_cpu = query_start_loc.detach().cpu().tolist()
    if (
        len(query_start_loc_cpu) != batch + 1
        or query_start_loc_cpu[0] != 0
        or query_start_loc_cpu[-1] != total_tokens
    ):
        raise ValueError("query_start_loc_cpu does not match the packed input")

    if validate_data:
        if conv_states is not None:
            _validate_indices(
                indices,
                batch,
                conv_states.shape[0],
                pad_slot_id,
            )
        elif bool((indices != pad_slot_id).any().item()):
            raise ValueError("cache_indices requires conv_states")

    audit_enabled = conv_states is not None and _cache_audit_enabled()
    if audit_enabled:
        valid = indices[(indices != pad_slot_id) & (indices >= 0)]
        audit_before = (
            conv_states.index_select(0, valid.to(torch.long))
            .float()
            .abs()
            .sum()
            .item()
            if valid.numel()
            else 0.0
        )

    query_start_loc = query_start_loc.to(
        device=x.device, dtype=torch.int32
    ).contiguous()
    weight = weight.contiguous()
    bias = None if bias is None else bias.float().contiguous()

    state_for_op = conv_states
    op_indices = indices
    compact_state = False
    is_ncw = True
    output = x.clone(memory_format=torch.contiguous_format)

    if conv_states is not None:
        state_nwc = conv_states.transpose(-1, -2)
        if state_nwc.is_contiguous():
            output = x.transpose(0, 1).contiguous()
            state_for_op = state_nwc
            is_ncw = False
        elif not conv_states.is_contiguous():
            # KDA splits Q/K/V from one contiguous NWC state pool. Each split
            # is strided, while the XPU kernel assumes a compact state tensor.
            # Gather only the active batch rows and scatter their final states
            # back after the op instead of copying the entire cache pool.
            safe_indices = torch.where(
                indices == pad_slot_id,
                torch.zeros_like(indices),
                indices,
            )
            state_for_op = conv_states.index_select(
                0, safe_indices.to(torch.long)
            ).contiguous()
            op_indices = torch.arange(
                batch, device=x.device, dtype=torch.int32
            )
            op_indices = torch.where(
                indices == pad_slot_id,
                torch.full_like(op_indices, pad_slot_id),
                op_indices,
            )
            compact_state = True

    status = kunlun_ops.causal_conv1d_fwd(
        x=output,
        weight=weight,
        bias=bias,
        conv_states=state_for_op,
        query_start_loc=query_start_loc,
        cache_indices=op_indices,
        has_initial_state=initial_state_mask,
        silu_activation=silu_activation,
        is_ncw=is_ncw,
        query_start_loc_cpu=query_start_loc_cpu,
        pad_slot_id=pad_slot_id,
    )
    if status != 0:
        raise RuntimeError(
            f"kunlun_ops.causal_conv1d_fwd failed with status={status}"
        )

    if compact_state:
        valid_rows = torch.nonzero(
            op_indices != pad_slot_id, as_tuple=False
        ).flatten()
        conv_states.index_copy_(
            0,
            indices.index_select(0, valid_rows).to(torch.long),
            state_for_op.index_select(0, valid_rows),
        )

    if audit_enabled:
        _audit_conv_state("prefill", conv_states, indices, audit_before)
    return output if is_ncw else output.transpose(0, 1).contiguous()
