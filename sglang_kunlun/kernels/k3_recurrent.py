"""KDA recurrent kernels and Torch-native references for Kunlun."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_recurrent_kda_decode(
    *, A_log, dt_bias, q, k, v, a, b, initial_state_source,
    initial_state_indices, cu_seqlens, scale=None, lower_bound=None,
) -> torch.Tensor:
    """Run the XSpeedGate KDA decode kernel with packed varlen inputs.

    Inputs use the wheel ABI layout ``[1, T, H, K/V]``.  The kernel updates
    ``initial_state_source`` in place and returns output in that layout.
    """
    import xspeedgate_ops  # noqa: F401  # Registers torch.ops.xspeedgate_ops.

    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"expected packed q [1, num_tokens, H, K], got {q.shape}")
    _, num_tokens, heads, key_dim = q.shape
    if k.shape != q.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} and {k.shape}"
        )
    if v.dim() != 4 or v.shape[:2] != (1, num_tokens):
        raise ValueError(f"expected packed v [1, num_tokens, H, V], got {v.shape}")
    if a.numel() != num_tokens * heads * key_dim:
        raise ValueError(
            f"gate must contain {num_tokens * heads * key_dim} elements, "
            f"got {a.shape}"
        )
    gate = a.reshape(1, num_tokens, heads, key_dim).contiguous()
    if b.numel() != num_tokens * heads:
        raise ValueError(
            f"beta must contain {num_tokens * heads} elements, got {b.shape}"
        )
    beta = b.reshape(1, num_tokens, heads).contiguous()
    if A_log is None or A_log.numel() != heads:
        raise ValueError(f"A_log must contain {heads} elements, got {A_log.shape}")
    if dt_bias is None or dt_bias.numel() != heads * key_dim:
        raise ValueError(
            f"dt_bias must contain {heads * key_dim} elements, got {dt_bias.shape}"
        )
    if initial_state_indices.numel() != cu_seqlens.numel() - 1:
        raise ValueError(
            "initial_state_indices must contain one slot per packed sequence"
        )

    op = torch.ops.xspeedgate_ops.fused_recurrent_kda_fwd.default
    output, _ = op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gate,
        beta,
        initial_state_source,
        cu_seqlens.contiguous(),
        initial_state_indices.contiguous(),
        scale,
        True,                 # inplace_final_state
        None,                 # num_accepted_tokens; decode has no speculative tokens
        True,                 # use_qk_l2norm_in_kernel
        A_log.reshape(-1).contiguous(),
        dt_bias.reshape(heads, key_dim).contiguous(),
        lower_bound,
        True,                 # use_gate_in_kernel
        # use_beta_sigmoid_in_kernel=True: on the decode path ``b`` arrives raw
        # (pre-activation), so the kernel must apply sigmoid itself. This is the
        # deliberate asymmetry vs fused_recurrent_kda_extend, which passes False
        # because the model's ordinary extend path already applied sigmoid.
        # Flipping either one is a *silent* precision bug, not a crash.
        True,                 # use_beta_sigmoid_in_kernel
        20.0,                 # softplus_threshold
        None,                 # out
    )
    return output


def fused_recurrent_kda_extend(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    g,
    beta,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    scale=None,
    lower_bound=None,
    sequence_lengths=None,
) -> torch.Tensor:
    """Run ordinary packed-varlen KDA extend with XSpeedGate.

    The wheel uses a two-dimensional state-index tensor for multi-token
    requests.  Each row belongs to one packed sequence.  Its first and last
    valid entries contain that sequence's state-pool slot; intermediate and
    padding entries are ``-1`` so the kernel keeps the recurrent state internal
    instead of writing a state snapshot for every token.

    ``beta`` is already sigmoid-activated by the Kimi-K3 model's ordinary
    extend path.  It is rounded to the Q/K/V dtype because the custom op
    requires Q/K/V/beta to share a dtype, and beta sigmoid is disabled in the
    kernel to avoid applying it twice.
    """
    import xspeedgate_ops  # noqa: F401  # Registers torch.ops.xspeedgate_ops.

    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"expected packed q [1, num_tokens, H, K], got {q.shape}")
    _, num_tokens, heads, key_dim = q.shape
    if k.shape != q.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")
    if v.dim() != 4 or v.shape[:3] != (1, num_tokens, heads):
        raise ValueError(f"expected packed v [1, num_tokens, H, V], got {v.shape}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q/k/v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if g.numel() != num_tokens * heads * key_dim:
        raise ValueError(
            f"gate must contain {num_tokens * heads * key_dim} elements, got {g.shape}"
        )
    gate = g.reshape(1, num_tokens, heads, key_dim).contiguous()
    if beta.numel() != num_tokens * heads:
        raise ValueError(
            f"beta must contain {num_tokens * heads} elements, got {beta.shape}"
        )
    beta_values = beta.reshape(1, num_tokens, heads).to(dtype=q.dtype).contiguous()
    if A_log is None or A_log.numel() != heads:
        raise ValueError(
            f"A_log must contain {heads} elements, got {getattr(A_log, 'shape', None)}"
        )
    if dt_bias is None or dt_bias.numel() != heads * key_dim:
        raise ValueError(
            "dt_bias must contain "
            f"{heads * key_dim} elements, got {getattr(dt_bias, 'shape', None)}"
        )
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be one-dimensional, got {cu_seqlens.shape}")

    num_sequences = cu_seqlens.numel() - 1
    if num_sequences <= 0:
        raise ValueError("packed KDA extend requires at least one sequence")
    if initial_state_indices.numel() != num_sequences:
        raise ValueError(
            "initial_state_indices must contain one slot per packed sequence"
        )

    # The scheduler already provides CPU sequence lengths on ordinary extend.
    # If a compatible upstream version omits them, using num_tokens as the row
    # width is still valid because cu_seqlens masks entries past each sequence.
    max_sequence_length = num_tokens
    if sequence_lengths is not None:
        if isinstance(sequence_lengths, torch.Tensor):
            lengths = sequence_lengths.detach().cpu().tolist()
        else:
            lengths = list(sequence_lengths)
        lengths = [int(length) for length in lengths]
        if len(lengths) != num_sequences:
            raise ValueError(
                "sequence_lengths must contain one length per packed sequence"
            )
        if any(length < 0 for length in lengths) or sum(lengths) != num_tokens:
            raise ValueError(
                "sequence_lengths must be non-negative and sum to num_tokens"
            )
        max_sequence_length = max(lengths, default=0)

    if max_sequence_length <= 0:
        raise ValueError("packed KDA extend requires at least one token")

    cu_seqlens_device = cu_seqlens.to(
        device=q.device, dtype=torch.int32
    ).contiguous()
    lengths_device = cu_seqlens_device[1:] - cu_seqlens_device[:-1]
    token_offsets = torch.arange(
        max_sequence_length, device=q.device, dtype=torch.int32
    ).reshape(1, -1)

    slots = initial_state_indices.to(device=q.device, dtype=torch.int32).reshape(
        num_sequences, 1
    )
    valid = token_offsets < lengths_device.reshape(-1, 1)
    boundary = (token_offsets == 0) | (
        token_offsets == lengths_device.reshape(-1, 1) - 1
    )
    state_indices = torch.where(
        valid & boundary,
        slots,
        torch.full_like(slots, -1),
    ).contiguous()

    op = torch.ops.xspeedgate_ops.fused_recurrent_kda_fwd.default
    output, _ = op(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gate,
        beta_values,
        initial_state_source,
        cu_seqlens_device,
        state_indices,
        scale,
        True,                 # inplace_final_state
        None,                 # num_accepted_tokens; ordinary extend only
        True,                 # use_qk_l2norm_in_kernel
        A_log.reshape(-1).contiguous(),
        dt_bias.reshape(heads, key_dim).contiguous(),
        lower_bound,
        True,                 # use_gate_in_kernel
        # use_beta_sigmoid_in_kernel=False: ``beta`` is already sigmoid-activated
        # by the Kimi-K3 model's ordinary extend path, so letting the kernel
        # apply sigmoid again would double-activate it. Mirror image of
        # fused_recurrent_kda_decode, which passes True. Silent precision bug if
        # swapped.
        False,                # beta is already sigmoid-activated by Kimi-K3
        20.0,                 # softplus_threshold
        None,                 # out
    )
    return output


def _prepare_gate(a, A_log, dt_bias, heads, key_dim, lower_bound):
    values = a.float().reshape(a.shape[0], a.shape[1], heads, key_dim)
    bias = dt_bias.reshape(heads, key_dim).float().view(1, 1, heads, key_dim)
    decay = A_log.reshape(-1)[:heads].float().exp().view(1, 1, heads, 1)
    if lower_bound is None:
        return -decay * F.softplus(values + bias)
    return lower_bound * torch.sigmoid(decay * (values + bias))


def _run_recurrent(
    q, k, v, gate, beta, state, scale, *, checkpoint_every=None, checkpoints=None
):
    qf = q.float()
    kf = k.float()
    qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    output = torch.empty_like(v)
    for step in range(q.shape[0]):
        if checkpoint_every is not None and step % checkpoint_every == 0:
            checkpoints.append(state.clone())
        state.mul_(torch.exp(gate[step]).unsqueeze(1))
        delta = v[step].float() - torch.einsum("hvk,hk->hv", state, kf[step])
        state.add_(beta[step].view(-1, 1, 1) * delta.unsqueeze(-1) * kf[step].unsqueeze(1))
        output[step] = torch.einsum("hvk,hk->hv", state, qf[step] * scale).to(v.dtype)
    return output


def _track_state_rows(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
) -> None:
    """Copy selected state rows while preserving overlapping source rows."""

    for row in range(batch_size):
        if not bool(mamba_track_mask[row].item()):
            continue
        src = int(cache_indices[row].item())
        dst = int(mamba_track_indices[row].item())
        if check_freed_slots and (src < 0 or dst < 0):
            continue
        conv_states[..., dst, :, :].copy_(conv_states[..., src, :, :].clone())
        ssm_states[..., dst, :, :, :].copy_(ssm_states[..., src, :, :, :].clone())


def track_mamba_states_if_needed_kunlun(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
) -> None:
    """Torch replacement for one-layer radix Mamba state tracking.

    Args:
        conv_states: Function parameter.
        ssm_states: Function parameter.
        cache_indices: Function parameter.
        mamba_track_mask: Function parameter.
        mamba_track_indices: Function parameter.
        batch_size: Function parameter.
        check_freed_slots: Function parameter."""

    _track_state_rows(
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        batch_size,
        check_freed_slots,
    )


def track_mamba_states_all_layers_kunlun(
    conv_states_pool: torch.Tensor,
    ssm_states_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
) -> None:
    """Torch replacement for fused all-layer radix Mamba state tracking.

    Args:
        conv_states_pool: Function parameter.
        ssm_states_pool: Function parameter.
        cache_indices: Function parameter.
        mamba_track_mask: Function parameter.
        mamba_track_indices: Function parameter.
        batch_size: Function parameter.
        check_freed_slots: Function parameter."""

    _track_state_rows(
        conv_states_pool,
        ssm_states_pool,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        batch_size,
        check_freed_slots,
    )


def recurrent_kda_extend(
    *, q, k, v, g, beta, initial_state, initial_state_indices,
    cu_seqlens=None, A_log=None, dt_bias=None, scale=None, lower_bound=None,
    beta_is_logits=False, return_intermediate_states=False, checkpoint_every=64,
):
    """Sequential KDA prefill reference for flattened varlen requests.

    Args:
        q: Function parameter.
        k: Function parameter.
        v: Function parameter.
        g: Function parameter.
        beta: Function parameter.
        initial_state: Function parameter.
        initial_state_indices: Function parameter.
        cu_seqlens: Function parameter.
        A_log: Function parameter.
        dt_bias: Function parameter.
        scale: Function parameter.
        lower_bound: Function parameter.
        beta_is_logits: Function parameter.
        return_intermediate_states: Function parameter.
        checkpoint_every: Function parameter."""
    if q.dim() != 4:
        raise ValueError(f"expected q [B, T, H, K], got {q.shape}")
    batch, tokens, heads, key_dim = q.shape
    if scale is None:
        scale = key_dim**-0.5
    if cu_seqlens is None:
        request_count = batch
        bounds = [(row, 0, tokens) for row in range(batch)]
    else:
        if batch != 1:
            raise ValueError("varlen KDA expects flattened batch dimension one")
        sequence_bounds = [int(item) for item in cu_seqlens.tolist()]
        request_count = len(sequence_bounds) - 1
        bounds = [
            (0, sequence_bounds[request], sequence_bounds[request + 1])
            for request in range(request_count)
        ]
    if initial_state_indices.numel() != request_count:
        raise ValueError("initial_state_indices must contain one slot per request")
    if initial_state.device != q.device:
        raise ValueError("initial_state and q must be on the same device")
    if initial_state_indices.numel() and (
        bool((initial_state_indices < 0).any().item())
        or bool((initial_state_indices >= initial_state.shape[0]).any().item())
    ):
        raise ValueError("initial_state_indices contains an out-of-range slot")
    if A_log is None or dt_bias is None:
        gate = g.float()
    else:
        gate = _prepare_gate(g, A_log, dt_bias, heads, key_dim, lower_bound)
    beta_values = beta.float()
    if beta_is_logits:
        beta_values = torch.sigmoid(beta_values)
    if beta_values.dim() == 3 and beta_values.shape[-1] == 1:
        beta_values = beta_values.expand(-1, -1, heads)
    output = torch.empty_like(v)
    checkpoints = [] if return_intermediate_states else None
    pending_states = []
    for request, (row, begin, end) in enumerate(bounds):
        slot = int(initial_state_indices[request].item())
        state = initial_state[slot].float().clone()
        output[row, begin:end] = _run_recurrent(
            q[row, begin:end], k[row, begin:end], v[row, begin:end],
            gate[0, begin:end] if gate.shape[0] == 1 else gate[row, begin:end],
            beta_values[0, begin:end] if beta_values.shape[0] == 1 else beta_values[row, begin:end],
            state,
            scale,
            checkpoint_every=checkpoint_every if return_intermediate_states else None,
            checkpoints=checkpoints,
        )
        pending_states.append((slot, state))
    for slot, state in pending_states:
        initial_state[slot].copy_(state.to(initial_state.dtype))
    if not return_intermediate_states:
        return output
    if not checkpoints:
        h = initial_state.new_empty((1, 0, *initial_state.shape[1:]))
    else:
        h = torch.stack(checkpoints, dim=0).unsqueeze(0)
    return output, h
