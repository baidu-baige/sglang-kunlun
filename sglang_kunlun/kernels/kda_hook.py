"""Kunlun replacements for the KDA / Mamba causal depthwise conv helpers.

34 of GLM-5-Next's 45 layers are KDA linear attention, and every one of them
runs ``causal_conv1d_fn`` (extend) or ``causal_conv1d_update`` (decode) from
``sglang.kernels.ops.mamba.causal_conv1d_triton``. Both are Triton, which cannot
compile on P800, so they are routed to ``kunlun_ops`` instead - these two ops
exist there with matching semantics
(``Y = act(causal_conv1d(concat(conv_states, x)) + bias)``), including the
varlen / cache-line / ``pad_slot_id`` contract.

Layout note: the upstream contract is ``conv_states[..., dim, state_len]``, but
the KDA pool allocates ``[lines, state_len, dim]`` and hands over a transposed
view, so the buffer is really NWC. ``kunlun_ops`` takes the layout explicitly,
so the view is un-transposed here instead of being made contiguous.
"""

import math
import os
from typing import List, Optional, Tuple, Union

import torch

import kunlun_ops
import xspeedgate_ops  # noqa: F401  -- registers torch.ops.xspeedgate_ops.*

from sglang_kunlun.kernels.kernel_ops import register_jit_op

# Escape hatch for A/B-ing the xspeedgate op paths below against the torch
# ports that preceded them. Set to 1 to force every hook back to torch.
_FORCE_TORCH = os.environ.get("SGLANG_KUNLUN_KDA_TORCH", "0") == "1"
# Narrower A/B: force ONLY the KDA recurrent decode step (the batched xspeedgate
# fused_sigmoid_gating_delta_rule_update op) back to torch, keeping prefill fast.
_FORCE_TORCH_DECODE = os.environ.get("SGLANG_KDA_DECODE_TORCH", "0") == "1"
# Zero the op output rows of CUDA-graph padding decode tokens (slot == -1); on by
# default. Set SGLANG_KDA_PAD_GUARD=0 to see the raw op output for localization.
_PAD_GUARD = os.environ.get("SGLANG_KDA_PAD_GUARD", "1") == "1"
# Route KDA decode to the KDA-dedicated op
# xspeedgate_ops.fused_recurrent_kda_packed_decode (the one vLLM-Kunlun/kimi-k3
# uses) instead of the generic fused_sigmoid_gating_delta_rule_update. ON by
# default (accuracy on par with generic+guard, handles cuda-graph padding itself).
# Set SGLANG_KDA_NATIVE_PACKED=0 to fall back to the generic op + padding guard.
_NATIVE_PACKED = os.environ.get("SGLANG_KDA_NATIVE_PACKED", "1") == "1"

# Upstream passes exactly this as ``scale`` into ``kda_gate_chunk_cumsum``
# (kda.py:1114); the xspeedgate op folds the same 1/ln(2) in internally, so the
# op path is only valid when the caller asks for that scale.
_RCP_LN2 = 1.0 / math.log(2.0)

_CONV_MODULE = "sglang.kernels.ops.mamba.causal_conv1d_triton"
_L2NORM_MODULE = "sglang.kernels.ops.attention.fla.l2norm"
_KDA_MODULE = "sglang.kernels.ops.attention.fla.kda"
_RECURRENT_MODULE = "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent"
_RECURRENT_MODULE_PACKED = "sglang.kernels.ops.attention.fla.fused_recurrent"
_NORM_GATE_MODULE = "sglang.kernels.ops.attention.fla.fused_norm_gate"

# kimi_delta_attention emits one intermediate-state row per this many tokens.
# Fixed by the op, and it has to match sglang's mamba_cache_chunk_size for the
# radix track path to index the right rows.
_KDA_STATE_BLOCK = 64


def _mamba_cache_chunk_size() -> int:
    from sglang.srt.runtime_context import mamba_cache_chunk_size

    return mamba_cache_chunk_size()


@register_jit_op(_NORM_GATE_MODULE, "layer_norm_gated_fwd")
def layer_norm_gated_fwd_kunlun(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    activation: str = "swish",
    eps: float = 1e-5,
    residual: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    is_rms_norm: bool = False,
):
    """Fused (layer|rms)norm + output-gate forward.

    KDA's ``o_norm`` runs this per layer (34 layers per token). ``xspeedgate_ops``
    has this op with a signature that matches upstream argument for argument, so
    the fast path is a direct call; the torch port below is kept as the reference
    / fallback (``SGLANG_KUNLUN_KDA_TORCH=1``) and mirrors
    ``layer_norm_gated_fwd_kernel``: optional residual add first (and that sum,
    not ``x``, is what gets normalized and returned as ``residual_out``), fp32
    statistics, then the gate ``y * g * sigmoid(g)`` for swish/silu.
    """

    assert x.dim() == 2, f"expected 2D x, got {tuple(x.shape)}"

    if not _FORCE_TORCH:
        # The op requires x/g/weight/bias to share a dtype and to be contiguous.
        dtypes = {t.dtype for t in (x, g, weight, bias) if t is not None}
        if len(dtypes) == 1:
            return torch.ops.xspeedgate_ops.layer_norm_gated_fwd(
                x.contiguous(),
                g.contiguous(),
                None if weight is None else weight.contiguous(),
                None if bias is None else bias.contiguous(),
                activation,
                eps,
                None if residual is None else residual.contiguous(),
                out_dtype,
                residual_dtype,
                is_rms_norm,
            )

    if residual is not None:
        residual_dtype = residual.dtype

    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = xf.to(residual_dtype)
    else:
        residual_out = None

    if is_rms_norm:
        mean = None
        var = xf.pow(2).mean(-1, keepdim=True)
        centered = xf
    else:
        mean_keep = xf.mean(-1, keepdim=True)
        centered = xf - mean_keep
        var = centered.pow(2).mean(-1, keepdim=True)
        mean = mean_keep.squeeze(-1)
    rstd_keep = torch.rsqrt(var + eps)

    y = centered * rstd_keep
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()

    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    else:
        raise NotImplementedError(f"kunlun layer_norm_gated: act {activation!r}")

    return y.to(out_dtype or x.dtype), mean, rstd_keep.squeeze(-1), residual_out


def _kda_gate_activation(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Natural-log-space KDA gate: fp32, per token, no cumsum and no log2 scale.

    This is the activation half of ``kda_gate_chunk_cumsum_vector_kernel``.
    """

    H, K = g.shape[-2], g.shape[-1]
    x = g.float()
    if dt_bias is not None:
        x = x + dt_bias.reshape(1, 1, H, K).float()
    a = A_log.reshape(H).float().exp().reshape(1, 1, H, 1)
    if lower_bound is None:
        # softplus with the kernel's linear cutoff at 20.0
        sp = torch.where(x < 20.0, torch.log1p(x.exp()), x)
        return -a * sp
    return lower_bound * torch.sigmoid(a * x)


def _chunk_local_cumsum_(out: torch.Tensor, chunk_size: int) -> None:
    """In-place chunk-local cumsum along dim 0 of ``out`` ([T, H, K])."""

    total = out.shape[0]
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        out[start:stop] = out[start:stop].cumsum(0)


@register_jit_op(_KDA_MODULE, "kda_gate_chunk_cumsum")
def kda_gate_chunk_cumsum_kunlun(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Fused KDA gate activation + chunk-local cumsum.

    ``xspeedgate_ops.fused_kda_gate_chunk_cumsum`` covers this, but it is the
    *wider* fusion: it also does ``sigmoid(raw_beta)`` and it folds the 1/ln(2)
    scale in unconditionally, so the op path only applies when the caller asked
    for that scale. beta is computed separately upstream, so a zero placeholder
    is passed and the second return value dropped.

    The torch reference below mirrors ``kda_gate_chunk_cumsum_vector_kernel``
    exactly: fp32 activation, then a cumsum that restarts at every ``chunk_size``
    boundary and never crosses a sequence boundary, then the ``scale``
    (log2(e)) multiply - in that order, since the downstream chunk kernels
    consume log2-space gates.
    """

    assert g.dim() == 4, f"expected [B, T, H, K] gate, got {tuple(g.shape)}"
    B, T, H, K = g.shape
    if cu_seqlens is not None:
        assert B == 1, "varlen KDA gate expects batch 1"

    if not _FORCE_TORCH and scale is not None and abs(scale - _RCP_LN2) < 1e-9:
        # dt_bias arrives flat ([H*K]); the op wants the [H, K] gate bias.
        g_bias = None if dt_bias is None else dt_bias.reshape(H, K).float().contiguous()
        raw_beta = g.new_zeros((B, T, H))
        gate, _beta = torch.ops.xspeedgate_ops.fused_kda_gate_chunk_cumsum(
            g.contiguous(),
            raw_beta,
            A_log.reshape(H).float().contiguous(),
            g_bias,
            1.0,  # softplus beta
            20.0,  # softplus linear-branch threshold, as in the torch port
            lower_bound,
            cu_seqlens,
            chunk_indices,
            chunk_size,
            output_dtype or torch.float32,
        )
        return gate

    out = _kda_gate_activation(g, A_log, dt_bias, lower_bound).contiguous()

    if cu_seqlens is None:
        for b in range(B):
            _chunk_local_cumsum_(out[b], chunk_size)
    else:
        bounds = cu_seqlens.tolist()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop > start:
                _chunk_local_cumsum_(out[0, start:stop], chunk_size)

    if scale is not None:
        out = out * scale
    return out.to(output_dtype or g.dtype)


# Upstream default; kept local so the import graph stays independent.
PAD_SLOT_ID = -1


def _kda_decode_xspeedgate(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor,
    lower_bound: Optional[float],
    HV: int,
    K: int,
) -> torch.Tensor:
    """KDA recurrent decode through the xspeedgate gating + delta-rule op.

    ``is_kda=True`` selects the per-K-channel gate, which is what KDA needs and
    what ``kunlun_ops.fused_sigmoid_gating_delta_rule_update`` cannot express.
    The op wants ``a`` / ``b`` flattened to ``[num_tokens, *]``, takes the state
    pool as ``[num_states, HV, V, K]`` (hence ``is_h0_transposed=True``), applies
    ``sigmoid`` to raw ``b`` itself, and leaves rows named by a negative index
    untouched (verified: only the addressed pool rows change).

    ``lower_bound`` (the safe gate, ``gate_lower_bound`` in the model config) has
    no argument on the op, which only knows the softplus gate
    ``g = -exp(A_log) * softplus(a + dt_bias)``. softplus is invertible over the
    safe gate's range ``(lower_bound, 0)``, so the target gate is fed back
    through it: with ``A_log' = 0`` and ``dt_bias' = 0``,
    ``a' = log(expm1(-g))`` makes the op reproduce
    ``g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`` exactly. Measured
    against the torch recurrence below at (4, 8, 128, 128): o cos 0.999999,
    final state cos 1.000000, max rel err 3.4e-3 (bf16 scale).
    """

    a_flat = a.reshape(-1, HV * K).float()
    b_flat = b.reshape(-1, HV).float().contiguous()

    if _NATIVE_PACKED:
        # Route to the KDA-dedicated native op (same one vLLM-Kunlun/kimi-k3 uses):
        # repack q/k/v into mixed_qkv and pass the RAW gate/beta + lower_bound
        # directly (native op implements the lower_bound gate, no log(expm1) hack).
        H = q.shape[-2]
        n_tok = q.reshape(-1, H, K).shape[0]
        mixed_qkv = torch.cat(
            [
                q.reshape(n_tok, H * K),
                k.reshape(n_tok, H * K),
                v.reshape(n_tok, -1),
            ],
            dim=-1,
        ).contiguous()
        raw_g = a.reshape(n_tok, HV, K).contiguous()
        raw_beta = b.reshape(n_tok, HV).contiguous()
        _slots_np = initial_state_indices.reshape(-1)[:n_tok].to(torch.int32)
        o = torch.ops.xspeedgate_ops.fused_recurrent_kda_packed_decode(
            mixed_qkv,
            raw_g,
            raw_beta,
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(HV, K).float().contiguous(),
            lower_bound,
            initial_state_source,
            _slots_np,
            scale,
            softplus_threshold,
        )
        return o

    if lower_bound is None:
        A_log_in = A_log.reshape(HV).float().contiguous()
        a_in = a_flat.contiguous()
        dt_in = dt_bias.reshape(HV * K).float().contiguous()
        beta_in, threshold_in = softplus_beta, softplus_threshold
    else:
        gate = lower_bound * torch.sigmoid(
            A_log.reshape(HV, 1).float().exp().expand(HV, K).reshape(1, HV * K)
            * (a_flat + dt_bias.reshape(1, HV * K).float())
        )
        # softplus_beta is pinned to 1 here because that is the inverse applied.
        a_in = torch.log(torch.expm1(-gate)).contiguous()
        A_log_in = torch.zeros(HV, dtype=torch.float32, device=a_in.device)
        dt_in = torch.zeros(HV * K, dtype=torch.float32, device=a_in.device)
        beta_in, threshold_in = 1.0, 20.0

    n_seq = cu_seqlens.numel() - 1
    _slots = initial_state_indices.reshape(-1)[:n_seq].to(torch.int32)
    _res = torch.ops.xspeedgate_ops.fused_sigmoid_gating_delta_rule_update(
        A_log_in,
        a_in,
        dt_in,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        b_flat,
        beta_in,
        threshold_in,
        initial_state_source,
        _slots,
        scale,
        use_qk_l2norm_in_kernel,
        True,  # is_h0_transposed: the KDA pool is [num_states, HV, V, K]
        True,  # is_kda: per-K-channel gate
        cu_seqlens.to(torch.int32),
    )
    # CUDA-graph padding guard: padded decode rows carry slot == -1 and read
    # stale q/k/v from the static replay buffers; the op can emit non-finite /
    # huge output for them, which then contaminates the real rows through
    # downstream tiled kernels ("!" 刷屏). Zero those rows. Graph-safe: pure
    # tensor ops, no device->host sync (no .any()/.item()); torch.where avoids
    # inf*0 -> NaN. Decode packs one token per sequence, so the leading dim of
    # the op output lines up with the per-sequence slots.
    # Set SGLANG_KDA_PAD_GUARD=0 to observe the raw (unguarded) op output while
    # localizing the underlying kernel defect.
    if _PAD_GUARD:
        _keep = (_slots >= 0).reshape(n_seq, 1)
        _res = torch.where(
            _keep, _res.reshape(n_seq, -1), _res.new_zeros(())
        ).reshape(_res.shape)
    return _res


@register_jit_op(_RECURRENT_MODULE, "fused_sigmoid_gating_delta_rule_update")
def fused_sigmoid_gating_delta_rule_update_kunlun(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Torch port of the fused gating + recurrent delta-rule decode step.

    ``kunlun_ops.fused_sigmoid_gating_delta_rule_update`` exists but is the
    Qwen3-Next / GDN shape: a per-head *scalar* gate (``a`` is
    ``[B, T, HV]``, ``A_log`` / ``dt_bias`` are ``[HV]``). KDA gates per K
    channel, which that op cannot express, so the recurrence is done in torch.
    Decode runs one token per sequence, so this is a handful of elementwise ops
    per layer rather than a real loop.

    Follows ``fused_sigmoid_gating_delta_rule_update_kernel`` step by step: gate,
    ``beta = sigmoid(b)``, optional q/k l2norm, ``q *= scale``, ``h *= exp(g)``,
    ``v -= h @ k``, ``v *= beta``, ``h += outer(k, v)``, ``o = h @ q``, then the
    final state is written back to the pool rows named by
    ``initial_state_indices`` (negative index = padding, left untouched).
    """

    if cache_ring:
        raise NotImplementedError(
            "kunlun fused_sigmoid_gating_delta_rule_update: the fused ReplaySSM "
            "ring-write is not wired up (needs --enable-linear-replayssm[-spec])"
        )
    if not is_kda:
        raise NotImplementedError(
            "kunlun fused_sigmoid_gating_delta_rule_update: only the KDA "
            "(per-channel gate) path is ported"
        )

    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5

    if (
        not _FORCE_TORCH
        and not _FORCE_TORCH_DECODE
        and cu_seqlens is not None
        and H == HV
        and intermediate_states_buffer is None
        and intermediate_state_indices is None
        and retrieve_parent_token is None
        and not disable_state_update
        and replayssm_rawv is None
        and replayssm_rawk is None
        and replayssm_g is None
        and replayssm_beta is None
    ):
        # ``xspeedgate_ops.fused_sigmoid_gating_delta_rule_update`` covers this
        # whole recurrence on the ``is_kda=True`` path (per-K-channel gate). It
        # has no intermediate-state cache and always commits the final state, so
        # the MTP verify shapes above stay on the torch recurrence below.
        return _kda_decode_xspeedgate(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            HV=HV,
            K=K,
        )

    n_seq = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if cu_seqlens is None:
        steps = T
    else:
        # Derive the per-sequence step count from shapes rather than reading
        # cu_seqlens back to host: a .tolist() here is a device->host sync, and
        # this function runs inside the decode CUDA-graph capture where such a
        # readback is not valid. Decode packs an equal number of tokens per
        # sequence (T == n_seq * steps), which is what the recurrence below
        # assumes anyway.
        assert B == 1, "varlen decode expects batch 1"
        assert (
            T % n_seq == 0
        ), f"ragged decode step is not supported: T={T} n_seq={n_seq}"
        steps = T // n_seq

    # [n_seq, steps, ...] views over the packed token axis.
    def per_seq(x, heads, dim):
        return x.reshape(n_seq, steps, heads, dim).float()

    q_s = per_seq(q, H, K)
    k_s = per_seq(k, H, K)
    v_s = per_seq(v, HV, V)
    a_s = per_seq(a, HV, K)
    # beta = sigmoid(b), matching `b_beta = 1.0 / (1.0 + tl.exp(-b_b))` in
    # fused_sigmoid_gating_delta_rule_update_kernel. The caller passes b raw
    # (the extend path sigmoids it inside chunk_kda), so skipping it here scales
    # the delta-rule write by the wrong factor -- and for b < 0 by the wrong
    # sign. Measured against a fp32 reference: with the sigmoid the output
    # matches the Triton kernel at cos 0.999993, without it at cos 0.4926.
    beta_s = torch.sigmoid(b.reshape(n_seq, steps, HV).float())


    if use_qk_l2norm_in_kernel:
        q_s = q_s * torch.rsqrt(q_s.pow(2).sum(-1, keepdim=True) + 1e-6)
        k_s = k_s * torch.rsqrt(k_s.pow(2).sum(-1, keepdim=True) + 1e-6)
    q_s = q_s * scale

    # GQA: every group of HV // H value heads shares one k/q head.
    if HV != H:
        repeat = HV // H
        q_s = q_s.repeat_interleave(repeat, dim=2)
        k_s = k_s.repeat_interleave(repeat, dim=2)

    x = a_s + dt_bias.reshape(1, 1, HV, K).float()
    a_coef = A_log.reshape(1, 1, HV, 1).float().exp()
    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(a_coef * x)
    else:
        bx = softplus_beta * x
        sp = torch.where(
            bx <= softplus_threshold, torch.log1p(bx.exp()) / softplus_beta, x
        )
        g = -a_coef * sp

    slots = initial_state_indices.reshape(-1)[:n_seq]
    valid = slots >= 0
    valid4 = valid.reshape(-1, 1, 1, 1)
    # Everything from here on must be shape-static and free of device->host
    # syncs: this runs inside the decode CUDA-graph capture, where a reduction
    # (.min()/.item()) is only *recorded*, so reading it back yields the
    # unexecuted kernel's garbage output, and a data-dependent shape
    # (h[valid]) turns that garbage into an absurd allocation. Padding rows
    # (slot -1) are clamped onto row 0 and masked out instead of being sliced
    # away.
    gather = slots.clamp_min(0).to(torch.long)
    # Pool rows are [HV, V, K]; padding rows read as zero state.
    orig = initial_state_source.index_select(0, gather).float()
    h = torch.where(valid4, orig, torch.zeros_like(orig))

    o = q.new_empty(B, T, HV, V)
    o_s = o.reshape(n_seq, steps, HV, V)

    # MTP verify: every draft step's post-update state is cached, and under a
    # draft *tree* each step restarts from its parent's cached state instead of
    # the running one (chain verify == parent is the previous step, so no
    # reload). The cache is addressed exactly as the kernel does it, flat:
    # row = cache_idx * cache_steps + step, each row [HV, V, K].
    cache_flat = None
    parent = None
    if intermediate_states_buffer is not None:
        assert intermediate_state_indices is not None
        cache_stride_steps = (
            cache_steps if cache_steps is not None else intermediate_states_buffer.shape[1]
        )
        cache_flat = intermediate_states_buffer.reshape(-1, HV, V, K)
        # reshape() silently copies a non-contiguous buffer, which would drop
        # every cached step on the floor instead of failing.
        assert (
            cache_flat.data_ptr() == intermediate_states_buffer.data_ptr()
        ), "intermediate_states_buffer must be flattenable as a view"
        c_idx = intermediate_state_indices.reshape(-1)[:n_seq]
        c_valid = c_idx >= 0
        c_base = c_idx.clamp_min(0).to(torch.long) * cache_stride_steps
        c_valid4 = c_valid.reshape(-1, 1, 1, 1)
        parent = (
            None
            if retrieve_parent_token is None
            else retrieve_parent_token[:n_seq, :steps].long()
        )

    for t in range(steps):
        if cache_flat is not None and parent is not None and t > 0:
            from_parent = cache_flat.index_select(
                0, c_base + parent[:, t]
            ).float()
            h = torch.where(c_valid4, from_parent, h)
        h = h * g[:, t].unsqueeze(-2).exp()
        kt = k_s[:, t]  # [n_seq, HV, K]
        vt = v_s[:, t] - (h * kt.unsqueeze(-2)).sum(-1)
        vt = vt * beta_s[:, t].unsqueeze(-1)
        h = h + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        o_s[:, t] = (h * q_s[:, t].unsqueeze(-2)).sum(-1).to(o.dtype)
        if cache_flat is not None:
            _masked_index_copy_(cache_flat, c_base + t, h, c_valid)

    if not disable_state_update:
        # index_copy_ leaves the write order undefined when indices repeat, and
        # every padding row clamps onto row 0 -- so if row 0 is also a real slot
        # in this batch, a padding write could clobber its update. Point the
        # padding rows at the first valid row *and* hand them that row's
        # payload, making the duplicates byte-identical and the order moot.
        first_valid = torch.argmax(valid.to(torch.uint8)).reshape(1)
        idx = torch.where(
            valid, gather, gather.index_select(0, first_valid).expand_as(gather)
        )
        val = torch.where(valid4, h, h.index_select(0, first_valid).expand_as(h))
        # An all-padding batch has no first valid row; rewrite what was read so
        # the pool is left untouched.
        val = torch.where(valid.any(), val, orig)
        initial_state_source.index_copy_(
            0, idx, val.to(initial_state_source.dtype)
        )
    return o


@register_jit_op(_KDA_MODULE, "chunk_kda_fwd")
def chunk_kda_fwd_kunlun(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Whole-pipeline replacement for the KDA extend chunk kernels.

    Upstream ``chunk_kda_fwd`` is a chain of five Triton kernels (intra, delta
    solve, state recurrence, inter output), none of which can compile on P800.
    ``kunlun_ops.kimi_delta_attention`` implements the same gated delta rule
    end to end, so the replacement is done at this level instead of per kernel.

    Semantics were pinned empirically against a serial torch reference
    (see /tmp probes): ``alpha`` is the q scale, ``g`` is expected per token in
    *natural*-log space (not the log2 / chunk-cumsum form the Triton chain
    wants), ``h0`` / ``ht`` are ``[num_seqs, H, V, K]`` fp32 with ``ht`` written
    to a separate buffer, and ``o`` is ``[1, T, H, V]`` in the activation dtype.

    ``output_intermediate_states`` (the mamba radix track path) needs the state
    at every 64-token block boundary, which the op exposes through its optional
    ``h`` out-param since kunlun_ops 0.1.237.
    """

    return _chunk_kda_fwd_kunlun_impl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_intermediate_states=output_intermediate_states,
    )


def _chunk_kda_fwd_kunlun_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if cu_seqlens is None:
        raise NotImplementedError("kunlun chunk_kda_fwd requires varlen cu_seqlens")

    assert q.dim() == 4 and q.shape[0] == 1, f"expected [1, T, H, K] q, got {tuple(q.shape)}"

    # kimi_delta_attention dtype contract: for bf16 q the gate / state / beta
    # tensors are fp32 (data bf16 + fp32 accumulators); for any other q dtype
    # (e.g. --dtype float16) the op only has a "pure" template where gate /
    # state / beta share q's dtype. Mixing fp32 accumulators with an fp16 q
    # trips a dtype check, so pick the accumulator dtype off q.
    _state_dtype = torch.float32 if q.dtype == torch.bfloat16 else q.dtype

    if A_log is not None:
        g = _kda_gate_activation(g, A_log, dt_bias, lower_bound).to(_state_dtype)
    else:
        # Caller already applied the activation; it is in natural-log space
        # (upstream converts to log2 inside the cumsum), which is what the
        # kunlun op wants.
        g = g.to(_state_dtype)

    cu_xpu = cu_seqlens.to(torch.int32)
    cu_cpu = cu_xpu.cpu()
    num_seqs = cu_cpu.numel() - 1

    # kimi_delta_attention takes states as a dense per-sequence stack, so the
    # pool rows selected by initial_state_indices are gathered in and the final
    # states scattered back (upstream updates the pool in place).
    slots = initial_state_indices.to(torch.long)[:num_seqs]
    h0 = initial_state.index_select(0, slots).to(_state_dtype).contiguous()
    ht = torch.empty_like(h0)
    h = None
    if output_intermediate_states:
        # The op packs one state row per 64-token block per sequence, in
        # cu_seqlens order: row `offset_s + j` is the state after j * 64 tokens
        # of sequence s (so row offset_s == h0[s]), and a partial trailing block
        # still gets its own row. Upstream indexes h on the
        # mamba_cache_chunk_size grid, so that has to be the same 64.
        chunk = _mamba_cache_chunk_size()
        assert chunk == _KDA_STATE_BLOCK, (
            f"kunlun chunk_kda_fwd: kimi_delta_attention emits intermediate "
            f"states on a {_KDA_STATE_BLOCK}-token grid, but "
            f"mamba_cache_chunk_size()={chunk}; the radix track path would read "
            f"the wrong rows"
        )
        seq_lens = cu_cpu[1:] - cu_cpu[:-1]
        num_state_blocks = int(
            ((seq_lens + _KDA_STATE_BLOCK - 1) // _KDA_STATE_BLOCK).sum().item()
        )
        h = h0.new_empty((num_state_blocks,) + tuple(h0.shape[1:]))
    o = torch.empty(
        q.shape[0], q.shape[1], q.shape[2], v.shape[-1], dtype=v.dtype, device=v.device
    )

    kunlun_ops.kimi_delta_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.to(_state_dtype).contiguous(),
        h0,
        ht,
        o,
        scale,
        cu_cpu,
        cu_xpu,
        False,
        h,
    )

    initial_state.index_copy_(0, slots, ht.to(initial_state.dtype))
    if output_intermediate_states:
        # Upstream's track path expects the [1, NT, H, V, K] batched layout.
        return o, h.unsqueeze(0)
    return o


def _normalize_activation(activation: Union[bool, str, None]) -> Optional[str]:
    """Map the upstream ``activation`` argument onto a kunlun_ops act name."""

    if activation is None or activation is False:
        return None
    if activation is True:
        return "silu"
    if activation in ("silu", "swish"):
        return "silu"
    raise NotImplementedError(f"kunlun causal_conv1d: unsupported act {activation!r}")


def _conv_state_as_nwc(conv_state: torch.Tensor) -> torch.Tensor:
    """Return the state buffer as ``[lines, state_len, dim]`` without a copy.

    Upstream passes ``[lines, dim, state_len]``. When that tensor is a
    transposed view of an NWC buffer (``stride(-2) == 1``) the transpose is
    simply undone; otherwise a contiguous NWC copy is unavoidable, and the
    caller must write it back.
    """

    if conv_state.stride(-2) == 1:
        return conv_state.transpose(-1, -2)
    return conv_state.transpose(-1, -2).contiguous()


def _state_in_dtype(state_nwc: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return the state in ``dtype``.

    kunlun_ops dispatches the conv on one dtype for x / y / weight / state, but
    the KDA pool may hold conv states in fp32 while activations are bf16. The
    state buffer is small (tens of MB for the whole pool), so a cast copy is
    acceptable; the caller writes it back.
    """

    if state_nwc.dtype == dtype:
        return state_nwc
    return state_nwc.to(dtype)


def _apply_activation(y: torch.Tensor, act: Optional[str]) -> torch.Tensor:
    """Apply the activation in torch.

    ``kunlun_ops`` accepts an ``act`` / ``silu_activation`` argument but this
    build ignores it: with ``act="silu"`` the output matches the *un*-activated
    reference to 1.9e-7. Silently dropping the activation would be a
    hard-to-spot accuracy bug, so the activation is always applied here and
    ``act`` is never handed to the kernel.
    """

    if act is None:
        return y
    assert act == "silu"
    return torch.nn.functional.silu(y)


@register_jit_op(_L2NORM_MODULE, "l2norm_fwd")
def l2norm_fwd_kunlun(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Row-wise ``x / sqrt(sum(x*x) + eps)`` over the last dim.

    ``chunk_kda`` calls this on q and k before the chunk pipeline. The
    ``xspeedgate_ops`` op is fp32-only with eps hardwired to 1e-6 and wants a 4D
    input, which is exactly the ``chunk_kda`` call shape - only the dtype needs
    bridging. The torch reference does the reduction in fp32 too, to match the
    Triton kernel, which accumulates in fp32 regardless of input dtype.
    """

    if not _FORCE_TORCH and x.dim() == 4 and abs(eps - 1e-6) < 1e-12:
        xf = x.float().contiguous()
        if x.dtype == torch.float16:
            # fp16 tops out at 65504; a KDA q/k projection outlier can overflow
            # that to +/-Inf, and the fp32 `infer_l2norm_fwd<float>` kernel
            # hardware-traps on non-finite input (KL_XID_KERNEL_EXCEPTION /
            # error 719 -> whole card down). L2-norm only encodes direction, so
            # clamp Inf back to the largest finite fp16 value (that element is
            # already dominant, so the unit-norm result is essentially
            # unchanged) and NaN->0. bf16/fp32 never enter this branch, so their
            # behaviour is untouched. Repro: retest_bad/l2norm_fp16_trap_ut.py.
            xf = torch.nan_to_num(xf, nan=0.0, posinf=65504.0, neginf=-65504.0)
        res = torch.ops.xspeedgate_ops.l2norm_fwd(xf)
        return res.to(output_dtype or x.dtype)

    flat = x.reshape(-1, x.shape[-1]).float()
    if x.dtype == torch.float16:
        flat = torch.nan_to_num(flat, nan=0.0, posinf=65504.0, neginf=-65504.0)
    normed = flat * torch.rsqrt(flat.pow(2).sum(-1, keepdim=True) + eps)
    return normed.to(output_dtype or x.dtype).view(x.shape)


@register_jit_op(_CONV_MODULE, "causal_conv1d_fn")
def causal_conv1d_fn_kunlun(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Varlen depthwise causal conv on ``x[dim, cu_seqlen]`` via kunlun_ops."""

    assert x.dim() == 2, f"kunlun causal_conv1d_fn expects 2D x, got {tuple(x.shape)}"
    assert x.stride(0) == 1, "kunlun causal_conv1d_fn needs channel-last x"
    dim, cu_seqlen = x.shape
    width = weight.shape[1]

    # The op addresses x as packed NWC ([tokens, dim] with row pitch == dim).
    # channel-last alone is not enough: the KDA backend passes
    # `mixed_qkv.transpose(0, 1)`, and mixed_qkv is a torch.split slice of the
    # fused qkvbfg projection, so the row pitch is the fused width (4160), not
    # dim (3072). stride(0) == 1 still holds, so this used to pass the assert and
    # then read 1088 elements of the neighbouring gate/beta block per row --
    # output with the right magnitude but uncorrelated (measured cos 0.013).
    if x.stride(1) != dim:
        x = x.transpose(0, 1).contiguous().transpose(0, 1)

    # Do not use empty_like: it would inherit the caller's (possibly non-packed)
    # strides for the same reason.
    out = torch.empty(cu_seqlen, dim, dtype=x.dtype, device=x.device).transpose(0, 1)
    state_nwc = _conv_state_as_nwc(conv_states)
    state_arg = _state_in_dtype(state_nwc, x.dtype)
    num_cache_lines, state_width, state_dim = state_arg.shape
    assert state_dim == dim, f"conv state dim {state_dim} != x dim {dim}"
    assert state_width >= width - 1

    batch_size = len(seq_lens_cpu)
    qsl_xpu = query_start_loc.to(torch.int32)
    qsl_cpu = qsl_xpu.cpu()
    cache_idx_xpu = None if cache_indices is None else cache_indices.to(torch.int32)
    cache_idx_cpu = None if cache_idx_xpu is None else cache_idx_xpu.cpu()
    init_xpu = None if has_initial_state is None else has_initial_state.to(torch.int32)
    init_cpu = None if init_xpu is None else init_xpu.cpu()

    kunlun_ops.causal_conv1d_fn(
        x,
        out,
        dim,
        cu_seqlen,
        weight.to(x.dtype),
        width,
        state_arg,
        num_cache_lines,
        state_width,
        qsl_cpu,
        qsl_xpu,
        batch_size,
        # The binding types the optional bias as float32 regardless of x dtype.
        bias=None if bias is None else bias.float(),
        cache_indices_cpu=cache_idx_cpu,
        cache_indices_xpu=cache_idx_xpu,
        has_initial_state_cpu=init_cpu,
        has_initial_state_xpu=init_xpu,
        act=None,
        pad_slot_id=pad_slot_id,
        is_ncw=False,
    )
    if state_arg.data_ptr() != conv_states.data_ptr():
        conv_states.copy_(state_arg.transpose(-1, -2))
    return _apply_activation(out, _normalize_activation(activation))


def _tree_verify_parent_map(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Derive ``retrieve_parent_token`` from the child/sibling links.

    Mirrors the fused prologue of ``_causal_conv1d_update_kernel``: walking the
    tokens in order, a token's first child inherits it as parent, and a token's
    first sibling inherits *that token's* parent. Token 0 keeps parent 0 (n/a).

    The loop bound is ``draft_token_num`` (a static python int), and every step
    is a scatter of device tensors -- no ``.item()`` on the link values, which
    would be a device->host sync and is invalid under cuda-graph capture.
    """

    nxt = retrieve_next_token.long()
    sib = retrieve_next_sibling.long()
    parent = torch.zeros_like(nxt[:, :num_tokens])
    for t in range(num_tokens):
        child = nxt[:, t : t + 1]
        c_idx = child.clamp_min(0)
        cur = parent.gather(1, c_idx)
        parent.scatter_(
            1, c_idx, torch.where(child >= 0, torch.full_like(cur, t), cur)
        )
        # The sibling must be resolved after the child, in this same iteration:
        # it copies parent[t], which earlier iterations have already finalised.
        sibling = sib[:, t : t + 1]
        s_idx = sibling.clamp_min(0)
        cur = parent.gather(1, s_idx)
        parent.scatter_(
            1, s_idx, torch.where(sibling >= 0, parent[:, t : t + 1], cur)
        )
    return parent


def _masked_index_copy_(
    dst: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """``dst[indices[i]] = values[i]`` for rows where ``valid[i]``, statically.

    ``index_copy_`` leaves the order undefined when indices repeat, and masked
    rows have to be aliased onto some real row rather than sliced away (a
    boolean-mask slice is a data-dependent shape, which cuda-graph capture
    cannot express). Aliasing both the index *and* the payload onto the first
    valid row makes every duplicate write byte-identical, so the order stops
    mattering. An all-masked batch rewrites each row with its own contents.
    """

    first = torch.argmax(valid.to(torch.uint8)).reshape(1)
    idx = torch.where(
        valid, indices, indices.index_select(0, first).expand_as(indices)
    )
    mask = valid.reshape(-1, *([1] * (values.dim() - 1)))
    val = torch.where(
        mask, values, values.index_select(0, first).expand_as(values)
    )
    val = torch.where(valid.any(), val, dst.index_select(0, idx))
    dst.index_copy_(0, idx, val.to(dst.dtype))


def _causal_conv1d_update_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: Optional[str],
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    retrieve_next_token: Optional[torch.Tensor],
    retrieve_next_sibling: Optional[torch.Tensor],
    retrieve_parent_token: Optional[torch.Tensor],
    pad_slot_id: int,
) -> torch.Tensor:
    """MTP verify depthwise causal conv (chain and draft tree), in torch.

    kunlun_ops.causal_conv1d_update covers the plain decode-step update, not the
    verify variant's per-step intermediate-window checkpointing or the tree
    taps, so this is ported. The one real difference from a chain: each draft
    token convolves against its *tree ancestors* rather than its left
    neighbours, so the taps are gathered along the parent chain -- and a chain
    is just the special case ``parent[i] == i-1``, so both share one path.

    Trick that keeps this simple: prepend the ``width-1`` prior state columns
    to the token axis, giving ``buf = [state[0..K-2], x[0..T-1]]``. In that
    index space "go to parent" is ``K-1 + parent[i]`` for a token ``i > 0`` and
    plain ``-1`` once the walk falls off token 0 into the state columns --
    which is exactly the reference kernel's col2/col1/col0 fallback chain.

    Matches the kernel's other three effects as well: the state writeback is
    the *linear* last ``width-1`` tokens (the tree fixup happens later, from
    the intermediate window), the intermediate window records the ancestor tap
    values (self at slot K-2, parent at K-3, ...), and ``retrieve_parent_token``
    is produced as a side output.
    """

    assert x.dim() == 3, f"verify conv expects [batch, dim, seqlen], got {tuple(x.shape)}"
    assert act == "silu", f"verify conv expects silu, got {act}"
    batch, dim, num_tokens = x.shape
    width = weight.shape[1]
    state_len = width - 1
    assert conv_state.shape[1] == dim
    assert conv_state.shape[2] >= state_len

    if retrieve_next_token is None:
        # Chain verify (eagle topk=1): no tree mask, so each token's parent is
        # simply its left neighbour -- which is exactly what the reference
        # kernel's non-tree branch (the sliding col0/col1/col2 shift) computes,
        # window snapshots included. The reference also leaves
        # retrieve_parent_token untouched in this mode, so neither do we.
        parent = (
            (torch.arange(num_tokens, device=x.device) - 1)
            .clamp_min(0)
            .unsqueeze(0)
            .expand(batch, num_tokens)
        )
    else:
        parent = _tree_verify_parent_map(
            retrieve_next_token, retrieve_next_sibling, num_tokens
        )
        if retrieve_parent_token is not None:
            # The reference kernel bails out before this store for padded rows,
            # leaving them stale; writing the (equally ignored) computed value
            # keeps the code branch-free.
            retrieve_parent_token[:, :num_tokens].copy_(
                parent.to(retrieve_parent_token.dtype)
            )

    coord = conv_state_indices.long()
    valid = coord != pad_slot_id
    gather = coord.clamp(0, conv_state.shape[0] - 1)
    # Read the prior window before the writeback below overwrites it.
    old = conv_state.index_select(0, gather)[:, :, :state_len].float()
    buf = torch.cat([old, x.float()], dim=2)  # [batch, dim, K-1 + T]

    # Tap-chain successor table over buf's token axis.
    steps = torch.arange(num_tokens, device=x.device)
    pmap = (
        (torch.arange(state_len + num_tokens, device=x.device) - 1)
        .clamp_min(0)
        .unsqueeze(0)
        .repeat(batch, 1)
    )
    pmap[:, state_len:] = torch.where(
        steps.unsqueeze(0) > 0,
        parent + state_len,
        torch.full_like(parent, state_len - 1),
    )

    acc = x.new_zeros(batch, dim, num_tokens, dtype=torch.float32)
    if bias is not None:
        acc = acc + bias.float().reshape(1, -1, 1)
    window = None
    if intermediate_conv_window is not None:
        window = x.new_zeros(batch, num_tokens, dim, state_len)

    cur = (steps + state_len).unsqueeze(0).expand(batch, num_tokens)
    for j in range(width):
        tap = torch.gather(buf, 2, cur.unsqueeze(1).expand(batch, dim, num_tokens))
        acc = acc + tap * weight[:, width - 1 - j].float().reshape(1, -1, 1)
        slot = width - 2 - j
        if window is not None and slot >= 0:
            window[:, :, :, slot] = tap.transpose(1, 2).to(window.dtype)
        cur = pmap.gather(1, cur)

    out = _apply_activation(acc, act).to(x.dtype)

    # Writeback: linear shift-left, i.e. the trailing state_len entries of
    # [prior window, x] -- the tree structure deliberately plays no part here.
    _masked_index_copy_(
        conv_state[:, :, :state_len], gather, buf[:, :, num_tokens:], valid
    )
    if window is not None:
        assert intermediate_state_indices is not None
        inter = intermediate_conv_window[:, :num_tokens]
        inter_idx = intermediate_state_indices.long().clamp(0, inter.shape[0] - 1)
        _masked_index_copy_(inter, inter_idx, window, valid)
    return out


@register_jit_op(_CONV_MODULE, "causal_conv1d_update")
def causal_conv1d_update_kunlun(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Decode-step depthwise causal conv update via kunlun_ops."""

    act = _normalize_activation(activation)
    if any(
        arg is not None
        for arg in (
            intermediate_state_indices,
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
        )
    ):
        # MTP verify: per-step intermediate-window checkpointing, and under a
        # draft tree taps that follow the tree -- neither is expressible with
        # the kunlun_ops op. Handled in torch instead.
        assert cache_seqlens is None, "verify + circular conv state cache"
        assert num_accept_tokens is None, (
            "verify with num_accept_tokens changes the effective state_len "
            "(width-1+seqlen-1); only the width-1 form is ported"
        )
        return _causal_conv1d_update_verify(
            x=x,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            act=act,
            conv_state_indices=conv_state_indices,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            pad_slot_id=pad_slot_id,
        )

    if act is None:
        raise NotImplementedError(
            "kunlun causal_conv1d_update only exposes the silu activation switch"
        )

    # kunlun_ops updates x in place; upstream call sites keep using their own
    # input afterwards, so operate on a private buffer.
    if x.dim() == 2:
        # clone() instead of contiguous(): for a [1, dim] tensor (or any tensor
        # whose singleton batch dim makes unsqueeze(1) "contiguous" by stride
        # rules), unsqueeze(1).contiguous() is a no-op that returns the caller's
        # storage, so the kernel writes straight back into mixed_qkv.
        work = x.unsqueeze(1).clone()  # [batch, 1, dim], NWC
        is_ncw = False
    elif x.dim() == 3:
        work = x.contiguous()  # [batch, dim, seqlen], NCW
        is_ncw = True
    else:
        raise AssertionError(f"unexpected x shape {tuple(x.shape)}")

    state_nwc = _conv_state_as_nwc(conv_state)
    state_arg = _state_in_dtype(state_nwc, work.dtype)
    kunlun_ops.causal_conv1d_update(
        work,
        state_arg,
        weight.to(work.dtype),
        bias=None if bias is None else bias.float(),
        silu_activation=False,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        intermediate_conv_window=intermediate_conv_window,
        is_ncw=is_ncw,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accept_tokens,
    )
    if state_arg.data_ptr() != conv_state.data_ptr():
        conv_state.copy_(state_arg.transpose(-1, -2))
    work = _apply_activation(work, act)
    return work.squeeze(1) if x.dim() == 2 else work


_CUMSUM_MODULE = "sglang.kernels.ops.attention.fla.cumsum"
_LN_GATED_MODULE = "sglang.kernels.ops.attention.fla.layernorm_gated"


def _chunk_bounds(
    length: int, chunk_size: int, cu_seqlens: Optional[torch.Tensor], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token start / end index of the chunk it belongs to, along the T axis.

    Chunks restart at every sequence boundary when ``cu_seqlens`` is given, and
    the last chunk of a sequence is truncated at that sequence's end.
    """

    pos = torch.arange(length, device=device)
    if cu_seqlens is None:
        start = torch.div(pos, chunk_size, rounding_mode="floor") * chunk_size
        stop = (start + chunk_size).clamp_max(length)
        return start, stop

    bounds = cu_seqlens.to(torch.int64).to(device)
    seq_id = torch.bucketize(pos, bounds[1:], right=False)
    seq_start = bounds.index_select(0, seq_id)
    seq_stop = bounds.index_select(0, seq_id + 1)
    offset = pos - seq_start
    start = seq_start + torch.div(offset, chunk_size, rounding_mode="floor") * chunk_size
    return start, torch.minimum(start + chunk_size, seq_stop)


def _chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    head_first: bool,
    output_dtype: Optional[torch.dtype],
    t_axis: int,
) -> torch.Tensor:
    x = g.float()
    if scale is not None:
        x = x * scale

    length = x.shape[t_axis]
    start, stop = _chunk_bounds(length, chunk_size, cu_seqlens, x.device)

    total = x.cumsum(dim=t_axis)
    shape = [1] * x.dim()
    shape[t_axis] = length
    if reverse:
        # suffix sum inside the chunk: cum[end - 1] - cum[t] + x[t]
        edge = total.index_select(t_axis, (stop - 1).clamp_min(0))
        out = edge - total + x
    else:
        # prefix sum inside the chunk: cum[t] - (cum[start] - x[start])
        exclusive = total - x
        out = total - exclusive.index_select(t_axis, start)
    return out.to(output_dtype or g.dtype)


def _chunk_local_cumsum_op(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    head_first: bool,
    output_dtype: Optional[torch.dtype],
    chunk_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``xspeedgate_ops.chunk_local_cumsum``, or None if it does not apply.

    Two measured limitations decide the guard, neither of them documented in the
    op stub:

    - 4D (vector) input is accepted but computed wrong (cos 0.0026 vs the torch
      reference), so only the 3D scalar layout may use the op.
    - ``cu_seqlens`` is ignored: the cumsum runs across sequence boundaries
      (cos 0.9967-0.9997, max abs diff 7-17). Passing the upstream-style
      ``chunk_indices`` alongside does not change the result, so there is no way
      to get correct varlen output out of it.

    The op also has no ``scale`` argument, so the scale is applied afterwards;
    cumsum is linear, so pre- and post-scaling agree.
    """

    if _FORCE_TORCH or chunk_size & (chunk_size - 1) != 0:
        return None
    if g.dim() != 3 or cu_seqlens is not None:
        return None

    out = torch.ops.xspeedgate_ops.chunk_local_cumsum(
        g.contiguous(),
        chunk_size,
        reverse,
        cu_seqlens,
        chunk_indices,
        head_first,
    )
    if scale is not None:
        out = out * scale
    return out.to(output_dtype or g.dtype)


@register_jit_op(_CUMSUM_MODULE, "chunk_local_cumsum_scalar")
def chunk_local_cumsum_scalar_kunlun(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """``chunk_local_cumsum_scalar`` (``[B, T, H]`` / ``[B, H, T]``)."""

    out = _chunk_local_cumsum_op(
        g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype, chunk_indices
    )
    if out is not None:
        return out

    return _chunk_local_cumsum(
        g,
        chunk_size,
        reverse,
        scale,
        cu_seqlens,
        head_first,
        output_dtype,
        t_axis=2 if head_first else 1,
    )


@register_jit_op(_CUMSUM_MODULE, "chunk_local_cumsum_vector")
def chunk_local_cumsum_vector_kunlun(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Torch port of ``chunk_local_cumsum_vector`` (``[B, T, H, S]`` / ``[B, H, T, S]``)."""

    out = _chunk_local_cumsum_op(
        g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype, chunk_indices
    )
    if out is not None:
        return out

    return _chunk_local_cumsum(
        g,
        chunk_size,
        reverse,
        scale,
        cu_seqlens,
        head_first,
        output_dtype,
        t_axis=2 if head_first else 1,
    )


def _gate_activation(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation in ("swish", "silu"):
        return z * torch.sigmoid(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise NotImplementedError(f"kunlun layernorm_gated: activation={activation!r}")


@register_jit_op(_LN_GATED_MODULE, "_layer_norm_fwd")
def layer_norm_fwd_kunlun(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """``_layer_norm_fwd`` (grouped, optionally gated, layer/RMS norm).

    ``kunlun_ops.rms_norm_gated`` covers the RMS + ungrouped + no-bias case and,
    unlike ``xspeedgate_ops.rms_norm_gated_fwd``, it accepts bf16 (measured
    cos 0.999997), so the op path applies to the shape this model actually runs.
    It only produces ``out``; the single upstream caller
    (``layernorm_gated.py:333``) discards ``mean`` / ``rstd``, so None is fine.

    In the torch reference, statistics are per ``(row, group)`` in fp32;
    ``mean`` / ``rstd`` keep the upstream ``[ngroups * M]`` flat layout with
    ``group * M + row`` indexing.
    """

    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size

    if out is None:
        out = torch.empty_like(x)

    if (
        not _FORCE_TORCH
        and is_rms_norm
        and bias is None
        and ngroups == 1
        and z is not None
        and activation in ("swish", "silu")
        and x.dtype == z.dtype == weight.dtype
    ):
        kunlun_ops.rms_norm_gated(
            x.contiguous(),
            out,
            z.contiguous(),
            weight.contiguous(),
            eps,
            None,
            norm_before_gate,
            True,
        )
        return out, None, None

    xf = x.float().reshape(M, ngroups, group_size)
    if z is not None and not norm_before_gate:
        xf = xf * _gate_activation(z.float().reshape(M, ngroups, group_size), activation)

    if is_rms_norm:
        mean = None
        centered = xf
    else:
        mean_g = xf.mean(dim=-1, keepdim=True)
        centered = xf - mean_g
        mean = mean_g.squeeze(-1).transpose(0, 1).reshape(-1).contiguous()

    rstd_g = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = centered * rstd_g
    rstd = rstd_g.squeeze(-1).transpose(0, 1).reshape(-1).contiguous()

    y = y.reshape(M, N) * weight.float()
    if bias is not None:
        y = y + bias.float()
    if z is not None and norm_before_gate:
        y = y * _gate_activation(z.float(), activation)

    out.copy_(y.to(out.dtype))
    return out, mean, rstd


@register_jit_op(_KDA_MODULE, "fused_recurrent_kda_fwd")
def fused_recurrent_kda_fwd_kunlun(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch port of the KDA recurrent forward (``g`` is already the log gate).

    ``h *= exp(g)``; ``v -= h @ k``; ``v *= beta``; ``h += outer(v, k)``;
    ``o = h @ q``, per token, per sequence.
    """

    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    n_seq = B if cu_seqlens is None else cu_seqlens.numel() - 1

    q_f = q.float().reshape(-1, H, K)
    k_f = k.float().reshape(-1, H, K)
    v_f = v.float().reshape(-1, HV, V)
    g_f = g.float().reshape(-1, HV, K)
    beta_f = beta.float().reshape(-1, HV)

    if use_qk_l2norm_in_kernel:
        q_f = q_f * torch.rsqrt(q_f.pow(2).sum(-1, keepdim=True) + 1e-6)
        k_f = k_f * torch.rsqrt(k_f.pow(2).sum(-1, keepdim=True) + 1e-6)
    q_f = q_f * scale
    if HV != H:
        repeat = HV // H
        q_f = q_f.repeat_interleave(repeat, dim=1)
        k_f = k_f.repeat_interleave(repeat, dim=1)

    if cu_seqlens is None:
        offsets = [(i * T, (i + 1) * T) for i in range(n_seq)]
    else:
        marks = cu_seqlens.tolist()
        offsets = list(zip(marks[:-1], marks[1:]))

    o = q.new_empty(B, T, HV, V)
    o_f = o.reshape(-1, HV, V)
    finals = []
    for i, (start, stop) in enumerate(offsets):
        h = initial_state[i].float()
        for t in range(start, stop):
            h = h * g_f[t].unsqueeze(-2).exp()
            kt = k_f[t]
            vt = (v_f[t] - (h * kt.unsqueeze(-2)).sum(-1)) * beta_f[t].unsqueeze(-1)
            h = h + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            o_f[t] = (h * q_f[t].unsqueeze(-2)).sum(-1).to(o.dtype)
        finals.append(h)

    final_state = torch.stack(finals).to(initial_state.dtype)
    if inplace_final_state:
        initial_state[:n_seq].copy_(final_state)
        final_state = initial_state
    return o, final_state


@register_jit_op(_RECURRENT_MODULE_PACKED, "fused_recurrent_kda_packed_decode")
def fused_recurrent_kda_packed_decode_kunlun(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch port of the packed T=1 KDA decode path.

    Splits ``mixed_qkv`` into q/k/v and reuses the gated delta-rule decode step,
    which already implements the identical recurrence and state write-back.
    """

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    H = (mixed_qkv.shape[1] - HV * V) // (2 * K)

    q = mixed_qkv[:, : H * K].reshape(B, 1, H, K)
    k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, 1, H, K)
    v = mixed_qkv[:, 2 * H * K :].reshape(B, 1, HV, V)

    o = fused_sigmoid_gating_delta_rule_update_kunlun(
        A_log=A_log,
        a=a.reshape(B, 1, HV * K),
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b.reshape(B, 1, HV),
        initial_state_source=initial_state,
        initial_state_indices=ssm_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=True,
        lower_bound=lower_bound,
    )
    out.copy_(o.reshape(out.shape).to(out.dtype))
    return out, initial_state
