"""Kunlun GDN / GatedDeltaNet interface replacements for Qwen3.5 (qwen3_5).

Each wrapper mirrors the upstream free-function contract but routes compute to
kunlun_ops, encoding the adapters verified by module-level replay against the
N-card golden (records/Qwen3.8-27B/p800-migrator):

- packed_decode: kunlun state layout is [HV, V, K] -> pass is_h0_transposed=True
  (this was the root cause of round-1 "精度不过"; with the flag it matches golden).
- fused_gdn_gating: kunlun op only produces g (= A_log.exp()*softplus(a+dt_bias));
  beta = sigmoid(b) is supplied by torch (L4 local fallback).
- chunk_gated_delta_rule: kunlun takes a dense initial_state (no initial_state_indices)
  -> gather ssm_states[cache_indices] before the call (prefill path).
- fused_sigmoid_mul / the qkvzba+qkv splits: pure elementwise / data-movement done
  with torch (non-specialised).

Registered as HookType.REPLACE against the CONSUMER-module bindings (the names the
production call sites actually resolve), not the definition modules.
"""
from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_HV_TO_H_DEFAULT = 3  # num_v_heads / num_k_heads for qwen3_5 GDN (48/16)


# --------------------------------------------------------------------------- #
# GDN gating: kunlun g + torch beta  (TORCH_FALLBACK[beta]: L4)
# --------------------------------------------------------------------------- #
def fused_gdn_gating_kunlun(A_log, a, b, dt_bias, beta_val: float = 1.0,
                            threshold: float = 20.0):
    """GDN gating -> (g, beta).

    kunlun_ops.fused_gdn_gating only produces g (= A_log.exp()*softplus(a+dt_bias));
    beta = sigmoid(b) is computed here in torch. Returns the pair (g, beta).
    """
    import kunlun_ops
    g = kunlun_ops.fused_gdn_gating(A_log, a, dt_bias, beta_val, threshold, True)
    g = g.view(1, *a.shape[-2:]) if g.dim() == 2 else g.unsqueeze(0)
    # TORCH_FALLBACK[fused_gdn_gating.beta]: kunlun fused_gdn_gating emits only g
    # (its signature has no `b`); beta = sigmoid(b) computed by torch. (L4)
    beta = torch.sigmoid(b.float()).view(1, *b.shape[-2:])
    return g, beta


# --------------------------------------------------------------------------- #
# GDN decode (packed):  is_h0_transposed=True  ([HV,V,K] state layout)
# --------------------------------------------------------------------------- #
def packed_decode_kunlun(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state,
                         out=None, ssm_state_indices=None,
                         use_qk_l2norm_in_kernel: bool = False, **_ignored):
    """GDN decode (packed recurrent) -> kunlun_ops.fused_recurrent_gated_delta_rule_packed_decode.

    Signature matches the fla call convention (kwargs + ``out`` buffer + dense
    ``initial_state`` pool + ``ssm_state_indices``). State layout is [num_states, HV, V, K]
    -> is_h0_transposed=True. If ``out`` is given, the result is written into it in place.
    """
    # Signature matches the fla fused_recurrent_gated_delta_rule_packed_decode call
    # convention used at gdn_triton.py (kwargs + `out` buffer + `initial_state` pool +
    # `ssm_state_indices`). initial_state is 4D [num_states, HV, V, K]; out is
    # [B, 1, HV, V] and is written in place (caller reads `out`, ignores return).
    import kunlun_ops
    HV, V, K = initial_state.shape[-3], initial_state.shape[-2], initial_state.shape[-1]
    B = mixed_qkv.shape[0]
    qkv_dim = mixed_qkv.shape[-1]
    H = (qkv_dim - HV * V) // (2 * K)          # qkv_dim = 2*H*K + HV*V
    mq3 = mixed_qkv.reshape(B, 1, qkv_dim).contiguous()
    a2 = a.reshape(B, HV).contiguous()
    b2 = b.reshape(B, HV).contiguous()
    idx = ssm_state_indices.reshape(-1).to(torch.int32)
    o = kunlun_ops.fused_recurrent_gated_delta_rule_packed_decode(
        A_log, a2, dt_bias, mq3, b2, initial_state, idx, H,
        1.0, 20.0, scale, True, use_qk_l2norm_in_kernel,
        True,   # is_h0_transposed: fla state layout is [num_states, HV, V, K] (round-2)
        None,
    )
    if out is not None:
        out.copy_(o.reshape(out.shape))   # fla contract: write result into out in place
        return out
    return o


# --------------------------------------------------------------------------- #
# GDN prefill (chunk): dense initial_state gather (no initial_state_indices)
# --------------------------------------------------------------------------- #
def chunk_gated_delta_rule_kunlun(q, k, v, g, beta, scale=None,
                                  initial_state=None, initial_state_indices=None,
                                  cu_seqlens=None, head_first=False,
                                  use_qk_l2norm_in_kernel=False):
    """GDN prefill (chunk) -> kunlun_ops.chunk_gated_delta_rule.

    The kunlun op takes a dense initial_state (no index addressing): gather the
    per-sequence slots (transposed to [N, HV, K, V]) and write the updated state back
    into the pool. cu_seqlens is moved to CPU int32 (the native kernel reads it
    host-side). Returns (o, None, None).
    """
    import kunlun_ops
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # kunlun chunk takes a DENSE initial_state (batch,H,K,V), no index addressing:
    # gather the per-sequence slots first when an index tensor is supplied.
    # PORTED from qwen3.5 working fork: the pool is [slots, HV, V, K] but the op takes
    # [N, HV, K, V] -> transpose(-1,-2) on gather. (K==V==128 here so the earlier no-transpose
    # form gave the SAME shape but V/K-swapped DATA -> wrong state.)
    if initial_state_indices is not None and initial_state is not None:
        dense_state = initial_state[initial_state_indices].transpose(-1, -2).contiguous()
    else:
        dense_state = initial_state
    o = torch.empty_like(v)
    ht = torch.empty_like(dense_state)
    # kunlun_ops.chunk_gated_delta_rule reads cu_seqlens HOST-side for chunk indexing:
    # it must be a CPU int32 tensor. Real engine dispatch passes query_start_loc on the
    # XPU device -> the native kernel dereferences a host pointer into device memory and
    # SIGSEGVs (eager-bringup/001). Move it to CPU int32 here. (verified round-5)
    cu = cu_seqlens.to(torch.int32).cpu().contiguous() if cu_seqlens is not None else None
    # Native kernel does raw pointer arithmetic -> force contiguous compute tensors.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if beta.dtype != torch.float32:
        beta = beta.float()
    if g.dtype != torch.float32:
        g = g.float()
    g = g.contiguous()
    beta = beta.contiguous()
    kunlun_ops.chunk_gated_delta_rule(
        q, k, v, g, beta, scale, dense_state, o, ht, cu,
        head_first, use_qk_l2norm_in_kernel,
    )
    # PORTED from qwen3.5: fancy-index gather is a COPY, so transpose the updated state
    # back to [slots, HV, V, K] and WRITE it into the pool (the earlier hook returned ht
    # but never wrote it back -> decode read a stale/zero ssm_state).
    if initial_state_indices is not None and initial_state is not None:
        initial_state[initial_state_indices] = ht.transpose(-1, -2).to(initial_state.dtype)
    return o, None, None


# --------------------------------------------------------------------------- #
# attn output gate:  x * sigmoid(gate)   (torch, non-specialised)
# --------------------------------------------------------------------------- #
def fused_sigmoid_mul_kunlun(x, gate, inplace: bool = True):
    """Attn output gate x * sigmoid(gate), computed in torch (non-specialised)."""
    out = x.mul_(torch.sigmoid(gate.float()).to(x.dtype)) if inplace \
        else x * torch.sigmoid(gate.float()).to(x.dtype)
    return out


# --------------------------------------------------------------------------- #
# Registrations — target the CONSUMER-module bindings the call sites resolve.
# --------------------------------------------------------------------------- #
plugin_hook(
    "sglang.srt.layers.attention.linear.gdn_backend.fused_gdn_gating",
    type=HookType.REPLACE,
)(fused_gdn_gating_kunlun)

plugin_hook(
    "sglang.srt.layers.attention.linear.kernels.gdn_triton.chunk_gated_delta_rule",
    type=HookType.REPLACE,
)(chunk_gated_delta_rule_kunlun)

plugin_hook(
    "sglang.srt.layers.attention.linear.kernels.gdn_triton."
    "fused_recurrent_gated_delta_rule_packed_decode",
    type=HookType.REPLACE,
)(packed_decode_kunlun)

plugin_hook(
    "sglang.srt.models.qwen3_5.fused_sigmoid_mul",
    type=HookType.REPLACE,
)(fused_sigmoid_mul_kunlun)
