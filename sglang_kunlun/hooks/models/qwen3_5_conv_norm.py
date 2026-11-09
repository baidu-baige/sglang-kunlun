"""Kunlun interface replacements for the remaining Qwen3.5 GDN path ops:
causal_conv1d (prefill fn / decode update), RMSNormGated, and the two GDN
split/reshape data-movement helpers.

Adapters verified by module-level replay against the N-card golden
(records/Qwen3.8-27B/p800-migrator/004 and /001):

- causal_conv1d_fn (prefill): kunlun_ops.causal_conv1d_fn uses a low-level
  signature (dim / cu_seqlen / num_cache_lines / state_width / dual cpu+xpu
  query_start_loc & cache_indices / has_initial_state as int32 / is_ncw). It does
  NOT apply the activation (verified: all act args give the identical pre-SiLU
  conv output; SiLU(y) then matches golden) -> we apply torch SiLU (L4 activation
  fallback). conv_states is [ncl, dim, width-1] NCW; layout is_ncw=True.
- causal_conv1d_update (decode): kunlun_ops.causal_conv1d_update (has its own
  silu_activation flag). NO golden capture point exists for this call (capture.md
  lists it but the .pt bundle lacks its input tensors) -> hook delivered, accuracy
  待确认 until golden is (re)captured.
- RMSNormGated: kunlun_ops.rms_norm_gated (norm_before_gate=True -> rms_norm(x)*silu(z)).
- the two split helpers: pure torch split/reshape/cat (verified bit-exact in 001).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


# --------------------------------------------------------------------------- #
# causal_conv1d prefill: kunlun conv + torch SiLU (TORCH_FALLBACK[activation] L4)
# --------------------------------------------------------------------------- #
def causal_conv1d_fn_kunlun(
    x, weight, bias=None, query_start_loc=None, cache_indices=None,
    has_initial_state=None, conv_states=None, activation="silu",
    pad_slot_id=-1, **kwargs,
):
    """GDN prefill causal conv1d -> kunlun_ops.causal_conv1d_fn.

    is_ncw=True (NCW varlen); the kunlun op does not apply the activation, so
    SiLU/Swish is applied here in torch. Returns the in-place updated x.
    """
    import kunlun_ops
    # x: (dim, cu_seqlen) varlen NCW; weight: (dim, width); conv_states: (ncl, dim, width-1)
    dim, cu_seqlen = x.shape[0], x.shape[1]
    width = weight.shape[-1]
    ncl = conv_states.shape[0]
    state_width = conv_states.shape[-1]
    batch_size = query_start_loc.shape[0] - 1

    qsl = query_start_loc.to(torch.int32)
    qsl_cpu = qsl.detach().cpu().contiguous()
    qsl_xpu = qsl.contiguous()

    if cache_indices is None:
        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=x.device)
    ci = cache_indices.to(torch.int32)
    ci_cpu = ci.detach().cpu().contiguous()
    ci_xpu = ci.contiguous()

    if has_initial_state is None:
        his = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
    else:
        his = has_initial_state.to(torch.int32)  # kunlun wants int32, not bool
    his_cpu = his.detach().cpu().contiguous()
    his_xpu = his.contiguous()

    x = x.contiguous()
    y = torch.empty_like(x)
    kunlun_ops.causal_conv1d_fn(
        x, y, dim, cu_seqlen, weight.contiguous(), width, conv_states, ncl,
        state_width, qsl_cpu, qsl_xpu, batch_size,
        bias, None, None, None, None, ci_cpu, ci_xpu, his_cpu, his_xpu,
        activation, pad_slot_id, -1, True, 1,
    )
    # TORCH_FALLBACK[causal_conv1d_fn.activation]: kunlun conv emits pre-activation
    # output; apply SiLU/Swish in torch to match CUDA semantics. (L4)
    if activation in ("silu", "swish"):
        y = F.silu(y.float()).to(x.dtype)
    x.copy_(y)  # upstream returns the (in-place) x
    return x


# --------------------------------------------------------------------------- #
# causal_conv1d decode update: kunlun_ops.causal_conv1d_update
# Contract (verified round-7, no golden -> structural): the kunlun UPDATE op only
# supports is_ncw=False, i.e. conv_state NWC [N, state_len, dim] (is_ncw=True /
# NCW fails with ret=1). But the engine mamba conv pool is NCW [N, dim, state_len]
# (same buffer the prefill causal_conv1d_fn consumes with is_ncw=True). So: gather
# the used slots, transpose NCW->NWC, run the update on the copy with identity
# indices, then transpose the updated window back and scatter into the NCW pool
# (the op mutates conv_state in-place; the pool must reflect the new window for the
# next decode step). Numerical correctness deferred to end-to-end (no golden).
# --------------------------------------------------------------------------- #
def causal_conv1d_update_kunlun(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None,
    conv_state_indices=None, pad_slot_id=-1, **kwargs,
):
    """GDN decode causal conv1d update -> kunlun_ops.causal_conv1d_update.

    The kunlun update op only supports is_ncw=False (NWC [N, state_len, dim]) while
    the engine conv-state pool is NCW: gather the used slots, transpose NCW->NWC, run
    the update on the copy with identity indices, then transpose the updated window
    back and scatter it into the NCW pool. Returns (batch, dim) or (batch, dim, seqlen).
    """
    import kunlun_ops
    # PORTED from qwen3.5 working fork (gdn_kunlun.py causal_conv1d_update_kunlun):
    # the kunlun op is is_ncw=False, so x MUST be NWC (batch, seqlen, dim) and conv_state
    # NWC (lines, state_len, dim). The engine hands x as (batch, dim) [seqlen=1] and the
    # pool as NCW (N, dim, state_len); the earlier hook passed x UN-transposed -> wrong
    # input layout for an is_ncw=False kernel (root of the decode conv hang). Also use a
    # pad-aware `local` index (torch.where) and bias.float(), matching the proven fork.
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    silu = activation in ("silu", "swish")
    num_accept_tokens = kwargs.get("num_accept_tokens", None)
    unsqueeze = x.dim() == 2
    xin = x.unsqueeze(-1) if unsqueeze else x            # (batch, dim, seqlen)
    batch = xin.shape[0]
    if conv_state_indices is None:
        idx = torch.arange(batch, device=conv_state.device, dtype=torch.long)
        local = torch.arange(batch, device=conv_state.device, dtype=torch.int32)
    else:
        idx = conv_state_indices.to(torch.long)
        local = torch.where(
            idx != pad_slot_id,
            torch.arange(batch, device=conv_state.device, dtype=torch.int32),
            pad_slot_id,
        )
    win = conv_state[idx].transpose(-1, -2).contiguous()  # (batch, state_len, dim) NWC
    out_nwc = xin.transpose(1, 2).contiguous()            # (batch, seqlen, dim) NWC
    kunlun_ops.causal_conv1d_update(
        out_nwc, win, weight.contiguous(),
        bias.float() if bias is not None else None,
        silu, None, local, None, False, pad_slot_id,
        num_accept_tokens.to(torch.int32) if num_accept_tokens is not None else None,
    )
    conv_state[idx] = win.transpose(-1, -2)               # scatter updated window back to NCW pool
    out = out_nwc.transpose(1, 2)                          # back to (batch, dim, seqlen)
    return out.squeeze(-1) if unsqueeze else out


# --------------------------------------------------------------------------- #
# RMSNormGated forward: kunlun_ops.rms_norm_gated  (norm(x)*silu(z))
# --------------------------------------------------------------------------- #
def rmsnorm_gated_forward_kunlun(self, x, z=None):
    """RMSNormGated forward -> kunlun_ops.rms_norm_gated (norm(x) * silu(z)).

    Replaces RMSNorm.forward for the GDN output norm; flattens to 2D, calls the op
    with the module's weight/eps/group_size/norm_before_gate, then restores the shape.
    """
    import kunlun_ops
    orig_shape = x.shape
    xf = x.reshape(-1, orig_shape[-1]).contiguous()
    out = torch.empty_like(xf)
    zf = z.reshape(-1, z.shape[-1]).contiguous() if z is not None else None
    gs = self.group_size if self.group_size is not None else None
    kunlun_ops.rms_norm_gated(
        xf, out, zf, self.weight, self.eps, gs, self.norm_before_gate, True, False,
    )
    return out.reshape(orig_shape)


# --------------------------------------------------------------------------- #
# GDN split/reshape helpers (pure torch data-movement; verified bit-exact 001)
# --------------------------------------------------------------------------- #
def fused_qkvzba_split_reshape_cat_contiguous_kunlun(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v,
):
    """Split mixed_qkvz / mixed_ba into (mixed_qkv, z, b, a) for the GDN in_proj.

    Pure torch split/reshape/cat (bit-exact vs golden); returned tensors are contiguous.
    """
    kd = num_heads_qk * head_qk
    vd = num_heads_v * head_v
    q, k, v, z = mixed_qkvz.split([kd, kd, vd, vd], dim=-1)
    b, a = mixed_ba.split([num_heads_v, num_heads_v], dim=-1)
    mixed_qkv = torch.cat([q, k, v], dim=-1).contiguous()
    z = z.reshape(z.shape[0], num_heads_v, head_v).contiguous()
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def fused_qkv_split_gdn_prefill_kunlun(
    mixed_qkv, num_q_heads, num_k_heads, num_v_heads, head_q, head_k, head_v,
):
    """Split the GDN prefill mixed_qkv into contiguous q/k/v (pure torch reshape)."""
    qd = num_q_heads * head_q
    kd = num_k_heads * head_k
    vd = num_v_heads * head_v
    T = mixed_qkv.shape[0]
    q, k, v = mixed_qkv.split([qd, kd, vd], dim=-1)
    q = q.reshape(1, T, num_q_heads, head_q).contiguous()
    k = k.reshape(1, T, num_k_heads, head_k).contiguous()
    v = v.reshape(1, T, num_v_heads, head_v).contiguous()
    return q, k, v


# --------------------------------------------------------------------------- #
# Registrations — consumer-module bindings the call sites resolve.
# --------------------------------------------------------------------------- #
plugin_hook("sglang.srt.layers.attention.linear.gdn_backend.causal_conv1d_fn",
            type=HookType.REPLACE)(causal_conv1d_fn_kunlun)
plugin_hook("sglang.srt.layers.attention.linear.gdn_backend.causal_conv1d_update",
            type=HookType.REPLACE)(causal_conv1d_update_kunlun)
plugin_hook("sglang.kernels.ops.attention.fla.layernorm_gated.RMSNorm.forward",
            type=HookType.REPLACE)(rmsnorm_gated_forward_kunlun)
plugin_hook("sglang.srt.models.qwen3_5.fused_qkvzba_split_reshape_cat_contiguous",
            type=HookType.REPLACE)(fused_qkvzba_split_reshape_cat_contiguous_kunlun)
plugin_hook("sglang.srt.layers.attention.linear.gdn_backend.fused_qkv_split_gdn_prefill",
            type=HookType.REPLACE)(fused_qkv_split_gdn_prefill_kunlun)
