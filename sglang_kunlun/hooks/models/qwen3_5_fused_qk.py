"""Kunlun interface replacement for Qwen3.5 full-attn QK preprocessing.

Replaces upstream ``fused_qk_gemma_rmsnorm_rope_gate`` (GemmaRMSNorm(1+w) +
partial NeoX RoPE + gate de-interleave) with
``torch.ops.xspeedgate_ops.split_norm_rope_neox``.

Adapters verified by module-level replay against the N-card golden
(records/Qwen3.8-27B/p800-migrator/003):

- dtype: the xsg op rejects bf16 ("not support dtype combination") -> cast
  qkv / weights / cos_sin_cache to fp16, cast outputs back to the model dtype.
  (bf16 support would remove this cast; logged in operator-gap-requests.md.)
- (1+w): the op applies Gemma's +1 internally -> pass the RAW q/k RMSNorm
  weights (passing 1+weight fails, verified).
- mrope: TEXT-ONLY. Qwen3.5 mrope positions are [3, T]; for pure text the three
  axes are identical, so we feed axis-0 (== sequence position) as 1D positions.
  **Enabling the vision tower needs a real interleaved-mrope variant** — this
  replacement is only valid on the text path.
- the op consumes the full fused qkv and also returns v_out; upstream calls the
  QK op with (q_gate, k) only, so we reconstruct qkv=[q_gate, k, dummy_v] and
  discard v_out (v_out does not affect q/k/gate).
"""
from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def fused_qk_gemma_rmsnorm_rope_gate_kunlun(
    q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps,
    num_q_heads, num_kv_heads, head_dim, rotary_dim,
    has_gate: bool = True, mrope_axis_map=None,
):
    """Full-attn QK preprocessing -> torch.ops.xspeedgate_ops.split_norm_rope_neox.

    Replaces fused_qk_gemma_rmsnorm_rope_gate: casts to fp16 (the op rejects bf16),
    passes the RAW Gemma q/k weights (the op adds +1 internally), collapses the
    3-axis mrope to 1D text positions, and discards the dummy V output.
    Returns (q, k, gate).
    """
    assert has_gate, "kunlun split_norm_rope_neox requires attn_output_gate=True"
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    T = q_gate.shape[0]
    dt = q_gate.dtype

    # TEXT-ONLY mrope collapse: [3, T] identical axes -> 1D sequence positions.
    if positions.dim() == 2:
        pos1d = positions[0].contiguous()
    else:
        pos1d = positions.contiguous()

    # Full fused qkv with a dummy V (v_out discarded; q/k/gate independent of V).
    v_dummy = q_gate.new_zeros(T, kv_size)
    qkv = torch.cat([q_gate, k, v_dummy], dim=-1).to(torch.float16).contiguous()

    q_out, k_out, _v_out, gate_out = torch.ops.xspeedgate_ops.split_norm_rope_neox(
        qkv,
        q_weight.to(torch.float16).contiguous(),
        k_weight.to(torch.float16).contiguous(),
        pos1d.to(torch.int64),
        cos_sin_cache.to(torch.float16).contiguous(),
        q_size, kv_size, num_q_heads, num_kv_heads, head_dim, rotary_dim,
        True,
    )
    q_out = q_out.to(dt)
    k_out = k_out.to(dt)
    gate_out = gate_out.to(dt).view(T, num_q_heads, head_dim)
    return q_out, k_out, gate_out


plugin_hook(
    "sglang.srt.models.qwen3_5.fused_qk_gemma_rmsnorm_rope_gate",
    type=HookType.REPLACE,
)(fused_qk_gemma_rmsnorm_rope_gate_kunlun)
