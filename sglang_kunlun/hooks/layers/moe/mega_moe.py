"""MegaMoE (hydrax) fused MoE for the Kunlun decode path.

Replaces the DeepEP low-latency `dispatch -> grouped GEMM -> SwiGLU -> down GEMM
-> combine` chain with one `int8_int4_mega_moe` call. On by default;
`SGLANG_KUNLUN_MEGA_MOE=0` restores the DeepEP path.

The weight layout, nibble order and scale convention were established against
the real checkpoint (see `文档/性能对标/megamoe接入摸底-1106.md`):

* `l1_weights = (w13, sf13)` with `w13 [E, 2*ffn, hidden // 2] int8` -- two 4-bit
  values per byte along K, logical element `2i` in the low nibble -- and
  `sf13 [E, 2*ffn] float32`.
* `l2_weights = (w2, sf2)` with `w2 [E, hidden, ffn // 2] int8` and
  `sf2 [E, hidden] float32`.
* gate/up rows must be interleaved at `MEGA_MOE_L1_INTERLEAVE_GRANULARITY` (256).
* scales are raw absmax, i.e. the checkpoint scale times
  `INT4_SIGNED_SCALE_MULT` (7) -- the same convention the Kunlun `moe_fc_v3`
  path already uses. Measured cosine 0.999858 / ratio 0.997 against a torch
  reference built from the production dequant.

Because the MegaMoE layout is not the layout `moe_fc_v3` wants (that one is
K-major and not interleaved) and the expert weights are far too large to hold
twice, a process keeps exactly one layout. So this is only enabled on a
decode-mode server, where every MoE call fits inside
`num_max_tokens_per_rank`; a prefill-sized call raises instead of silently
computing with the wrong layout.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# hydrax.xaccelerator.mega_moe.MEGA_MOE_L1_INTERLEAVE_GRANULARITY
INTERLEAVE_GRANULARITY = 256
# sglang_kunlun.hooks.layers.quantization.compressed_tensors.INT4_SIGNED_SCALE_MULT
INT4_SCALE_MULT = 7.0

# One buffer per distinct MoE shape: DSPARK runs a target model (384 experts,
# topk 6) and a draft model (128 experts, topk 3), so a single buffer cannot
# serve both. Each entry keeps its own fixed address for graph capture.
_SYMM_BUFFERS = {}


def mega_moe_enabled() -> bool:
    """Whether to route MoE through MegaMoE. On by default.

    Set `SGLANG_KUNLUN_MEGA_MOE=0` to fall back to the DeepEP path, which also
    restores the `moe_fc_v3` weight layout -- the two layouts differ by the
    gate/up 256-row interleave and a process keeps only one, so this is a
    start-up decision, not a per-forward one.
    """
    return os.environ.get("SGLANG_KUNLUN_MEGA_MOE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def min_experts_for_mega() -> int:
    """Expert count below which a layer stays on DeepEP.

    DSPARK runs two MoE models: the target (384 experts, top-6) and the draft
    (128 experts, top-3). Each distinct shape needs its own symmetric buffer,
    and a second one does not fit -- `get_symm_buffer` for the draft fails with
    `Cannot allocate XLink head/tail and 128 KiB barrier area: L3 flag heap
    full` after the target's buffer is up. The target's VERIFY stage is where
    the MoE cost actually sits, so MegaMoE takes the target and the draft keeps
    the DeepEP path (which also keeps the draft weights in the layout
    `moe_fc_v3` wants).
    """
    try:
        return max(1, int(os.environ.get("SGLANG_KUNLUN_MEGA_MOE_MIN_EXPERTS", 256)))
    except ValueError:
        return 256


def max_tokens_per_rank() -> int:
    """Token capacity of the symmetric buffer, rounded up by the op itself.

    Decode on this model needs `cuda_graph_max_bs_decode * draft_token_num`
    rows at most (40 * 6 = 240), so the 384 default covers it with headroom.
    """
    try:
        return max(1, int(os.environ.get("SGLANG_KUNLUN_MEGA_MOE_MAX_TOKENS", 384)))
    except ValueError:
        return 384


def interleave_rows(a: torch.Tensor, b: torch.Tensor, gran: int = INTERLEAVE_GRANULARITY):
    """`[N, ...] x2 -> [2N, ...]`: `a[0:gran], b[0:gran], a[gran:2gran], ...`"""
    n = a.shape[0]
    if n % gran:
        raise ValueError(f"MegaMoE interleave needs rows divisible by {gran}, got {n}")
    ab = a.reshape(n // gran, gran, *a.shape[1:])
    bb = b.reshape(n // gran, gran, *b.shape[1:])
    return torch.stack([ab, bb], 1).reshape(2 * n, *a.shape[1:]).contiguous()


def pack_to_mega_int4(packed_int32: torch.Tensor) -> torch.Tensor:
    """`[..., K // 8] int32` (unsigned nibbles) -> `[..., K // 2] int8` (signed).

    `bitwise_xor(0x88)` rebiases both nibbles of every byte from the
    checkpoint's unsigned `value + 8` to signed two's complement, exactly as
    `process_weights_after_loading` does for `moe_fc_v3`. No transpose: MegaMoE
    wants K on the last axis, which is how the checkpoint already stores it.
    """
    return packed_int32.view(torch.int8).bitwise_xor(0x88)


def _interleave_gate_up(t: torch.Tensor, gran: int = INTERLEAVE_GRANULARITY):
    """`[E, 2F, ...]` with gate in the first F rows -> gate/up interleaved by `gran`."""
    two_f = t.shape[1]
    f = two_f // 2
    if f % gran:
        raise ValueError(f"MegaMoE interleave needs ffn divisible by {gran}, got {f}")
    tail = t.shape[2:]
    g = t[:, :f].reshape(t.shape[0], f // gran, gran, *tail)
    u = t[:, f:].reshape(t.shape[0], f // gran, gran, *tail)
    return torch.stack([g, u], 2).reshape(t.shape[0], two_f, *tail).contiguous()


def convert_layer_weights(layer) -> bool:
    """Rewrite one MoE layer's expert weights into the MegaMoE layout, in place.

    The layer stores the checkpoint transposed (`is_transposed: True`):
    `w13_weight_packed [E, hidden // 8, 2*ffn] int32` and
    `w2_weight_packed [E, ffn // 8, hidden] int32`. `transpose(1, 2).view(int8)`
    turns those into `[E, 2*ffn, hidden // 2]` / `[E, hidden, ffn // 2]`, which
    is exactly what MegaMoE wants -- the same step `moe_fc_v3` already does.
    The only extra work is interleaving gate/up rows by 256; the `x7` scale
    convention is shared with `moe_fc_v3`.

    Returns:
        True when the layer was converted, False when it stays on DeepEP.
    """
    num_experts = int(getattr(layer, "num_experts", 0) or 0)
    if num_experts < min_experts_for_mega():
        logger.info(
            "MegaMoE skips a %d-expert layer (threshold %d); it keeps the DeepEP "
            "path and the moe_fc_v3 weight layout",
            num_experts, min_experts_for_mega(),
        )
        return False

    for name, scale_name in (
        ("w13_weight_packed", "w13_weight_scale"),
        ("w2_weight_packed", "w2_weight_scale"),
    ):
        weight = getattr(layer, name)
        packed = weight.data.transpose(1, 2).contiguous().view(torch.int8)
        packed.bitwise_xor_(0x88)

        scale = getattr(layer, scale_name)
        sf = scale.data.transpose(1, 2).contiguous().to(torch.float32)
        sf = sf.mul_(INT4_SCALE_MULT)
        if sf.dim() == 3:
            if sf.shape[2] != 1:
                raise ValueError(
                    "MegaMoE needs per-channel scales; got group scales with "
                    f"{sf.shape[2]} groups for {scale_name}"
                )
            sf = sf.squeeze(2)

        if name == "w13_weight_packed":
            packed = _interleave_gate_up(packed)
            sf = _interleave_gate_up(sf)

        weight.data = packed
        scale.data = sf.contiguous()

    layer._mega_moe_layout = True
    return True


def symm_buffer(hidden: int, intermediate: int, num_experts: int, topk: int):
    """Symmetric workspace for one MoE shape, built once and cached.

    A captured decode graph records reads of these buffers, so an entry must
    outlive capture and never be reallocated. Keyed by shape because DSPARK's
    draft model has its own expert count and top-k.
    """
    key = (hidden, intermediate, num_experts, topk, max_tokens_per_rank())
    hit = _SYMM_BUFFERS.get(key)
    if hit is not None:
        return hit

    from hydrax.xaccelerator.mega_moe_fusion import get_symm_buffer

    from sglang.srt.distributed import get_tp_group

    buf = get_symm_buffer(
        get_tp_group().device_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=max_tokens_per_rank(),
        num_topk=topk,
        hidden=hidden,
        intermediate_hidden=intermediate,
        num_shared_experts=0,
        use_fp8_dispatch=False,
    )
    _SYMM_BUFFERS[key] = buf
    logger.info(
        "MegaMoE symmetric buffer #%d: experts=%d topk=%d hidden=%d ffn=%d max_tokens=%d",
        len(_SYMM_BUFFERS), num_experts, topk, hidden, intermediate,
        max_tokens_per_rank(),
    )
    return buf


def forward(layer, hidden_states: torch.Tensor, topk_output) -> torch.Tensor:
    """One fused MegaMoE call in place of dispatch / GEMMs / combine.

    Args:
        layer: the `DeepEPMoE` whose expert weights were converted by
            `convert_layer_weights`.
        hidden_states: `[num_tokens, hidden]` in the serving dtype.
        topk_output: carries `topk_ids` / `topk_weights` for these rows.

    Returns:
        `[num_tokens, hidden]` in `hidden_states.dtype`.
    """
    from hydrax.xaccelerator.mega_moe_fusion import int8_int4_mega_moe

    if not getattr(layer, "_mega_moe_layout", False):
        raise RuntimeError(
            "MegaMoE forward on a layer whose weights were not converted; "
            "SGLANG_KUNLUN_MEGA_MOE must be set before weight loading"
        )

    num_tokens, hidden = hidden_states.shape
    cap = max_tokens_per_rank()
    if num_tokens > cap:
        raise RuntimeError(
            f"MegaMoE got {num_tokens} rows but the symmetric buffer holds {cap}. "
            "This build keeps only the MegaMoE weight layout, so a prefill-sized "
            "MoE call cannot fall back; enable it on a decode-mode server only, "
            "or raise SGLANG_KUNLUN_MEGA_MOE_MAX_TOKENS."
        )

    # Per-token symmetric int8 with a raw-absmax scale, which is the convention
    # the op documents for `x_sf` (it applies the 1/127 itself).
    flat = hidden_states.reshape(num_tokens, hidden)
    absmax = flat.abs().amax(-1).to(torch.float32).clamp_min(1e-6)
    x_q = (
        (flat.to(torch.float32) / absmax[:, None] * 127.0)
        .round()
        .clamp_(-127, 127)
        .to(torch.int8)
    )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    sb = symm_buffer(
        hidden=hidden,
        intermediate=layer.w2_weight_packed.shape[2] * 2,
        num_experts=layer.num_experts,
        topk=topk_ids.shape[1],
    )

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=flat.device)
    int8_int4_mega_moe(
        y,
        x=x_q,
        x_sf=absmax,
        topk_idx=topk_ids.to(torch.int64),
        topk_weights=topk_weights.to(torch.float32),
        l1_weights=(layer.w13_weight_packed.data, layer.w13_weight_scale.data),
        l2_weights=(layer.w2_weight_packed.data, layer.w2_weight_scale.data),
        sym_buffer=sb,
    )
    return y.to(hidden_states.dtype)
