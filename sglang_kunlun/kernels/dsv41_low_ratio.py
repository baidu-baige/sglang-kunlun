"""Torch replacements for the CUDA/Triton kernels on the DeepSeek-V4.1 low-ratio path.

The runtime reports ``torch.version.cuda = "12.9"`` and ``tensor.is_cuda = True``
on Kunlun/XPU (the platform aliases the device type to `cuda`), so upstream's
CUDA guards take the kernel branch and call ops that do not exist on P800. Every
symbol registered here has an identical name and signature to its upstream
kernel so `kernel_ops.install()` can patch it in place; the bodies are the
mathematically equivalent torch expressions.

Registered symbols:
  * sglang.kernels.ops.attention.dsv4.rmsnorm_fp32.rmsnorm_fp32
  * sglang.kernels.ops.attention.dsv4.pair_pool_decode.pair_pool_decode
  * sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4.rope_tail_fake_quant_fp4
  * sglang.kernels.ops.embeddings.engram_gate.fused_engram_gate
    (XSpeedGate `fused_engram_gate`, torch fallback for the inputs it rejects)
  * sglang.kernels.ops.embeddings.engram_gather.engram_gather
  * sglang.kernels.ops.embeddings.engram_hash.engram_hash_ids
  * sglang.kernels.ops.embeddings.engram_hash.engram_hash_ids_and_commit
  * sglang.kernels.ops.embeddings.engram_hash.engram_hash_extend_and_commit
  * sglang.kernels.ops.embeddings.engram_hash.engram_commit_history
    (XSpeedGate `engram_commit_history`, torch fallback for the inputs it rejects)
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import numpy as np
import torch

from sglang_kunlun.kernels.kernel_ops import (
    note_op_path,
    op_probe,
    op_probe_shape,
    register_jit_op,
    register_triton_op,
)


@register_jit_op("sglang.kernels.ops.layernorm.rmsnorm_fp32", "rmsnorm_fp32")
def rmsnorm_fp32_kunlun(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``weight * (x.float() * rsqrt(mean(x.float()**2) + eps))`` in x's dtype.

    fp32 statistics and fp32 weight multiply, cast back at the end, exactly the
    arithmetic `DeepseekV41Compressor`/`DeepseekV41Indexer` use on their torch
    fallback (`srt/layers/attention/dsv4/dsv41_sparse.py:RMSNorm.forward`).
    """
    dtype = x.dtype
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return ((xf * inv_rms) * weight.float()).to(dtype)


# NOTE: the ratio-2 ``pair_pool_decode`` replacement was dropped: upstream has no
# ``sglang.kernels.ops.attention.dsv4.pair_pool_decode`` module in 0.5.19 or 0.5.22,
# so the registration only ever logged "kernel_ops: ... not found".


def _fp32_pow2(exp: torch.Tensor) -> torch.Tensor:
    """2**exp for an int32 exponent tensor, via the IEEE exponent field."""
    return ((exp + 127) << 23).view(torch.float32)


def _e4m3fn_to_fp32(bits: np.ndarray) -> np.ndarray:
    """Decode float8_e4m3fn bytes to fp32, for platforms without the fp8 cast.

    Layout: 1 sign, 4 exponent (bias 7), 3 mantissa. Exponent 0 is subnormal
    (mantissa * 2**-9); 0b01111111 / 0b11111111 is NaN.
    """
    b = bits.astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    normal = (1.0 + mant / 8.0) * np.exp2(exp - 7.0)
    subnormal = (mant / 8.0) * 2.0**-6
    val = np.where(exp == 0, subnormal, normal) * sign
    return np.where((exp == 0xF) & (mant == 0x7), np.nan, val)


def _round_to_e4m3_rne(v: torch.Tensor) -> torch.Tensor:
    """Round positive fp32 to the E4M3 grid with ties-to-even, as fp32.

    Equivalent to ``v.to(torch.float8_e4m3fn).float()`` for the positive range
    used by the compressed-KV scale. XPU implements neither the fp8 dtype cast
    nor a bit-exact substitute, so the rounding is done on the exponent field:
    an E4M3 normal has three explicit mantissa bits (ulp = 2**(e - 3)), and
    values below 2**-6 use the subnormal step 2**-9. `torch.round` is
    round-half-to-even, matching the cast.
    """
    v = v.clamp(min=2.0**-9, max=448.0)
    exp = ((v.view(torch.int32) >> 23) & 0xFF) - 127
    step = _fp32_pow2(torch.clamp(exp - 3, min=-9))
    return torch.round(v / step) * step


def _fake_quant_compressed_kv_torch(x: torch.Tensor) -> torch.Tensor:
    """FP4 round-trip with one E4M3 scale per 16 compressed-KV elements."""
    from sglang.kernels.ops.quantization.fp4_fake_quant import FP4_MAX, round_fp4

    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _round_to_e4m3_rne((amax * (1.0 / FP4_MAX)).clamp(min=2.0**-9, max=448.0))
    scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
    deq = round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


@register_jit_op(
    "sglang.kernels.ops.quantization.fp4_fake_quant", "fake_quant_compressed_kv"
)
def fake_quant_compressed_kv_kunlun(x: torch.Tensor) -> torch.Tensor:
    """XPU-safe replacement for the E4M3-scaled FP4 round-trip."""
    return _fake_quant_compressed_kv_torch(x)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant",
    "rope_tail_fake_quant_fp4",
)
def rope_tail_fake_quant_fp4_kunlun(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    block_size: int = 32,
    *,
    compressed_kv: bool = False,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotate the last `rope_dim` features, then FP4 round-trip.

    Delegates to the repo's torch oracle for the RoPE and the per-32 UE8M0 FP4
    step; the compressed-KV (per-16 E4M3) step uses the XPU-safe E4M3 rounding
    above because the oracle's fp8 dtype cast is unimplemented on P800.

    ``positions`` (0.5.22): when given, ``freqs`` is the whole table and row ``t``
    reads ``freqs[positions[t]]`` -- the fused kernel gathers it itself. The
    torch path has to do the gather, exactly as ``dsv41_sparse._rope_fq4`` does
    on its own non-fused branch; dropping it would rope every row with the
    table's leading entries.
    """
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
    from sglang.kernels.ops.quantization.fp4_fake_quant import (
        FP4_AMAX_FLOOR,
        fake_quant_fp4,
    )

    if positions is not None:
        assert positions.dim() == 1 and positions.shape[0] == x.shape[0], (
            positions.shape,
            x.shape,
        )
        freqs = freqs[positions]
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-1] in (128, 512)
        and rope_dim == 64
        and (compressed_kv or block_size == 32)
        and hasattr(torch.ops.xspeedgate_ops, "rope_tail_fake_quant_fp4_kernel")
    ):
        # Gather positions at the new wrapper boundary before invoking the old
        # operator ABI, which receives per-token frequencies.
        packed = x.contiguous()
        out = torch.empty_like(packed)
        if packed.numel() == 0:
            return out
        x_rows = packed.reshape(-1, packed.shape[-1])
        out_rows = out.reshape_as(x_rows)
        f = torch.view_as_real(freqs).contiguous()
        torch.ops.xspeedgate_ops.rope_tail_fake_quant_fp4_kernel(
            x_rows, f, out_rows, x_rows.stride(0), out_rows.stride(0),
            f.stride(0), x_rows.shape[0] // packed.shape[0],
            packed.shape[-1], rope_dim, 16 if compressed_kv else block_size,
            FP4_AMAX_FLOOR, inverse, compressed_kv, x_rows.shape[0],
        )
        return out
    rotated = rope_tail(x, freqs, rope_dim, inverse)
    if compressed_kv:
        return _fake_quant_compressed_kv_torch(rotated)
    return fake_quant_fp4(rotated, block_size)


# Keep the 0.5.22 RoPE wrapper above: it owns the new positions contract.
_ENGRAM_GATE_V2_COMBOS = frozenset(
    {
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
    }
)


def _engram_gate_v2_usable(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> bool:
    """Whether `xspeedgate_ops::fused_engram_gate` accepts these inputs.

    Mirrors `check_fused_engram_gate_inputs`
    (`XSpeedGate/wrapper/src/fusion/fused_engram_gate.cpp:113`): io dtype
    (`x`/`kv`) and weight dtype each in bf16/fp16/fp32, `x.dtype == kv.dtype`,
    q/k weights sharing one dtype, the `(io, weight)` pair in
    `_ENGRAM_GATE_V2_COMBOS`, everything contiguous and on the same device,
    `eps > 0`. Anything else keeps the torch path instead of tripping a
    TORCH_CHECK deep inside a forward.

    The `hasattr` comes first because the op is newer than the deployed wheel:
    without this check a run would die with `AttributeError: '_OpNamespace' ...
    has no attribute 'fused_engram_gate'` instead of falling back. Same guard
    style as `_dsv4_native_rope_supported`.
    """
    if not hasattr(torch.ops.xspeedgate_ops, "fused_engram_gate"):
        return False
    if x.dtype is not kv.dtype:
        return False
    if q_weight.dtype is not k_weight.dtype:
        return False
    if (x.dtype, q_weight.dtype) not in _ENGRAM_GATE_V2_COMBOS:
        return False
    if not all(t.is_contiguous() for t in (x, kv, q_weight, k_weight)):
        return False
    if any(t.device != x.device for t in (kv, q_weight, k_weight)):
        return False
    return eps > 0.0


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_gate", "fused_engram_gate"
)
def fused_engram_gate_kunlun(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """FP32 RMS(x) * RMS(key), weighted dot, sigmoid gate, value residual.

    x [T, HC, D], kv [T, (HC+1)*D] laid out head-major with the value segment last.

    Runs on the XSpeedGate `fused_engram_gate` op when the inputs fit its
    contract — that is the production path, the shape dispatch (feature-parallel
    / row / token) lives inside the op. The torch expression below stays as the
    fallback for the cases the op rejects (mixed fp16/fp32 io, non-contiguous
    inputs), since `engram_gate`'s own CUDA guard does not cover every caller.

    0.5.23 added ``input_ids`` / ``image_token_id``: rows whose id equals
    ``image_token_id`` keep ``x`` unchanged -- no gate, no value residual. The
    XSpeedGate op has no such input (it takes six arguments), so the passthrough
    is applied to its output in torch afterwards; it is one device-side
    ``torch.where`` on a ``[T,1,1]`` mask, no host sync, and it leaves the op on
    the production path instead of falling back to torch for every multimodal
    batch.
    """
    if _engram_gate_v2_usable(x, kv, q_weight, k_weight, eps):
        note_op_path("fused_engram_gate", "xspeedgate:fused_engram_gate")
        out = torch.ops.xspeedgate_ops.fused_engram_gate(
            x, kv, q_weight, k_weight, eps, clamp_value
        )
        return _keep_image_rows(out, x, input_ids, image_token_id)
    note_op_path("fused_engram_gate", "torch")
    out = _fused_engram_gate_torch(
        x, kv, q_weight, k_weight, eps, clamp_value, input_ids, image_token_id
    )
    return out


def _keep_image_rows(
    out: torch.Tensor,
    x: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    image_token_id: Optional[int],
) -> torch.Tensor:
    """0.5.23: image-token rows pass through ungated."""
    if input_ids is None:
        return out
    assert image_token_id is not None and input_ids.shape == (x.shape[0],)
    keep = (input_ids == image_token_id).reshape(x.shape[0], 1, 1)
    return torch.where(keep, x, out)


def _fused_engram_gate_torch(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Torch transcription of the engram gate, used when the op declines."""
    t, hc, d = x.shape
    xf = x.float()
    kvf = kv.float()
    # [T, HC, D] key heads, [T, D] value.
    key = kvf[:, : hc * d].reshape(t, hc, d)
    value = kvf[:, hc * d :]
    weight = q_weight.float() * k_weight.float()  # [HC, D]
    inv_d = 1.0 / d

    rstd = torch.rsqrt(xf.square().sum(-1, keepdim=True) * inv_d + eps) * torch.rsqrt(
        key.square().sum(-1, keepdim=True) * inv_d + eps
    )
    dot = ((xf * weight) * key).sum(-1, keepdim=True) * rstd * (d**-0.5)
    gate = torch.sigmoid(torch.copysign(torch.sqrt(dot.abs().clamp_min(clamp_value)), dot))
    out = (xf + gate * value.unsqueeze(1)).to(x.dtype)
    return _keep_image_rows(out, x, input_ids, image_token_id)


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_hash", "engram_hash_ids"
)
def engram_hash_ids_kunlun(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    mode: int,
    history: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    num_real: "int | None" = None,
    req_slots: "torch.Tensor | None" = None,
    block: int = 1,
    row: "torch.Tensor | None" = None,
    starts: "torch.Tensor | None" = None,
    image_token_id: "int | None" = None,
    mm_pad_shift: int = 0,
    block_t: int = 32,
):
    """Torch transposition of `_engram_hash_kernel`.

    The predecessor table and the blocked mask are rebuilt exactly as the kernel
    computes them; the hash arithmetic itself is delegated to
    `engram.compute_engram_hash_ids`, the same helper the hasher's own torch path
    uses, so the ids are bit-identical to the reference by construction.
    """
    from sglang.srt.layers.engram import compute_engram_hash_ids

    mode_decode, mode_verify, mode_extend = 0, 1, 2
    num_tokens = input_ids.shape[0]
    num_layers, n = multipliers.shape
    n_heads = primes.shape[-1]
    device = input_ids.device
    out = torch.empty(
        num_tokens, num_layers, (n - 1) * n_heads, dtype=torch.int64, device=device
    )
    tokens_out = torch.empty(num_tokens, n, dtype=torch.int32, device=device)
    if num_tokens == 0:
        return out, tokens_out
    if num_real is None:
        num_real = num_tokens

    ids = input_ids.to(torch.int64)
    t = torch.arange(num_tokens, device=device)
    if mode == mode_decode:
        req_of_row, off = t, torch.zeros_like(t)
    elif mode == mode_verify:
        req_of_row, off = t // block, t - (t // block) * block
    else:
        assert mode == mode_extend and row is not None and starts is not None
        req_of_row = row.to(torch.int64)
        off = t - starts.to(torch.int64)[req_of_row]
    real = t < num_real
    hrow = req_slots.to(torch.int64)[req_of_row] if req_slots is not None else req_of_row

    shifts = torch.arange(n, device=device)
    # Predecessor at shift s: an earlier token of the same run when s <= off, else
    # the history row, which is oldest-first.
    from_batch = ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
    hist_col = (n - 2 - (shifts.unsqueeze(0) - off.unsqueeze(-1) - 1)).clamp(0, n - 2)
    from_hist = history.to(torch.int64)[hrow].gather(1, hist_col)
    tok = torch.where(shifts.unsqueeze(0) <= off.unsqueeze(-1), from_batch, from_hist)
    tok = torch.where(real.unsqueeze(-1), tok, torch.zeros_like(tok))

    blocked = positions.to(torch.int64).unsqueeze(-1) < shifts.unsqueeze(0)
    blocked = blocked | (~real).unsqueeze(-1)
    if image_token_id is not None:
        # Scheduler-provided history still carries the multimodal pad ids.
        tok = torch.where(tok >= mm_pad_shift, torch.full_like(tok, image_token_id), tok)
        blocked = blocked | (tok == image_token_id)
    blocked = blocked.to(torch.int32).cumsum(-1) > 0

    tokens_out.copy_(tok.to(torch.int32))
    # The kernel reads the map unmasked; keep indices in range so a stray id
    # cannot fault where the kernel would have read adjacent memory.
    safe_tok = tok.clamp(0, token_map.numel() - 1)
    out.copy_(
        compute_engram_hash_ids(
            safe_tok, blocked, pad_id, token_map, multipliers, primes, offsets
        )
    )
    return out, tokens_out


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_hash", "engram_hash_extend_and_commit"
)
def engram_hash_extend_and_commit_kunlun(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    history: torch.Tensor,
    commit_history: torch.Tensor,
    req_slots: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    num_real: int,
    history_via_slots: bool,
    out_cache_loc: Optional[torch.Tensor],
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    image_token_id: Optional[int] = None,
    mm_pad_shift: int = 0,
) -> torch.Tensor:
    """Eager extend: finish every hash against old history before committing.

    Live request slots must be distinct, as in upstream. Boolean selection here
    is intentionally eager-only; decode/verify graph paths use other entries.

    Args:
        input_ids: Flattened extend tokens, including optional trailing padding.
        positions: Absolute token positions used for n-gram history masking.
        history: Old history, indexed by request or by request-pool slot.
        commit_history: Persistent history to update after all hashes are read.
        req_slots: Distinct pool slots for live requests.
        starts: Per-request offsets into the flattened token array.
        lengths: Per-request extend lengths; zero-length requests do not commit.
        num_real: Number of non-padding tokens.
        history_via_slots: Whether history rows are addressed through req_slots.
        out_cache_loc: Optional mask; a zero final-token slot prevents commit.
        token_map: Original-to-compressed vocabulary mapping.
        multipliers: Per-layer n-gram hash multipliers.
        primes: Hash moduli for each layer, n-gram size and head.
        offsets: Per-head embedding-table offsets.
        pad_id: Compressed padding token ID.
        image_token_id: Optional image token that blocks history contribution.
        mm_pad_shift: Threshold used to normalize multimodal placeholder IDs.

    Returns:
        Hash IDs for every token and layer, with commit_history updated in place.
    """
    bs = req_slots.numel()
    num_tokens = input_ids.numel()
    width = multipliers.shape[1] - 1
    assert 0 <= num_real <= num_tokens
    assert starts.numel() == lengths.numel() == bs
    assert history.is_contiguous() and commit_history.is_contiguous()
    assert all(t.ndim == 1 and t.stride(0) == 1 for t in (req_slots, starts, lengths))
    lens = lengths.to(torch.int64)
    starts64 = starts.to(torch.int64)
    # The existing torch hash gathers even padded rows before masking them.
    # Give it a full row map, and a safe dummy source when no real tokens exist.
    row = torch.zeros(num_tokens, dtype=torch.int64, device=input_ids.device)
    hash_history = history
    hash_slots = req_slots if history_via_slots else None
    hash_starts = starts64
    if num_real:
        row[:num_real] = torch.repeat_interleave(
            torch.arange(bs, device=input_ids.device), lens, output_size=num_real
        )
        row[num_real:] = row[0]
    else:
        hash_history = history.new_zeros((1, width))
        hash_slots = None
        hash_starts = starts64.new_zeros(1)
    out, tokens = engram_hash_ids_kunlun(
        input_ids, positions, mode=2, history=hash_history,
        token_map=token_map, multipliers=multipliers, primes=primes,
        offsets=offsets, pad_id=pad_id, num_real=num_real,
        req_slots=hash_slots, row=row, starts=hash_starts,
        image_token_id=image_token_id, mm_pad_shift=mm_pad_shift,
    )
    if num_tokens and bs:
        last = (starts64 + lens - 1).clamp(0, num_tokens - 1)
        live = lens > 0
        if out_cache_loc is not None:
            live = live & (out_cache_loc[last] != 0)
        # Do not redirect inactive requests to the spare row: it stays untouched.
        commit_history[req_slots.to(torch.int64)[live]] = (
            tokens[last[live], :width].flip(-1).to(commit_history.dtype)
        )
    return out


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_hash", "engram_hash_ids_and_commit"
)
def engram_hash_ids_and_commit_kunlun(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    history: torch.Tensor,
    req_slots: torch.Tensor,
    out_cache_loc: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    image_token_id: "int | None" = None,
    mm_pad_shift: int = 0,
    block_t: int = 32,
) -> torch.Tensor:
    """Decode-phase hash id + in-place history advance, torch version.

    Upstream folded the commit into `_engram_hash_kernel` (`write_tokens=False` and passing
    `out_cache_loc`); `EngramHasher.forward` returns early on that path, so covering only
    `engram_hash_ids` is not enough: without registering this symbol, every decode step would enter triton,
    reporting `AssertionError: libcuda.so cannot found!`.

    The commit expression matches the hasher's own torch branch: the new row = this token plus its `width-1`
    most recent predecessors, oldest-first. Rows with `out_cache_loc == 0` are graph padding and must not be
    written to the table.
    """
    out, tokens = engram_hash_ids_kunlun(
        input_ids,
        positions,
        mode=0,  # MODE_DECODE
        history=history,
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        pad_id=pad_id,
        num_real=None,
        req_slots=req_slots,
        block=1,
        row=None,
        starts=None,
        image_token_id=image_token_id,
        mm_pad_shift=mm_pad_shift,
        block_t=block_t,
    )
    width = history.shape[1]
    if width:
        # Unconditionally element-wise, no host sync, no data-dependent shapes, so it can enter the CUDA graph.
        # Rows with `out_cache_loc == 0` are graph padding; redirect them to the spare slot at the end of history
        # (`init_history` allocates `num_req_slots + 1` rows, and its comment literally says "a spare row for
        # graph padding", with `pad_row = num_req_slots = history.shape[0] - 1`).
        # This is the same formulation as upstream's own torch branch (`engram.py:371`:
        # `commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)`).
        #
        # Previously we selected rows with a boolean mask (`req_slots[out_cache_loc != 0]`) +
        # `rows.numel()` emptiness check,
        # which during capture is a data-dependent shape + host read: in practice the decode graph capture allocated
        # 10.14 GiB at this line and OOM'd directly.
        pad_row = history.shape[0] - 1
        rows = torch.where(out_cache_loc == 0, pad_row, req_slots.to(torch.int64))
        history[rows] = tokens[:, :width].flip(-1).to(history.dtype)
    return out


# Two capacity limits from `engram_commit_history.cpp`, measured empirically: bs=4097 reports "exceeds SM
# cache capacity 4096", width=513 reports "exceeds LM capacity 512".
_ECH_MAX_BS, _ECH_MAX_WIDTH = 4096, 512


def _engram_commit_history_op_args(
    history: torch.Tensor,
    verify_ids: torch.Tensor,
    req_slots: torch.Tensor,
    commit_lens: torch.Tensor,
) -> "tuple | None":
    """Normalized (verify_ids, req_slots, commit_lens) for the op, or None when
    `xspeedgate_ops::engram_commit_history` cannot serve this call.

    The op's contract (`XSpeedGate/wrapper/src/embedding/engram_commit_history.cpp`):
    `history` / `verify_ids` 2D int32 with `stride(1) == 1` (row padding is fine),
    `req_slots` 1D contiguous int32/int64, `commit_lens` 1D contiguous int32, one
    row each per request, `bs <= 4096`, `width <= 512`.

    `verify_ids` arrives int64 -- `verify_ids_2d` is built from the int64 draft
    token ids -- and is cast here; it is [bs, verify_block], so the copy is
    negligible. `history` is never cast: the op updates it in place, so handing
    over a converted copy would silently drop the commit.

    The documented `commit <= width` precondition is deliberately not enforced.
    Checking it means reading `commit_lens` on the host, and the kernel was
    verified bit-exact against the torch reference for `commit` up to the verify
    block (width 3, commits 4/5/6), i.e. it implements the same general formula
    upstream does. Distinct `req_slots` stays the caller's invariant -- one row
    per live request -- exactly as upstream's own kernel assumes.
    """
    if not hasattr(torch.ops.xspeedgate_ops, "engram_commit_history"):
        return None
    if not history.is_cuda or history.dim() != 2 or verify_ids.dim() != 2:
        return None
    if history.dtype is not torch.int32 or history.stride(1) != 1:
        return None
    if req_slots.dim() != 1 or commit_lens.dim() != 1:
        return None
    if any(t.device != history.device for t in (verify_ids, req_slots, commit_lens)):
        return None
    bs, num_verify = verify_ids.shape
    if req_slots.shape[0] != bs or commit_lens.shape[0] != bs:
        return None
    if num_verify < 1 or bs > _ECH_MAX_BS or history.shape[1] > _ECH_MAX_WIDTH:
        return None

    verify = verify_ids
    if verify.dtype is not torch.int32:
        verify = verify.to(torch.int32)
    if verify.stride(1) != 1:
        verify = verify.contiguous()
    slots = req_slots
    if slots.dtype not in (torch.int32, torch.int64):
        slots = slots.to(torch.int64)
    return verify, slots.contiguous(), commit_lens.to(torch.int32).contiguous()


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_hash", "engram_commit_history"
)
def engram_commit_history_kunlun(
    history: torch.Tensor,
    verify_ids: torch.Tensor,
    req_slots: torch.Tensor,
    commit_lens: torch.Tensor,
    live_rows: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """Anchor + accepted drafts slid into the per-request history ring.

    Runs on the XSpeedGate `engram_commit_history` op, which updates `history`
    in place; `_engram_commit_history_torch` stays as the fallback for the calls
    its contract rejects. Only reachable through
    `EngramHasher.commit_after_verify`, i.e. the DSpark verify path.
    """
    if req_slots.numel() == 0 or history.shape[1] == 0:
        return
    args = _engram_commit_history_op_args(history, verify_ids, req_slots, commit_lens)
    if args is None:
        note_op_path("engram_commit_history", "torch")
        _engram_commit_history_torch(
            history, verify_ids, req_slots, commit_lens, live_rows, gate
        )
        return
    note_op_path("engram_commit_history", "xspeedgate:engram_commit_history")
    keep = _engram_commit_keep_rows(req_slots, live_rows, gate)
    if keep is None:
        torch.ops.xspeedgate_ops.engram_commit_history(history, *args)
        return
    # The op has no gating parameters and writes `history` in place, so the
    # skipped rows are put back afterwards rather than letting the op decline --
    # declining would move the DSpark verify path, which always passes
    # `live_rows`, onto the torch transcription for every decode step.
    # `req_slots` is distinct by contract, so the scatter is well defined.
    req = req_slots.to(torch.int64)
    previous = history[req].clone()
    torch.ops.xspeedgate_ops.engram_commit_history(history, *args)
    history[req] = torch.where(keep.unsqueeze(-1), history[req], previous)


def _engram_commit_keep_rows(
    req_slots: torch.Tensor,
    live_rows: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Which rows 0.5.23 wants committed, or None when no gating applies.

    Rows whose `live_rows` entry is 0 are skipped, and `gate[0] == 0` skips every
    row. Upstream keys the whole gate on `live_rows` being present
    (`GATED=live_rows is not None`, engram_hash.py:92), so a `gate` passed on its
    own is ignored here too. Both are device tensors and stay on device: a
    `.item()` here would sync the decode critical path and is illegal under graph
    capture.
    """
    if live_rows is None:
        return None
    keep = live_rows.reshape(-1)[: req_slots.numel()] != 0
    if gate is not None:
        keep = keep & gate.reshape(-1)[:1].to(torch.bool)
    return keep


def _engram_commit_history_torch(
    history: torch.Tensor,
    verify_ids: torch.Tensor,
    req_slots: torch.Tensor,
    commit_lens: torch.Tensor,
    live_rows: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """Torch transcription, used when the op declines.

    `history[slot]` then this batch's accepted ids form one window; the new row is
    `window[commit : commit + width]`. Same expression as
    `EngramHasher.commit_after_verify`'s torch fallback.

    0.5.23 added ``live_rows`` / ``gate``: rows whose ``live_rows`` entry is 0 are
    skipped, and a ``gate[0] == 0`` skips the whole update. Both are device
    tensors, so the gate is applied with ``torch.where`` rather than a host read
    (a ``.item()`` here would sync the decode critical path and is illegal under
    graph capture).
    """
    width = history.shape[1]
    req = req_slots.to(torch.int64)
    window = torch.cat(
        [history[req], verify_ids.to(history.dtype)]
        if verify_ids.dtype is not history.dtype
        else [history[req], verify_ids],
        dim=1,
    )
    cols = commit_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
        width, device=window.device
    )
    updated = window.gather(1, cols)
    keep = _engram_commit_keep_rows(req_slots, live_rows, gate)
    if keep is not None:
        updated = torch.where(keep.unsqueeze(-1), updated, history[req])
    history[req] = updated


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_gather", "engram_gather"
)
def engram_gather_kunlun(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
    sort_ids: bool = False,
) -> torch.Tensor:
    """Gather fp8 rows over a host pointer and dequantize with e8m0 block scales.

    `weight_ptr`/`scale_ptr` address host memory (the engram host table), so the
    rows are read through ctypes and only the gathered values cross to the
    device. Ids outside [row_lo, row_hi) produce a zero row.

    0.5.23 added ``sort_ids``, which makes the Triton kernel read rows in address
    order to improve host-page locality and scatter each row back to its original
    position. It does not change the result, and this implementation already
    walks ``ids`` row by row on the host, so the flag is accepted and ignored.
    """
    del sort_ids
    n = ids.numel()
    if n == 0:
        return out
    ids_cpu = ids.detach().to("cpu", torch.int64).reshape(-1).tolist()
    nblocks = dim // block_size
    # `block_size == dim` means one scale per row: the W8A8-INT8 table layout
    # (int8 payload + bf16 per-row scale). Otherwise it is the fp8 layout with
    # e8m0 scales, one per `block_size` elements (DIM // BLK per row).
    int8_layout = block_size == dim
    w_t = np.zeros((n, dim), dtype=np.float32)
    s_t = np.zeros((n, 1 if int8_layout else nblocks), dtype=np.float32)
    u8 = ctypes.c_ubyte
    # Both formats are decoded from their bits: XPU implements neither the fp8
    # dtype cast nor `float8_e8m0fnu`.
    # The pointer is the base of this rank's shard: the kernel reads
    # `w + (idx - row_lo) * DIM`, so subtract row_lo before addressing.
    for i, rid in enumerate(ids_cpu):
        if rid < row_lo or rid >= row_hi:
            continue
        local = rid - row_lo
        if int8_layout:
            w_t[i] = np.frombuffer(
                (u8 * dim).from_address(weight_ptr + local * dim), dtype=np.int8
            ).astype(np.float32)
            sbits = np.frombuffer(
                (u8 * 2).from_address(scale_ptr + local * 2), dtype=np.uint16
            )
            # bf16 -> fp32 is a 16-bit left shift of the bit pattern.
            s_t[i, 0] = (sbits.astype(np.uint32) << 16).view(np.float32)[0]
            continue
        wb = np.frombuffer(
            (u8 * dim).from_address(weight_ptr + local * dim), dtype=np.uint8
        )
        sb = np.frombuffer(
            (u8 * nblocks).from_address(scale_ptr + local * nblocks), dtype=np.uint8
        )
        w_t[i] = _e4m3fn_to_fp32(wb)
        # e8m0: 2**(e - 127); byte 0 encodes 2**-127, exactly as the kernel.
        exps = sb.astype(np.int32)
        s_t[i] = np.where(exps == 0, 2.0**-127, np.exp2(exps - 127.0))
    values = torch.from_numpy(
        (w_t.reshape(n, nblocks, block_size) * s_t[..., None]).reshape(n, dim)
    )
    out.view(n, dim).copy_(values.to(out.device).to(out.dtype))
    return out


@register_jit_op(
    "sglang.kernels.ops.quantization.fp4_indexer_quant", "quantize_fp4_indexer_tensor"
)
def quantize_fp4_indexer_tensor_kunlun(
    x: torch.Tensor, rne: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-32 UE8M0 FP4 quantize into (packed int8 [n, 64], packed scales [n] int32).

    Four 32-element groups per 128-wide row; group scale is
    ``ceil_pow2_exp(max(max(|x|)/6, 1e-4))`` stored as one byte each in the int32, and the
    payload packs two 4-bit e2m1 codes per byte (low nibble = even column).

    Use the native bf16 op when installed, otherwise transcribe upstream
    ``fp4_indexer.quantize_fp4_indexer_row`` with device-side torch operations.
    kunlun_ops.indexer_k_quant_and_cache only supports FP8, not this format.
    """
    assert x.shape[-1] == 128
    if (
        x.is_cuda and x.dtype == torch.bfloat16
        and hasattr(torch.ops.xspeedgate_ops, "quantize_fp4_indexer_tensor")
    ):
        return torch.ops.xspeedgate_ops.quantize_fp4_indexer_tensor(x, rne)
    rows = x.contiguous().view(-1, 128)
    payload = torch.empty((rows.shape[0], 64), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows.shape[0],), dtype=torch.int32, device=x.device)
    # Bound temporary storage during prefill; chunk sizes depend only on shape.
    for start in range(0, rows.shape[0], 4096):
        blocks = rows[start:start + 4096].float().reshape(-1, 4, 32)
        sf = (blocks.abs().amax(-1) / 6.0).clamp_min(1e-4)
        bits = sf.contiguous().view(torch.int32)
        exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).clamp(1, 254)
        scale = (exponent << 23).view(torch.float32)
        values = blocks / scale.unsqueeze(-1)
        magnitude = values.abs().clamp(max=6.0)
        code = torch.zeros_like(values, dtype=torch.int32)
        # At boundaries the default chooses the smaller magnitude; RNE chooses
        # the even E2M1 code. Negative zero is canonicalized to code zero.
        for i, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
            up = magnitude >= boundary if rne and i % 2 else magnitude > boundary
            code = code + up.to(torch.int32)
        code = code | (((values < 0) & (code != 0)).to(torch.int32) << 3)
        code = code.reshape(-1, 128)
        payload[start:start + 4096] = (code[:, ::2] | (code[:, 1::2] << 4)).to(torch.int8)
        scales[start:start + 4096] = (
            exponent[:, 0] | (exponent[:, 1] << 8)
            | (exponent[:, 2] << 16) | (exponent[:, 3] << 24)
        )
    return payload, scales


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer", "flat_selection_slots"
)
def flat_selection_slots_kunlun(
    raw_indices: torch.Tensor,
    request_starts: torch.Tensor,
    slots: torch.Tensor,
    page_indices: torch.Tensor,
) -> None:
    """``page_indices[r, j] = slots[request_starts[r] + raw_indices[r, j]]``, -1
    where the request-local position is negative.

    New in 0.5.23 (``v41_indexer.scoring.write_page_indices``). Upstream's
    version is Triton-only -- no device gate, no fallback -- so on Kunlun it dies
    with "0 compatible backends for target (cuda)". The expression is upstream's
    own docstring; the index clamp only keeps the masked-out lanes in range.
    """
    rows, k = raw_indices.shape
    if not rows or not k:
        return
    idx = request_starts.reshape(-1, 1).to(torch.int64) + raw_indices.to(torch.int64)
    gathered = slots.reshape(-1)[idx.clamp_(0, slots.numel() - 1)]
    page_indices[:rows, :k] = torch.where(
        raw_indices >= 0,
        gathered.to(page_indices.dtype),
        torch.full((), -1, dtype=page_indices.dtype, device=page_indices.device),
    )


@register_jit_op(
    "sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache", "store_fp4_index_k_cache"
)
def store_fp4_index_k_cache_kunlun(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    """Scatter the quantized index-K payload and its 4 scale bytes into the paged
    cache: payload at ``page, page_offset*64``, scales at ``page, page_size*64 +
    page_offset*4``.

    Prefer ``xspeedgate_ops::store_fp4_index_k_cache`` when available, otherwise
    scatter bytes with torch, including caches with padded page strides.
    One difference from the Triton kernel it replaces: the Kunlun path treats
    ``loc <= 0`` and out-of-range slots as a whole-row no-op (slot 0 is padding),
    instead of writing them.
    """
    assert input.shape[-1] == 128
    assert page_size > 0
    assert cache.ndim == 2 and cache.shape[1] == page_size * 68
    assert cache.dtype == torch.uint8 and cache.stride(1) == 1
    assert cache.device == input.device == loc.device
    assert loc.ndim == 1 and loc.dtype in (torch.int32, torch.int64)
    assert loc.numel() == input.numel() // 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor_kunlun(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    if n_tokens == 0:
        return
    assert cache.shape[0] > 0
    if (
        input.is_cuda and cache.is_contiguous()
        and hasattr(torch.ops.xspeedgate_ops, "store_fp4_index_k_cache")
    ):
        torch.ops.xspeedgate_ops.store_fp4_index_k_cache(
            k_fp4, k_sf, cache, loc.contiguous(), page_size
        )
        return
    payload_lane = torch.arange(64, device=cache.device)
    scale_lane = torch.arange(4, device=cache.device)
    for start in range(0, n_tokens, 4096):
        locations = loc[start:start + 4096].to(torch.int64)
        valid = (locations > 0) & (locations < cache.shape[0] * page_size)
        # Slot zero is reserved. Invalid rows write its original bytes back;
        # no boolean compaction, host synchronization or data-dependent shapes.
        safe = torch.where(valid, locations, 0)
        page = (safe // page_size).unsqueeze(-1)
        slot = (safe % page_size).unsqueeze(-1)
        pcol = slot * 64 + payload_lane
        scol = page_size * 64 + slot * 4 + scale_lane
        data = k_fp4[start:start + 4096].view(torch.uint8)
        sf = ((k_sf[start:start + 4096, None] >> (scale_lane * 8)) & 255).to(torch.uint8)
        # XPU where does not support uint8; select losslessly in int32.
        cache[page, pcol] = torch.where(
            valid[:, None], data.to(torch.int32), cache[page, pcol].to(torch.int32)
        ).to(torch.uint8)
        cache[page, scol] = torch.where(
            valid[:, None], sf.to(torch.int32), cache[page, scol].to(torch.int32)
        ).to(torch.uint8)


def _fp32_to_e4m3_bytes(v: torch.Tensor) -> torch.Tensor:
    """Encode fp32 to float8_e4m3fn bytes (uint8) with round-to-nearest-even,
    saturating at +-448.

    XPU has no fp8 dtype cast, so the encoding is done on the bits. Validated
    against CPU `torch.to(torch.float8_e4m3fn)` in the operator tests.
    """
    sign = (v < 0).to(torch.int32)
    a = v.abs().clamp(max=448.0)
    bits = a.contiguous().view(torch.int32)
    # Round the 23-bit mantissa to 3 bits (RNE) by adding half an ulp of the
    # target precision plus the tie-to-even adjustment, then masking.
    rounded = bits + (1 << 19) - 1 + ((bits >> 20) & 1)
    rounded = rounded & ~((1 << 20) - 1)
    exp = ((rounded >> 23) & 0xFF) - 127
    mant = (rounded >> 20) & 0x7
    normal = ((exp + 7) << 3) | mant
    # Subnormal (exp < -6): round value / 2**-9, one byte 0..7.
    sub = torch.round(a / (2.0**-9)).to(torch.int32).clamp(0, 7)
    byte = torch.where(exp < -6, sub, normal)
    byte = torch.where(a == 0.0, torch.zeros_like(byte), byte)
    byte = byte.clamp(0, 0x7E)  # 448 = 0b0_1111_110; above it saturates
    return (byte | (sign << 7)).to(torch.uint8)


_FLASHMLA_BYTES_PER_TOKEN = 584  # 448 fp8 nope + 7 ue8m0 scale + 1 pad + 128 B bf16 rope
_FLASHMLA_SLOT_BYTES = 576


def _flashmla_page_bytes(page_size: int) -> int:
    """``ceil_div(584 * page_size, 576) * 576`` -- the op's (and CUDA's) page stride."""
    raw = _FLASHMLA_BYTES_PER_TOKEN * page_size
    return -(-raw // _FLASHMLA_SLOT_BYTES) * _FLASHMLA_SLOT_BYTES


def _fused_store_flashmla_unsupported(
    input: torch.Tensor, cache_u8: torch.Tensor, indices: torch.Tensor, page_size: int
) -> Optional[str]:
    """Return why ``fused_store_flashmla_cache`` cannot run this call, else None.

    Mirrors ``check_inputs`` in
    ``XSpeedGate/wrapper/src/attention/fused_store_flashmla_cache.cpp``. The
    ``hasattr`` comes first because the op is newer than the deployed wheel
    (``xspeedgate_ops-1.6.0`` does not export it), so without the guard the
    FlashMLA store would die with ``AttributeError: '_OpNamespace' ...`` instead
    of keeping the torch transcription. Same guard style as
    ``_engram_gate_v2_usable``.
    """
    if not hasattr(torch.ops.xspeedgate_ops, "fused_store_flashmla_cache"):
        return "xspeedgate_ops has no fused_store_flashmla_cache"
    if input.dtype is not torch.bfloat16:
        return f"input dtype {input.dtype} is not bfloat16"
    if input.dim() != 2 or input.shape[1] != 512:
        return f"input shape {tuple(input.shape)} is not [num_tokens, 512]"
    if indices.dim() != 1 or indices.dtype not in (torch.int32, torch.int64):
        return "indices must be 1D int32/int64"
    if indices.shape[0] != input.shape[0]:
        return "indices length does not match the input rows"
    if page_size <= 0 or page_size & (page_size - 1):
        return f"page_size={page_size} is not a power of two"
    if cache_u8.dim() != 2 or not cache_u8.is_contiguous():
        return "cache must be a contiguous 2D byte tensor"
    if cache_u8.shape[1] < _flashmla_page_bytes(page_size):
        return (
            f"cache page is {cache_u8.shape[1]} B, the op needs "
            f"{_flashmla_page_bytes(page_size)} B"
        )
    if indices.device != input.device or cache_u8.device != input.device:
        return "input/cache/indices are on different devices"
    return None


def _fused_store_latent_unsupported(
    input: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor, page_size: int
) -> Optional[str]:
    """Return why ``fused_store_latent_cache`` cannot run this call, else None.

    Mirrors ``check_fused_store_latent_cache_args`` in
    ``XSpeedGate/wrapper/src/attention/fused_store_latent_cache.cpp``. This is the
    XSpeedGate op for the **Kunlun half cache** layout: the plain fp16
    ``[num_pages, page_size*512]`` latent that ``dsv4_create_buffer_kunlun``
    allocates, i.e. exactly the torch scatter-copy the half-cache branch below
    runs (no quantization, no scale). The ``hasattr`` guard comes first so wheels
    older than ``1.6.0+c9bac3b`` (which only export ``fused_store_flashmla_cache``)
    keep the torch transcription instead of dying with ``AttributeError``.
    """
    if not hasattr(torch.ops.xspeedgate_ops, "fused_store_latent_cache"):
        return "xspeedgate_ops has no fused_store_latent_cache"
    if input.dtype is not torch.bfloat16:
        return f"input dtype {input.dtype} is not bfloat16"
    if input.dim() != 2 or input.shape[1] != 512:
        return f"input shape {tuple(input.shape)} is not [num_tokens, 512]"
    if indices.dim() != 1 or indices.dtype not in (torch.int32, torch.int64):
        return "indices must be 1D int32/int64"
    if indices.shape[0] != input.shape[0]:
        return "indices length does not match the input rows"
    if page_size <= 0 or page_size & (page_size - 1):
        return f"page_size={page_size} is not a power of two"
    if cache.dtype is not torch.float16:
        return f"cache dtype {cache.dtype} is not float16"
    if cache.dim() != 2 or not cache.is_contiguous():
        return "cache must be a contiguous 2D fp16 tensor"
    if cache.shape[0] < 1:
        return "cache must have at least one page"
    if cache.shape[1] != page_size * 512:
        return (
            f"cache row is {cache.shape[1]}, the op needs "
            f"page_size*512={page_size * 512}"
        )
    if indices.device != input.device or cache.device != input.device:
        return "input/cache/indices are on different devices"
    return None


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.attn", "fused_store_cache"
)
def fused_store_cache_kunlun(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: str,
    layout=None,
    freqs_cis: Optional[torch.Tensor] = None,
) -> None:
    """Fused FP8 quantize + paged scatter for the DeepSeek-V4 caches.

    Torch transcription of `triton_fused_store_flashmla` (the HIP reference for
    the JIT `store.cuh` kernel this replaces on XPU). MLA slot layout per page:
    `page_size` slots of 576 bytes (448 fp8 nope + 128 bytes bf16 rope) followed
    by `page_size` 8-byte scale rows holding one UE8M0 exponent per 64-wide nope
    tile (7 written, the 8th is padding).

    Chunked over tokens: this runs while the KV pool is already resident (the card
    had ~0.5 GiB free), so the whole `[n, 512]` fp32 upcast and the `n * 64`
    int64 index tensors are built per 4096-token block instead of all at once.
    """
    if type == "indexer":
        raise NotImplementedError(
            "fused_store_cache type='indexer' has no Kunlun implementation; the "
            "V4.1 low-ratio path uses set_index_k_fp4 instead"
        )
    n = input.shape[0]
    if n == 0:
        return
    # Entry probe comes first: xspeedgate's fused_store_flashmla_cache only covers the 584B byte
    # layout branch, while the Kunlun half cache returns early below. To know "whether this path is reached", first
    # check whether the function itself was called and what dtype the cache is.
    if freqs_cis is not None:
        # V4.1's CUDA kernel folds the rotation of the 64-dim RoPE tail into the store, so fp4/fp8 only round
        # once. The Kunlun pool is plain fp16 with no fusable quantization — but here upstream passes an **unrotated**
        # latent, and silently storing it would be a wrong-value path.
        raise NotImplementedError(
            "fused_store_cache(freqs_cis=...) reaches the Kunlun half-cache path; "
            "the caller must pass the already-rotated latent (freqs_cis=None)"
        )
    if cache.dtype in (torch.float16, torch.bfloat16):
        # Kunlun half-cache layout. `DeepSeekV4SingleKVPool` has been hooked on this platform
        # (`hooks/mem_cache/common.py::dsv4_create_buffer_kunlun` /
        # `dsv4_get_bytes_per_token_kunlun`): one page is
        # `page_size * (qk_nope_head_dim + qk_rope_head_dim)` `store_dtype` **elements**,
        # i.e. plain fp16/bf16 latent, not the N-card FlashMLA 584 B/token packing
        # (448 fp8 nope + 7 ue8m0 scale + 1 pad + 128 B bf16 rope). Both sides of XSpeedGate
        # (write `set_k_and_s_v4_with_mapping`, read `compressed_attention`) use this same layout.
        #
        # Writing the FlashMLA byte packing into this buffer was exactly why every CSA2 (ratio 1/2) layer's attention
        # output NaN: the compressed KV pool has only `set_extra_key_buffer_fused` →
        # `set_key_buffer_fused` → here, a single write path; the layout mismatch shows up as rows near
        # the fp16 limit (≈65000) with NaN, then a fully-zero final hidden state and completely uniform logits.
        dim = input.shape[-1]
        num_pages = cache.shape[0]
        assert cache.numel() == num_pages * page_size * dim, (
            f"half-cache page mismatch: cache{tuple(cache.shape)} vs "
            f"page_size={page_size} dim={dim}"
        )
        reason = _fused_store_latent_unsupported(input, cache, indices, page_size)
        if reason is None:
            # XSpeedGate's fp16 scatter-write op: equivalent to the torch advanced indexing below
            # (out-of-range rows are skipped),
            # but avoids the two page/slot int64 tensors and one `input.to(fp16)` intermediate tensor.
            note_op_path("fused_store_cache", "xspeedgate:fused_store_latent_cache")
            torch.ops.xspeedgate_ops.fused_store_latent_cache(
                input.contiguous(), cache, indices.contiguous(), page_size
            )
            return
        # The page pool is already a plain 512-element half cache. Reuse the
        # byte-preserving writer when the BF16-to-FP16 fused writer above does
        # not cover this dtype pair. Keep the original cache-dtype conversion.
        if (
            dim == 512
            and input.ndim == 2
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and indices.ndim == 1
            and indices.dtype in (torch.int32, torch.int64)
            and indices.numel() == input.shape[0]
            and cache.ndim == 2
            and cache.is_contiguous()
            and input.device == indices.device == cache.device
            and input.is_cuda
            and hasattr(torch.ops.xspeedgate_ops, "set_k_and_s_v4")
        ):
            # Legal Torch locations are [-capacity, capacity). Preserve negative
            # page indexing as well as dummy slot 0. Allocator locations must
            # satisfy that contract; out-of-range locations are not supported.
            loc = indices.to(torch.int64).reshape(-1)
            loc = loc.remainder(num_pages * page_size).contiguous()
            torch.ops.xspeedgate_ops.set_k_and_s_v4(
                cache.view(torch.uint8), loc, input.to(cache.dtype).contiguous(), page_size
            )
            note_op_path("fused_store_cache", "xspeedgate:set_k_and_s_v4")
            return
        note_op_path("fused_store_cache", f"torch half-cache ({reason})")
        loc = indices.to(torch.int64).reshape(-1)
        page = loc // page_size
        slot = loc - page * page_size
        view = cache.view(num_pages, page_size, dim)
        view[page, slot] = input.to(cache.dtype)
        return
    u8 = cache.view(torch.uint8)
    reason = _fused_store_flashmla_unsupported(input, u8, indices, page_size)
    if reason is None:
        # XSpeedGate fuses the fp8 quantize + ue8m0 scales + rope copy into one
        # kernel, writing the same 584 B/token FlashMLA slot layout the torch
        # transcription below builds byte by byte.
        note_op_path("fused_store_cache", "xspeedgate:fused_store_flashmla_cache")
        torch.ops.xspeedgate_ops.fused_store_flashmla_cache(
            input.contiguous(), u8, indices.contiguous(), page_size
        )
        return
    note_op_path("fused_store_cache", f"torch ({reason})")
    flat = u8.reshape(-1)
    bpp = u8.shape[1]
    tile = 64
    slot_bytes = 576
    nope_dim = 448
    scales_per_token = 8
    loc = indices.to(torch.int64).reshape(-1)

    chunk = 4096
    lane = torch.arange(tile, device=flat.device)
    rope_lane = torch.arange(2 * tile, device=flat.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        page = loc[start:end] // page_size
        slot = loc[start:end] - page * page_size
        x = input[start:end].to(torch.float32)
        for t in range(nope_dim // tile):
            blk = x[:, t * tile : (t + 1) * tile]
            abs_max = blk.abs().amax(dim=1, keepdim=True)
            scale = torch.maximum(abs_max, torch.full_like(abs_max, 1.0e-8)) / 448.0
            # ceil(log2(scale)) from the exponent bits, exactly (no libm).
            sbits = scale.contiguous().view(torch.int32)
            ceil_log2 = ((sbits >> 23) & 0xFF) - 127 + ((sbits & 0x7FFFFF) != 0).to(
                torch.int32
            )
            inv = torch.exp2(-ceil_log2.to(torch.float32))
            bytes_ = _fp32_to_e4m3_bytes((blk * inv).clamp(-448.0, 448.0))
            base = page * bpp + slot * slot_bytes + t * tile
            flat[(base[:, None] + lane[None, :]).reshape(-1)] = bytes_.reshape(-1)
            ue8m0 = (ceil_log2 + 127).clamp(0, 255).to(torch.uint8).reshape(-1)
            flat[
                page * bpp + page_size * slot_bytes + slot * scales_per_token + t
            ] = ue8m0
        # Rope: a straight bf16 copy at byte 448 of the slot, 128 bytes wide.
        rope = (
            input[start:end, nope_dim:]
            .contiguous()
            .view(torch.uint8)
            .reshape(-1, 2 * tile)
        )
        rbase = page * bpp + slot * slot_bytes + nope_dim
        flat[(rbase[:, None] + rope_lane[None, :]).reshape(-1)] = rope.reshape(-1)
