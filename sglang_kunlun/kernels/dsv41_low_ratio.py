"""Torch replacements for the CUDA/Triton kernels on the DeepSeek-V4.1 low-ratio path.

The runtime reports ``torch.version.cuda = "12.9"`` and ``tensor.is_cuda = True``
on Kunlun/XPU (the platform aliases the device type to `cuda`), so upstream's
CUDA guards take the kernel branch and call ops that do not exist on P800. Every
symbol registered here has an identical name and signature to its upstream
kernel so `kernel_ops.install()` can patch it in place; the bodies are the
mathematically equivalent torch expressions.

Registered symbols:
  * sglang.kernels.ops.layernorm.rmsnorm_fp32.rmsnorm_fp32
  * sglang.kernels.ops.attention.dsv4.c2_decode_pool.c2_decode_pool
  * sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant.rope_tail_fake_quant_fp4
  * sglang.kernels.ops.embeddings.engram_gate.fused_engram_gate
  * sglang.kernels.ops.embeddings.engram_gather.engram_gather
  * sglang.kernels.ops.embeddings.engram_hash.engram_hash_ids
  * sglang.kernels.ops.embeddings.engram_hash.engram_hash_ids_and_commit
  * sglang.kernels.ops.embeddings.engram_hash.engram_commit_history
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import numpy as np
import torch

from sglang_kunlun.kernels.kernel_ops import register_jit_op


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


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.c2_decode_pool", "c2_decode_pool"
)
def c2_decode_pool_kunlun(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    pad_row: int,
    *,
    ring_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ratio-2 decode pair pooling and state update in torch.

    Mirrors `_c2_decode_pool_kernel` step for step. All state reads happen
    before any write (the kernel never has two rows sharing a slot in one
    launch), so plain gather-then-scatter reproduces it without the racy
    per-program ordering. Softmax runs in fp32 with the same pair-axis order as
    `DeepseekV41Compressor.pool_pairs`.
    """
    kv = kv.contiguous().float()
    score = score.contiguous().float()
    n, _ = kv.shape
    req = req.to(torch.int64).reshape(-1)
    pos = pos.to(torch.int64).reshape(-1)
    raw_loc = raw_out_loc.to(torch.int64).reshape(-1)
    out_loc_ = out_loc.to(torch.int64).reshape(-1)

    odd = (pos % 2) == 1

    if ring_size:
        read_row = torch.where(
            (raw_loc == 0) | (pos == 0),
            torch.full_like(req, pad_row),
            req * ring_size + (pos - 1) % ring_size,
        )
    else:
        read_row = torch.where(raw_loc == 0, torch.full_like(req, pad_row), req)

    # Read the partner state before writing anything.
    p_kv = state_kv.index_select(0, read_row).clone()
    p_score = state_score.index_select(0, read_row).clone()

    if ring_size:
        write_row = req * ring_size + pos % ring_size
        live = (raw_loc != 0).unsqueeze(-1)
        # A pad row is never written. Index-copy handles the write_row being a
        # permutation (one decode row per live request).
        kv_out = torch.where(live, kv, state_kv.index_select(0, write_row))
        score_out = torch.where(live, score, state_score.index_select(0, write_row))
        state_kv.index_copy_(0, write_row, kv_out)
        state_score.index_copy_(0, write_row, score_out)
    else:
        odd_c = odd.unsqueeze(-1)
        state_kv.index_copy_(
            0, read_row, torch.where(odd_c, p_kv, kv)
        )
        state_score.index_copy_(
            0, read_row, torch.where(odd_c, p_score, score)
        )

    m = torch.maximum(p_score, score)
    e0 = torch.exp(p_score - m)
    e1 = torch.exp(score - m)
    denom = e0 + e1
    pooled = p_kv * (e0 / denom) + kv * (e1 / denom)

    group_pos = torch.where(odd, pos - 1, pos)
    slots = torch.where(out_loc_ >= 0, out_loc_, torch.zeros_like(out_loc_))
    return pooled, group_pos, slots


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
    from sglang.kernels.ops.attention.dsv4.torch_quant import FP4_MAX, round_fp4

    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _round_to_e4m3_rne((amax * (1.0 / FP4_MAX)).clamp(min=2.0**-9, max=448.0))
    scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
    deq = round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.torch_quant", "fake_quant_compressed_kv"
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
) -> torch.Tensor:
    """Rotate the last `rope_dim` features, then FP4 round-trip.

    Delegates to the repo's torch oracle for the RoPE and the per-32 UE8M0 FP4
    step; the compressed-KV (per-16 E4M3) step uses the XPU-safe E4M3 rounding
    above because the oracle's fp8 dtype cast is unimplemented on P800.
    """
    from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

    rotated = rope_tail(x, freqs, rope_dim, inverse)
    if compressed_kv:
        return _fake_quant_compressed_kv_torch(rotated)
    return fake_quant_fp4(rotated, block_size)


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
) -> torch.Tensor:
    """FP32 RMS(x) * RMS(key), weighted dot, sigmoid gate, value residual.

    x [T, HC, D], kv [T, (HC+1)*D] laid out head-major with the value段 last.
    """
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
    return (xf + gate * value.unsqueeze(1)).to(x.dtype)


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
    """Decode 期的 hash id + 原地推进 history，torch 版。

    上游把 commit 融进了 `_engram_hash_kernel`（`write_tokens=False` 且给
    `out_cache_loc`），`EngramHasher.forward` 在这条路上提前 return，所以只覆盖
    `engram_hash_ids` 不够：不注册这个符号，每个 decode step 都会走进 triton，
    报 `AssertionError: libcuda.so cannot found!`。

    commit 的表达式与 hasher 自己的 torch 分支一致：新行 = 该 token 及其 `width-1` 个
    最近前驱，oldest-first。`out_cache_loc == 0` 的行是 graph padding，不能落表。
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
        # 无条件逐元素，不做 host 同步、不产生数据相关形状，可以进 CUDA graph。
        # `out_cache_loc == 0` 的行是 graph padding，重定向到 history 末尾那行保留槽位
        # （`init_history` 按 `num_req_slots + 1` 分配，注释写的就是 "a spare row for
        # graph padding"，`pad_row = num_req_slots = history.shape[0] - 1`）。
        # 这和上游自己的 torch 分支同一套写法（`engram.py:371`:
        # `commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)`）。
        #
        # 先前用布尔掩码选行（`req_slots[out_cache_loc != 0]`）+ `rows.numel()` 判空，
        # 在 capture 期是数据相关形状 + host 读：实测 decode graph 捕获时在这一行
        # 申请 10.14 GiB 直接 OOM。
        pad_row = history.shape[0] - 1
        rows = torch.where(out_cache_loc == 0, pad_row, req_slots.to(torch.int64))
        history[rows] = tokens[:, :width].flip(-1).to(history.dtype)
    return out


@register_jit_op(
    "sglang.kernels.ops.embeddings.engram_hash", "engram_commit_history"
)
def engram_commit_history_kunlun(
    history: torch.Tensor,
    verify_ids: torch.Tensor,
    req_slots: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Anchor + accepted drafts slid into the per-request history ring, in torch.

    `history[slot]` then this batch's accepted ids form one window; the new row is
    `window[commit : commit + width]`. Same expression as
    `EngramHasher.commit_after_verify`'s torch fallback.
    """
    width = history.shape[1]
    if req_slots.numel() == 0 or width == 0:
        return
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
    history[req] = window.gather(1, cols)


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
) -> torch.Tensor:
    """Gather fp8 rows over a host pointer and dequantize with e8m0 block scales.

    `weight_ptr`/`scale_ptr` address host memory (the engram host table), so the
    rows are read through ctypes and only the gathered values cross to the
    device. Ids outside [row_lo, row_hi) produce a zero row.
    """
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


def _ceil_ue8m0_exp_torch(sf: torch.Tensor) -> torch.Tensor:
    """`_ceil_ue8m0_exp`: biased exponent, rounded up when a mantissa is present,
    clamped to [1, 254]."""
    bits = sf.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code_torch(v: torch.Tensor, rne: bool) -> torch.Tensor:
    """`_fp4_e2m1_code` / `_fp4_e2m1_code_rne`: e2m1 code with the sign in bit 3."""
    ax = v.abs().clamp_max(6.0)
    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    if rne:
        idx = torch.zeros_like(ax, dtype=torch.int32)
        is_boundary = torch.zeros_like(ax, dtype=torch.bool)
        for t in thresholds:
            idx = idx + (ax >= t).to(torch.int32)
            is_boundary = is_boundary | (ax == t)
        # Round-half-to-even: an odd index exactly on a boundary drops to the even one.
        idx = torch.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    else:
        idx = torch.zeros_like(ax, dtype=torch.int32)
        for t in thresholds:
            idx = idx + (ax > t).to(torch.int32)
    sign = ((v < 0) & (idx != 0)).to(torch.int32)
    return idx | (sign << 3)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer", "quantize_fp4_indexer_tensor"
)
def quantize_fp4_indexer_tensor_kunlun(
    x: torch.Tensor, rne: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-32 UE8M0 FP4 quantize into (packed int8 [n, 64], packed scales [n] int32).

    Four 32-element groups per 128-wide row; group scale is
    ``ceil_pow2_exp(max(|x|)/6)`` stored as one byte each in the int32, and the
    payload packs two 4-bit e2m1 codes per byte (low nibble = even column).
    """
    assert x.shape[-1] == 128
    x = x.contiguous().view(-1, 128)
    n = x.shape[0]
    x_fp4 = torch.empty((n, 64), device=x.device, dtype=torch.int8)
    x_sf = torch.empty((n,), device=x.device, dtype=torch.int32)
    if n == 0:
        return x_fp4, x_sf
    xf = x.to(torch.float32).view(n, 4, 32)
    sf = (xf.abs().amax(dim=2) / 6.0).clamp_min(1.0e-4)
    exp = _ceil_ue8m0_exp_torch(sf)
    x_sf.copy_(
        exp[:, 0] | (exp[:, 1] << 8) | (exp[:, 2] << 16) | (exp[:, 3] << 24)
    )
    scale = (exp << 23).view(torch.float32).unsqueeze(-1)
    code = _fp4_e2m1_code_torch(xf / scale, rne).view(n, 64, 2)
    x_fp4.copy_(((code[..., 0] & 0x0F) | ((code[..., 1] & 0x0F) << 4)).to(torch.int8))
    return x_fp4, x_sf


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer", "store_fp4_index_k_cache"
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
    page_offset*4``."""
    assert input.shape[-1] == 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor_kunlun(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    if n_tokens == 0:
        return
    assert cache.shape[1] == page_size * (64 + 4)
    u8 = cache.view(torch.uint8)
    flat = u8.reshape(-1)
    loc = loc.to(torch.int64)
    page = loc // page_size
    page_offset = loc - page * page_size

    payload_base = page * u8.stride(0) + page_offset * 64
    payload_idx = payload_base[:, None] + torch.arange(64, device=cache.device)
    flat[payload_idx] = k_fp4.view(torch.uint8)

    scale_base = page * u8.stride(0) + page_size * 64 + page_offset * 4
    scale_idx = scale_base[:, None] + torch.arange(4, device=cache.device)
    scale_bytes = torch.stack(
        [(k_sf >> (8 * i)) & 0xFF for i in range(4)], dim=1
    ).to(torch.uint8)
    flat[scale_idx] = scale_bytes


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
    if freqs_cis is not None:
        # V4.1 的 CUDA kernel 把 64 维 RoPE 尾部的旋转折进了 store，让 fp4/fp8 只 round
        # 一次。昆仑池是 plain fp16，没有可融合的量化——但此时上游传进来的是**未旋转**的
        # latent，静默存下去就是错值路径。
        raise NotImplementedError(
            "fused_store_cache(freqs_cis=...) reaches the Kunlun half-cache path; "
            "the caller must pass the already-rotated latent (freqs_cis=None)"
        )
    if cache.dtype in (torch.float16, torch.bfloat16):
        # 昆仑半 cache 布局。`DeepSeekV4SingleKVPool` 在本平台被 hook 过
        # （`hooks/mem_cache/common.py::dsv4_create_buffer_kunlun` /
        # `dsv4_get_bytes_per_token_kunlun`）：一页是
        # `page_size * (qk_nope_head_dim + qk_rope_head_dim)` 个 `store_dtype` **元素**，
        # 即 plain fp16/bf16 latent，不是 N 卡 FlashMLA 的 584 B/token 打包
        # （448 fp8 nope + 7 ue8m0 scale + 1 pad + 128 B bf16 rope）。XSpeedGate 两侧
        # （写 `set_k_and_s_v4_with_mapping`、读 `compressed_attention`）也是这个布局。
        #
        # 把 FlashMLA 字节打包写进这个 buffer，就是当初每个 CSA2（ratio 1/2）层 attention
        # 输出 NaN 的原因：压缩 KV 池只有 `set_extra_key_buffer_fused` →
        # `set_key_buffer_fused` → 这里一条写入路径，布局不匹配的表现是池子里的行接近
        # fp16 上限（≈65000）且带 NaN，然后是最终 hidden state 全 0、logits 完全均匀。
        dim = input.shape[-1]
        num_pages = cache.shape[0]
        assert cache.numel() == num_pages * page_size * dim, (
            f"half-cache page mismatch: cache{tuple(cache.shape)} vs "
            f"page_size={page_size} dim={dim}"
        )
        loc = indices.to(torch.int64).reshape(-1)
        page = loc // page_size
        slot = loc - page * page_size
        view = cache.view(num_pages, page_size, dim)
        view[page, slot] = input.to(cache.dtype)
        return
    u8 = cache.view(torch.uint8)
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
