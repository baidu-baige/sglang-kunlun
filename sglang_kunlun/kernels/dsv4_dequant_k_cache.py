# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun replacements for the DSV4 paged-KV-cache dequant kernels.

``sglang.kernels.ops.attention.dsv4.dequant_k_cache`` lowers ``tl.exp2`` to a
libm ``ldexpf`` call, which the XPU device library ``libdevice-xpu3.bc`` does
not define; the kernels then fail at ``xpu3-elfconv`` link time and every
scheduler dies on the first forward.

Two exponentials are involved, and both are reproduced without ``exp2``:

* a ue8m0 scale byte is exactly ``2 ** (byte - 127)`` — a bit shift (the
  ``_ue8m0_to_fp32`` helper, copied verbatim);
* an e2m1 magnitude exponent is one of ``{0.5, 1, 2, 4}`` — a small
  ``tl.where`` chain.

The kernels are registered ``call_style="direct"`` so the upstream wrappers,
which launch them as ``kernel[grid](...)``, pick the Kunlun kernels up with no
change to the wrapper. ``token_req_indices`` (the ``-714`` pad fix) is a plain
Python helper and is registered as a JIT op.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang_kunlun.kernels.kernel_ops import register_jit_op, register_triton_op

_DEQUANT_MODULE = "sglang.kernels.ops.attention.dsv4.dequant_k_cache"


@triton.jit
def _ue8m0_to_fp32(scale_u8):
    # Follows torch.float8_e8m0fnu: byte 0 is the denormal 2 ** -127, byte 255
    # is NaN, otherwise 2 ** (byte - 127).
    normal = (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    denormal = tl.full(scale_u8.shape, 0x00400000, tl.int32).to(
        tl.float32, bitcast=True
    )
    scale = tl.where(scale_u8 == 0, denormal, normal)
    return tl.where(scale_u8 == 255, float("nan"), scale)


@triton.jit
def _e2m1_code_to_fp32(code):
    # The 4-bit e2m1 code: bit 3 is the sign, bits 0-2 index into
    # [0, 0.5, 1, 1.5, 2, 3, 4, 6]. The magnitude's exponent e = m >> 1 is one
    # of {0,1,2,3}, so 2 ** (e - 1) is a four-way select — no tl.exp2 (which
    # would lower to the missing ldexpf).
    m = code & 7
    e = m >> 1
    f = (m & 1).to(tl.float32)
    pow2 = tl.where(
        e == 0, 0.5, tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
    )
    mag = tl.where(e == 0, 0.5 * f, pow2 * (1.0 + 0.5 * f))
    # Set the sign bit directly: a negated zero must stay -0.0 (code 0x8).
    sign = (code & 8).to(tl.int32) << 28
    return (mag.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@register_triton_op(_DEQUANT_MODULE, "_dequantize_k_cache_paged_kernel",
                    metadata={"call_style": "direct"})
@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = _ue8m0_to_fp32(scale_u8)

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


@register_triton_op(_DEQUANT_MODULE, "_dequantize_k_cache_paged_v41_fp8_kernel",
                    metadata={"call_style": "direct"})
@triton.jit
def _dequantize_k_cache_paged_v41_fp8_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # V41: 512 e4m3 values per token, then 16 ue8m0 scales (one per 32 values).
    tl.static_assert(DATA_BYTES == 512 and SCALE_BYTES == 16 and TILE_SIZE == 32)
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * DATA_BYTES
    token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * SCALE_BYTES

    offs = tl.arange(0, DATA_BYTES)
    vals = tl.load(buf_fp8_ptr + token_data_base + offs).to(tl.float32)
    scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + offs // TILE_SIZE)
    out = vals * _ue8m0_to_fp32(scale_u8)
    tl.store(
        output_ptr + token_id * output_stride_0 + offs,
        out.to(output_ptr.dtype.element_ty),
    )


@register_triton_op(_DEQUANT_MODULE, "_dequantize_k_cache_paged_v41_fp4_kernel",
                    metadata={"call_style": "direct"})
@triton.jit
def _dequantize_k_cache_paged_v41_fp4_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # V41_FP4: 512 e2m1 codes packed two per byte (even index in the low nibble),
    # then 32 e4m3 scales (one per 16 values).
    tl.static_assert(DATA_BYTES == 256 and SCALE_BYTES == 32 and TILE_SIZE == 16)
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * DATA_BYTES
    token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * SCALE_BYTES

    boffs = tl.arange(0, DATA_BYTES)
    packed = tl.load(buf_uint8_ptr + token_data_base + boffs)
    # Byte j holds elements 2j (low nibble) and 2j + 1, both in tile (2j) // 16.
    scale = tl.load(buf_fp8_ptr + token_scale_base + (2 * boffs) // TILE_SIZE).to(
        tl.float32
    )
    lo = _e2m1_code_to_fp32(packed & 0xF) * scale
    hi = _e2m1_code_to_fp32(packed >> 4) * scale
    out_base = output_ptr + token_id * output_stride_0
    tl.store(out_base + 2 * boffs, lo.to(output_ptr.dtype.element_ty))
    tl.store(out_base + 2 * boffs + 1, hi.to(output_ptr.dtype.element_ty))


@register_triton_op(_DEQUANT_MODULE, "_gather_dequant_requant_fp8_paged_kernel",
                    metadata={"call_style": "direct"})
@triton.jit
def _gather_dequant_requant_fp8_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = _ue8m0_to_fp32(scale_u8)

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(
        output_ptr + out_row_base + DIM_NOPE + rope_offs,
        rope_data.to(output_ptr.dtype.element_ty),
    )


_TOKEN_REQ_CACHE = None


def _host_token_req_index(forward_batch, lens_cpu, total, num_tokens, device):
    """``repeat_interleave(req, extend_seq_lens)`` built as a host list.

    One ``repeat_interleave_fwd<long>`` launch costs 178 us at the 8x60k shape
    -- launch bound, not bandwidth bound: the whole expansion is 4096 int64
    (32 KB), and the eight indexer layers of a forward ask for the identical
    tensor, so the device tensor is kept in a one-entry cache keyed on every
    input.  Returns ``None`` (caller keeps ``repeat_interleave``) when the
    ``req_pool_indices_cpu`` mirror is missing or disagrees with the host
    total.
    """
    rows_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
    if rows_cpu is None or rows_cpu.numel() < len(lens_cpu):
        return None
    rows = [int(r) for r in rows_cpu.tolist()[: len(lens_cpu)]]
    key = (
        str(device),
        total,
        num_tokens,
        tuple(rows),
        tuple(int(n) for n in lens_cpu),
    )
    global _TOKEN_REQ_CACHE
    hit = _TOKEN_REQ_CACHE
    if hit is not None and hit[0] == key:
        return hit[1]
    ids = []
    for r, n in zip(rows, lens_cpu):
        n = int(n)
        if n > 0:
            ids.extend([r] * n)
    if len(ids) != total or not ids:
        return None
    if num_tokens is not None and total < num_tokens:
        ids.extend([ids[-1]] * (num_tokens - total))
    out = torch.tensor(ids, dtype=torch.int64, device=device)
    _TOKEN_REQ_CACHE = (key, out)
    return out


@register_jit_op("sglang.srt.layers.attention.dsv4.dsv41_sparse", "token_req_indices")
def token_req_indices(forward_batch, *, num_tokens=None) -> torch.Tensor:
    """Kunlun ``token_req_indices`` — the ``-714`` pad fix.

    With the all-to-all on, ``ForwardBatch.prepare_mlp_sync_batch`` rounds
    ``num_tokens`` (positions) up to an ``ep_size`` multiple, but pads
    ``extend_seq_lens`` only to the batch size, so ``sum(extend_seq_lens) <
    num_tokens``. Upstream's ``repeat_interleave(..., output_size=num_tokens)``
    then leaves an uninitialised tail, and the downstream
    ``index_select_mt<int64>`` gathers out of bounds (device ``-714``). Repeat
    the last valid index over the pad rows, exactly as ``ExpandPrefillCausally``
    already does.
    """
    req = forward_batch.req_pool_indices.to(torch.int64)
    if forward_batch.forward_mode.is_decode():
        return req
    if forward_batch.forward_mode.is_target_verify():
        block = int(forward_batch.spec_info.draft_token_num)
        # Eager PD verify pads the token rows to a tensor-parallel tier. The
        # padding is *global*: bs * block real rows followed by the trailing
        # tier rows, so distributing num_tokens over the requests is wrong
        # whenever it is not divisible by bs (5 requests x 6 rows -> 32).
        # Build the real rows first and repeat the last request's index over
        # the trailing rows: this function only needs an in-range index there
        # (the -714 fix), and the padded rows' outputs are discarded.
        indices = torch.repeat_interleave(req, block)
        if num_tokens is not None and req.numel() > 0:
            if indices.numel() < num_tokens:
                indices = torch.cat(
                    [indices, req[-1:].repeat(num_tokens - indices.numel())]
                )
            elif indices.numel() > num_tokens:
                indices = indices[:num_tokens]
        return indices
    assert forward_batch.forward_mode.is_extend(), (
        "the V4.1 torch attention path serves extend, target-verify and decode"
    )
    extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int64)
    # ``repeat_interleave`` without ``output_size`` makes ATen compute
    # ``repeats.sum().item()`` -- two device synchronises per call, 8 calls per
    # forward.  ``extend_seq_lens_cpu`` is the host mirror of the same list that
    # the batch builder already publishes (it is padded in lock-step with the
    # device tensor), so the total can be read from the host instead.  The
    # gpu-only path hands device tensors in directly and leaves the mirror
    # unset: fall back to the old expression there.
    lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    total = None
    if lens_cpu is not None and len(lens_cpu) == extend_seq_lens.numel():
        host_total = int(sum(lens_cpu))
        # ``sum(lens) <= num_tokens`` always holds for a real batch; a mirror
        # that reads larger is stale (e.g. a captured batch view) and the
        # device expression is used instead.
        if host_total >= 0 and (
            num_tokens is None or host_total <= num_tokens
        ):
            total = host_total
    if total is None:
        out = torch.repeat_interleave(req, extend_seq_lens)
        if num_tokens is not None and out.numel() < num_tokens:
            out = torch.cat([out, out[-1:].expand(num_tokens - out.numel())])
        return out
    # ``repeat_interleave`` with a tensor ``repeats`` decomposes into
    # ``cumsum`` -> ``index_select`` -> ``copy_`` on this build (11.4 ms of
    # device time per 8x60k capture, 178 us per call); the same permutation is
    # a 32 KB host list that all eight indexer layers of a forward share.
    built = _host_token_req_index(forward_batch, lens_cpu, total, num_tokens, req.device)
    if built is not None:
        return built
    out = torch.repeat_interleave(req, extend_seq_lens, output_size=total)
    if num_tokens is not None and total < num_tokens:
        out = torch.cat([out, out[-1:].expand(num_tokens - total)])
    return out
