"""DSA mixed INT8 KV-cache row: ``[nope | fp32 scales | bf16 rope]``.

Port of ``sglang_kunlun/kernels/dsv4_mixed_kv.py`` (0.5.17) onto the DSA/NSA
pool this tree owns.  Per-token layout for the target model
(``kv_lora_rank=512``, ``qk_rope_head_dim=64``, ``quant_block_size=128``)::

    nope   kv_lora_rank                            int8    512 B
    scale  kv_lora_rank // quant_block_size * 4     fp32     16 B
    rope   qk_rope_head_dim * 2                     bf16    128 B
                                                    total   656 B

``nope`` here is the *compressed latent* (``kv_lora_rank``), not
``qk_nope_head_dim``. The two are different quantities and only coincide for
the DSV4 single-KV pool 0.5.17 was written against (448/64/64 -> 604 B). Every
call site passes the pool field it actually stores, and the width has no
default, so a copy of this module cannot silently inherit the wrong tiling.

Two differences from the 0.5.17 module, both deliberate:

* **No ``move_scale_to_tail``.** The 0.5.17 DSV4 readers derive
  ``[nope | rope | scale]``, so that tree had to rotate the quantiser's tail.
  Measured on this image with ``kunlun_ops.quantize_mla_kv_cache_split`` at
  512/64/128: the row it emits is already ``[nope | scale | rope]``. Reading
  the same bytes at the other offsets costs max|err| 1.0e19, and rotating the
  tail the way ``move_scale_to_tail`` did costs 1.7e38, against 0.017 for the
  order below -- so there is nothing to rotate. This is also the row 0.5.8
  production reads (`memory_pool.py`
  ``[int8 nope(512) | fp32 scales(4*4) | bf16 rope(64*2)]``). The writer is
  therefore a single kernel call and the reader owns the order.
* **The reader is parameterised**, not pinned to module constants, because a
  read has to work for whatever dims the pool was built with.
"""

from __future__ import annotations

import torch


def mixed_int8_bytes_per_token(nope_dim: int, rope_dim: int, group_size: int) -> int:
    """Return the byte-packed mixed INT8 footprint of one token.

    ``nope_dim`` is the width of the *stored* noPE part: ``kv_lora_rank`` for
    the DSA pool (``KunlunNSATokenToKVPool``) and ``qk_nope_head_dim`` for the
    0.5.17 DSV4 single-KV pool.

    ``nope_dim % 4`` is required, not cosmetic: the trailing fp32 scales are
    materialised by viewing the row at byte offset ``nope_dim`` as
    ``torch.float32``, and a torch dtype view rejects an unaligned offset.
    """
    if nope_dim % group_size:
        raise ValueError(
            f"nope_dim={nope_dim} is not divisible by group_size={group_size}"
        )
    if nope_dim % 4:
        raise ValueError(
            f"nope_dim={nope_dim} is not a multiple of 4; the fp32 scale slice "
            "would start at an unaligned byte offset"
        )
    return nope_dim + nope_dim // group_size * 4 + rope_dim * 2


def mixed_int8_scale_bytes(nope_dim: int, group_size: int) -> int:
    """Bytes of the fp32 per-group scales that follow the int8 noPE part."""
    return nope_dim // group_size * 4


def write_mixed_int8_rows(
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    dst_rows: torch.Tensor,
    loc: torch.Tensor,
    group_size: int,
) -> None:
    """Quantize one MLA write into the mixed INT8 rows at ``loc``.

    ``dst_rows`` is the pool byte-viewed as ``[slots, stride]`` int8 and only
    the rows named by ``loc`` are touched; the fp32 scales land inside each
    written row, which is why the caller never sees a ``kv_scale_buffer``.

    The rope half is kept in bf16: the kernel stores it as raw input bytes
    while every reader interprets those bytes as bf16, so a model served in
    fp16 has to be cast up here. Casting the whole pack is also what the 0.5.8
    production write does (``DSATokenToKVPool.set_mla_kv_buffer``,
    ``.to(torch.bfloat16)`` on the concatenated nope+rope) -- the noPE half is
    quantized right after, so its cast is free.
    """
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = 0 if cache_k_rope is None else cache_k_rope.shape[-1]
    stride = mixed_int8_bytes_per_token(nope_dim, rope_dim, group_size)
    if dst_rows.ndim != 2 or dst_rows.shape[-1] != stride:
        raise ValueError(
            f"expected [N, {stride}] int8 cache rows, got "
            f"{tuple(dst_rows.shape)} of {dst_rows.dtype}"
        )
    num_tokens = cache_k_nope.shape[0]
    packed = torch.cat(
        [
            cache_k_nope.reshape(num_tokens, -1),
            cache_k_rope.reshape(num_tokens, -1),
        ],
        dim=-1,
    ).to(torch.bfloat16).contiguous()
    # Clamp both ends the way the 0.5.17 writer does: `swa_loc` can carry a
    # stale/garbage value, and the vendor op skips negatives but would still
    # scribble on a too-large positive index.
    loc_i32 = loc.clamp(min=0, max=dst_rows.shape[0] - 1).to(torch.int32).contiguous()

    import kunlun_ops

    kunlun_ops.quantize_mla_kv_cache_split(
        packed,
        loc_i32,
        dst_rows,
        nope_dim,
        rope_dim,
        group_size,
    )


def unpack_mixed_int8(
    cache: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
    group_size: int,
    dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize packed mixed INT8 rows back to a dense ``[N, nope + rope]``.

    Inverse of :func:`write_mixed_int8_rows`, and the only way this tree can
    feed the packed rows to ``xspeedgate_ops.sparse_attn_fwd``: that op has no
    scale argument and no int8 instantiation (its schema is
    ``(q, kv, indices, sm_scale, d_v, topk_length, attn_sink=None)`` and it
    requires ``kv.dtype == q.dtype``), so a mixed row has to arrive as plain
    dequantized values. The 0.5.8/0.5.17 kernels that read the row natively
    (``sparse_mixed_attn_int8`` / ``hybrid_attention_with_mixed_cache``) take
    the scales from inside the row and are a different entry point.

    ``cache`` is the pool byte-viewed as ``[slots, stride]`` int8. ``out`` lets
    the caller hand in a reused buffer: the dense table is a full copy of one
    layer's pool, so a per-call allocation inside a captured graph would pin one
    such copy per capture bucket. When ``out`` is given the returned tensor
    aliases it and is only valid until the next call that writes it.
    """
    scale_bytes = mixed_int8_scale_bytes(nope_dim, group_size)
    stride = mixed_int8_bytes_per_token(nope_dim, rope_dim, group_size)
    if cache.ndim != 2 or cache.shape[-1] != stride:
        raise ValueError(
            f"expected [N, {stride}] int8 cache rows, got {tuple(cache.shape)} "
            f"of {cache.dtype}"
        )
    num_groups = nope_dim // group_size
    scale = (
        cache[:, nope_dim : nope_dim + scale_bytes]
        .contiguous()
        .view(torch.float32)
        .reshape(-1, num_groups)
    )
    out_shape = (cache.shape[0], nope_dim + rope_dim)
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=cache.device)
    elif out.shape != out_shape or out.dtype is not dtype:
        raise ValueError(
            f"scratch must be {out_shape} of {dtype}, got "
            f"{tuple(out.shape)} of {out.dtype}"
        )
    # One fp32 staging buffer for the noPE half, dequantized in place; the
    # copy into `out` is where the cast to the reader's dtype happens.
    nope = cache[:, :nope_dim].to(torch.float32).reshape(-1, num_groups, group_size)
    nope.mul_(scale.unsqueeze(-1))
    out[:, :nope_dim] = nope.reshape(-1, nope_dim)
    if rope_dim:
        out[:, nope_dim:] = (
            cache[:, nope_dim + scale_bytes : stride]
            .contiguous()
            .view(torch.bfloat16)
        )
    return out
