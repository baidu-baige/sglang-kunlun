"""Kunlun replacements for the DSA kpool host helpers.

GLM-5-Next's DSA indexer runs with ``index_kpool = 4``, so every kpool code path
in ``sglang.srt.layers.attention.dsa`` is on the critical path. Those helpers
launch Triton kernels, and Triton cannot run on P800: because Kunlun reports
itself as CUDA, ``triton`` picks the nvidia backend and dies in
``parse_options`` with ``'_CudaTarget' object has no attribute 'arch'`` - a
compile-time failure, so it cannot be masked with an env var.

Each replacement here is a plain-torch port of the host wrapper (not the
kernel), which keeps the patch surface at one symbol per helper. These are
correctness-first ports; hot ones get mapped onto ``kunlun_ops`` /
``xspeedgate_ops`` once the whole path runs.
"""

from typing import Tuple

import torch

import xspeedgate_ops  # noqa: F401  -- registers torch.ops.xspeedgate_ops.*

from sglang_kunlun.kernels.kernel_ops import register_jit_op

_KPOOL_MODULE = "sglang.srt.layers.attention.dsa.kpool_fp8_index"


def _u8(buf: torch.Tensor) -> torch.Tensor:
    """Present the paged index cache as uint8, which every writer op requires.

    The draft (EAGLE) model's index buffer comes back as int8 while the target
    model's is uint8. ``view`` shares storage, so the ops still write in place.
    """

    return buf if buf.dtype == torch.uint8 else buf.view(torch.uint8)


@register_jit_op(_KPOOL_MODULE, "kpool_build_ragged_layout")
def kpool_build_ragged_layout_kunlun(
    full_page_table: torch.Tensor,
    cu_pages_excl: torch.Tensor,
    ragged_pool_pages: torch.Tensor,
    cu_q_len_excl: torch.Tensor,
    ragged_q_len: torch.Tensor,
    pooled_seq_lens_expanded: torch.Tensor,
    slots_per_page: int,
    total_pool_pages: int,
    total_q: int,
    pool_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``xspeedgate_ops`` replacement for ``kpool_build_ragged_layout``.

    A pooled page is every ``pool_size``-th column of the real page table, so
    row ``k`` contributes ``ragged_pool_pages[k]`` pages starting at
    ``cu_pages_excl[k]``. Each q row of that group sees keys in
    ``[page_start * slots_per_page, min(start + pooled_len, end_of_pages))``.
    Positions outside the written ranges are left uninitialized, matching the
    Triton version.

    The op's argument order and the three returned tensors line up with the
    upstream host wrapper, and it normalizes the five metadata tensors to
    contiguous int32 itself, so nothing is cast here.
    """

    return torch.ops.xspeedgate_ops.kpool_build_ragged_layout(
        full_page_table,
        cu_pages_excl,
        ragged_pool_pages,
        cu_q_len_excl,
        ragged_q_len,
        pooled_seq_lens_expanded,
        slots_per_page,
        total_pool_pages,
        total_q,
        pool_size,
    )


@register_jit_op(_KPOOL_MODULE, "kpool_assemble_softmax_rotate_write_cache")
def kpool_assemble_softmax_rotate_write_cache_kunlun(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor = None,
    round_scale: bool = False,
) -> None:
    """``xspeedgate_ops`` replacement for ``kpool_assemble_softmax_rotate_write_cache``.

    Each output row pools ``index_kpool`` index-keys: the first ``n_from_tail``
    come from the request's circular tail ring, the rest from this step's chunk
    buffer. Scores get the learned per-slot APE added, then a *per channel*
    softmax over the pool's slots weights the keys; the pooled key is rotated and
    int8-quantized into the paged index cache (upstream stores fp8, which P800
    cannot represent - the ops store int8 with ``scale = absmax / 127`` instead).

    The op writes ``buf`` in place and returns nothing, matching the upstream
    host wrapper. It is strict about dtypes: the index-plan tensors must be int32
    and ``ape`` fp32, so they are cast here rather than at the call sites.
    ``round_scale=True`` is rejected by the op; that error is the correct
    outcome, so the flag is forwarded unchanged instead of being silenced.
    """

    if req_pool_idx.shape[0] == 0:
        return

    def _plan_i32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32).contiguous()

    # tail_k / tail_score / chunk_k / chunk_score must share one dtype. The tail
    # ring is persistent bf16; under --dtype float16 the chunk key/score arrive
    # as Half, so coerce them to the buffers' dtype (bf16 index path throughout).
    _kdtype = tail_k.dtype
    torch.ops.xspeedgate_ops.kpool_assemble_softmax_rotate_write_cache(
        _u8(buf),
        tail_k.contiguous(),
        tail_score.contiguous(),
        chunk_k.to(_kdtype).contiguous(),
        chunk_score.to(_kdtype).contiguous(),
        ape.to(torch.float32).contiguous(),
        _plan_i32(req_pool_idx),
        _plan_i32(n_from_tail),
        _plan_i32(chunk_src_start),
        _plan_i32(tail_logical_base),
        loc.contiguous(),
        write_mask.contiguous() if write_mask is not None else None,
        pool.slots_per_page,
        round_scale=False,
    )

@register_jit_op(_KPOOL_MODULE, "kpool_write_tail_and_maybe_compress")
def kpool_write_tail_and_maybe_compress_kunlun(
    pool,
    buf: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    round_scale: bool,
    effective_n_per_batch: torch.Tensor = None,
) -> None:
    """``xspeedgate_ops`` replacement for ``kpool_write_tail_and_maybe_compress``.

    Two steps per request: append this step's index key/score to the circular
    tail ring, then, for every pool the append just closed, pool it (channel-wise
    softmax over the ring slots plus APE), rotate, quantize and write it to the
    index cache. Requests whose first ``out_cache_loc`` is 0 are skipped, matching
    the kernel's padding convention.

    ``pool`` is not an op argument - ``pool_size`` comes from ``ape.size(0)``,
    ``slots_per_page`` from ``buf.size(1) / 132`` and ``bs`` from
    ``key.size(0) / num_draft_tokens``.

    Same three dtype/flag contracts as the decode-side op, tighter than upstream's:
    ``score`` must match ``key``'s dtype (upstream allows any of
    ``KPOOL_SCORE_DTYPES``); ``req_pool_indices`` / ``write_start`` /
    ``tail_logical_start`` / ``out_cache_loc`` / ``effective_n_per_batch`` must all
    share one dtype, so they are normalized to int64; and ``round_scale=True`` is
    rejected by the int8 path, so it is forced off.
    """

    if key.shape[0] == 0:
        return

    # The op requires bn % num_draft_tokens == 0 (it derives bs that way), and so
    # does the upstream host wrapper. The kunlun MTP path still reaches here with
    # fewer rows than one draft block from ``_draft_extend_for_decode`` (the draft
    # runner gets the real accept length, not a padded block), where
    # ``bs = bn // num_draft_tokens`` is 0. The torch port this replaced skipped
    # those calls as well - its ``rows_b`` came out empty - so keep that behaviour
    # rather than feed the op a plan that does not describe this batch. Only the
    # draft model's own index cache is affected, i.e. proposal quality.
    if key.shape[0] % num_draft_tokens != 0:
        return

    def _i64(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int64).contiguous()

    # tail_k / tail_score / key / score must all share one dtype. The tail ring
    # is persistent bf16 (the DSA index path is bf16 throughout); under
    # --dtype float16 the incoming key/score are Half, so coerce them down to
    # the buffers' dtype -- same as scatter_kpool_tail_updates / the decode op.
    _kdtype = tail_k.dtype
    torch.ops.xspeedgate_ops.kpool_write_tail_and_maybe_compress(
        _u8(buf),
        tail_k,
        tail_score,
        key.to(_kdtype).contiguous(),
        score.to(_kdtype).contiguous(),
        ape.to(torch.float32).contiguous(),
        _i64(req_pool_indices),
        _i64(write_start),
        _i64(tail_logical_start),
        write_loc.contiguous(),
        _i64(out_cache_loc),
        _i64(effective_n_per_batch) if effective_n_per_batch is not None else None,
        num_draft_tokens,
        round_scale=False,
    )


@register_jit_op(_KPOOL_MODULE, "update_kpool_write_plan_cuda_graph")
def update_kpool_write_plan_cuda_graph_kunlun(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out=None,
    seqlens_per_q_out=None,
    *,
    pool_size: int,
    num_draft_tokens: int,
    slots_per_page: int,
) -> None:
    """Fills the per-request kpool write plan in place: which pool the next write
    starts at, where that pool's tail begins, and the physical cache location of
    each closed pool. All outputs are preallocated by the caller (the name says
    cuda_graph because the buffers are reused across replays).
    """

    if write_start.shape[0] == 0:
        return

    torch.ops.xspeedgate_ops.update_kpool_write_plan(
        write_start.contiguous(),
        req_pool_indices.contiguous(),
        real_page_table.contiguous(),
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out,
        seqlens_per_q_out,
        pool_size=pool_size,
        num_draft_tokens=num_draft_tokens,
        slots_per_page=slots_per_page,
    )


@register_jit_op(_KPOOL_MODULE, "kpool_softmax_rotate_write_cache")
def kpool_softmax_rotate_write_cache_kunlun(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor = None,
    round_scale: bool = False,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """``xspeedgate_ops`` replacement for ``kpool_softmax_rotate_write_cache``.

    Same pooling math as the extend-side assemble port, but the pool slots are
    already gathered into ``slot_k`` / ``slot_score`` ``[rows, pool_size, 128]``
    and ``loc`` is a cache slot number.

    Two differences from the upstream host wrapper are bridged here. The op takes
    the compressed outputs as pre-allocated in/out tensors rather than a
    ``return_compressed`` flag, so they are allocated and returned here - as int8,
    not fp8, since P800 has no fp8 (``scale = absmax / 127``, the same convention
    as the other kpool writers). And ``page_size``
    is an explicit argument instead of being read off a ``pool`` object.
    """

    rows, _, head_dim = slot_k.shape
    if rows == 0:
        if not return_compressed:
            return None
        return (
            torch.empty((0, head_dim), dtype=torch.int8, device=slot_k.device),
            torch.empty((0,), dtype=torch.float32, device=slot_k.device),
        )

    compressed_k = compressed_scale = None
    if return_compressed:
        compressed_k = torch.empty(
            (rows, head_dim), dtype=torch.int8, device=slot_k.device
        )
        compressed_scale = torch.empty(
            (rows,), dtype=torch.float32, device=slot_k.device
        )

    torch.ops.xspeedgate_ops.kpool_softmax_rotate_write_cache(
        _u8(buf),
        slot_k.contiguous(),
        slot_score.contiguous(),
        ape.to(torch.float32).contiguous(),
        loc.contiguous(),
        write_mask.contiguous() if write_mask is not None else None,
        compressed_k,
        compressed_scale,
        pool.page_size,
        round_scale,
        write_cache,
    )

    if return_compressed:
        return compressed_k, compressed_scale
    return None



@register_jit_op(_KPOOL_MODULE, "gather_index_k_scale_prefix_into")
def gather_index_k_scale_prefix_into_kunlun(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    """``xspeedgate_ops`` replacement for ``gather_index_k_scale_prefix_into``.

    Gathers the first ``seq_len`` tokens of one request's index cache out of the
    paged buffer into dense ``k_out`` (quantized bytes) / ``scale_out`` (fp32).

    The op fills both outputs in place and leaves rows past ``seq_len`` untouched,
    matching the upstream kernel. ``head_dim`` and ``page_size`` come from
    ``k_out.size(1)`` and the pool, and it requires ``k_out``'s dtype to equal
    ``buf``'s (the copy is bitwise), hence the view rather than ``_u8``.

    The upstream kernel also honours ``PRESHUFFLE_TILE``, an AITER/ROCm-only
    layout (``_preshuffle_tile()`` returns 0 unless
    ``aiter_can_use_preshuffle_paged_mqa()``), so the linear layout assumed here
    is the one this platform gets.
    """

    if seq_len == 0:
        return

    torch.ops.xspeedgate_ops.gather_index_k_scale_prefix_into(
        buf.contiguous().view(k_out.dtype),
        page_indices.contiguous(),
        seq_len,
        k_out,
        scale_out,
        pool.page_size,
    )


@register_jit_op(_KPOOL_MODULE, "scatter_kpool_tail_updates")
def scatter_kpool_tail_updates_kunlun(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """Copies this step's unpooled tail keys / scores into each request's circular
    tail ring; row ``r`` writes its first ``n_write[r]`` slots only.

    The op derives ``pool_size`` from ``tail_k.shape[1]`` (both are 4 in the
    GLM-5-Next config, i.e. ``pool.index_kpool``), so ``pool`` is unused.
    """

    if req_pool_idx.shape[0] == 0:
        return

    dtype = tail_k.dtype
    torch.ops.xspeedgate_ops.scatter_kpool_tail_updates_kernel(
        chunk_k.to(dtype).contiguous(),
        chunk_score.to(dtype).contiguous(),
        tail_k,
        tail_score,
        req_pool_idx.contiguous(),
        dst_logical_start.contiguous(),
        chunk_src_start.contiguous(),
        n_write.contiguous(),
    )


@register_jit_op(_KPOOL_MODULE, "kpool_decode_update_and_maybe_write_cache")
def kpool_decode_update_and_maybe_write_cache_kunlun(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool = False,
) -> None:
    """``xspeedgate_ops`` replacement for ``kpool_decode_update_and_maybe_write_cache``.

    One decode token per request: append its index key / score into the circular
    tail ring, and whenever that token closes a pool (``pos % pool_size ==
    pool_size - 1``) pool the ring's ``pool_size`` slots and write the rotated,
    quantized result into the index cache.

    ``pool`` is not an op argument - ``pool_size`` comes from ``ape.size(0)`` and
    ``slots_per_page`` from ``buf.size(1) / 132``, both of which the caller's
    tensors already carry.

    Three input contracts are tighter than the upstream host wrapper's, and each
    is met here rather than left to fail:

    - ``req_pool_indices`` / ``positions`` / ``seq_lens`` / ``out_cache_loc`` must
      all share one dtype, but the decode caller mixes int64 tensors with
      ``get_seqlens_int32()``, so all four are normalized to int64.
    - ``tail_k`` / ``tail_score`` / ``slot_score`` must all share ``key``'s
      dtype. The tail ring buffers are persistent and allocated bf16 (the DSA
      index path is bf16 throughout -- the KV cache is force-pinned to bf16 even
      under ``--dtype float16``), while under ``--dtype float16`` the incoming
      ``key`` / ``slot_score`` arrive as Half (the model's activation dtype). So
      instead of casting to ``key.dtype`` we coerce ``key`` and ``slot_score``
      down to the tail buffers' dtype, keeping the whole index op bf16 -- same
      approach as ``scatter_kpool_tail_updates``.
    - ``round_scale=True`` is rejected outright ("int8 path stores
      scale = max(absmax/127, 1e-4) without power-of-two rounding"), and upstream
      enables it whenever ``scale_fmt`` is set, so it is forced off - same as in
      ``kpool_assemble_softmax_rotate_write_cache``.
    """

    if key.shape[0] == 0:
        return

    def _i64(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int64).contiguous()

    _kdtype = tail_k.dtype
    torch.ops.xspeedgate_ops.kpool_decode_update_and_maybe_write_cache(
        _u8(buf),
        tail_k,
        tail_score,
        key.to(_kdtype).contiguous(),
        slot_score.to(_kdtype).contiguous(),
        ape.to(torch.float32).contiguous(),
        block_tables.contiguous(),
        _i64(req_pool_indices),
        _i64(positions),
        _i64(seq_lens),
        _i64(out_cache_loc),
        round_scale=False,
    )


@register_jit_op(_KPOOL_MODULE, "append_kpool_tail_to_topk")
def append_kpool_tail_to_topk_kunlun(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """``xspeedgate_ops`` replacement for ``append_kpool_tail_to_topk``.

    Widens ``topk_result`` by ``pool_size - 1`` columns: columns below
    ``history_len = min(pool_lens * pool_size, n_cols)`` keep the selected
    expanded-history tokens, the next ``seq_len % pool_size`` columns address the
    still-unpooled tail (optionally translated through ``page_table`` or shifted
    by ``topk_offsets``), and everything else is ``-1``.

    The op is int32-only for all four index tensors, and it accepts
    ``topk_offsets`` as ``[n_rows]`` or ``[n_rows, 1]``, so no squeeze is needed.
    ``pool_size == 1`` returns a copy rather than the same tensor.
    """

    if pool_size - 1 == 0:
        return topk_result

    def _i32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32).contiguous()

    return torch.ops.xspeedgate_ops.append_kpool_tail_to_topk_kunlun(
        _i32(topk_result),
        _i32(seq_lens),
        _i32(pool_lens),
        pool_size,
        _i32(page_table) if page_table is not None else None,
        _i32(topk_offsets) if topk_offsets is not None else None,
    )


@register_jit_op(
    "sglang.kernels.ops.moe.kpool_topk_transform", "fast_kpool_topk_transform_fused"
)
def fast_kpool_topk_transform_fused_kunlun(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table=None,
    topk_indices_offset=None,
    row_starts=None,
    seq_lens=None,
    page_table_row_index=None,
) -> torch.Tensor:
    """Pool-level top-k + expand, on ``xspeedgate_ops``.

    Upstream nvcc-compiles ``dsa/kpool_topk_transform.cuh`` on first call, which
    P800 cannot do, so the whole symbol is redirected to the backend op.

    The backend op now natively supports ``row_starts`` (per-row score window
    start; selected pool ids are shifted back) and ``page_table_row_index`` (row
    into the shared ``req_to_token``; defaults to the batch index), so both are
    passed straight through — the old ``torch.gather`` window-slide and the
    post-hoc flat-index page mapping are gone (A/B verified bit-identical on the
    ragged+paged and ragged+offset paths).

    The one remaining fixup is the ``page_table``-without-``page_table_row_index``
    path (decode, row-aligned): the op's default batch-index branch drops the
    unpooled tail when ``lengths == num_pools`` (its page-gather loop only covers
    ``[0, num_pools * pool_size)``). One extra never-selected score column lifts
    ``num_pools`` by one so the loop reaches the tail; ``page_table`` is
    right-padded to the ``>= num_pools * pool_size`` width the op then demands.
    No column past ``seq_lens`` is read, so the pad value never reaches output.
    """

    # 新版 xspeedgate_ops.fast_kpool_topk_transform_fused 已原生支持 row_starts 与
    # page_table_row_index（A/B 实测：prefill ragged+paged、ragged+offset 两路，与旧的
    # gather / 事后 flat-index 映射 bit 一致），故这两段前后处理全部透传给算子。
    #
    # 唯一保留：page_table 存在但无 page_table_row_index（decode 行对齐路径）时，op 的
    # 默认 batch-index 分支在 lengths == num_pools 时会漏掉未池化 tail（A/B 实测 decode
    # 路径差 tail 几列），所以这一路仍做「score 补 1 列 / 右 pad page_table」的 fixup。
    if page_table is not None and page_table_row_index is None:
        score = torch.nn.functional.pad(score, (0, 1), value=-1e30)
        needed = score.shape[1] * pool_size
        if page_table.shape[1] < needed:
            page_table = torch.nn.functional.pad(
                page_table, (0, needed - page_table.shape[1]), value=-1
            )
        page_table = page_table.contiguous()

    return torch.ops.xspeedgate_ops.fast_kpool_topk_transform_fused(
        score,
        lengths,
        pool_size,
        topk,
        page_table,
        topk_indices_offset,
        row_starts,
        seq_lens,
        page_table_row_index,
    )
