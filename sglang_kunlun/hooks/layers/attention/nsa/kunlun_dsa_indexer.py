"""Kunlun implementation of the DeepSeek sparse-attention indexer.

Single patch point: ``Indexer.forward_cuda`` / ``Indexer.forward_xpu`` are replaced
with :func:`forward_cuda` below through ``plugin_hook``. Everything else here is a
plain module-level function called directly from ``forward_cuda`` - we deliberately
do NOT hook the individual ``Indexer._get_*`` methods, so the Kunlun code path is
fully self contained and an upstream upgrade cannot silently mix our implementation
with upstream control flow.

Dependencies on the upstream ``Indexer`` instance are listed in
:data:`REQUIRED_INDEXER_ATTRS` and checked once per forward by
:func:`_resolve_deps`; that list is the whole contract to re-verify when
upgrading.

Aligned with upstream sglang 0.5.14 ``dsa/dsa_indexer.py``
(ported from the 0.5.8 ``nsa/nsa_indexer.py`` Kunlun implementation). Known, intentional
differences vs upstream:
  1. max_k logits layout + explicit ``mask_for_I8_mqa_logits`` (upstream uses a
     total_k layout);
  2. int8 quantization instead of fp8_e4m3 (hence the ``/ 127`` corrections);
  3. free-memory driven chunking of the ragged MQA logits;
  4. dual-stream is never enabled, so those upstream branches are dropped.
When upgrading, diff the *upstream* forward_cuda between versions and port the
new semantics here, rather than diffing this file.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import os

import torch
import kunlun_ops
from einops import rearrange

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# Experimental DSA prefix attention-sink: number of leading positions to pin
# into the selected top-k, and the additive logit boost that guarantees it.
# ``DSV4_HISPARSE_PREFIX_SINK`` tokens, 0 = disabled (default). See the sink
# block in ``_get_topk_paged``.
_PREFIX_SINK = int(os.environ.get("DSV4_HISPARSE_PREFIX_SINK", "0"))
_PREFIX_SINK_BOOST = 1.0e4

try:
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
except ImportError:  # dsv4.1 删掉了 DSA prefill-CP 的 in-seq-split 模式
    def is_dsa_prefill_cp_in_seq_split() -> bool:
        """上游只保留 interleave / round-robin 模式，这个符号守着的分支永远不会激活。"""
        return False
from sglang.srt.layers.attention.mqa_logits_utils import mqa_logits_should_chunk
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip

_IS_HIP = is_hip()

from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.cp.utils import get_cp_strategy, is_cp_active
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _broadcast_indexer_topk_from_rank0,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk


if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


# Upstream's ``_is_cuda`` branch projects the head gates with
# ``torch.mm(..., out_dtype=torch.float32)`` (``aten::mm.dtype``), which has no
# Kunlun kernel. The replacement lives in ``kernel_ops``; only the registration
# belongs here.
@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._weights_proj_bf16_in_fp32_out", type=HookType.REPLACE
)
def dsa_weights_proj_bf16_in_fp32_out_kunlun(self, x: torch.Tensor) -> torch.Tensor:
    """Project the DSA indexer head gates to fp32 without ``aten::mm.dtype``.

    Kunlun reports itself as CUDA, so upstream takes the ``_is_cuda`` branch and
    calls ``torch.mm(x, w.t(), out_dtype=torch.float32)``. That overload has no
    Kunlun kernel, so run the projection in the weight dtype and upcast, matching
    what the Kunlun indexer's own ``_get_logits_head_gate`` does.
    """

    _, weights = _indexer_wk_weights(self, x)
    return weights.float()


# Attributes we read off the upstream Indexer instance. This is the full coupling
# surface between the Kunlun indexer and upstream.
REQUIRED_INDEXER_ATTRS = (
    "_get_k_bf16",
    "alt_stream",
    "block_size",
    "head_dim",
    "index_topk",
    "k_norm",
    "n_heads",
    # 0.5.22 不再在 C4Indexer 上快照 ``dsa_enable_prefill_cp``；下面改成直接问
    # ``dsa/utils`` 的自由函数。
    "rope_head_dim",
    "rotary_emb",
    "softmax_scale",
    "wq_b",
)

# Checked structurally rather than by name: 0.5.23 builds the key/gate
# projections in one of two layouts and which one depends on the platform.
PROJECTION_LAYOUTS = ("wk+weights_proj", "wk_weights_proj")


def _indexer_projection_layout(indexer):
    """Which key/gate projection layout upstream built on this instance."""
    if getattr(indexer, "wk_weights_proj", None) is not None:
        return "wk_weights_proj"
    if getattr(indexer, "wk", None) is not None and (
        getattr(indexer, "weights_proj", None) is not None
    ):
        return "wk+weights_proj"
    return None


def _indexer_wk_weights(indexer, x: torch.Tensor):
    """``(key, weights)`` from whichever projection layout upstream built.

    0.5.19 and 0.5.22 always built ``wk`` and ``weights_proj`` separately.
    0.5.23's ``Indexer.__init__`` folds them into one ``wk_weights_proj`` Linear
    of width ``head_dim + n_heads`` whenever ``use_dsa_indexer_fusion`` holds --
    ``_is_cuda and not SGLANG_DISABLE_DSA_INDEXER_FUSION and not is_neox_style``
    (``dsa/dsa_indexer.py:310-314``). Kunlun reports itself as CUDA, so the
    fused layout is the one built here and the two separate modules never exist.
    Upstream splits the fused output the same way (``dsa_indexer.py:605``,
    ``1090``, ``1121``). Both layouts stay accepted so a build with the fusion
    switched off needs no further change.

    Cost note, deliberate: the two call sites need these separately (the key on
    the tensor-parallel path, the gate weights on a possibly requantized copy of
    ``x``), so the fused layout runs the projection twice per layer -- one
    ``[rows, hidden] x [hidden, head_dim + n_heads]`` GEMV. That is a
    performance follow-up, not a correctness one: forcing the non-fused layout
    would need a server arg or an env switch, and this upgrade may add neither
    (R5/R6).
    """
    orig_dtype = x.dtype
    fused = getattr(indexer, "wk_weights_proj", None)
    if fused is not None:
        # Cast first: 0.5.23 builds this Linear with `params_dtype=bfloat16`
        # regardless of the model dtype (`dsa_indexer.py:353-358`), while this
        # run serves GLM-5.3 at `--dtype float16`. Handing it an fp16 activation
        # makes the Kunlun linear kernel reject the call outright
        # (`Check ret == 0 failed ... linear custom kernel call Failed`).
        # Prefer upstream's own accessor so the split order and the dtype cast
        # stay theirs (`dsa_indexer.py:605-607`, `kw.split([head_dim,
        # n_heads])` -> key, weights).
        x = x.to(fused.weight.dtype)
        split = getattr(indexer, "_fused_k_weights", None)
        if split is not None:
            key, weights = split(x)
        else:
            out, _ = fused(x)
            key, weights = torch.split(
                out, [indexer.head_dim, indexer.n_heads], dim=-1
            )
        # Stay in the projection's own dtype (bf16). The DSA indexer chain on
        # this platform is bf16 end to end -- the Hadamard rotation is bf16-only
        # and `prefill_attention_dsa_opt` checks that q, kv, out and lse_out all
        # agree (`Check failed, prefill_attention_dsa_opt not support q/kv/out/
        # lse_out/indices dtype` when they do not). Casting back to the model
        # dtype here makes the indexer the only fp16 link in a bf16 chain.
        del orig_dtype
        return key, weights
    # Non-fused layout (0.5.19/0.5.22 and any 0.5.23 build with the fusion off):
    # the two separate projections, exactly as before.
    weights, _ = indexer.weights_proj(x.to(indexer.weights_proj.weight.dtype))
    key, _ = indexer.wk(x)
    return key, weights


def _resolve_deps(indexer) -> None:
    """Fail loudly (once per call) if upstream renamed something we depend on."""
    missing = [a for a in REQUIRED_INDEXER_ATTRS if not hasattr(indexer, a)]
    if _indexer_projection_layout(indexer) is None:
        missing.append(f"one of {PROJECTION_LAYOUTS}")
    if missing:
        raise AttributeError(
            f"kunlun dsa indexer requires Indexer attributes {missing}; upstream "
            "layout changed - update REQUIRED_INDEXER_ATTRS and the call sites"
        )


def xspeedgate_hadamard_transform(
    x: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """xspeedgate hadamard transform (replaces sgl_kernel fast_hadamard_transform).

    NOTE: this is a different kernel from :func:`hadamard_transform` above (which
    materializes the matrix and calls ``kunlun_ops.matmul``). The DSA indexer's
    q/k rotation has always used this one, so it is kept separate rather than
    unified - switching it would change indexer numerics.
    """

    if scale is None:
        return torch.ops.xspeedgate_ops.hadamard_transform(x.contiguous())
    return torch.ops.xspeedgate_ops.hadamard_transform(x.contiguous(), scale)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate activations with a scaled Hadamard transform.

    The Hadamard op is bf16-only and the assert used to be unconditional, which
    held as long as every producer handed it bf16. 0.5.23's fused
    ``wk_weights_proj`` is built at ``params_dtype=bfloat16`` while this run
    serves GLM-5.3 at ``--dtype float16``, so the projections upstream folds
    together now come back at different dtypes and the caller has to pick one
    for the rotary embedding (it requires q and k to agree). Rather than force
    both to bf16 -- which would move the projection's output dtype away from
    the model dtype -- the transform keeps its own bf16 requirement internally
    and returns the caller's dtype.
    """
    out_dtype = x.dtype
    x = x.to(torch.bfloat16)
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return xspeedgate_hadamard_transform(x, hidden_size**-0.5).to(out_dtype)


def chunk_cu_seqlens(
    cu_seqlens: torch.Tensor,  # shape: (1, batch_size + 1)
    chunk_size: int,
):
    """Split the original sequence ranges into chunks of ``chunk_size`` rows.

    Returns a list of ``(new_cu_seqlens, interval_bounds, chunk_range)`` tuples.
    Used by the DSA ragged MQA-logits chunk path to bound peak logits memory.
    """

    cu = cu_seqlens.squeeze(0)
    total_len = cu[-1].item()

    seg_starts = cu[:-1]
    seg_ends = cu[1:]

    results = []
    for chunk_start in range(0, total_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_len)

        inter_starts = torch.clamp(seg_starts, min=chunk_start)
        inter_ends = torch.clamp(seg_ends, max=chunk_end)

        valid_mask = inter_starts < inter_ends

        valid_inter_starts = inter_starts[valid_mask]
        valid_inter_ends = inter_ends[valid_mask]
        inter_lens = valid_inter_ends - valid_inter_starts

        new_cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(inter_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        interval_bounds = torch.cat([valid_inter_starts[:1], valid_inter_ends])
        chunk_range = torch.tensor(
            [valid_inter_starts[0].item(), valid_inter_ends[-1].item()],
            dtype=torch.int32,
        )

        results.append((new_cu_seqlens, interval_bounds, chunk_range))

    return results


@torch.compile(dynamic=True)
def _get_logits_head_gate(indexer, x: torch.Tensor, q_scale: torch.Tensor):
    """ _get_logits_head_gate """
    _, weights = _indexer_wk_weights(indexer, x)
    # GLM 5掉点
    weights = weights.float()
    weights = weights * indexer.n_heads**-0.5
    weights = weights * q_scale * indexer.softmax_scale
    return weights


def _live_row_index(
    extend_lens: List[int], rows: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Rows of a padded per-request metadata layout that carry real q tokens."""
    q_offset = sum(extend_lens)
    bs = len(extend_lens)
    if rows == q_offset or bs == 0 or rows % bs != 0:
        return None
    stride = rows // bs
    if any(length > stride for length in extend_lens):
        return None
    return torch.cat(
        [
            torch.arange(i * stride, i * stride + length, device=device)
            for i, length in enumerate(extend_lens)
            if length > 0
        ]
    )


def _metadata_shapes(attn_metadata) -> str:
    """One-line dump of the DSA metadata row shapes, for alignment failures."""
    fields = (
        "cache_seqlens_int32",
        "dsa_cache_seqlens_int32",
        "dsa_seqlens_expanded",
        "page_table_1",
        "real_page_table",
        "cu_seqlens_q",
        "cu_seqlens_k",
    )
    parts = []
    for name in fields:
        value = getattr(attn_metadata, name, None)
        parts.append(f"{name}={None if value is None else tuple(value.shape)}")
    parts.append(f"max_seq_len_q={getattr(attn_metadata, 'max_seq_len_q', None)}")
    parts.append(f"max_seq_len_k={getattr(attn_metadata, 'max_seq_len_k', None)}")
    parts.append(
        f"dsa_extend_seq_lens_list={getattr(attn_metadata, 'dsa_extend_seq_lens_list', None)}"
    )
    return "md(" + ", ".join(parts) + ")"


def _require_continuous_index_k(pool, where: str) -> None:
    """Fail loudly on the 0.5.22 index-K API change (in-seq-split CP path only).

    0.5.22 removed the per-request ``get_index_k_continuous`` /
    ``get_index_k_scale_continuous`` gather. The replacement is the batched
    ``get_index_k_scale_buffer(...)`` -> ``IndexKeyCache.get_k_and_scale(...)``;
    ``_get_topk_ragged`` is ported onto it above. ``_get_topk_ragged_with_cp``
    is not: it gathers a sub-range of one request per ``cp_index`` entry and
    rebuilds ``ks``/``ke`` around those row offsets, so it needs its offset
    bookkeeping reworked as well. 0.5.23 cannot reach it anyway --
    ``is_dsa_prefill_cp_in_seq_split()`` is False here, so CP prefill takes the
    interleave path through ``_get_topk_ragged``. Readable message instead of an
    AttributeError.
    """
    if hasattr(pool, "get_index_k_continuous"):
        return
    raise NotImplementedError(
        f"{where}: the 0.5.22 index-K pool dropped "
        "get_index_k_continuous/get_index_k_scale_continuous. Port this path "
        "onto get_index_k_scale_buffer()/IndexKeyCache.get_k_and_scale() "
        "before using DSA ragged prefill or CP prefill on this build."
    )


def _align_rows_to_q(tensor, extend_lens: List[int], q_offset: int):
    """Reduce a per-request-padded metadata tensor to the live q rows."""
    if tensor is None or tensor.shape[0] == q_offset:
        return tensor
    index = _live_row_index(extend_lens, tensor.shape[0], tensor.device)
    if index is not None:
        return tensor[index]
    if tensor.shape[0] > q_offset:
        return tensor[:q_offset]
    return tensor


_PAGED_TOPK_FALLBACK_LOGGED = False


def _log_paged_topk_fallback(exc: BaseException) -> None:
    """Warn once that the metadata torch port is standing in for the kernel."""
    global _PAGED_TOPK_FALLBACK_LOGGED
    if _PAGED_TOPK_FALLBACK_LOGGED:
        return
    _PAGED_TOPK_FALLBACK_LOGGED = True
    import logging

    logging.getLogger(__name__).warning(
        "sglang-kunlun: xspeedgate_ops.topk_transform unusable on the paged "
        "path (%s); falling back to KunlunDSAIndexerMetadata.topk_transform. "
        "The selected set is the same, but the trace then lacks "
        "xinfer3::topk_transform_decode_kernelv2 and shows "
        "topk_find_kth_converted_clusterwise / topk_select_element_clusterwise "
        "/ masked_fill_scalar instead.",
        exc,
    )


def _paged_topk_transform(
    logits: torch.Tensor,
    topk: int,
    metadata,
    page_table_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """Select ``topk`` positions per row and map them to page-1 cache slots.

    0.5.8 ran the vendor fused kernel here
    (``sgl_kernel/patch/layers/attention/nsa_backend.py:500``, reached from
    ``patch/.../nsa_indexer.py:162 _get_topk_paged``), and the baseline decode
    trace shows it launching as
    ``xinfer3::topk_transform_decode_kernelv2<float, 2048>`` from
    ``cudaGraphLaunch`` under ``eagle_draft_cuda_graph_runner.py:175`` -- i.e. it
    is safe to capture, which is why the call is made unconditionally rather than
    being gated on capture mode.

    ``KunlunDSAIndexerMetadata.topk_transform``'s PAGED branch
    (``kunlun_nsa_backend.py``) replaced it with a torch port because the score
    is page aligned (``real_page_table.shape[1] * page_size``) while
    ``page_table_1`` is only ``max_seqlen_k`` wide. 0.5.8 hit that same mismatch
    and fixed it by right-padding the table with -1
    (``nsa_backend.py:480-497``): -1 is never dereferenced, because ``lengths``
    bounds each row's gather to its real sequence length.
    """
    # Same lengths/table selection the metadata's own PAGED branch would make
    # (`kunlun_nsa_backend.py` `KunlunDSAIndexerMetadata.topk_transform`, and
    # 0.5.8 `nsa_backend.py:471`), so the only thing that changes here is which
    # kernel runs.
    seq_lens_topk = metadata.get_seqlens_expanded()
    page_table_1 = metadata.attn_metadata.page_table_1
    if page_table_index is not None:
        page_table_1 = page_table_1[page_table_index]
    # The vendor kernel takes the row count off `score` and indexes `lengths`
    # with it, so a short `lengths` would read out of bounds instead of failing.
    # The torch port raised on the same mismatch (broadcasting
    # `lengths.unsqueeze(1)` against the score rows), so keep it an error.
    if seq_lens_topk.shape[0] != logits.shape[0]:
        raise RuntimeError(
            "kunlun DSA paged top-k: seqlens rows do not match the score rows "
            f"(lengths={tuple(seq_lens_topk.shape)}, "
            f"score={tuple(logits.shape)})"
        )
    try:
        return _paged_topk_transform_vendor(
            score=logits,
            lengths=seq_lens_topk,
            src_page_table=page_table_1,
            topk=topk,
            cu_seqlens_q=metadata.attn_metadata.cu_seqlens_q,
        )
    except Exception as exc:  # noqa: BLE001 - keep serving on the torch port
        _log_paged_topk_fallback(exc)
        return metadata.topk_transform(
            logits, topk, batch_idx_list=page_table_index
        )


def _paged_topk_transform_vendor(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused select + page-1 lookup in one launch.

    Contract from the vendor stub (``xspeedgate_ops/_topk.pyi:152-179``):
    ``score`` float32 with ``stride(1) == 1``, ``lengths`` int32 contiguous,
    ``src_page_table`` int32, ``dst_page_table`` int32 ``[rows, topk]``,
    contiguous and written in place.
    """
    rows, width = score.shape
    # The kernel indexes ``src_page_table`` by the selected column and writes
    # ``topk`` entries per row, so the table must cover both the score width and
    # ``topk``. 0.5.8 spelled the first term ``align64(page_table_1.width)``,
    # which equals the page-aligned score width when both tables are built from
    # the same ``max_seqlen_k``.
    need = max(topk, width)
    table = src_page_table
    if table.shape[1] < need:
        table = torch.nn.functional.pad(table, (0, need - table.shape[1]), value=-1)
    dst_page_table = score.new_empty((rows, topk), dtype=torch.int32)
    torch.ops.xspeedgate_ops.topk_transform(
        score=score if score.stride(1) == 1 else score.contiguous(),
        lengths=lengths.to(torch.int32).contiguous(),
        src_page_table=table.to(torch.int32).contiguous(),
        dst_page_table=dst_page_table,
        topk=topk,
        # One score row per request on decode, one per query token on
        # target-verify / draft-extend; the kernel expands the table rows from
        # this, which is why 0.5.8 passed it on every PAGED call.
        cu_seqlens_q=(
            None if cu_seqlens_q is None else cu_seqlens_q.to(torch.int32).contiguous()
        ),
    )
    return dst_page_table


def _get_topk_paged(
    indexer,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """ _get_topk_paged """
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"

    block_tables = metadata.get_page_table_64()
    max_seq_len = block_tables.shape[1] * page_size

    kv_cache_fp8 = get_token_to_kv_pool().get_index_k_with_scale_buffer(
        layer_id=layer_id
    )

    block_kv = 64
    num_heads_kv = 1
    
    # One context length per q row. Decode has exactly one q row per request, so
    # ``cache_seqlens`` is the per-row length there; target-verify /
    # draft-extend-v2 contribute several q rows per request and use the expanded
    # lengths. NOTE: ``dsa_seqlens_expanded`` is *not* refreshed by the decode
    # graph metadata builder, so decode must not read it.
    if (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        seqlens_32 = metadata.get_seqlens_expanded()
        seqlens_32_cpu = metadata.get_seqlens_expanded_cpu()
    else:
        seqlens_32 = metadata.get_seqlens_int32()
        seqlens_32_cpu = metadata.get_seqlens_int32_cpu()

    assert len(q_fp8.shape) == 3, f"q_fp8.shape: {q_fp8.shape}"
    q_fp8 = q_fp8.unsqueeze(1)

    if len(weights.shape) == 2:
        weights = weights.unsqueeze(1)

    # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
    # and it is necessary to extract the actual q length.
    extend_lens = list(metadata.get_dsa_extend_len_cpu())
    q_offset = sum(extend_lens)

    # ``dsa_paged_mqa_logits`` takes its batch from ``q.size(0)`` (== q_offset
    # below) and requires the gathered scale, the page table and the context
    # lengths to have exactly that many rows. Draft-extend metadata pads the page
    # table to ``max_extend_len`` rows per request, so pick the live rows per
    # request instead of a prefix -- a prefix would misalign every request that
    # follows a partially-accepted one.
    seqlens_32 = _align_rows_to_q(seqlens_32, extend_lens, q_offset)
    seqlens_32_cpu = _align_rows_to_q(seqlens_32_cpu, extend_lens, q_offset)
    block_tables = _align_rows_to_q(block_tables, extend_lens, q_offset)
    if seqlens_32.shape[0] != q_offset or block_tables.shape[0] != q_offset:
        backend = get_attn_backend()
        raise RuntimeError(
            "kunlun DSA paged top-k: cannot align metadata rows with the q rows. "
            "The metadata does not describe this batch -- usually it was built for "
            "a different (earlier/smaller) batch. "
            f"mode={forward_batch.forward_mode}, q_offset={q_offset}, "
            f"q_rows={q_fp8.shape[0]}, extend_lens={extend_lens}, "
            f"context_lens={seqlens_32.shape[0]}, page_table={block_tables.shape[0]}, "
            f"fb.batch_size={getattr(forward_batch, 'batch_size', None)}, "
            f"fb.extend_seq_lens_cpu={getattr(forward_batch, 'extend_seq_lens_cpu', None)}, "
            f"fb.seq_lens_cpu={getattr(forward_batch, 'seq_lens_cpu', None)}, "
            f"fb.seq_lens_rows={forward_batch.seq_lens.numel()}, "
            f"{_metadata_shapes(metadata.attn_metadata)}, "
            f"active_backend=({id(backend)}, {type(backend).__name__}), "
            f"fb_id={id(forward_batch)}, "
            f"provenance={getattr(forward_batch, '_kunlun_draft_meta_provenance', None)}"
        )
        
    head_dim = get_token_to_kv_pool().index_head_dim
    k_fp8 = kv_cache_fp8[:, :page_size * head_dim].view(
        kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim
    ).to(torch.int8)
    
    k_scale = torch.ops.xspeedgate_ops.gather_paged_scale(
        kv_cache=kv_cache_fp8,
        block_tables=block_tables,
        page_size=page_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        scale_offset=page_size * head_dim,
        context_lens=seqlens_32,
    )

    # NOTE: reverted luno-4142 (1830d8ea) -- scale the gathered k scales here
    # instead of relying on the kernel's q_is_int8 argument.
    if q_fp8.dtype is torch.int8:
        k_scale /= 127
        
    fused_kv_cache = [k_fp8.contiguous(), k_scale.contiguous()]
    logits = torch.empty(
        (q_offset, q_fp8.shape[1], max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device
    )

    kunlun_ops.I8_paged_mqa_logits(
        q=q_fp8[:q_offset],
        fused_kv_cache=fused_kv_cache,
        weights=weights[:q_offset],
        context_lens=[seqlens_32_cpu, seqlens_32],
        block_table=block_tables,
        max_context_len=max_seq_len,
        clean_logits=False,
        out=logits,
        use_xfa_boost=False
    )
    
    logits = logits.reshape(-1, max_seq_len)

    # --- DSA prefix "attention sink" (experimental, env-gated) --------------
    # DSV4-Flash's DSA indexer selects only ``index_topk`` (=512) positions per
    # query. Once a request's sequence exceeds 512 tokens, the prompt-prefix --
    # which carries global format instructions ("no commas", "all lowercase") --
    # scores low against the content-generation query and is dropped from the
    # selected set, so the model forgets the constraint. Empirically ifeval
    # format-constraint cases fail 0%->100% as the output grows past ~512 total
    # tokens, while gsm8k/longbench (content, not format) are unaffected.
    #
    # Force the first ``_PREFIX_SINK`` positions (the prompt head, where the
    # instruction usually sits) to always win selection by additively boosting
    # their indexer logits. Additive keeps invalid (-inf) positions excluded and
    # preserves ordering among the sunk prefix. Decode only; off unless the env
    # var is set, so it is a clean A/B.
    if _PREFIX_SINK > 0 and logits.shape[1] > 0:
        sink = min(_PREFIX_SINK, logits.shape[1])
        logits[:, :sink] += _PREFIX_SINK_BOOST

    # ``topk_transform`` maps the selected positions through
    # ``attn_metadata.page_table_1``, which is padded per request just like the
    # page-64 table. ``batch_idx_list`` is the upstream seam for selecting its
    # rows, so reuse it rather than letting the gather blow up on mismatched rows.
    page_table_1 = metadata.get_page_table_1()
    page_table_index = None
    if page_table_1 is not None and page_table_1.shape[0] != q_offset:
        page_table_index = _live_row_index(
            extend_lens, page_table_1.shape[0], logits.device
        )
        if page_table_index is None or page_table_index.numel() != q_offset:
            raise RuntimeError(
                "kunlun DSA paged top-k: page_table_1 rows cannot be aligned with "
                f"the q rows (q_offset={q_offset}, extend_lens={extend_lens}, "
                f"page_table_1={tuple(page_table_1.shape)}, "
                f"mode={forward_batch.forward_mode}, "
                f"{_metadata_shapes(metadata.attn_metadata)}, "
                f"provenance={getattr(forward_batch, '_kunlun_draft_meta_provenance', None)})"
            )

    topk_result = _paged_topk_transform(
        logits, indexer.index_topk, metadata, page_table_index
    )
    # Restore possible padding exist in the hidden states.
    if q_offset < q_fp8.shape[0]:
        pad_len = q_fp8.shape[0] - q_offset
        padding = torch.full(
            (pad_len, topk_result.shape[1]),
            -1,
            dtype=topk_result.dtype,
            device=topk_result.device,
        )
        topk_result = torch.cat([topk_result, padding], dim=0)
    del logits
    return topk_result


def _get_topk_ragged(
    indexer,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: "BaseIndexerMetadata",
    topk_result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kunlun ragged indexer.

    ``enable_dual_stream`` / ``topk_result`` exist for upstream 0.5.14 signature
    compatibility; the Kunlun path never uses the alternate stream and always
    allocates its own ``topk_result``.

    The K/scale gather is the batched 0.5.23 accessor, as upstream uses it
    (``dsa/dsa_indexer.py:1548``), not the 0.5.19 per-request loop: the index-K
    pool dropped ``get_index_k_continuous`` / ``get_index_k_scale_continuous``
    (only ``get_index_k_scale_buffer`` survives, ``memory_pool.py:5562``) and the
    first disaggregated ragged prefill raised ``NotImplementedError`` here.
    ``get_index_k_scale_buffer`` builds the identical ragged layout in one
    launch -- per request its first ``seq_len_i`` K rows, then that request's
    scales, requests concatenated in order -- which is exactly what the loop's
    ``torch.cat`` produced for ``I8_mqa_logits``'s ``context_k_lens``
    (``cu_seqlens_k``).
    """
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    assert forward_batch.forward_mode.is_extend_without_speculative()

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    if len(weights.shape) == 3:
        weights = weights.squeeze(-1)
    block_tables = metadata.get_page_table_64()

    assert (
        forward_batch.seq_lens_cpu is not None
        and forward_batch.extend_seq_lens_cpu is not None
    )

    batch_size = len(block_tables)
    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device
    topk_result = torch.full(
        (token_nums, indexer.index_topk), -1, device=device, dtype=torch.int32
    )
    if batch_size == 0:
        return topk_result

    indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    assert len(indexer_seq_lens_cpu) == batch_size
    # ``get_k_and_scale`` wants the length vector on the device (one program per
    # sequence in ``_get_k_and_s_triton_kernel``) and a 2-D page table it indexes
    # as ``page_indices[batch_id * num_pages + page]``. Take the device vector the
    # metadata already carries instead of copying the host mirror; ``sum``/``max``
    # stay on the host mirror, so they replace the loop's per-request ``.item()``
    # reads without adding a sync.
    indexer_seq_lens = metadata.get_indexer_seq_len()
    if indexer_seq_lens is None:
        indexer_seq_lens = indexer_seq_lens_cpu.to(device, non_blocking=True)
    assert indexer_seq_lens.shape[0] == batch_size, (
        "kunlun DSA ragged top-k: indexer seq_lens rows do not match the page "
        f"table rows (seq_lens={tuple(indexer_seq_lens.shape)}, "
        f"page_table={tuple(block_tables.shape)})"
    )

    # One physical index page is ``[int8 K bytes | fp32 scales]`` with
    # ``scale_offset = page_size * index_head_dim`` bytes
    # (`dsa_index_k_cache.py` ``S_OFFSET_NBYTES_IN_PAGE``, the same split
    # ``_get_topk_paged`` reads through ``gather_paged_scale``), so the K half is
    # ``(sum seq_len, head_dim)`` int8 and the scale half is ``(sum seq_len, 4)``
    # bytes holding one fp32 per token.
    k_fp8, k_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
        layer_id=layer_id,
        seq_len_tensor=indexer_seq_lens,
        page_indices=block_tables,
        seq_len_sum=int(indexer_seq_lens_cpu.sum().item()),
        max_seq_len=int(indexer_seq_lens_cpu.max().item()),
    )
    # ``I8_mqa_logits`` wants K in ``q``'s dtype, int8 on the int8 path
    # (``kunlun_ops/_deepgemm.py:I8_mqa_logits``), while the gather hands back
    # bytes (``dsa_index_k_cache.py`` allocates ``uint8``). Reinterpret, do not
    # convert: ``.to(int8)`` would copy the whole K block.
    k_fp8 = k_fp8.view(torch.int8)
    k_scale = k_scale.view(torch.float32).squeeze(-1)
    if q_fp8.dtype is torch.int8:
        k_scale /= 127
    ks, ke = metadata.get_indexer_kvcache_range()
    seq_lens_expanded = metadata.get_seqlens_expanded()
    token_to_batch_idx = metadata.get_token_to_batch_idx()
    q_offset = ks.shape[0]
    k_offset = k_fp8.shape[0]

    # Suppose there are two requests, with extend_seq_len = [3, 2]
    # and seq_lens = [10, 4]
    # The logits matrix looks like this, with * representing the valid logits
    # and - representing the invalid logits:
    #
    #  ********--|----
    #  *********-|----
    #  **********|----
    #  ----------|***-
    #  ----------|****
    #
    # ks = [0, 0, 0, 10, 10]
    # ke = [8, 9, 10, 13, 14]

    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device

    # Check if we need to chunk to avoid OOM
    # 0.5.23 moved this off the instance onto `mqa_logits_should_chunk`
    # (`layers/attention/mqa_logits_utils.py:71`), with the free-memory budget
    # behind a callable so the host sync only happens when the static gate
    # passes -- the same behaviour the 0.5.19 method had inline.
    need_chunk, free_mem = mqa_logits_should_chunk(
        num_rows=q_offset,
        num_cols=k_offset,
        get_budget_bytes=lambda: indexer._get_mqa_logits_budget_bytes(device),
        rocm=_IS_HIP,
    )
    if need_chunk:
        import logging
        logging.warning(f"[NSA] chunk path triggered: q_offset={q_offset}, \
            k_offset={k_offset}, free_mem={free_mem:.0f}")

    context_q_lens_xpu = metadata.get_cu_seqlens_q()
    context_q_lens_cpu = metadata.get_cu_seqlens_q_cpu()
    context_k_lens_xpu = metadata.get_cu_seqlens_k()
    context_k_lens_cpu = metadata.get_cu_seqlens_k_cpu()
    if not need_chunk:
        assert q_fp8[:q_offset].shape[0] != 0

        # Use max_k layout: logits shape = (q_total, max_k_per_batch).
        # I8_mqa_logits handles multi-batch correctly with this layout
        # (each batch's results written from column 0), avoiding the
        # total_k layout bug where batch 1+ are skipped.
        # This also makes fast_topk_transform_fused work correctly:
        # valid K at columns [0, k_len_b) matches LOCAL seq_lens_expanded
        # and page_table indexing, eliminating per-batch loops entirely.
        max_k = int(indexer_seq_lens_cpu.max().item())
        logits = torch.full((q_offset, max_k), -1.0, dtype=torch.float32, device=q_fp8.device)

        kunlun_ops.I8_mqa_logits(
            q=q_fp8[:q_offset],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[:q_offset],
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )

        # Mask logits outside causal range [0, ke_local) per token.
        # In max_k layout each batch's K starts from column 0, so
        # ks_local = 0 for all tokens and ke_local = seq_lens_expanded.
        ks_local = torch.zeros(q_offset, dtype=ks.dtype, device=ks.device)
        ke_local = seq_lens_expanded.to(dtype=ks.dtype)
        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=max_k,
            cu_seqlen_ks=ks_local.contiguous(),
            cu_seqlen_ke=ke_local.contiguous(),
            logits=logits,
        )

        assert logits.shape[0] == len(seq_lens_expanded)

        page_table_1 = metadata.get_page_table_1()
        cu_seqlens_q_topk = metadata.attn_metadata.cu_seqlens_q
        dst_page_table = logits.new_empty((logits.size(0), indexer.index_topk), dtype=torch.int32)
        torch.ops.xspeedgate_ops.topk_transform(
            score=logits.contiguous(),
            lengths=seq_lens_expanded.contiguous(),
            src_page_table=page_table_1.contiguous(),
            dst_page_table=dst_page_table,
            topk=indexer.index_topk,
            cu_seqlens_q=cu_seqlens_q_topk.contiguous(),
        )
        topk_result[:q_offset] = dst_page_table

        del logits
        del ks, q_fp8, k_fp8, k_scale

        return topk_result

    # Chunk path: split only the *query rows*; every row still scores its full K
    # range. Layout, masking and top-k are identical to the non-chunk branch above
    # (max_k layout, local ks/ke, page-table transform) so the two paths cannot
    # drift apart.
    #
    # NOTE: an earlier version also ran ``chunk_cu_seqlens`` over the *K* lengths
    # with the query chunk size, which handed each query chunk only a
    # ``max_rows``-wide slice of K. On a 24k-token prompt that limited the indexer
    # to the first ~3k tokens of the document and the answers silently degraded.
    max_k = int(indexer_seq_lens_cpu.max().item())
    bytes_per_row = max_k * 4  # float32 logits
    # Reserve 50% of free memory for logits
    max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
    max_rows = min(max_rows, q_offset)

    page_table_1 = metadata.get_page_table_1()
    cu_q_cpu = context_q_lens_cpu.reshape(-1)
    cu_k_cpu = context_k_lens_cpu.reshape(-1)
    cu_k_xpu = context_k_lens_xpu.reshape(-1)

    q_interval_cpu = chunk_cu_seqlens(context_q_lens_cpu, chunk_size=max_rows)
    q_interval_xpu = chunk_cu_seqlens(context_q_lens_xpu, chunk_size=max_rows)
    for i in range(len(q_interval_cpu)):
        chunk_q_cpu, _, chunk_range = q_interval_cpu[i]
        chunk_q_xpu, _, _ = q_interval_xpu[i]
        start = int(chunk_range[0].item())
        end = int(chunk_range[1].item())

        # Requests this row window belongs to; K stays addressed by the original
        # (absolute) cu_seqlens_k so each row keeps its whole context.
        b0 = int(torch.searchsorted(cu_q_cpu, torch.tensor(start), right=True)) - 1
        b1 = int(torch.searchsorted(cu_q_cpu, torch.tensor(end - 1), right=True)) - 1
        chunk_k_cpu = cu_k_cpu[b0 : b1 + 2].contiguous()
        chunk_k_xpu = cu_k_xpu[b0 : b1 + 2].contiguous()
        chunk_max_k = int(
            (chunk_k_cpu[1:] - chunk_k_cpu[:-1]).max().item()
        )

        logits_chunk = torch.full(
            (end - start, chunk_max_k),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        kunlun_ops.I8_mqa_logits(
            q=q_fp8[start:end],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[start:end],
            context_q_lens=(chunk_q_cpu, chunk_q_xpu),
            context_k_lens=(chunk_k_cpu, chunk_k_xpu),
            logits=logits_chunk,
            clean_logits=True,
            use_xfa_boost=False,
        )

        # max_k layout: each request's K starts at column 0, so the causal bound
        # is [0, seq_lens_expanded) for every row.
        lengths_chunk = seq_lens_expanded[start:end]
        ks_local = torch.zeros(end - start, dtype=ks.dtype, device=ks.device)
        ke_local = lengths_chunk.to(dtype=ks.dtype)
        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=chunk_max_k,
            cu_seqlen_ks=ks_local.contiguous(),
            cu_seqlen_ke=ke_local.contiguous(),
            logits=logits_chunk,
        )

        dst_page_table = logits_chunk.new_empty(
            (logits_chunk.size(0), indexer.index_topk), dtype=torch.int32
        )
        torch.ops.xspeedgate_ops.topk_transform(
            score=logits_chunk.contiguous(),
            lengths=lengths_chunk.contiguous(),
            src_page_table=page_table_1[b0 : b1 + 1].contiguous(),
            dst_page_table=dst_page_table,
            topk=indexer.index_topk,
            cu_seqlens_q=chunk_q_xpu.to(device=device, dtype=torch.int32).contiguous(),
        )
        topk_result[start:end] = dst_page_table
        del logits_chunk
    del ks, q_fp8, k_fp8, k_scale
    del q_interval_cpu, q_interval_xpu
    return topk_result

def _get_topk_ragged_with_cp(
    indexer,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: "BaseIndexerMetadata",
    kv_len: int,
    actual_seq_q: int,
    cp_index: List[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    _require_continuous_index_k(
        get_token_to_kv_pool(), "_get_topk_ragged_with_cp"
    )
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    if len(weights.shape) == 3:
        weights = weights.squeeze(-1)
    k_fp8_list = []
    k_scale_list = []
    ks_list = []
    ke_offset_list = []
    offset = 0
    actual_seq_q_list = []
    batch_idx_list = []

    block_tables = metadata.get_page_table_64()

    assert (
        forward_batch.seq_lens_cpu is not None
        and forward_batch.extend_seq_lens_cpu is not None
    )
    if cp_index is not None:
        # TODO Multi-batch support has accuracy issues
        for batch_idx, start_seq_position, end_seq_position in cp_index:
            pre_chunk_offset = (
                forward_batch.seq_lens_cpu[batch_idx].item()
                - forward_batch.extend_seq_lens_cpu[batch_idx]
            )
            start_seq_position += pre_chunk_offset
            end_seq_position += pre_chunk_offset
            if offset == 0 and batch_idx != 0:
                offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
            k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                layer_id,
                end_seq_position,
                block_tables[batch_idx],
            )
            k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                layer_id,
                end_seq_position,
                block_tables[batch_idx],
            )

            extend_seq_len = end_seq_position - start_seq_position
            ks = torch.full(
                (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
            )
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_offset = torch.arange(
                start_seq_position + 1,
                end_seq_position + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke_offset_list.append(ke_offset)
            actual_seq_q = torch.tensor(
                [extend_seq_len], dtype=torch.int32, device="cuda"
            )
            actual_seq_q_list.append(actual_seq_q)
            batch_idx_list.append(batch_idx)

        k_fp8 = torch.cat(k_fp8_list, dim=0)#.view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        k_scale /= 127
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke_offset = torch.cat(ke_offset_list, dim=0)
        ke = ks + ke_offset
        actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
        
        context_q_lens_xpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_q_lens_cpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cpu")
        context_k_lens_xpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_k_lens_cpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cpu")

        logits = torch.zeros((q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device)
        kunlun_ops.I8_mqa_logits(
            q=q_fp8,
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights,
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )
        topk_result = metadata.topk_transform(
            logits,
            indexer.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
            batch_idx_list=batch_idx_list,
        )
    else:
        kv_len = (
            forward_batch.seq_lens_cpu[0].item()
            - forward_batch.extend_seq_lens_cpu[0]
            + kv_len
        )
        k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )
        k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )

        #k_fp8 = k_fp8.view(torch.float8_e4m3fn)
        k_scale = k_scale.view(torch.float32).squeeze(-1)
        #k_scale /= 127
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
        ke_offset = torch.arange(
            (kv_len - actual_seq_q) + 1,
            kv_len + 1,
            dtype=torch.int32,
            device="cuda",
        )
        ke = ks + ke_offset

        context_q_lens_xpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_q_lens_cpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cpu")
        context_k_lens_xpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_k_lens_cpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cpu")

        logits = torch.zeros((q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device)
        kunlun_ops.I8_mqa_logits(
            q=q_fp8,
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights,
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )
        actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
            device="cuda", non_blocking=True
        )
        topk_result = metadata.topk_transform(
            logits,
            indexer.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
        )

    return topk_result


# 0.5.8 production kept an escape hatch on the vendor key-LayerNorm
# (``patch/layers/attention/nsa/nsa_indexer.py:119``); keep the same name so a
# run can be put back on upstream's torch path without a code change.
_KLX_KNORM = os.environ.get("KLX_KNORM", "1") != "0"


def _k_norm(indexer, key: torch.Tensor) -> torch.Tensor:
    """LayerNorm the indexer key with the vendor op instead of ``F.layer_norm``.

    ``indexer.k_norm`` is upstream's ``LayerNorm(head_dim, dtype=torch.float32)``
    (``dsa/dsa_indexer.py:378-385``), and on this platform its ``forward_cuda``
    falls straight through to ``forward_native``: ``flashinfer.norm.layernorm``
    does not exist in this image, so ``_flashinfer_layernorm_available`` is
    False (``layers/layernorm.py:1042-1049``). ``forward_native`` is up-cast to
    fp32 -> ``F.layer_norm`` -> down-cast, three kernels, and the middle one
    traces as ``xpukernel_xpu3::layer_norm_fwd`` -- torch_xmlir's own kernel
    (``layer_norm_fwd`` is in ``torch_xmlir/libxpuapi.so``; it does not exist in
    ``kunlun_ops/libapiinfer.so`` at all).

    0.5.8 replaced exactly this call with the vendor op:
    ``sgl-kernel/python/sgl_kernel/patch/layers/attention/nsa/nsa_indexer.py:122-139``
    (``_k_norm_``) does ``kunlun_ops.layernorm(key, self.k_norm.weight, key,
    self.k_norm.variance_epsilon, bias=self.k_norm.bias)``, gated on
    ``KLX_KNORM``, ``key.dim() == 2`` and ``key.is_contiguous()``, and defers to
    ``self.k_norm(key)`` otherwise. ``kunlun_ops.layernorm``
    (``kunlun_ops/_xflashinfer_norm.py:363``) -> ``xpu_flash_ops.layernorm`` ->
    ``optimized_ops::layer_norm<T, TW>``, which is the only thing that launches
    ``xpuops3::infer_layer_norm_no_ada`` (disassembled
    ``optimized_ops::layer_norm<float16, float>``: it calls
    ``__device_stub__infer_layer_norm_no_ada<float16, float>``). ``<T, TW>`` is
    (activation, weight/bias); ``<bfloat16, float>`` and ``<float16, float>``
    are both instantiated, so a half-precision key with the fp32 weight and bias
    is the kernel's native argument shape and it accumulates in fp32 inside --
    which is what the up-cast in ``forward_native`` was there for.

    Two deviations from the 0.5.8 guard, both forced by 0.5.23:

    * 0.5.23 folds ``wk`` and ``weights_proj`` into one Linear
      (:func:`_indexer_wk_weights`), so ``key`` arrives as a column slice of the
      fused output and is *not* contiguous. The vendor op rejects a strided
      input (``layernorm, Tensor x is not contiguous!``), so the slice is packed
      here rather than handed back to torch.
    * The result goes to a separate buffer instead of aliasing ``key``. The
      caller mutates the norm's output afterwards (``key[..., :rope_head_dim] =
      k_rope``) and upstream's ``forward_native`` returned a fresh tensor, so
      writing in place would start mutating the fused projection's output; it
      also keeps the fallback below usable after a failed call.

    ``index_k_norm_type == "rms"`` builds an ``RMSNorm`` here instead
    (``dsa_indexer.py:375-376``), which is different math and has no bias, so
    the affine/bias attributes are the gate rather than the class name.
    """

    norm = indexer.k_norm
    if (
        not _KLX_KNORM
        or key.dim() != 2
        or not getattr(norm, "elementwise_affine", False)
        or not getattr(norm, "use_bias", False)
        or getattr(norm, "weight", None) is None
        or getattr(norm, "bias", None) is None
    ):
        return norm(key)
    packed = key.contiguous()
    out = torch.empty_like(packed)
    try:
        kunlun_ops.layernorm(
            packed,
            norm.weight,
            out,
            norm.variance_epsilon,
            bias=norm.bias,
        )
    except Exception:
        return norm(key)
    return out


def _get_q_k_bf16(
    indexer,
    q_lora: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
):
    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        # for kunlun ops
        query, _ = indexer.wq_b(q_lora.to(indexer.wq_b.weight.dtype))
        # The DSA indexer chain runs in bf16: the Hadamard rotation is bf16-only
        # and `prefill_attention_dsa_opt` requires q/kv/out/lse_out to agree
        # (measured: passing fp16 q with bf16 kv raises `Check failed,
        # prefill_attention_dsa_opt not support q/kv/out/lse_out/indices dtype`,
        # and passing fp16 q with bf16 k raises `query and key must have same
        # dtype` in the rotary). 0.5.23 builds the fused `wk_weights_proj` at
        # `params_dtype=bfloat16` while this run serves GLM-5.3 at
        # `--dtype float16`, so the projection's own dtype is the one to follow
        # rather than the model dtype.
        query = query.to(torch.bfloat16)
        query = rearrange(query, "l (h d) -> l h d", d=indexer.head_dim)
        q_rope, _ = torch.split(
            query,
            [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
            dim=-1,
        )
        with torch.cuda.stream(indexer.alt_stream):
            # TODO we should also put DeepGEMM half SM here?
            key, _ = _indexer_wk_weights(indexer, x)
            key = _k_norm(indexer, key)

            k_rope, _ = torch.split(
                key,
                [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
                dim=-1,
            )

        current_stream.wait_stream(indexer.alt_stream)
    else:
        query, _ = indexer.wq_b(q_lora.to(indexer.wq_b.weight.dtype))
        # The DSA indexer chain runs in bf16: the Hadamard rotation is bf16-only
        # and `prefill_attention_dsa_opt` requires q/kv/out/lse_out to agree
        # (measured: passing fp16 q with bf16 kv raises `Check failed,
        # prefill_attention_dsa_opt not support q/kv/out/lse_out/indices dtype`,
        # and passing fp16 q with bf16 k raises `query and key must have same
        # dtype` in the rotary). 0.5.23 builds the fused `wk_weights_proj` at
        # `params_dtype=bfloat16` while this run serves GLM-5.3 at
        # `--dtype float16`, so the projection's own dtype is the one to follow
        # rather than the model dtype.
        query = query.to(torch.bfloat16)
        query = rearrange(query, "l (h d) -> l h d", d=indexer.head_dim)
        q_rope, _ = torch.split(
            query, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
        )
        key, _ = _indexer_wk_weights(indexer, x)
        key = _k_norm(indexer, key)
        k_rope, _ = torch.split(
            key, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
        )

    q_rope, k_rope = indexer.rotary_emb(positions, q_rope, k_rope)

    query[..., : indexer.rope_head_dim] = q_rope
    key[..., : indexer.rope_head_dim] = k_rope

    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        query = rotate_activation(query)

        with torch.cuda.stream(indexer.alt_stream):
            key = rotate_activation(key)
        current_stream.wait_stream(indexer.alt_stream)
    else:
        query = rotate_activation(query)
        key = rotate_activation(key)

    # allgather+rerrange
    if is_cp_active(forward_batch):
        # Strategy CP (``--cp-strategy interleave``): the strategy owns the gather, so
        # it restores the original token order and drops the per-rank padding.
        # ``cp_all_gather_rerange_output`` cannot do that here: for round-robin
        # it returns ``local_rows * cp_size`` rows, and the local rows are padded
        # to ``max_rank_len`` whenever ``total_tokens % cp_size != 0``. The K
        # store pairs one key row with one ``forward_batch.out_cache_loc`` entry,
        # so the extra padded rows make the Kunlun store abort with
        # "set_k_and_s_triton: k_num != loc_num".
        key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
    elif dsa_use_prefill_cp(forward_batch):
        key = cp_all_gather_rerange_output(
            key.contiguous(),
            # 0.5.23 dropped the `cp_size` snapshot from Indexer; the live
            # value is on the parallel state (`kunlun_deepseek_v4_backend.py`
            # reads it the same way).
            get_parallel().attn_cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )

    return query, key


def _forward_cuda_k_only(
    indexer,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    act_quant,
    enable_dual_stream: bool,
    metadata: "BaseIndexerMetadata",
    return_indices: bool = True,
):
    assert forward_batch.forward_mode.is_extend_without_speculative()

    # Fast path: only compute and store k cache, skip all q and weights ops
    # 0.5.19 dropped the ``enable_dual_stream`` parameter from ``_get_k_bf16``.
    key = indexer._get_k_bf16(x, positions)
    key_shape = key.shape
    key = key.view(-1, indexer.block_size)
    k_fp8 = torch.empty(
        key.shape,
        device=key.device,
        dtype=torch.int8,
    )
    k_scale = torch.empty(
        [key.shape[0], 1],
        device=key.device,
        dtype=torch.float32,
    )
    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

    get_token_to_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )

    # MHA doesn't need topk_indices
    if not return_indices:
        return None

    # MLA: every position is selected when the sequence is not longer than
    # index_topk, so the fused top-k result is just each token's own cache slots
    # in order, padded with -1. Build that page table directly instead of running
    # the topk kernel on dummy logits: on 0.5.14 the Kunlun topk_transform
    # requires the score width to fit inside the page-1 table, which
    # `[tokens, index_topk]` dummy logits violate for short sequences.
    seq_lens_expanded = metadata.get_seqlens_expanded()
    page_table_1 = metadata.get_page_table_1()
    token_to_batch_idx = metadata.get_token_to_batch_idx()
    topk = indexer.index_topk
    device = page_table_1.device
    width = page_table_1.shape[1]

    positions = torch.arange(topk, device=device)
    src_index = token_to_batch_idx.to(torch.int64).unsqueeze(1) * width + (
        positions.clamp(max=width - 1)
    )
    slots = (
        page_table_1.reshape(-1)
        .index_select(0, src_index.reshape(-1))
        .view(-1, topk)
    )
    valid = positions.unsqueeze(0) < seq_lens_expanded.unsqueeze(1)
    return torch.where(valid, slots, -torch.ones_like(slots)).to(torch.int32)


# ``MultiPlatformOp`` dispatches to ``forward_xpu`` on this platform (0.5.14 dropped
# ``forward_native``), so both entry points resolve to the Kunlun implementation.
@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda",
    type=HookType.REPLACE,
)
def forward_cuda(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    return_indices: bool = True,
    # 0.5.23 threaded this through from the piecewise/breakable graph path; the
    # Kunlun body refuses that path outright (see the assertion below), so the
    # value cannot change the result -- it only has to be accepted so a caller
    # passing it by keyword does not raise TypeError.
    use_cuda_graph: bool = True,
) -> Optional[torch.Tensor]:
    """forward_cuda"""
    # TODO: try to remove this patch

    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)
    # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
    x_meta = x[0] if isinstance(x, tuple) else x

    # Kunlun does not support upstream's TC piecewise cuda graph path (it splits the
    # graph around store_k_cache/mqa_logits and fetches metadata inside custom ops).
    # Fail loudly rather than silently running with metadata=None.
    assert (
        not is_in_tc_piecewise_cuda_graph()
    ), "kunlun dsa indexer does not support TC piecewise cuda graph"

    metadata = get_attn_backend().get_indexer_metadata(
        layer_id, forward_batch
    )

    _resolve_deps(indexer)

    # skip DSA if attention backend choose to skip this batch
    if metadata is None:
        return None

    # Determine if should skip topk based on sequence length
    # We can only skip the logits computation if cuda graph is not involved
    skip_logits_computation = False
    if forward_batch.forward_mode.is_extend_without_speculative():
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            skip_logits_computation = max_kv_len <= indexer.index_topk

    # Optimization: fast path when skipping topk computation
    if skip_logits_computation and (not is_dsa_enable_prefill_cp()):
        topk_result = _forward_cuda_k_only(indexer,
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            False,  # dual stream is never enabled on kunlun
            metadata,
            return_indices,
        )
        # 0.5.14 additions: TP broadcast + state capture on every return path
        topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
        return maybe_capture_indexer_topk(layer_id, topk_result)

    query, key = _get_q_k_bf16(indexer,
            q_lora, x, positions, False, forward_batch=forward_batch
        )
    # ``_get_q_k_bf16`` already gathered the key to the full sequence (CP) and
    # restored the token order, so it pairs row-for-row with the (also full)
    # ``forward_batch.out_cache_loc``.
    key_shape = key.shape
    key = key.view(-1, indexer.block_size)
    # TODO: try to remove this to act_quant
    query_shape = query.shape
    query = query.view(-1, indexer.block_size)
    q_fp8 = torch.empty(
        query.shape,
        device=query.device,
        dtype=torch.int8,
    )
    q_scale = torch.empty(
        [query.shape[0], 1],
        device=query.device,
        dtype=torch.float32,
    )
    k_fp8 = torch.empty(
        key.shape,
        device=key.device,
        dtype=torch.int8,
    )
    k_scale = torch.empty(
        [key.shape[0], 1],
        device=key.device,
        dtype=torch.float32,
    )

    kunlun_ops.quant2d(query, q_fp8, q_scale, force_sdnn=True)
    q_fp8 = q_fp8.view(query_shape)
    q_scale = q_scale.view(query_shape[0], -1)

    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)

    # Under prefill CP the gathered K covers the CP-padded token count (the round-robin
    # split requires a multiple of cp_size) while out_cache_loc only lists the real
    # tokens. The rerange restores token order, so the padding is the tail.
    store_k_fp8, store_k_scale = k_fp8, k_scale
    loc_rows = forward_batch.out_cache_loc.shape[0]
    if store_k_fp8.shape[0] != loc_rows:
        if store_k_fp8.shape[0] < loc_rows:
            raise RuntimeError(
                "kunlun DSA indexer: fewer K rows than cache locations "
                f"(key={tuple(store_k_fp8.shape)}, loc={loc_rows}, "
                f"cp_size={getattr(indexer, 'cp_size', None)}, "
                f"cp_meta={forward_batch.attn_cp_metadata is not None})"
            )
        store_k_fp8 = store_k_fp8[:loc_rows]
        store_k_scale = store_k_scale[:loc_rows]

    # `_get_logits_head_gate` expects a Tensor. For tuple activations, dequantize
    # to a float tensor here (callsite), keeping `_get_logits_head_gate` backend-agnostic.
    if isinstance(x, tuple):
        assert len(x) in (
            2,
            3,
        ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
        x_q, x_s = x[0], x[1]
        if (
            x_s is not None
            and x_q.dim() == 2
            and x_s.dim() == 2
            and x_q.shape[0] == x_s.shape[0]
        ):
            m, n = x_q.shape
            ng = x_s.shape[1]
            if ng > 0 and n % ng == 0:
                group = n // ng
                x_for_gate = (
                    x_q.to(torch.float32)
                    .view(m, ng, group)
                    .mul_(x_s.to(torch.float32).unsqueeze(-1))
                    .view(m, n)
                    .to(torch.bfloat16)
                )
            else:
                x_for_gate = x_q.to(torch.bfloat16)
        else:
            x_for_gate = x_q.to(torch.bfloat16)
    else:
        x_for_gate = x
    weights = _get_logits_head_gate(indexer, x_for_gate, q_scale)
    # k_fp8: (seq_len, head_dim) fp8_e4m3fn
    # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
    # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
    # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
    get_token_to_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=store_k_fp8,
        index_k_scale=store_k_scale,
    )
    # The old first line here read `forward_batch.seq_lens_cpu[0].item()` and
    # stored it in a `seq_len` nothing ever used, *before* the assert that
    # guards the same attribute. On 0.5.23's decode path the CPU mirror is
    # legitimately None (the metadata is device-only), so the dead read became
    # `TypeError: 'NoneType' object is not subscriptable` at the first decode
    # step. The assert below is the real contract; keep it, drop the dead read.
    if q_fp8.dtype is torch.int8:
        weights /= 127

    assert forward_batch.seq_lens_cpu is not None

    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        topk_result = _get_topk_paged(indexer,
            forward_batch, layer_id, q_fp8, weights, metadata
        )
    else:
        if (
            forward_batch.attn_cp_metadata is not None
            and is_dsa_prefill_cp_in_seq_split()
        ):
            kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
            kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
            actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
            actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]

            # TODO support mutil-batch
            # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
            # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
            # TODO prev, next, combined into a single call
            q_fp8_prev, q_fp8_next = torch.split(
                q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
            )
            weights_prev, weights_next = torch.split(
                weights, (weights.shape[0] + 1) // 2, dim=0
            )
            topk_result_prev = _get_topk_ragged_with_cp(indexer,
                forward_batch,
                layer_id,
                q_fp8_prev,
                weights_prev,
                metadata,
                kv_len_prev,
                actual_seq_q_prev,
            )

            topk_result_next = _get_topk_ragged_with_cp(indexer,
                forward_batch,
                layer_id,
                q_fp8_next,
                weights_next,
                metadata,
                kv_len_next,
                actual_seq_q_next,
            )
            topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)
            topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
            return maybe_capture_indexer_topk(layer_id, topk_result)
        else:
            topk_result = _get_topk_ragged(indexer,
                False,  # enable_dual_stream: never used on kunlun
                forward_batch, layer_id, q_fp8, weights, metadata
            )
    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
    return maybe_capture_indexer_topk(layer_id, topk_result)
