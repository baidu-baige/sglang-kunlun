"""Share eager prefill geometry across layers through the 0.5.22 scoring seam.

Selection and candidate publication remain upstream-owned. The opt-in cache
stores only request geometry, never projected queries, keys or scores.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend."
_ACTIVE_BACKEND = ContextVar("kunlun_prefill_indexer_backend", default=None)
_CACHE_ATTR = "_kunlun_lri_prep"
logger = logging.getLogger(__name__)
_LOGGED_FUSED_PREFILL = False


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.v41_indexer.has_dense_fp4_indexer",
    type=HookType.AROUND,
)
def dense_fp4_indexer_layout_guard(original_fn):
    """The 0522 factory must not send int8 rows to a packed FP4 reader."""
    from sglang_kunlun.hooks.mem_cache.common import low_ratio_index_k_int8_enabled

    return False if low_ratio_index_k_int8_enabled() else original_fn()


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.v41_indexer.full_topk.FullTopKIndexer.topk_prefill",
    type=HookType.AROUND,
)
def full_topk_prefill_fused_kunlun(original_fn, self, inputs):
    """Pack 0519's fused logits/top-k at the 0522 full-top-k boundary.

    Candidate publishers and consumers have separate entrypoints in 0522 and
    retain their upstream semantics. CP supplies per-request local row counts.
    """
    if _full_topk_prefill_fused(self, inputs):
        return None
    return original_fn(self, inputs)


def _full_topk_prefill_fused(self, inputs) -> bool:
    if os.environ.get("SGLANG_KUNLUN_DSV41_FUSED_PREFILL_TOPK", "0") != "1":
        return False
    if not hasattr(torch.ops.xspeedgate_ops, "dsv4_low_ratio_index_topk_prefill"):
        return False
    indexer = inputs.indexer
    if indexer.is_candidate_source or indexer.uses_candidates:
        return False
    geo = _build_prefill_geometry(self, inputs)
    if geo is None:
        return False
    inputs.reset_outputs()
    if geo["empty"]:
        return True
    page, raw = _geometry_topk(geo)
    rows, topk = geo["rows"], geo["topk"]
    inputs.out_raw_indices[:rows, :topk] = raw
    if inputs.out_page_indices is not None:
        inputs.out_page_indices[:rows, :topk] = page
    global _LOGGED_FUSED_PREFILL
    if not _LOGGED_FUSED_PREFILL:
        _LOGGED_FUSED_PREFILL = True
        logger.info(
            "Low-ratio prefill: fused logits + top-k (%d requests, %d rows)",
            len(geo["req_ids"]), rows,
        )
    return True


_DENSE = "sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks.DenseBlocksBackend."


@plugin_hook(_DENSE + "publish_prefill", type=HookType.AROUND)
def publish_prefill_fused_kunlun(original_fn, self, inputs):
    """Candidate source: one whole-forward fused logits + top-k pass, then the
    upstream block selection on the same fp32 logits.
    """
    if not _fused_candidates_usable():
        return original_fn(self, inputs)
    geo = _build_prefill_geometry(self, inputs)
    if geo is None:
        return original_fn(self, inputs)
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        select_candidate_block_ids,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds

    inputs.reset_outputs()
    dev = inputs.positions.device
    rows_total = inputs.positions.shape[0]
    if geo["empty"]:
        return BlockIds(
            blocks=torch.zeros((rows_total, 0), dtype=torch.int32, device=dev),
            rows_per_request=inputs.rows_per_request,
        )
    page, raw = _geometry_topk(geo)
    rows, topk = geo["rows"], geo["topk"]
    inputs.out_raw_indices[:rows, :topk] = raw
    if inputs.out_page_indices is not None:
        inputs.out_page_indices[:rows, :topk] = page

    # Block selection reuses the logits the top-k just read. The op leaves the
    # positions past a row visible length undefined, so mask them exactly like
    # upstream _publish_tile_blocks does before selecting (in place: the top-k is
    # already done with this buffer).
    lo = geo["lo"].tolist()
    picked, width, start = [], 0, 0
    for i, (cnt, length) in enumerate(zip(geo["row_counts"], geo["lengths"])):
        if length:
            view = geo["logits"][lo[i] : lo[i] + cnt * length].view(cnt, length)
            lens_i = geo["visible"][start : start + cnt].to(torch.int64)[:, None]
            cols = torch.arange(length, device=dev)
            view.masked_fill_(cols[None, :] >= lens_i, float("-inf"))
            ids = select_candidate_block_ids(
                view,
                lens_i,
                topk_blocks=self.topk_blocks,
                block_size=self.block_size,
            )
            width = max(width, ids.shape[1])
            picked.append((start, cnt, ids))
            if _VERIFY:
                _verify_view(view, lens_i.squeeze(1), raw[start : start + cnt], topk)
        start += cnt
    blocks = torch.full((rows_total, width), -1, dtype=torch.int32, device=dev)
    for start, cnt, ids in picked:
        blocks[start : start + cnt, : ids.shape[1]] = ids
    return BlockIds(blocks=blocks, rows_per_request=inputs.rows_per_request)


@plugin_hook(_DENSE + "consume_prefill", type=HookType.AROUND)
def consume_prefill_fused_kunlun(original_fn, self, inputs, published):
    """Candidate consumer: when the published blocks cover every block of the visible
    context, topk_among_blocks degenerates to a full-width top-k, so one fused
    whole-forward pass computes the same selection. Otherwise fall through.
    """
    if not _fused_candidates_usable() or published is None:
        return original_fn(self, inputs, published)
    if not _candidates_unconstrained(self, inputs):
        return original_fn(self, inputs, published)
    geo = _build_prefill_geometry(self, inputs)
    if geo is None:
        return original_fn(self, inputs, published)
    inputs.reset_outputs()
    if geo["empty"]:
        return None
    page, raw = _geometry_topk(geo)
    rows, topk = geo["rows"], geo["topk"]
    inputs.out_raw_indices[:rows, :topk] = raw
    if inputs.out_page_indices is not None:
        inputs.out_page_indices[:rows, :topk] = page
    if _VERIFY:
        lo = geo["lo"].tolist()
        start = 0
        for i, (cnt, length) in enumerate(zip(geo["row_counts"], geo["lengths"])):
            if length:
                view = geo["logits"][lo[i] : lo[i] + cnt * length].view(cnt, length)
                lens_i = geo["visible"][start : start + cnt].to(torch.int64)
                cols = torch.arange(length, device=view.device)
                view.masked_fill_(cols[None, :] >= lens_i[:, None], float("-inf"))
                _verify_view(view, lens_i, raw[start : start + cnt], topk)
            start += cnt
    return None


@plugin_hook(_BACKEND + "_low_ratio_index_topk", type=HookType.AROUND)
def index_topk_context(original_fn, self, *args, **kwargs):
    """Expose the active backend during top-k selection and restore it afterward.

    Args:
        original_fn: Upstream backend top-k method.
        args: Positional arguments forwarded to the method.
        kwargs: Keyword arguments forwarded to the method.

    Returns:
        The upstream top-k result.
    """
    token = _ACTIVE_BACKEND.set(self)
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        _ACTIVE_BACKEND.reset(token)


def _reset(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    self._kunlun_lri_prep = {}
    return result


@plugin_hook(_BACKEND + "init_forward_metadata_prefill", type=HookType.AROUND)
def _reset_prefill(original_fn, self, *args, **kwargs):
    return _reset(original_fn, self, *args, **kwargs)


@plugin_hook(_BACKEND + "init_forward_metadata_decode", type=HookType.AROUND)
def _reset_decode(original_fn, self, *args, **kwargs):
    return _reset(original_fn, self, *args, **kwargs)


@plugin_hook(_BACKEND + "init_forward_metadata_target_verify", type=HookType.AROUND)
def _reset_verify(original_fn, self, *args, **kwargs):
    return _reset(original_fn, self, *args, **kwargs)


def _prep(backend, inputs, req_to_token):
    req, pos, ratio = inputs.req_rows, inputs.positions, inputs.compress_ratio
    meta = backend.forward_metadata
    cache = getattr(backend, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(backend, _CACHE_ATTR, cache)
    hit = cache.get(ratio)
    # Tensor versions also reject in-place replay updates to otherwise identical
    # tensor objects; metadata identity rejects a new forward altogether.
    key = (req._version, pos._version, req_to_token._version)
    if hit is not None and hit[0] is meta and hit[1] is req and hit[2] is pos:
        if hit[3] is req_to_token and hit[4] == key:
            return hit[5]
    lens_all = (pos + 1) // ratio
    geometry = []
    for r in torch.unique_consecutive(req).tolist():
        tok = (req == r).nonzero().squeeze(1)
        lens = lens_all[tok]
        lc = int(lens.max().item())
        if lc == 0:
            continue
        columns = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, columns * ratio].to(torch.int64) // ratio
        geometry.append((lc, columns, tok, lens, slots))
    cache[ratio] = (meta, req, pos, req_to_token, key, geometry)
    return geometry


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.v41_indexer.scoring.prefill_requests",
    type=HookType.AROUND,
)
def prefill_requests_kunlun(original_fn, *, inputs, token_to_kv_pool, req_to_token):
    """Reuse eligible prefill request geometry while computing fresh layer scores.

    Args:
        original_fn: Upstream generator used when geometry caching is unavailable.
        inputs: Indexer inputs and output state for this layer.
        token_to_kv_pool: Pool providing dequantized low-ratio index keys.
        req_to_token: Request-to-physical-token mapping used to build key slots.

    Yields:
        RequestScores geometry paired with its score-chunk iterator.
    """
    if inputs.req_rows is None:
        from msgspec.structs import replace

        # CP interleaves the flattened batch, preserving request order locally.
        # These counts already include each request's global starting offset
        # (or its late-layer tail layout); positions already include KV prefixes.
        counts = inputs.rows_per_request
        if (
            counts is None
            or len(counts) != inputs.req_pool_indices.numel()
            or any(n < 0 for n in counts)
            or sum(counts) != inputs.positions.numel()
        ):
            raise ValueError("CP indexer requires per-request counts matching local rows")
        req_rows = torch.repeat_interleave(
            inputs.req_pool_indices,
            torch.tensor(counts, dtype=torch.int64, device=inputs.req_pool_indices.device),
            output_size=inputs.positions.numel(),
        )
        # PrefillInputs is frozen; retain the caller's output buffers and local
        # row order, and keep CP geometry out of the optional eager cache.
        yield from original_fn(
            inputs=replace(inputs, req_rows=req_rows),
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
        )
        return
    backend = _ACTIVE_BACKEND.get()
    if (
        os.environ.get("DSV4_LRI_TORCH_PREP_CACHE", "0") != "1"
        or backend is None
        or inputs.req_rows is None
        or torch.is_inference(inputs.req_rows)
        or torch.is_inference(inputs.positions)
        or torch.is_inference(req_to_token)
    ):
        yield from original_fn(
            inputs=inputs, token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token
        )
        return
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
        RequestScores,
        _score_chunks,
    )

    inputs.reset_outputs()
    indexer = inputs.indexer
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[inputs.positions])
    weights = indexer.head_weights(inputs.x)
    for lc, columns, tok, lens, slots in _prep(backend, inputs, req_to_token):
        request = RequestScores(
            lc=lc, k=min(indexer.index_topk, lc), columns=columns,
            tok=tok, lens=lens, slots=slots,
        )
        keys = token_to_kv_pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
        yield request, _score_chunks(
            indexer=indexer, q=q, weights=weights, index_k=keys, request=request
        )


# ---------------------------------------------------------------------------
# Candidate publish / consume through the fused prefill top-k op
# ---------------------------------------------------------------------------

# DSV4_INDEXER_FUSED_CANDIDATES=0 restores the upstream torch publish/consume.
_FUSED_CANDIDATES = os.environ.get("DSV4_INDEXER_FUSED_CANDIDATES", "1") == "1"
# DSV4_INDEXER_VERIFY=1 runs the upstream selection beside the fused one. Top-k ties
# may legitimately pick different positions, so the picked scores are compared, not
# the positions; positional disagreement is reported separately as a tie counter.
_VERIFY = os.environ.get("DSV4_INDEXER_VERIFY", "0") == "1"
# DSV4_INDEXER_FUSED_LOGITS_BUDGET_BYTES caps the whole-forward flat fp32 logits
# (sum over requests of rows*length, 4 bytes/element). Above it the fused path
# falls back to the upstream torch chunked path instead of materializing a
# multi-GB score matrix (1M context would be ~8 GB per rank).
_FUSED_LOGITS_BUDGET = int(
    os.environ.get("DSV4_INDEXER_FUSED_LOGITS_BUDGET_BYTES", str(2 * 2**30))
)
_verify = {
    "chunks": 0,
    "rows": 0,
    "tie_rows": 0,
    "count_bad_rows": 0,
    "rows_score_diff": 0,
    "max_score_diff": 0.0,
}


def _fused_candidates_usable() -> bool:
    return _FUSED_CANDIDATES and hasattr(
        torch.ops.xspeedgate_ops, "dsv4_low_ratio_index_topk_prefill"
    )


def _candidates_unconstrained(backend, inputs) -> bool:
    # ceil(lc / block_size) <= topk_blocks means select_candidate_block_ids keeps every
    # block, so the candidate mask admits all visible positions. seq_lens_cpu is the
    # host-side visible context per request, so this needs no device sync.
    bs = int(getattr(backend, "block_size", 0) or 0)
    topk_blocks = int(getattr(backend, "topk_blocks", 0) or 0)
    if bs < 1 or topk_blocks < 1:
        return False
    lens = inputs.seq_lens_cpu
    if not lens:
        return False
    ratio = int(inputs.compress_ratio)
    lc_max = max((int(n) + 1) // ratio for n in lens)
    return -(-lc_max // bs) <= topk_blocks


def _verify_view(scores, lens, raw, topk) -> None:
    # Row-chunk the reference top-k: a whole candidate-source view is
    # [rows, lc] = [4096, 16640] fp32 and torch.topk over it needs several
    # hundred MB of scratch, which trips the allocator mid-forward.
    length = max(1, int(scores.shape[1]))
    step = max(1, (1 << 26) // (length * 4))
    for i in range(0, scores.shape[0], step):
        _verify_rows(scores[i : i + step], lens[i : i + step], raw[i : i + step], topk)


def _verify_rows(scores, lens, raw, topk) -> None:
    # Compare the fused pick against a torch top-k on the very same fp32 logits.
    # Both sides are -1 padded / may point past a row visible length; those slots
    # carry no score, so they are folded to -inf on both sides before sorting,
    # otherwise a padded slot would be compared against a real score.
    length = scores.shape[1]
    k = min(int(topk), length)
    ref = scores.topk(k, dim=-1, sorted=False).indices
    got = raw[:, :k].to(torch.int64)
    neg = torch.finfo(torch.float32).min
    lens2 = lens[:, None]
    ref_ok = (ref >= 0) & (ref < lens2)
    got_ok = (got >= 0) & (got < lens2)
    s_ref = torch.where(ref_ok, scores.gather(1, ref.clamp(0, length - 1)), neg)
    s_got = torch.where(got_ok, scores.gather(1, got.clamp(0, length - 1)), neg)
    diff = (s_ref.sort(dim=-1).values - s_got.sort(dim=-1).values).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    ties = (
        torch.where(ref_ok, ref, torch.full_like(ref, -1)).sort(dim=-1).values
        != torch.where(got_ok, got, torch.full_like(got, -1)).sort(dim=-1).values
    ).any(dim=-1)
    _verify["chunks"] += 1
    _verify["rows"] += int(scores.shape[0])
    _verify["tie_rows"] += int(ties.sum().item())
    _verify["count_bad_rows"] += int((ref_ok.sum(-1) != got_ok.sum(-1)).sum().item())
    _verify["max_score_diff"] = max(_verify["max_score_diff"], float(diff.max().item()))
    _verify["rows_score_diff"] += int((diff > 1e-3).any(dim=-1).sum().item())
    if _verify["chunks"] % 20 == 0:
        logger.info("[kunlun] indexer verify: %s", dict(_verify))



def _build_prefill_geometry(backend, inputs):
    """Whole-forward geometry for the fused prefill ops, or None when uncovered.

    One flat fp32 logits buffer covering every row of every request, plus the
    offset vectors the fused ops index it with. Batching is the point: the
    upstream torch path scores and top-k's per request (R calls of each), while
    here one batched scoring + one batched fused top-k covers all rows at once --
    that batching is where the short-context win comes from.

    The flat buffer is capped by DSV4_INDEXER_FUSED_LOGITS_BUDGET_BYTES (default
    2 GiB): above it the fused path would materialize a multi-GB score matrix
    (1M context -> ~8 GB per rank), so those fall back to the upstream torch
    chunked path.
    """
    indexer = inputs.indexer
    rows = inputs.x.shape[0]
    topk = int(indexer.index_topk)
    if rows == 0 or topk <= 0 or inputs.positions.numel() != rows:
        return None
    raw_buf, page_buf = inputs.out_raw_indices, inputs.out_page_indices
    if raw_buf.shape[0] < rows or raw_buf.shape[1] < topk:
        return None
    if page_buf is not None and (page_buf.shape[0] < rows or page_buf.shape[1] < topk):
        return None

    req = inputs.req_rows
    if req is None:
        counts = inputs.rows_per_request
        if (
            counts is None
            or len(counts) != inputs.req_pool_indices.numel()
            or any(n < 0 for n in counts)
            or sum(counts) != rows
        ):
            return None
        pairs = [(r, n) for r, n in zip(inputs.req_pool_indices.tolist(), counts) if n]
        req_ids, row_counts = [r for r, _ in pairs], [n for _, n in pairs]
    else:
        if req.numel() != rows:
            return None
        ids, counts = torch.unique_consecutive(req, return_counts=True)
        req_ids, row_counts = ids.tolist(), counts.tolist()
        if len(set(req_ids)) != len(req_ids):
            return None
    if not req_ids or len(req_ids) > 64:
        return None

    ratio = inputs.compress_ratio
    pos = inputs.positions
    visible = ((pos + 1) // ratio).to(torch.int32)
    lengths, slots, start = [], [], 0
    for request, count in zip(req_ids, row_counts):
        length = int(visible[start : start + count].max().item())
        if length < 0 or length * ratio > backend.req_to_token.shape[1]:
            return None
        lengths.append(length)
        if length:
            columns = torch.arange(length, device=pos.device)
            slots.append(
                backend.req_to_token[request, columns * ratio].to(torch.int64) // ratio
            )
        start += count
    if not slots:
        return {"empty": True}

    sizes = [n * length for n, length in zip(row_counts, lengths)]
    # The flat fp32 logits is the one big buffer this path materializes; cap it
    # so long contexts fall back to the upstream torch chunked path instead of
    # allocating a multi-GB score matrix. The int32 bound is the vendor kernels'
    # offset limit (a hard ceiling independent of the tunable budget).
    if sum(sizes) * 4 > _FUSED_LOGITS_BUDGET or sum(sizes) > 2**31 - 1:
        return None

    slots = torch.cat(slots)
    keys = backend.token_to_kv_pool.get_low_ratio_index_k_dequant(
        inputs.layer_id, slots
    )
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    from sglang_kunlun.hooks.layers.attention.dsv41_sparse import _fused_logits_usable

    if not _fused_logits_usable(q, keys, weights):
        return None

    def offsets(counts):
        values = [0]
        for count in counts:
            values.append(values[-1] + count)
        return torch.tensor(values, device=pos.device, dtype=torch.int32)

    qo, ko, lo = offsets(row_counts), offsets(lengths), offsets(sizes)
    logits = torch.empty(sum(sizes), device=pos.device, dtype=torch.float32)
    w_op = (
        weights if weights.dtype in (torch.bfloat16, torch.float32) else weights.float()
    )
    torch.ops.xdll_ops.dsv4_low_ratio_index_logits(
        q.to(torch.bfloat16).contiguous(),
        keys.to(torch.bfloat16).contiguous(),
        w_op.contiguous(),
        qo,
        ko,
        visible.contiguous(),
        logits,
        lo,
    )
    return {
        "empty": False,
        "logits": logits,
        "qo": qo,
        "ko": ko,
        "lo": lo,
        "visible": visible.contiguous(),
        "slots": slots.contiguous(),
        "req_ids": req_ids,
        "row_counts": row_counts,
        "lengths": lengths,
        "rows": rows,
        "topk": topk,
    }


def _geometry_topk(geo):
    return torch.ops.xspeedgate_ops.dsv4_low_ratio_index_topk_prefill(
        geo["logits"],
        geo["lo"],
        geo["qo"],
        geo["ko"],
        geo["visible"],
        geo["slots"],
        geo["topk"],
    )
