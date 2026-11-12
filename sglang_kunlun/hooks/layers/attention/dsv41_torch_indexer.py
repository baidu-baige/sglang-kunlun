"""Share eager prefill geometry across layers through the 0.5.22 scoring seam.

Selection and candidate publication remain upstream-owned. The opt-in cache
stores only request geometry, never projected queries, keys or scores.
"""

from __future__ import annotations

import os
import logging
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
    rows = inputs.x.shape[0]
    topk = int(indexer.index_topk)
    if rows == 0 or topk <= 0 or inputs.positions.numel() != rows:
        return False
    raw, page = inputs.out_raw_indices, inputs.out_page_indices
    if raw.shape[0] < rows or raw.shape[1] < topk:
        return False
    if page is not None and (page.shape[0] < rows or page.shape[1] < topk):
        return False

    req = inputs.req_rows
    if req is None:
        counts = inputs.rows_per_request
        if (
            counts is None or len(counts) != inputs.req_pool_indices.numel()
            or any(n < 0 for n in counts) or sum(counts) != rows
        ):
            return False
        pairs = [(r, n) for r, n in zip(inputs.req_pool_indices.tolist(), counts) if n]
        req_ids, row_counts = [r for r, _ in pairs], [n for _, n in pairs]
    else:
        if req.numel() != rows:
            return False
        ids, counts = torch.unique_consecutive(req, return_counts=True)
        req_ids, row_counts = ids.tolist(), counts.tolist()
        if len(set(req_ids)) != len(req_ids):
            return False
    if not req_ids or len(req_ids) > 64:
        return False
    ratio = inputs.compress_ratio
    pos = inputs.positions
    visible = ((pos + 1) // ratio).to(torch.int32)
    lengths, slots, start = [], [], 0
    for request, count in zip(req_ids, row_counts):
        length = int(visible[start:start + count].max().item())
        if length < 0 or length * ratio > self.req_to_token.shape[1]:
            return False
        lengths.append(length)
        if length:
            columns = torch.arange(length, device=pos.device)
            slots.append(self.req_to_token[request, columns * ratio].to(torch.int64) // ratio)
        start += count
    if not slots:
        inputs.reset_outputs()
        return True
    slots = torch.cat(slots)
    keys = self.token_to_kv_pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    from sglang_kunlun.hooks.layers.attention.dsv41_sparse import _fused_logits_usable

    if not _fused_logits_usable(q, keys, weights):
        return False

    def offsets(counts):
        values = [0]
        for count in counts:
            values.append(values[-1] + count)
        return torch.tensor(values, device=pos.device, dtype=torch.int32)

    sizes = [n * length for n, length in zip(row_counts, lengths)]
    # Both vendor kernels address logits with int32 offsets.
    if sum(sizes) > 2**31 - 1:
        return False
    qo, ko, lo = offsets(row_counts), offsets(lengths), offsets(sizes)
    logits = torch.empty(sum(sizes), device=pos.device, dtype=torch.float32)
    w_op = weights if weights.dtype in (torch.bfloat16, torch.float32) else weights.float()
    torch.ops.xdll_ops.dsv4_low_ratio_index_logits(
        q.to(torch.bfloat16).contiguous(), keys.to(torch.bfloat16).contiguous(),
        w_op.contiguous(), qo, ko, visible.contiguous(), logits, lo,
    )
    selected_page, selected_raw = torch.ops.xspeedgate_ops.dsv4_low_ratio_index_topk_prefill(
        logits, lo, qo, ko, visible.contiguous(), slots.contiguous(), topk,
    )
    inputs.reset_outputs()
    raw[:rows, :topk] = selected_raw
    if page is not None:
        page[:rows, :topk] = selected_page
    global _LOGGED_FUSED_PREFILL
    if not _LOGGED_FUSED_PREFILL:
        _LOGGED_FUSED_PREFILL = True
        logger.info(
            "Low-ratio prefill: fused logits + top-k (%d requests, %d rows)",
            len(req_ids), rows,
        )
    return True


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


def _scoring_fwd_kwargs(original_fn, req_to_token):
    """0.5.23 dropped `req_to_token` from `prefill_requests`; pass it only if taken."""
    import inspect

    try:
        takes = "req_to_token" in inspect.signature(original_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        takes = False
    return {"req_to_token": req_to_token} if takes and req_to_token is not None else {}


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.v41_indexer.scoring.prefill_requests",
    type=HookType.AROUND,
)
def prefill_requests_kunlun(
    original_fn, *, inputs, token_to_kv_pool, req_to_token=None
):
    """Reuse eligible prefill request geometry while computing fresh layer scores.

    Args:
        original_fn: Upstream generator used when geometry caching is unavailable.
        inputs: Indexer inputs and output state for this layer.
        token_to_kv_pool: Pool providing dequantized low-ratio index keys.
        req_to_token: Request-to-physical-token mapping used to build key slots.
            0.5.23 no longer passes it -- upstream reads the slots off
            ``inputs`` -- so it defaults to None and the eager prep cache takes
            it from the active backend instead.

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
            **_scoring_fwd_kwargs(original_fn, req_to_token),
        )
        return
    backend = _ACTIVE_BACKEND.get()
    if req_to_token is None and backend is not None:
        req_to_token = getattr(backend, "req_to_token", None)
    if (
        os.environ.get("DSV4_LRI_TORCH_PREP_CACHE", "0") != "1"
        or backend is None
        or inputs.req_rows is None
        or req_to_token is None
        or torch.is_inference(inputs.req_rows)
        or torch.is_inference(inputs.positions)
        or torch.is_inference(req_to_token)
    ):
        yield from original_fn(
            inputs=inputs,
            token_to_kv_pool=token_to_kv_pool,
            **_scoring_fwd_kwargs(original_fn, req_to_token),
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
