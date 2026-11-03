"""Share eager prefill geometry across layers through the 0.5.22 scoring seam.

Selection and candidate publication remain upstream-owned. The opt-in cache
stores only request geometry, never projected queries, keys or scores.
"""

from __future__ import annotations

import os
from contextvars import ContextVar

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend."
_ACTIVE_BACKEND = ContextVar("kunlun_prefill_indexer_backend", default=None)
_CACHE_ATTR = "_kunlun_lri_prep"


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
