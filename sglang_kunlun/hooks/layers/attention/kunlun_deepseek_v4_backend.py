"""Kunlun DeepSeek-V4 compressed-attention backend.

The upstream 0.5.14 backend owns metadata preparation and graph lifecycle.  This
module only replaces the FlashMLA-specific metadata/forward boundary with the
Kunlun compressed-attention operator.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import functools
import inspect
import logging
import os
import cocopod  # noqa: F401  # Preserve target operator registration.
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as upstream
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4AttnMetadata,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import (
    _dsv4_topk_torch_fallback,
    build_causal_swa_page_indices_kunlun,
    dsv4_c4_paged_mqa_logits_torch,
    dsv4_compressed_attention_torch,
    dsv4_quant_k_cache_kunlun,
    dsv4_set_k_and_s_with_mapping_kunlun,
    get_paged_mqa_logits_metadata_kunlun,
    page_table_positions_kernel,
)

_DSV4_TENSOR_DUMP_CALLBACK = None

def _hybrid_kernel_note(stage, layer, capturing, capture_phase, detail=""):
    """Compatibility no-op; capture bookkeeping and exit reports are removed."""
    return None


def set_dsv4_tensor_dump_callback(callback):
    """Sets a callback to dump DSV4 attention tensors."""
    global _DSV4_TENSOR_DUMP_CALLBACK
    _DSV4_TENSOR_DUMP_CALLBACK = callback
    setattr(torch, "_dsv4_tensor_dump_callback", callback)


def _dsv4_dump_backend_layers():
    raw = os.getenv("DSV4_DEBUG_TENSOR_DUMP_LAYERS", "")
    layers = set()
    for token in raw.replace(",", " ").split():
        try:
            layers.add(int(token))
        except ValueError:
            continue
    return layers


def dsv4_stable_dsa_enabled() -> bool:
    """One-switch entry point for a reproducible DSA forward pass.

    The DSV4 compressed-attention path consumes two XSpeedGate operators whose
    results depend on the KV physical page layout (measured on P800 with
    DeepSeek-V4-Flash W8A8: the same request returns different tokens depending
    on which pages the allocator handed out, which makes greedy output
    non-reproducible and shows up as score jitter in thinking-mode evals):

      * c4 indexer scores  -- ``mqa_logits_paged`` / ``c4a_paged_mqa_logits``
      * compressed attention -- ``compressed_attention``

    Setting ``DSV4_KUNLUN_STABLE_DSA=1`` routes exactly those two operators
    through their Torch reference implementations (equivalent to setting
    ``DSV4_KUNLUN_REFERENCE_C4=1`` plus ``DSV4_KUNLUN_REFERENCE_ATTN=1``), which
    makes repeated runs bit-identical at a modest throughput cost.  Everything
    else keeps using the vendor operators.

    Measured 2026-10-23 (P800, TP8, greedy, same request repeated):

      prompt tokens   default            + STABLE_DSA        + STABLE_DSA and
                                                              BKCL_STABLE_NCORES
      899             top-3 logit jitter  exact               exact
      7911            jitter ~0.05-0.18   jitter ~0.05         jitter ~0.005
      45138           jitter ~0.10,       jitter ~0.10         jitter ~0.10
                      argmax flips

    So two independent noise sources exist.  The short-context one is the XCCL
    ring all-reduce: ``bkcl.h`` only promises an identical accumulation order
    across devices for ``bkcl_all_reduce_stable``, and ``BKCL_STABLE_NCORES=16``
    switches ``ring_all_reduce`` to that kernel (``device_wrapper_kl3.cpp``).
    The long-context one lives in vendor kernels that have no reference
    implementation (compressor store and the main paged attention): it survives
    ``DSV4_KUNLUN_REFERENCE_ONLY=1``, ``--disable-cuda-graph``, non-speculative
    decoding, ``SGLANG_OPT_USE_COMPRESSOR_V2=0``, ``--boundary-reduction ar`` and
    ``--chunked-prefill-size 262144``, so it needs layout-invariant vendor
    kernels.

    NOTE: the compressor operators (``dsv4_compress_forward_v2`` /
    ``dsv4_compress_norm_rope_store_v2``) have no Torch reference hook yet, so a
    small residual run-to-run variation can still remain; a fully deterministic
    forward needs the vendor kernels to become layout-invariant.
    """
    return os.environ.get("DSV4_KUNLUN_STABLE_DSA", "0") == "1"


def _dsv4_dump_enabled() -> bool:
    """Whether the attention-chain dump is on.

    Call sites must check this before building probe values: the values cost
    `torch.unique` / `index_select` / scalar H2D pushes per layer per forward.
    """
    return os.getenv("TENSOR_DUMP_DSV4_ATTN_CHAIN", "0") == "1"


def _dsv4_dump_backend_tensor(backend, name, value, layer=None):
    if os.getenv("TENSOR_DUMP_DSV4_ATTN_CHAIN", "0") != "1":
        return
    selected = _dsv4_dump_backend_layers()
    layer_id = getattr(layer, "layer_id", None)
    if selected and layer_id is not None and layer_id not in selected:
        # Every DSV4 layer reaches this boundary; without the filter the last
        # layer would overwrite the selected layer's keys under the same name.
        return
    callback = getattr(torch, "_dsv4_tensor_dump_callback", None)
    if callback is None:
        callback = _DSV4_TENSOR_DUMP_CALLBACK
    if callback is not None and isinstance(value, torch.Tensor):
        callback(name, value)


logger = logging.getLogger(__name__)


def _clamp_c128_prefill_topk(
    compressed_topk: int,
    kv_lens_cpu: Optional[torch.Tensor],
    extend_seq_lens_cpu: Optional[List[int]],
    compress_ratio: int,
) -> int:
    if kv_lens_cpu is None or kv_lens_cpu.numel() == 0 or not extend_seq_lens_cpu:
        return compressed_topk
    chunk_start_pos = int(kv_lens_cpu[0]) - max(extend_seq_lens_cpu)
    if chunk_start_pos <= 0:
        return compressed_topk
    return min(compressed_topk, max(chunk_start_pos // compress_ratio, 1))


_TOPK_VEC_PAGE_CAP_WARNED: set = set()


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.indexer.topk_transform_pytorch_vectorized",
    type=HookType.AROUND,
)
def topk_transform_pytorch_vectorized_kunlun(
    original_fn,
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    out_raw_indices=None,
):
    """Clamp seq_lens to the page-table-addressable length so top-k cannot pick
    positions beyond the page table and read out of bounds.

    Upstream `_topk_transform_vectorized` applies only a lower bound to page_idx,
    no upper bound:

        page_idx = raw_indices >> page_bits
        physical_pages = torch.gather(page_tables, 1, page_idx.clamp(min=0).long())

    The `c4_seq_lens` passed in from the C4 indexer decode/verify path
    (`run_topk_transform` in `indexer.py`) is the live compressed length, while
    `logits` / `page_table` are buffers fixed at graph-capture time. Under long
    context the live length can exceed what the page table covers, so the chosen
    position is shifted right out of the page table and the gather goes
    out-of-bounds immediately -- on Kunlun this surfaces as gather_2d's
    "load/store operation exceed memory size", reported as
    status=700 illegal memory access, and the scheduler exits directly.

    Clamping rationale: positions beyond `page_tables.shape[1] * page_size` have
    no page-table entry to address anyway. After clamping they are masked to
    -inf in `valid_mask` and never enter top-k. A hit warns once -- at that
    point the top-k range for the long-context rows has been truncated, which
    loses precision, so it must be visible in the logs.
    """
    width = int(scores.shape[1])
    cap = width
    if page_tables is not None and page_tables.dim() == 2:
        page_cap = int(page_tables.shape[1]) * int(page_size)
        if page_cap < cap:
            cap = page_cap
            key = (page_cap, width)
            if key not in _TOPK_VEC_PAGE_CAP_WARNED:
                _TOPK_VEC_PAGE_CAP_WARNED.add(key)
                logger.warning(
                    "DSV4 topk: page table addresses %d compressed tokens but the "
                    "score buffer is %d wide; clamping seq_lens to the page table "
                    "(long-context rows get truncated)",
                    page_cap,
                    width,
                )
    return original_fn(
        scores,
        seq_lens.clamp(max=cap),
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
    )


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend._create_flashmla_metadata",
    type=HookType.REPLACE,
)
def _create_flashmla_metadata_kunlun():
    """Disable CUDA FlashMLA metadata on Kunlun.

    The companion registration on
    ``deepseek_v4_backend_hip_radix._create_flashmla_metadata`` is gone: that
    module is ROCm-only and does not import on this box (``No module named
    'triton.experimental'``), so the hook could never resolve and every boot
    logged "Failed to apply hooks". Kunlun never selects the HIP radix backend.
    """

    return None


def _flash_mla_with_kvcache_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int = 512,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run sparse FlashMLA attention with a Torch implementation.

    Kunlun does not provide the CUDA ``flash_mla_with_kvcache`` entrypoint.
    The cache layout is already token-addressable after flattening, so this
    follows the 0514 reference path and returns the same ``(output, lse)``
    tuple as the original kernel.
    """
    batch_size, query_len, num_heads, head_dim_qk = q.shape
    topk = indices.shape[-1]
    if softmax_scale is None:
        softmax_scale = head_dim_qk**-0.5

    def gather_cache(cache, cache_indices, length):
        cache_flat = cache.reshape(-1, head_dim_qk).float()
        safe_indices = cache_indices.clamp_min(0).clamp_max(cache_flat.shape[0] - 1)
        gathered = cache_flat[safe_indices.reshape(-1)].reshape(
            batch_size, query_len, -1, head_dim_qk
        )
        invalid = cache_indices < 0
        if length is not None:
            positions = torch.arange(
                cache_indices.shape[-1], device=q.device
            ).view(1, 1, -1)
            invalid = invalid | (positions >= length.view(batch_size, 1, 1))
        return gathered, invalid

    gathered_kv, invalid_mask = gather_cache(k_cache, indices, topk_length)
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        extra_kv, extra_invalid = gather_cache(
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )
        gathered_kv = torch.cat((gathered_kv, extra_kv), dim=2)
        invalid_mask = torch.cat((invalid_mask, extra_invalid), dim=2)

    total_topk = gathered_kv.shape[2]
    gathered_kv = torch.nan_to_num(gathered_kv).reshape(
        batch_size * query_len, total_topk, head_dim_qk
    )
    q_float = q.float().reshape(batch_size * query_len, num_heads, head_dim_qk)
    invalid_mask = invalid_mask.reshape(
        batch_size * query_len, 1, total_topk
    )

    scores = torch.bmm(
        q_float, gathered_kv.transpose(1, 2)
    ) * softmax_scale
    scores.masked_fill_(invalid_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    output = torch.bmm(weights, gathered_kv[..., :head_dim_v])

    output = output.reshape(batch_size, query_len, num_heads, head_dim_v)
    lse = lse.reshape(batch_size, query_len, num_heads)
    if attn_sink is not None:
        sink_scale = torch.sigmoid(lse - attn_sink.view(1, 1, num_heads))
        output = output * sink_scale.unsqueeze(-1)

    lonely = lse == float("-inf")
    output = output.masked_fill(lonely.unsqueeze(-1), 0.0)
    lse = lse.masked_fill(lonely, float("inf"))
    return output, lse.transpose(1, 2)


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_with_kvcache",
    type=HookType.REPLACE,
)
@plugin_hook(
    "sgl_kernel.flash_mla_with_kvcache",
    type=HookType.REPLACE,
)
def flash_mla_with_kvcache_kunlun(*args, **kwargs):
    """Kunlun equivalent; does not execute the CUDA FlashMLA kernels."""
    if args:
        raise TypeError("Kunlun FlashMLA replacement requires keyword arguments")
    q = kwargs.get("q")
    fallback_to_reference = kwargs.pop("kunlun_fallback_to_reference", True)
    result = _try_flash_mla_hybrid(**kwargs)
    if result is not None or not fallback_to_reference:
        return result
    # Neither hybrid nor this reference can reinterpret packed bytes as K/V.
    for name in ("k_cache", "extra_k_cache"):
        cache = kwargs.get(name)
        if cache is not None and (
            cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or cache.shape[-1] != kwargs["q"].shape[-1]
        ):
            raise ValueError("Kunlun FlashMLA reference requires typed unpacked KV")
    result = _flash_mla_with_kvcache_reference(**kwargs)
    return result


def _hybrid_trace_before(arguments, layer_id, context):
    """Compatibility seam for migrated callers; production tracing is removed."""
    return None


def _hybrid_trace_after(record, arguments, rc):
    """No diagnostic synchronization or tensor inspection in production."""
    return None


_KUNLUN_HYBRID_SEQ_LENS_ATTR = "_kunlun_hybrid_seq_lens_i32"
_KUNLUN_HYBRID_SEQ_LENS_FRESH_ATTR = "_kunlun_hybrid_seq_lens_fresh"


def _hybrid_seq_lens_cache(owner):
    cache = getattr(owner, _KUNLUN_HYBRID_SEQ_LENS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(owner, _KUNLUN_HYBRID_SEQ_LENS_ATTR, cache)
    return cache


def _hybrid_seq_lens_key(seq_lens, n):
    return (seq_lens.device.type, seq_lens.device.index, n)


def _hybrid_seq_lens_prerefresh_enabled() -> bool:
    """``SGLANG_KUNLUN_DSV4_HYBRID_PREREFRESH``: A/B switch for the hoisted mirror.

    Default on.  Set to ``0`` and ``init_forward_metadata_out_graph`` stops
    refreshing the int32 ``seq_lens`` mirror, so the decode layer loop falls
    back to casting per layer and the capture records one copy per hybrid layer
    again -- the pre-hoist behaviour, same binary, for measuring what the
    hoist is worth.
    """

    return os.environ.get("SGLANG_KUNLUN_DSV4_HYBRID_PREREFRESH", "1") != "0"


def _report_topk_route(route, raw_indices, capture_enabled, hisparse_decode) -> None:
    """Compatibility no-op; does not alter the selected top-k implementation."""
    return None


def _low_ratio_index_bmm_enabled() -> bool:
    """``DSV4_LOW_RATIO_INDEX_BMM``: batched-matmul form of the indexer scores.

    Default on.  Set to ``0`` to restore the broadcast-multiply-then-sum form,
    which is the same contraction but materialises an ``[rows, heads, chunk,
    head_dim]`` intermediate -- the reason the low-ratio decode indexer runs
    out of memory when the chunk or the row block grows.  Kept as an A/B
    because the two forms accumulate in different precision.
    """

    return os.environ.get("DSV4_LOW_RATIO_INDEX_BMM", "1") != "0"


def _refresh_hybrid_seq_lens_i32(owner, forward_batch) -> None:
    """Refresh the hybrid adapter's int32 ``seq_lens`` mirror for this forward.

    The decode layer loop used to cast per layer: the buffer is keyed by
    ``(device, bucket)`` and *not* by layer, so all ~38 hybrid layers wrote
    byte-identical lengths into the same address and the capture recorded every
    one of those copies.  Replaying them is pure overhead.  One refresh here
    serves every layer and lets the capture record nothing.

    Stays device-only on pointer-stable tensors because this hook is itself
    capturable (``MetadataGlueGraph`` records ``init_forward_metadata_out_graph``
    per replay key): a host-computed source would replay frozen at its
    capture-time value.  ``forward_batch.seq_lens`` is the runner's static graph
    buffer, already padded to the bucket with ``seq_len_fill_value``, which is
    what the per-layer copy read too -- so the padded tail keeps its old values.

    The stamp records which bucket was refreshed; ``_hybrid_seq_lens_i32``
    refuses to trust a buffer this hook did not just fill.  Like
    ``_kunlun_capture_phase`` next to it, this relies on the hook running once
    per forward ahead of the layers, which is the contract of
    ``init_forward_metadata``/the graph runner's replay prep.
    """

    setattr(owner, _KUNLUN_HYBRID_SEQ_LENS_FRESH_ATTR, None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    if seq_lens is None or seq_lens.numel() == 0:
        return
    n = seq_lens.numel()
    cache = _hybrid_seq_lens_cache(owner)
    key = _hybrid_seq_lens_key(seq_lens, n)
    buffer = cache.get(key)
    if buffer is None:
        buffer = torch.empty(n, dtype=torch.int32, device=seq_lens.device)
        cache[key] = buffer
    buffer.copy_(seq_lens)
    setattr(owner, _KUNLUN_HYBRID_SEQ_LENS_FRESH_ATTR, key)


def _hybrid_seq_lens_i32(owner, seq_lens, n, capturing):
    """Fixed-address int32 mirror of ``seq_lens[:n]``.

    ``ForwardBatch.seq_lens`` is int64 in sglang's graph slot while the operator
    wants contiguous int32, and a per-call ``.to(torch.int32)`` would hand
    capture a fresh allocation on every replay.  So one buffer is kept per
    ``(device, n)`` and ``_refresh_hybrid_seq_lens_i32`` fills it once per
    forward, outside the graph; the layer loop below is then a pure lookup and
    the capture records no copy at all.

    The in-place copy is the fallback for callers that never run that hook
    (standalone users and tests).  It is a device-side op, so under capture it
    records once per layer -- exactly the cost the refresh exists to remove.

    Returns ``None`` when no buffer exists for this shape under capture; the
    capture itself must never allocate.
    """

    cache = _hybrid_seq_lens_cache(owner)
    key = _hybrid_seq_lens_key(seq_lens, n)
    buffer = cache.get(key)
    if (
        buffer is not None
        and getattr(owner, _KUNLUN_HYBRID_SEQ_LENS_FRESH_ATTR, None) == key
    ):
        return buffer
    if buffer is None:
        if capturing:
            return None
        buffer = torch.empty(n, dtype=torch.int32, device=seq_lens.device)
        cache[key] = buffer
    buffer.copy_(seq_lens[:n])
    return buffer


def _kunlun_hybrid_decode_enabled() -> bool:
    """``SGLANG_KUNLUN_DSV4_HYBRID_DECODE``: kill switch for the hybrid decode adapter.

    Default on (``"1"``), so an unset environment keeps the current behaviour.
    Set to ``0`` and ``_try_flash_mla_hybrid`` bails on every call, which drops
    decode onto the legacy ``xspeedgate_ops.compressed_attention`` kernel. The
    DSV4 call site passes ``kunlun_fallback_to_reference=False``, so this is a
    true A/B against the legacy kernel -- it does NOT divert to the slow torch
    reference.

    The guard lives inside the adapter rather than at the call sites so it also
    covers the upstream ``flash_mla_with_kvcache`` REPLACE hook, not just the
    direct DSV4 decode call.
    """
    return os.environ.get("SGLANG_KUNLUN_DSV4_HYBRID_DECODE", "1") == "1"


def _try_flash_mla_hybrid(
    q, k_cache, indices, *, head_dim_v=512, softmax_scale=None,
    attn_sink=None, topk_length=None, extra_k_cache=None,
    extra_indices_in_kvcache=None, extra_topk_length=None,
    kunlun_decode_lengths=None, kunlun_compress_ratio=None,
    kunlun_layer_id=None, kunlun_forward_batch=None, kunlun_backend=None, **_,
):
    """Eager decode adapter; None retains the caller's existing fallback.

    Only typed, full 512-dimensional K=V caches are supported. FlashMLA FP8
    (528/584 bytes) and Kunlun mixed INT8 (604 bytes) are NOT this layout.
    These length/index checks synchronize; never use them in graph capture.
    """
    capturing = q.is_cuda and torch.cuda.is_current_stream_capturing()
    # ``capturing`` is False during the two eager warmup passes sglang runs
    # before every graph capture, yet those passes are exactly as synthetic as
    # the capture itself: ``seq_lens`` is the graph fill value and the window
    # lengths come from a dummy page table.  The backend records the phase from
    # ``init_forward_metadata_out_graph(..., in_capture=...)``, which runs once
    # per replay with False, so use that -- not ``capturing`` -- to decide
    # whether the data-dependent checks below are looking at a real batch.
    capture_phase = bool(getattr(kunlun_backend, "_kunlun_capture_phase", False))

    def bail(reason):
        """Keep the fallback decision without diagnostic bookkeeping."""
        return None

    if not _kunlun_hybrid_decode_enabled():
        return bail("kill_switch")

    if (
        kunlun_compress_ratio not in (1, 2, 4, 128)
        or q.ndim != 4 or q.shape[1] != 1 or q.shape[0] == 0
        or q.shape[-1] != 512 or head_dim_v != 512
        or q.dtype not in (torch.float16, torch.bfloat16)
        or extra_k_cache is None or extra_indices_in_kvcache is None
        or topk_length is None or extra_topk_length is None
    ):
        return bail("unsupported-shape-or-metadata")
    n, _, heads, dim = q.shape
    if attn_sink is not None and (
        attn_sink.shape != (heads,) or attn_sink.device != q.device
    ):
        return bail("attn-sink-shape")
    caches = (k_cache, extra_k_cache)
    for name, cache in zip(("win", "com"), caches):
        if (
            cache.dtype != q.dtype or cache.device != q.device
            or cache.ndim < 2 or cache.shape[-1] != dim
            or cache.numel() == 0 or not cache.is_contiguous()
        ):
            return bail(f"cache-layout:{name}")
    lengths = kunlun_decode_lengths
    if lengths is not None:
        if (
            lengths.device.type != "cpu" or lengths.numel() != n
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            return bail("decode-lengths-host-mismatch")
        # ``kunlun_decode_lengths`` is the host-side compatibility argument.
        # The scheduler commonly publishes it as CPU int64.  Converting that
        # host tensor to contiguous CPU int32 is capture-safe; unlike a
        # device-side conversion, it does not enqueue work on the captured
        # CUDA stream.  The authoritative replay-time lengths remain
        # ``kunlun_forward_batch.seq_lens`` below.
        lengths = lengths.reshape(n).to(torch.int32).contiguous()
    win_idx, com_idx = indices, extra_indices_in_kvcache
    for name, idx in zip(("win", "com"), (win_idx, com_idx)):
        if (
            idx.ndim != 3 or idx.shape[:2] != (n, 1)
            or idx.shape[-1] == 0 or idx.dtype != torch.int32
            or idx.device != q.device
        ):
            return bail(f"index-layout:{name}")
    # A padded index width is a stride, NOT the valid length. The scheduler
    # already publishes these lengths together with the index tensors. Do not
    # copy indices to the host or compare device metadata from every layer:
    # that turns a one-token decode into a per-layer synchronization point.
    # The scheduler's device sequence lengths are the authoritative per-step
    # values.  Do not manufacture a device tensor from the host copy in the
    # layer loop: the ForwardBatch already owns this tensor.
    if kunlun_forward_batch is not None:
        raw_seq_lens = kunlun_forward_batch.seq_lens
        if raw_seq_lens.device != q.device or raw_seq_lens.numel() < n:
            return bail("device-seq-lens-mismatch")
        # int64 in sglang's graph slot, int32 contiguous for the operator; the
        # mirror keeps one address for the whole capture/replay lifetime.
        device_lengths = _hybrid_seq_lens_i32(
            kunlun_backend or kunlun_forward_batch, raw_seq_lens, n, capturing
        )
        if device_lengths is None:
            return bail("no-int32-seq-lens-buffer-for-this-shape")
    else:
        if lengths is None:
            return bail("no-decode-lengths")
        if capturing:
            return bail("capture-without-forward-batch")
        device_lengths = lengths.to(q.device, dtype=torch.int32)
    if not capturing and bool((device_lengths <= 0).any()):
        return bail("non-positive-seq-lens")
    expected = (
        device_lengths.clamp_max(win_idx.shape[-1]),
        (device_lengths // kunlun_compress_ratio).clamp_max(com_idx.shape[-1]),
    )
    for name, idx, explicit, inferred, cache in zip(
        ("win", "com"), (win_idx, com_idx), (topk_length, extra_topk_length),
        expected, caches
    ):
        if (
            explicit.numel() != n
            or explicit.device != q.device
            or explicit.dtype not in (torch.int32, torch.int64)
        ):
            return bail(f"topk-length-tensor:{name}")
        if capturing and (
            explicit.dtype != torch.int32 or not explicit.is_contiguous()
        ):
            return bail(f"topk-length-layout-in-capture:{name}")
        # Both checks stay on the device.  The scalar reductions are deliberate:
        # this adapter has no asynchronous "try and report invalid metadata"
        # operator, so silently omitting them would turn malformed dynamic
        # indices into out-of-bounds reads.  Unlike the old implementation,
        # they do not copy an index tensor or a length tensor to the host.
        # Skipped for the whole capture phase, not merely for ``capturing``:
        # during it both sides are placeholders -- the dummy batch's ``seq_lens``
        # is the graph fill value while the window lengths come from a dummy page
        # table -- so the comparison says nothing about the batch the graph will
        # replay.  Replay re-runs ``init_forward_metadata_out_graph`` and never
        # calls this adapter, and real eager decode still reports
        # ``capture_phase`` False, so the guard keeps its teeth where it matters.
        if not capture_phase:
            if not torch.equal(explicit.reshape(n).to(torch.int32), inferred):
                # Eager only, so the D2H here is allowed and worth the diagnosis:
                # the capture warmup runs on a dummy batch whose ``seq_lens`` is
                # the graph fill value, and this check compares it against the
                # scheduler's real top-k lengths.
                return bail(
                    f"topk-lengths-disagree-with-seq-lens:{name} "
                    f"explicit={explicit.reshape(n)[:4].tolist()} "
                    f"inferred={inferred[:4].tolist()} seq_lens="
                    f"{device_lengths[:4].tolist()}"
                )
            active = (
                torch.arange(idx.shape[-1], device=q.device)[None, None, :]
                < inferred.reshape(n, 1, 1)
            )
            rows = cache.numel() // dim
            if bool((((idx < 0) | (idx >= rows)) & active).any()):
                return bail(f"index-out-of-range:{name}")
            if bool((inferred <= 0).any()):
                return bail(f"zero-inferred-topk:{name}")
    try:
        import kunlun_ops
    except ImportError:
        return bail("kunlun_ops-missing")
    op = getattr(kunlun_ops, "hybrid_attention", None)
    if op is None:
        return bail("hybrid_attention-op-missing")
    q3 = q.squeeze(1)
    if capturing and not q3.is_contiguous():
        return bail("q-non-contiguous-in-capture")
    graph_buffer_key = (
        kunlun_layer_id, n, heads, dim, q.dtype, q.device.type, q.device.index
    )
    if capturing:
        # CUDA graph replay requires the exact same storage addresses.  These
        # buffers are allocated during the eager warmup call and keyed by the
        # layer and graph bucket.  A graph capture that arrives before warmup
        # falls back instead of allocating from inside capture.
        graph_buffers = getattr(
            kunlun_backend, "_kunlun_hybrid_graph_buffers", None
        )
        if graph_buffers is None and kunlun_forward_batch is not None:
            graph_buffers = getattr(
                kunlun_forward_batch, "_kunlun_hybrid_graph_buffers", None
            )
        buffers = (
            graph_buffers.get(graph_buffer_key)
            if graph_buffers is not None else None
        )
        if buffers is None:
            return bail("no-graph-buffers-for-this-shape")
        out, lse, max_logits = buffers
    else:
        q3 = q3.contiguous()
        out = torch.empty_like(q3)
        lse = torch.empty((n, heads), dtype=torch.float32, device=q.device)
        max_logits = torch.empty_like(lse)
        if kunlun_backend is not None or kunlun_forward_batch is not None:
            owner = kunlun_backend or kunlun_forward_batch
            graph_buffers = getattr(owner, "_kunlun_hybrid_graph_buffers", None)
            if graph_buffers is None:
                graph_buffers = {}
                setattr(owner, "_kunlun_hybrid_graph_buffers", graph_buffers)
            graph_buffers[graph_buffer_key] = (out, lse, max_logits)
    if kunlun_forward_batch is not None:
        metadata_owner = kunlun_backend or kunlun_forward_batch
        metadata_cache = getattr(
            metadata_owner, "_kunlun_hybrid_decode_metadata", None
        )
        if metadata_cache is None:
            metadata_cache = {}
            setattr(
                metadata_owner,
                "_kunlun_hybrid_decode_metadata",
                metadata_cache,
            )
        # LoD is shape metadata, not request metadata.  Reuse it by device and
        # bucket; sequence lengths are refreshed below from ForwardBatch on
        # every call and are never keyed by data_ptr (ForwardBatch tensors are
        # commonly updated in-place between steps).
        bucket = n
        cache_key = (q.device.type, q.device.index, bucket)
        metadata = metadata_cache.get(cache_key)
        if metadata is None and capturing:
            return bail("no-lod-metadata-for-this-bucket")
        if metadata is None:
            lod_cpu = torch.arange(bucket + 1, dtype=torch.int32)
            metadata = (
                lod_cpu,
                lod_cpu.to(q.device),
            )
            metadata_cache[cache_key] = metadata
        lod_cpu, lod_xpu = metadata
        # Reuse the mirror built above: re-reading ``seq_lens`` here would
        # reintroduce the int64/layout mismatch this replaced.
        lengths_xpu = device_lengths
    else:
        # Standalone callers/tests do not have a ForwardBatch to own the
        # lifetime. The serving path above is the one that eliminates the
        # per-layer H2D copies.
        lod_cpu = torch.arange(n + 1, dtype=torch.int32)
        lod_xpu = lod_cpu.to(q.device)
        lengths_xpu = device_lengths
    # FlashMLA returns KV-only LSE. Fold sink ourselves so the contract does
    # not depend on whether hybrid includes sink in its returned LSE.
    op_arguments = dict(
        q=q3, win_kv_cache=k_cache.reshape(-1, dim),
        win_indices=win_idx.squeeze(1).contiguous(),
        com_kv_cache=extra_k_cache.reshape(-1, dim),
        com_indices=com_idx.squeeze(1).contiguous(),
        o=out, max_logits=max_logits, lse=lse,
        qlod_cpu=lod_cpu, qlod_xpu=lod_xpu,
        kvseqlen_cpu=lengths if lengths is not None else device_lengths,
        kvseqlen_xpu=lengths_xpu,
        sm_scale=dim**-0.5 if softmax_scale is None else softmax_scale,
        is_causal=True, max_window_size=win_idx.shape[-1],
        compress_ratio=kunlun_compress_ratio, com_topk=com_idx.shape[-1],
        attn_sink=None, side_stream=-1,
    )
    trace = _hybrid_trace_before(op_arguments, kunlun_layer_id, {
        "topk_length": topk_length, "extra_topk_length": extra_topk_length,
        "sink_folded_after_kernel": attn_sink,
    })

    _hybrid_kernel_note(
        "enter", kunlun_layer_id, capturing, capture_phase,
        f"ratio={kunlun_compress_ratio} bs={n}",
    )
    rc = op(**op_arguments)
    _hybrid_kernel_note(
        "exit", kunlun_layer_id, capturing, capture_phase, f"rc={rc}"
    )
    _hybrid_trace_after(trace, op_arguments, rc)
    # Never retry after a failed launch: the stream/device may be invalid.
    if isinstance(rc, int) and rc != 0:
        raise RuntimeError(f"kunlun_ops.hybrid_attention failed with rc={rc}")
    if attn_sink is not None:
        out = (out.float() * torch.sigmoid(
            lse - attn_sink.float().view(1, heads)
        ).unsqueeze(-1)).to(q.dtype)
    return out.unsqueeze(1), lse.unsqueeze(-1)


_KUNLUN_SPARSE_PREFILL_RATIOS = (0, 128)


def _kunlun_sparse_prefill_enabled() -> bool:
    """``SGLANG_KUNLUN_DSV4_SPARSE_PREFILL``: kill switch for the prefill branch.

    Needs the `flash_mla_sparse_fwd` entrypoint from kunlun_ops 0.1.237 (with
    the matching xpytorch, for the `dsa_prefill_with_mixed_cache` XFA symbol).
    On 0.1.227 the branch dies on the first prefill: that build compiles the
    kernel out below XHPC 20260820 and returns r=1.
    """
    return os.environ.get("SGLANG_KUNLUN_DSV4_SPARSE_PREFILL", "1") == "1"


def _resolve_flash_mla_sparse_fwd():
    """Return the ``flash_mla_sparse_fwd`` entrypoint, whichever module carries it.

    Newer builds moved this entrypoint out of kunlun_ops into chen_ops, so take
    whichever module actually carries it instead of assuming either.
    """
    import kunlun_ops

    sparse_fwd = getattr(kunlun_ops, "flash_mla_sparse_fwd", None)
    if sparse_fwd is None:
        try:
            import chen_ops
        except ImportError as exc:
            raise AttributeError(
                "flash_mla_sparse_fwd is missing from kunlun_ops and chen_ops "
                "is not installed"
            ) from exc
        sparse_fwd = getattr(chen_ops, "flash_mla_sparse_fwd", None)
        if sparse_fwd is None:
            raise AttributeError(
                "flash_mla_sparse_fwd is missing from both kunlun_ops and chen_ops"
            )
    return sparse_fwd


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
    type=HookType.REPLACE,
)
@plugin_hook(
    "sgl_kernel.flash_mla_sparse_fwd",
    type=HookType.REPLACE,
)
def flash_mla_sparse_fwd_kunlun(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kunlun ``flash_mla_sparse_fwd`` on ``kunlun_ops.flash_mla_sparse_fwd``.

    The CUDA op is ``sm90::fwd::sparse_attn_fwd_kernel`` from the FlashMLA fork
    (``torch.ops.sgl_kernel.sparse_prefill_fwd``); on Kunlun ``sgl_kernel`` is
    the stub package, so every caller would get a ``_StubModule``. kunlun_ops
    0.1.237 added a same-named entrypoint built to match that contract, with
    native ``topk_length`` and ``attn_sink`` -- measured against a torch
    reference at bf16 noise level (4e-3) for both. The older
    ``sparse_prefill_fwd`` is not used: it is compiled out below XHPC 20260820
    ("DO NOT SUPPORT dsa_prefill"), and ``sparse_prefill_fwd_opt`` leaves every
    lane of ``indices`` live, which the ragged ``topk_length`` rows need.

    The allocating Kunlun ABI takes indices as [s_q, h_kv, topk]; the older
    out-buffer ABI takes [s_q, topk]. Both use kv as [s_kv, 1, d_qk].
    ``is_causal`` stays off and the cu_seqlens stay None: the indices are
    rebased into a gathered workspace, so neither positional masking nor the
    batch split means anything to them.

    Args:
        q: Queries with shape [s_q, h_q, d_qk].
        kv: Gathered key/value workspace with one KV head.
        indices: Workspace indices per query, with an optional singleton head axis.
        sm_scale: Scale applied to attention logits.
        d_v: Value dimension of the attention output.
        attn_sink: Optional per-head attention sink.
        topk_length: Optional valid index-prefix length for each query.

    Returns:
        Attention output, maximum logits, and log-sum-exp tensors.
    """
    sparse_fwd = _resolve_flash_mla_sparse_fwd()

    if q.dim() != 3:
        raise ValueError(f"q must be [s_q, h_q, d_qk], got {tuple(q.shape)}")
    if indices.dim() == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"kunlun sparse prefill needs one kv head, got {tuple(indices.shape)}"
            )
    elif indices.dim() == 2:
        # The op now documents indices as [s_q, h_kv, topk]; keep the kv-head
        # axis the FlashMLA signature carries instead of squeezing it away, so
        # both caller shapes land on the one layout it expects.
        indices = indices.unsqueeze(1)
    else:
        raise ValueError(
            f"indices must be [s_q, h_kv, topk] or [s_q, topk], got {tuple(indices.shape)}"
        )
    kv3 = kv.reshape(kv.shape[0], 1, kv.shape[-1])

    # The kernel's own arguments, hoisted so the probe and the shape dump record
    # exactly what it receives (post-``contiguous`` / post-cast) rather than the
    # caller's views.
    q_arg = q.contiguous()
    kv_arg = kv3.contiguous()
    indices_arg = indices.to(torch.int32).contiguous()
    topk_length_arg = (
        topk_length.to(torch.int32).contiguous() if topk_length is not None else None
    )
    sink_arg = (
        attn_sink.to(torch.float32).contiguous() if attn_sink is not None else None
    )

    # kunlun_ops 0.1.237+ce32dc7a reshaped this entrypoint: the outputs are
    # allocated by the op and returned, the lods are renamed to
    # ``cu_seqlens_*``, and batch_num / max_seq_q / max_seq_kv / head_num /
    # kv_lora_rank / rope_head_dim / topk are all inferred from the shapes
    # instead of being passed. ``is_causal`` must stay explicit: it defaults to
    # True there but the indices are rebased into a gathered workspace, so
    # positional masking means nothing to them (same reason the cu_seqlens stay
    # None).
    if _sparse_mla_allocates_outputs():
        out, max_logits, lse, _ = sparse_fwd(
            q_arg,
            kv_arg,
            indices_arg,
            sm_scale,
            d_v=d_v,
            is_causal=False,
            topk_length=topk_length_arg,
            attn_sink=sink_arg,
        )
    else:
        # Earlier 0.1.237 wheels expose the explicit out-buffer ABI. Both
        # variants share a package version, so inspect the callable contract.
        tokens, heads, qk_dim = q_arg.shape
        if not 0 < d_v <= qk_dim:
            raise ValueError(f"invalid sparse MLA value width {d_v} for {qk_dim}")
        out = torch.empty((tokens, heads, d_v), dtype=q_arg.dtype, device=q_arg.device)
        max_logits = torch.empty((tokens, heads), dtype=torch.float32, device=q_arg.device)
        lse = torch.empty_like(max_logits)
        status = sparse_fwd(
            q=q_arg, kv=kv_arg, indices=indices_arg.squeeze(1),
            o=out, max_logits=max_logits, lse=lse,
            qlod_cpu=None, qlod_xpu=None, kvlod_cpu=None, kvlod_xpu=None,
            batch_num=1, max_seq_q=tokens, max_seq_kv=kv_arg.shape[0],
            head_num=heads, kv_lora_rank=d_v, rope_head_dim=qk_dim - d_v,
            softmax_scale=sm_scale, is_causal=False, topk=indices_arg.shape[-1],
            topk_length=topk_length_arg, attn_sink=None,
        )
        if status != 0:
            raise RuntimeError(f"Kunlun sparse MLA failed with status {status}")
        # The out-buffer wheel leaves empty rows unwritten and ignores its
        # attn_sink argument. Compute sink normalization from its natural-log
        # LSE explicitly, always passing None so newer out-buffer wheels cannot
        # apply the sink twice.
        columns = torch.arange(indices_arg.shape[-1], device=q_arg.device)
        live = indices_arg.squeeze(1) >= 0
        if topk_length_arg is not None:
            live = live & (columns[None, :] < topk_length_arg.reshape(-1, 1))
        nonempty = live.any(dim=-1)
        out.masked_fill_(~nonempty[:, None, None], 0)
        max_logits.masked_fill_(~nonempty[:, None], -float("inf"))
        lse.masked_fill_(~nonempty[:, None], -float("inf"))
        if sink_arg is not None:
            sink = sink_arg.reshape(1, heads)
            factor = torch.sigmoid(lse - sink)
            out.copy_((out.float() * factor[:, :, None]).to(out.dtype))
            max_logits.copy_(torch.maximum(max_logits, sink))
            lse.copy_(torch.logaddexp(lse, sink))
    return out, max_logits, lse


@functools.cache
def _sparse_mla_allocates_outputs() -> bool:
    sparse_fwd = _resolve_flash_mla_sparse_fwd()
    try:
        return "d_v" in inspect.signature(sparse_fwd).parameters
    except (ValueError, TypeError):
        # pybind11 builtins (e.g. the chen_ops build) may not expose a
        # signature; those follow the allocating ABI.
        return True


# ``get_paged_mqa_logits_metadata`` used to be replaced here with an empty
# placeholder: the upstream JIT kernel needs C++20 ``std::bit_cast`` (gcc-11+
# libstdc++) which this build env cannot provide. XSpeedGate now ships the op
# (``kernel_ops.get_paged_mqa_logits_metadata_kunlun``), and
# ``KunlunDeepseekV4AttnBackend.init_forward_metadata_indexer`` calls it to fill
# ``deep_gemm_metadata`` -- upstream's ``__post_init__`` short-circuits that field
# to ``None`` on Kunlun (``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`` is set), so the
# registered op would otherwise never run.


class KunlunDSV4AttnMetadata(DSV4AttnMetadata):
    """DSV4 metadata without CUDA FlashMLA scheduler objects."""

    def init_flashmla_related(
        self, is_prefill: bool = False, low_ratio_buffers=None
    ) -> None:
        """Initialize Kunlun sparse metadata without FlashMLA schedulers.

        Mirrors the upstream ratio split. V4-Flash stages carry ratio 4 (and
        possibly 128); the V4.1 checkpoint carries only ratios 0/1/2, so the c4
        buffers must stay unset there and the c1/c2 buffers must be allocated --
        otherwise the low-ratio path silently degrades to window-only attention
        (and an absent c4 pool crashes this method outright).

        ``low_ratio_buffers`` (0.5.22): the V4.1 decode path pre-builds the
        capture-shaped c1/c2 top-k buffers once
        (``build_low_ratio_metadata``) and hands the tuple in by keyword, so
        this method must adopt them instead of allocating fresh ones -- a fresh
        allocation on a graph replay writes into a tensor nothing captured
        reads. Upstream slices ``[2:4]`` for c1 and ``[6:8]`` for c2 under
        ``not is_prefill and low_ratios == (1, 2)``; mirror exactly that.
        """
        # 4.1 renamed the DSV4 metadata field (and the backend attribute) from
        # c4_sparse_topk/c4_topk to index_topk; the value is the same
        # `hf_text_config.index_topk` fallback to C4_TOPK.
        if self.index_topk not in (512, 1024):
            raise ValueError(f"unsupported Kunlun top-k: {self.index_topk}")
        if 4 in self.present_ratios:
            self.c4_sparse_topk_lengths = torch.clamp(
                self.c4_topk_lengths_clamp1, max=self.index_topk
            )
            self.c4_sparse_page_indices = torch.full(
                (self.c4_topk_lengths_clamp1.size(0), self.index_topk),
                -1,
                dtype=torch.int32,
                device=self.c4_topk_lengths_clamp1.device,
            )
            self.c4_sparse_page_indices = upstream._pad_last_dim(
                self.c4_sparse_page_indices
            )
        else:
            self.c4_sparse_topk_lengths = None
            self.c4_sparse_page_indices = None
        # Kunlun's custom prefill attention consumes physical page indices only.
        # Keep this unset to match the patched xspeedgate top-k path; a raw
        # buffer would force the 0.5.14 JIT wrapper into its PyTorch fallback.
        self.c4_sparse_raw_indices = None
        # V4.1 low-ratio (1/2) sparse top-k buffers. The indexer fills them in
        # place; the attention op reads them as (slot, -1-padded) extra indices.
        if low_ratio_buffers is not None:
            assert not is_prefill and tuple(self.low_ratios) == (1, 2), (
                "low_ratio_buffers is the V4.1 decode contract: "
                f"is_prefill={is_prefill} low_ratios={tuple(self.low_ratios)}"
            )
            (
                self.c1_sparse_topk_lengths,
                self.c1_sparse_page_indices,
            ) = low_ratio_buffers[2:4]
            (
                self.c2_sparse_topk_lengths,
                self.c2_sparse_page_indices,
            ) = low_ratio_buffers[6:8]
            self.c1_sparse_raw_indices = None
            self.c2_sparse_raw_indices = None
            self.c0_flashmla_metadata = None
            self.c1_flashmla_metadata = None
            self.c2_flashmla_metadata = None
            self.c4_flashmla_metadata = None
            self.c128_flashmla_metadata = None
            return
        for ratio in self.low_ratios:
            # 0.5.19's upstream metadata has only c4/c128 fields.  The
            # c1/c2 fields exist only on the V4.1 low-ratio metadata layout,
            # so do not even evaluate those attributes for the V4-Flash
            # ratio set (0/4/128).
            lengths = getattr(self, f"c{ratio}_topk_lengths_clamp1", None)
            assert lengths is not None, (
                f"present low ratio {ratio} without its topk lengths"
            )
            setattr(
                self,
                f"c{ratio}_sparse_topk_lengths",
                torch.clamp(lengths, max=self.index_topk),
            )
            setattr(
                self,
                f"c{ratio}_sparse_page_indices",
                upstream._pad_last_dim(
                    torch.full(
                        (lengths.size(0), self.index_topk),
                        -1,
                        dtype=torch.int32,
                        device=lengths.device,
                    )
                ),
            )
            # V4.1 prefill writes request-local positions as well as pool slots.
            # Decode keeps its page-only buffers, including graph-owned storage.
            setattr(
                self,
                f"c{ratio}_sparse_raw_indices",
                torch.full_like(
                    getattr(self, f"c{ratio}_sparse_page_indices"), -1
                ) if is_prefill else None,
            )
        self.c0_flashmla_metadata = None
        self.c1_flashmla_metadata = None
        self.c2_flashmla_metadata = None
        self.c4_flashmla_metadata = None
        self.c128_flashmla_metadata = None


def _copy_host_lengths_(destination, source, fill_value: int) -> None:
    count = min(source.numel(), destination.numel())
    destination[:count].copy_(source[:count])
    if count < destination.numel():
        destination[count:].fill_(fill_value)


def _as_packed(tensor: torch.Tensor, what: str) -> torch.Tensor:
    """Pack a tensor before handing it to a vendor op, warning once if it copied.

    The XSpeedGate / kunlun_ops kernels take raw ``data_ptr()`` and recompute the
    row stride from the declared shape; a strided view silently makes every row
    but the first read the wrong memory. Column slices of page tables are the
    usual source, so warn the first time each site actually needs the copy -- a
    silent ``.contiguous()`` would hide a real metadata-layout change.
    """
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _as_packed_out(tensor: torch.Tensor, what: str):
    """Return ``(kernel_target, writeback)`` for a vendor-op *output* tensor.

    ``out.contiguous()`` on a strided output is a silent data loss: the kernel
    fills the temporary copy and the caller keeps reading zeros. Hand back the
    packed buffer plus the original so the caller can copy the result home.
    """
    if tensor.is_contiguous():
        return tensor, None
    return tensor.contiguous(), tensor


def _is_graph_extend_mode(forward_mode) -> bool:
    return forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()


def _alloc_graph_extend_aux(
    batch_size: int, num_queries: int, device: torch.device
):
    num_queries_per_request = num_queries // batch_size if batch_size else 1
    q_lod_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32) * num_queries_per_request
    )
    q_lod = q_lod_cpu.to(device, non_blocking=False)
    kv_lens_cpu = torch.ones(batch_size, dtype=torch.int32)
    kv_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    query_lens = torch.full(
        (batch_size,),
        num_queries_per_request,
        dtype=torch.int32,
        device=device,
    )
    return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens, query_lens


def _get_graph_extend_aux(
    cache, batch_size: int, num_queries: int, device: torch.device
):
    if batch_size <= 0 or num_queries % batch_size:
        raise ValueError(
            "Kunlun DSV4 graph extend requires uniform queries per request: "
            f"batch_size={batch_size}, num_queries={num_queries}"
        )
    key = (batch_size, num_queries, device)
    aux = cache.get(key)
    if aux is None:
        aux = _alloc_graph_extend_aux(batch_size, num_queries, device)
        cache[key] = aux
    return aux


def _verify_row_lod_enabled(forward_batch) -> bool:
    """Eager TARGET_VERIFY must use per-row LoD, matching sglang 0.5.8.

    ``xspeedgate_ops.compressed_attention`` has no access to ``win_lengths`` /
    ``extra_lengths``; its only per-row mask comes from ``kv_lens`` plus the LoD
    layout. Treating gamma+1 query rows as one item that shares a request-level
    ``kv_len`` lets every row read the whole window -- including the draft
    tokens at the later verify positions, so the target directly copies the
    drafts it was supposed to verify.

    The 0.5.8 Kunlun backend (whose operator is already validated) uses the
    decode-style contract: ``qlod = arange(n+1)`` and
    ``kvseqlen = seq_lens_casual`` (the exact causal length L+j+1 at each verify
    position). ``_make_cp_prefill_lod`` constructs exactly this, so reuse it.

    During capture the graph-specific implementation is used (see
    ``_make_verify_row_lod_graph_safe``), because this construction needs H2D
    and cannot be captured into the graph. ``DSV4_VERIFY_ROW_LOD=0`` A/Bs back
    to the old behavior.
    """
    if forward_batch is None or not forward_batch.forward_mode.is_target_verify():
        return False
    if os.environ.get("DSV4_VERIFY_ROW_LOD", "1") != "1":
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return True


def _seq_lens_cpu_i32(forward_batch) -> torch.Tensor:
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.to("cpu", non_blocking=False)
        return seq_lens_cpu[: forward_batch.batch_size].to(torch.int32)

    # The pinned ``seq_lens_cpu`` mirror is not refreshed per CUDA-graph replay:
    # ``DecodeCudaGraphRunner`` fills its buffer once with ``seq_len_fill_value``
    # and only the eagle draft runners copy the live lengths in. Under DSPARK the
    # replay therefore reported length 1 per row, so ``context_lens_cpu`` for the
    # C4 operator came out as 0 and the target attended to nothing of its real
    # history (accept length 1.0 vs 1.8 on 0.5.19). The device tensor *is*
    # refreshed before replay, so read it instead -- once per forward, cached on
    # the batch object, because the C4 path asks per layer. Capture must not sync,
    # and the extend/prefill runners do refresh their mirror, so both are left on
    # the original path.
    mode = forward_batch.forward_mode
    if (
        (mode.is_decode_or_idle() or mode.is_target_verify())
        and not torch.cuda.is_current_stream_capturing()
    ):
        cached = getattr(forward_batch, "_dsv4_host_seq_lens", None)
        if cached is not None and cached.numel() == forward_batch.batch_size:
            return cached
        live = (
            forward_batch.seq_lens[: forward_batch.batch_size]
            .to("cpu", non_blocking=False)
            .to(torch.int32)
        )
        try:
            forward_batch._dsv4_host_seq_lens = live
        except Exception:  # noqa: BLE001 - slotted batch objects
            pass
        return live
    return seq_lens_cpu[: forward_batch.batch_size].to(torch.int32)


def _dsa_prefill_cp_enabled() -> bool:
    """True only when DSA prefill context parallel is switched on.

    Every CP-specific adjustment below is gated on this so that a CP-off server
    keeps the exact pre-CP contract.
    """
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

    return is_dsa_enable_prefill_cp()


_cp_bcg_replay_logged = False


def _log_cp_bcg_replay_once(parallel) -> None:
    """Announce the first replay that trusts the captured CP geometry."""
    global _cp_bcg_replay_logged
    if _cp_bcg_replay_logged:
        return
    _cp_bcg_replay_logged = True
    logger.warning(
        "Kunlun DSA prefill CP: breakable-graph replay keeps the captured "
        "CP-local geometry (attn_cp_rank=%s, attn_cp_size=%s) while the live "
        "batch is smaller; using the pinned CP metadata.",
        parallel.attn_cp_rank,
        parallel.attn_cp_size,
    )


def _dsa_cp_prefill_ranks(forward_batch) -> Optional[tuple]:
    """Return ``(cp_rank, cp_size)`` when DSA prefill CP round-robin is active.

    Upstream ``DeepseekV4ForCausalLM.forward`` calls ``apply_cp_reindex()`` before
    the layers run, so every per-token metadata field the Kunlun boundary consumes
    is already CP-local while ``forward_batch.extend_seq_lens_cpu`` stays global.
    """
    # CP-V1 deprecation renamed round-robin to interleave
    # (`can_dsa_prefill_cp_round_robin_split` -> `can_dsa_prefill_cp_interleave`);
    # the criterion and token->rank mapping are unchanged.
    try:
        from sglang.srt.layers.attention.dsa.utils import (
            can_dsa_prefill_cp_interleave as _can_cp_split,
        )
    except ImportError:
        from sglang.srt.layers.attention.dsa.utils import (
            can_dsa_prefill_cp_round_robin_split as _can_cp_split,
        )
    # [fix] BCG & CP conflict
    batch_active = _can_cp_split(forward_batch)
    # ``can_dsa_prefill_cp_round_robin_split`` reads the *live* extend lengths,
    # which undershoot under breakable-graph replay: the graph keeps the
    # captured CP-local row geometry while the live batch carries only this
    # step's tokens (a one-token extend sums below cp_size). The CP metadata
    # that the BCG capture input pins at replay is authoritative then.
    replay_active = is_in_breakable_cuda_graph() and (
        getattr(forward_batch, "attn_cp_metadata", None) is not None
    )
    if not (batch_active or replay_active):
        return None

    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.attn_cp_size <= 1:
        return None
    if replay_active and not batch_active:
        _log_cp_bcg_replay_once(parallel)
    return parallel.attn_cp_rank, parallel.attn_cp_size


def _dsa_cp_local_extend_lens(extend_seq_lens_cpu, cp_rank: int, cp_size: int):
    """Round-robin split the global extend lengths onto this CP rank.

    Token ``i`` of the flattened batch belongs to rank ``i % cp_size``, so the
    remainder of one request carries over into the next request's window.
    """
    local_lens = []
    carry = 0
    for length in extend_seq_lens_cpu:
        total = int(length) + carry
        local = total // cp_size + int(total % cp_size > cp_rank)
        local_lens.append(local)
        carry = total - local * cp_size
    return local_lens


def _cp_short_prompt_needs_reference(forward_batch) -> bool:
    """True for extend batches that CP leaves unsplit.

    ``can_dsa_prefill_cp_round_robin_split`` bails out when the batch has fewer
    tokens than ``cp_size``, so such a batch keeps the pre-CP request-level LoD
    while its query rows are still padded to the CP/page alignment. Feeding those
    padding rows to ``compressed_attention`` wedges the device (observed with the
    one-token internal health check, which pads to a full 256-row page and never
    returns). Real traffic is long enough to be CP-split and stays on the
    operator; only these degenerate batches take the Torch reference path.
    ``DSV4_KUNLUN_CP_SHORT_REFERENCE=0`` restores the previous behaviour.
    """
    if forward_batch is None or os.environ.get(
        "DSV4_KUNLUN_CP_SHORT_REFERENCE", "1"
    ) != "1":
        return False
    if not _dsa_prefill_cp_enabled():
        return False
    mode = forward_batch.forward_mode
    if not mode.is_extend() or _is_graph_extend_mode(mode):
        return False
    return _dsa_cp_prefill_ranks(forward_batch) is None


def _make_cp_prefill_lod(core_metadata, num_queries: int, device: torch.device):
    """CP round-robin contract: every local Q token is its own batch item.

    ``seq_lens_casual`` is per-token and already CP-reindexed, so it is the exact
    KV context length for each local query row.
    """
    # Memoize per forward: seq_lens_casual is identical across layers; skip under BCG (it bakes addresses).
    memo = core_metadata is not None and not is_in_breakable_cuda_graph()
    if memo:
        cached = getattr(core_metadata, "_kunlun_cp_lod_memo", None)
        if cached is not None and cached[0] == num_queries:
            return cached[1]
    kv_lens_cpu = core_metadata.seq_lens_casual.to(
        "cpu", non_blocking=False
    ).to(torch.int32)
    if kv_lens_cpu.shape[0] > num_queries:
        kv_lens_cpu = kv_lens_cpu[:num_queries]
    elif kv_lens_cpu.shape[0] < num_queries:
        kv_lens_cpu = torch.cat(
            [
                kv_lens_cpu,
                torch.ones(
                    num_queries - kv_lens_cpu.shape[0], dtype=torch.int32
                ),
            ]
        )
    kv_lens_cpu = torch.clamp(kv_lens_cpu, min=1).contiguous()
    q_lod_cpu = torch.arange(num_queries + 1, dtype=torch.int32)
    result = (
        q_lod_cpu,
        q_lod_cpu.to(device, non_blocking=False),
        kv_lens_cpu,
        kv_lens_cpu.to(device, non_blocking=False),
    )
    if memo:
        try:
            core_metadata._kunlun_cp_lod_memo = (num_queries, result)
        except AttributeError:
            pass
    return result


def _refresh_graph_host_lengths(
    forward_batch,
    attention_decode_aux,
    c4_decode_aux,
    graph_extend_aux,
) -> None:
    """Refresh pointer-stable backend lengths before graph replay."""
    seq_lens = _seq_lens_cpu_i32(forward_batch)
    batch_size = forward_batch.batch_size

    if forward_batch.forward_mode.is_decode_or_idle():
        # The runner replays the graph captured for the padded bucket, and the
        # bucket is not observable here, so refresh every cached entry.
        for cached_bs, attention_aux in attention_decode_aux.items():
            if cached_bs < batch_size:
                continue
            _copy_host_lengths_(attention_aux[2], seq_lens, fill_value=1)
            attention_aux[3].copy_(attention_aux[2])

        c4_context_lens = (seq_lens // 4) * 4
        for key, aux in c4_decode_aux.items():
            cached_bs, _, _, _rows_per_req = key
            if cached_bs < batch_size:
                continue
            _copy_host_lengths_(aux[2], c4_context_lens, fill_value=4)
            aux[3].copy_(aux[2])
        return

    if forward_batch.forward_mode.is_target_verify():
        # DSPARK verify replays a graph captured for `bs * num_draft` rows, but
        # the host `context_lens_cpu` buffer is pointer-stable and was only ever
        # filled at capture time, so every replay fed the kernel the placeholder
        # length. The device tensor is refreshed inside the graph, the host one
        # cannot be -- which is exactly why eager verify was right and the
        # captured verify path silently diverged. Expand each request's length to
        # its draft rows to match the row-shaped page table.
        c4_context_lens = (seq_lens // 4) * 4
        for key, aux in c4_decode_aux.items():
            cached_bs, _, _, rows_per_req = key
            rows = batch_size * max(rows_per_req, 1)
            if cached_bs < rows:
                continue
            lens = c4_context_lens
            if rows_per_req > 1:
                lens = c4_context_lens.repeat_interleave(rows_per_req)
            _copy_host_lengths_(aux[2], lens, fill_value=4)
            aux[3].copy_(aux[2])
        return

    if not _is_graph_extend_mode(forward_batch.forward_mode):
        return

    for (cached_bs, _, _), aux in graph_extend_aux.items():
        if cached_bs < batch_size:
            continue
        _, _, kv_lens_cpu, kv_lens, query_lens = aux
        _copy_host_lengths_(kv_lens_cpu, seq_lens, fill_value=1)
        kv_lens.copy_(kv_lens_cpu)
        torch.maximum(kv_lens, query_lens, out=kv_lens)
def _page_slack() -> int:
    """One page of slack for the page-table span (see _dsv4_page_table_span)."""
    return upstream.PAGE_INDEX_ALIGNED_SIZE if hasattr(upstream, "PAGE_INDEX_ALIGNED_SIZE") else 64


def _draft_token_count() -> int:
    """Number of draft tokens a verify row carries (0 when not speculative."""
    try:
        from sglang.srt.runtime_context import get_spec

        return int(get_spec().speculative_num_draft_tokens or 0)
    except Exception:  # noqa: BLE001 - spec context is absent in unit tests
        return 0


def _dsv4_page_table_span(max_seq_len: int, seq_lens_casual: torch.Tensor) -> int:
    """Return a page-table span that covers every row's causal length.

    TARGET_VERIFY passes the committed max_seq_len while its rows reach
    committed + num_draft_tokens, so on a page boundary the draft positions land in
    a page the table does not describe. The vendor paged kernels then read past
    the last page-table column and pick up uninitialised ``req_to_token`` memory,
    which surfaces as ``flash_attention_infer_decoder kernel error! block_id must
    be non-negtive!`` followed by an illegal memory access. Widening the span is a
    correctness requirement, so it is on by default; set the env to 0 to A/B it.
    """
    if os.environ.get("DSV4_PAGE_TABLE_COVER_CASUAL", "1") != "1":
        return max_seq_len
    bound = max_seq_len + _draft_token_count() + _page_slack()
    if torch.cuda.is_current_stream_capturing():
        # During capture the lens hold placeholder data; take the static upper bound.
        return bound
    if seq_lens_casual.numel() == 0:
        return max_seq_len
    needed = int(seq_lens_casual.max().item())
    if needed <= max_seq_len:
        return max_seq_len
    # Clamp: capture on a non-current stream (DSPARK verify graphs) reports
    # capturing=False while the lens are still uninitialised, which sized the
    # table from garbage and OOMed with a 2 TB allocation.
    return min(needed, bound)


def _forward_c4_indexer_impl(
    original_fn,
    self,
    x,
    q_lora,
    c4_indexer,
    forward_batch,
    alt_streams,
    enable_multi_stream,
    q_lora_ready,
    skip_compressor,
):
    # ``original_fn`` is the bound method, so ``self`` is already bound; it stays
    # in the signature only so the break keeps a strong reference to the backend.
    return original_fn(
        x,
        q_lora,
        c4_indexer,
        forward_batch,
        alt_streams,
        enable_multi_stream,
        q_lora_ready,
        skip_compressor,
    )


# Eager-break form of the C4 indexer call (see the wrapper method below).
_eager_forward_c4_indexer = eager_on_graph(True)(_forward_c4_indexer_impl)



_STEP_BARRIER_GROUP = None


def _cross_rank_step_barrier() -> None:
    """Rendezvous every TP rank before the next replay is launched.

    ``_serialize_graph_replay`` only fences *this* device, so a fast rank can
    enter replay N+1 while a peer is still finishing replay N and its
    collectives. That skew is the remaining suspect for the XCCL all_reduce
    deadlock (``kl3_all_reduce_Entry ... task timeout`` on every rank while all
    of them sit in ``cudaGraphLaunch``), so DSV4_KUNLUN_STEP_BARRIER=1 adds a
    global step boundary. The rendezvous uses the group's CPU (gloo) group, so
    the barrier itself never touches XCCL. Diagnostic; costs one host round
    trip per step.
    """
    global _STEP_BARRIER_GROUP
    if os.environ.get("DSV4_KUNLUN_STEP_BARRIER", "0") != "1":
        return
    if _STEP_BARRIER_GROUP is None:
        try:
            import torch.distributed as dist

            from sglang.srt.distributed.parallel_state import get_tp_group

            group = get_tp_group().cpu_group
            _STEP_BARRIER_GROUP = group if group is not None else "none"
        except Exception:
            _STEP_BARRIER_GROUP = "none"
    if _STEP_BARRIER_GROUP == "none":
        return
    torch.distributed.barrier(group=_STEP_BARRIER_GROUP)


def _serialize_graph_replay() -> None:
    """Wait for the previous replay before building the next one's metadata.

    The target decode graph replays the compressor store, C4/C128 indexer and
    attention without a device-wide fence at the end. The out-of-graph Python
    that builds the next iteration's metadata (and the next replay itself) can
    therefore start while the previous iteration's async work is still in
    flight. At long context this race corrupts the compress state and faults
    the device (``kl3ChannelCheckErrors status=719`` / ``invalid program
    counter``); a single stream sync is not enough because the compressor uses
    a different stream. Serializing the replay is the verified fix. Set
    ``DSV4_KUNLUN_GRAPH_SYNC=0`` to trade correctness for overlap.
    """
    if os.environ.get("DSV4_KUNLUN_GRAPH_SYNC", "1") != "1":
        _cross_rank_step_barrier()
        return
    torch.cuda.synchronize()
    _cross_rank_step_barrier()


def _build_paged_mqa_logits_metadata_kunlun(indexer_metadata):
    """Build the paged-MQA SM schedule through the Kunlun XSpeedGate op.

    Upstream ``PagedIndexerMetadata.__post_init__`` (``dsv4/metadata.py``) sets
    ``deep_gemm_metadata = None`` whenever ``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH``
    is set -- and the Kunlun launch scripts set it -- so the registered
    ``sglang.kernels.ops.attention.dsv4.attn.get_paged_mqa_logits_metadata``
    replacement never runs. Rebuild the schedule here, mirroring upstream's
    ``row_chunk`` / ``rows_per_chunk`` splitting, so the field matches the
    non-Kunlun contract instead of staying ``None``.
    """
    import deep_gemm

    compressed_seq_lens = indexer_metadata.compressed_seq_lens.to(torch.int32)
    if compressed_seq_lens.dim() == 1:
        compressed_seq_lens = compressed_seq_lens.unsqueeze(-1)
    page_size = indexer_metadata.compressed_page_size
    num_sm = deep_gemm.get_num_sms()
    num_rows = compressed_seq_lens.shape[0]
    if indexer_metadata.row_chunk > 0:
        return torch.stack(
            [
                get_paged_mqa_logits_metadata_kunlun(
                    compressed_seq_lens[start : start + indexer_metadata.row_chunk],
                    page_size,
                    num_sm,
                )
                for start in range(0, num_rows, indexer_metadata.row_chunk)
            ]
        )
    if indexer_metadata.rows_per_chunk is not None:
        return [
            get_paged_mqa_logits_metadata_kunlun(
                compressed_seq_lens[start : start + indexer_metadata.rows_per_chunk],
                page_size,
                num_sm,
            )
            for start in range(0, num_rows, indexer_metadata.rows_per_chunk)
        ]
    return get_paged_mqa_logits_metadata_kunlun(compressed_seq_lens, page_size, num_sm)


class KunlunDeepseekV4AttnBackend(DeepseekV4AttnBackend):
    """0.5.14 DSV4 control flow with the Kunlun attention boundary."""

    # Kunlun DSV4 graph-extend needs uniform queries per request; ragged
    # (compact) verify runs on the eager per-row LoD path instead.
    supports_ragged_verify_graph: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do NOT force ``needs_cpu_seq_lens`` here. Upstream clears it under DSPARK
        # (``deepseek_v4_backend.py:1241``) and 0.5.19 left that alone, which is
        # what makes the whole backend read its lengths from the device tensor.
        # Forcing it on gave every consumer a *pinned host mirror* instead, and
        # ``DecodeCudaGraphRunner`` never refreshes that mirror per replay (only the
        # eagle draft runners do) -- so a DSPARK verify replay saw the capture-time
        # fill value, the C4 compressed lengths came out of uninitialised memory,
        # and PD decode degenerated into a repetition loop with accept length 1.0.
        # The ``_make_lod`` deadlock that motivated the override was the *blocking
        # H2D* of the LoD, fixed separately with a pinned non-blocking upload.
        # The Kunlun metadata builder was written against the 4.1 field names.
        # 0.5.22's upstream backend publishes ``index_topk`` itself
        # (``deepseek_v4_backend.py:1179``) and no longer has the older
        # ``c4_topk`` spelling anywhere, so the former compatibility fallback
        # ``getattr(self, "index_topk", self.c4_topk)`` is both dead and an
        # AttributeError (the default is evaluated eagerly). Assert instead:
        # losing this field silently would mean an unbounded top-k.
        assert hasattr(self, "index_topk"), (
            "upstream DSV4 backend no longer publishes index_topk"
        )
        ratios = tuple(
            getattr(self.model_runner.model_config, "compress_ratios", ())
        )
        if not ratios:
            ratios = tuple(
                getattr(
                    self.model_runner.model_config.hf_text_config,
                    "compress_ratios",
                    (),
                )
            )
        self.present_ratios = tuple(sorted(set(ratios)))
        self.low_ratios = tuple(r for r in (1, 2) if r in self.present_ratios)
        self._verify_row_lod_aux = {}
        self._attention_decode_aux = {}
        self._attention_graph_extend_aux = {}
        self._c4_decode_aux = {}

    def init_forward_metadata_indexer(
        self,
        core_attn_metadata,
        *,
        compress_ratio: int = 4,
        use_prefill_cuda_graph: bool = False,
    ):
        """Build the indexer metadata and fill the Kunlun paged-MQA schedule.

        ``super()`` leaves ``deep_gemm_metadata`` at ``None`` on Kunlun (the
        launch scripts set ``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH``), which
        short-circuits the registered ``get_paged_mqa_logits_metadata`` op. Build
        it here through that op so the field matches the upstream contract.

        Args:
            core_attn_metadata: Core metadata for the current attention batch.
            compress_ratio: Compression ratio used to construct indexer metadata.
            use_prefill_cuda_graph: Whether metadata is built for prefill capture.

        Returns:
            Indexer metadata with its paged-MQA schedule populated.
        """
        indexer_metadata = super().init_forward_metadata_indexer(
            core_attn_metadata,
            compress_ratio=compress_ratio,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )
        if indexer_metadata.deep_gemm_metadata is None:
            indexer_metadata.deep_gemm_metadata = (
                _build_paged_mqa_logits_metadata_kunlun(indexer_metadata)
            )
        return indexer_metadata

    _WINDOW_WARNED = False

    def _low_ratio_index_topk(
        self, layer, x, q_lora, req, pos, forward_batch, *, rows_per_request=None
    ):
        # 0.5.22 owns candidate metadata in v41_indexer. Adapt the fused decode
        # path at this live entry point, preserving newer sparse-table schemes.
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            BlockIds,
            DenseBlocksBackend,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            skip_candidate_filtering,
            skip_low_ratio_indexer,
        )

        mode = forward_batch.forward_mode
        candidate = layer.indexer.is_candidate_source or layer.indexer.uses_candidates
        if (
            not (mode.is_decode() or mode.is_target_verify())
            or skip_candidate_filtering()
            or (candidate and not isinstance(self.decode_candidates, DenseBlocksBackend))
        ):
            return super()._low_ratio_index_topk(
                layer, x, q_lora, req, pos, forward_batch,
                rows_per_request=rows_per_request,
            )
        if skip_low_ratio_indexer(layer.compress_ratio):
            return
        published = self.forward_metadata.candidate_metadata
        self.candidate_masks = (
            {"rows": published.decode_mask}
            if isinstance(published, BlockIds) and published.decode_mask is not None
            else None
        )
        if layer.indexer.uses_candidates and self.candidate_masks is None:
            return super()._low_ratio_index_topk(
                layer, x, q_lora, req, pos, forward_batch,
                rows_per_request=rows_per_request,
            )
        self._low_ratio_index_topk_sm90_decode(layer, x, q_lora, req, pos)
        if layer.indexer.is_candidate_source and self.candidate_masks is not None:
            # The fused op publishes and consumes the row mask directly, and the
            # only reader of this metadata is the `decode_mask` branch above, so
            # the block ids the upstream scheme derives from the mask (pad ->
            # any -> top-k) are dead work: measured 16 ms per decode round at
            # bs=32. Publish the mask with an empty id table instead of wrong
            # ids, so a future reader fails loudly rather than silently.
            mask = self.candidate_masks["rows"]
            self._publish_candidate_metadata(
                BlockIds(
                    blocks=mask.new_empty((mask.shape[0], 0), dtype=torch.int32),
                    decode_mask=mask,
                )
            )

    def _low_ratio_fp4_lut(self, device) -> torch.Tensor:
        """Device-resident E2M1 fp4 code->value table for the fused indexer op.

        Cached per device so graph capture records at most one tiny H2D copy
        (the table is a module-level CPU constant). Shares the exact values the
        pool's dequant readback uses, keeping the op bit-identical to the native
        row-wise path.
        """
        cache = getattr(self, "_low_ratio_fp4_lut_cache", None)
        if cache is not None and cache.device == device:
            return cache
        from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

        lut = DSV4_DEQUANT_FP4_TABLE.to(device=device, dtype=torch.float32).contiguous()
        self._low_ratio_fp4_lut_cache = lut
        return lut

    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
        env = os.environ.get("DSV4_LOW_RATIO_INDEXER_KUNLUN", "1")
        use_kunlun = (
            env not in ("0", "false", "False")
            if env is not None
            else True
        )
        if use_kunlun:
            return self._low_ratio_index_topk_sm90_decode_kunlun(
                layer, x, q_lora, req, pos
            )
        return self._low_ratio_index_topk_sm90_decode_native(
            layer, x, q_lora, req, pos
        )

    def _low_ratio_index_topk_sm90_decode_kunlun(self, layer, x, q_lora, req, pos) -> None:
        """Low-ratio decode/verify indexer top-k via the fused XSpeedGate
        ``dsv41_low_ratio_indexer`` kernel on Kunlun.
        """
        from sglang_kunlun.hooks.mem_cache.common import low_ratio_index_k_int8_enabled

        use_int8 = low_ratio_index_k_int8_enabled()
        op_name = "dsv41_low_ratio_indexer_int8" if use_int8 else "dsv41_low_ratio_indexer"
        ratio = layer.compress_ratio
        indexer = layer.indexer
        core = self.forward_metadata.core_metadata
        pool = self.token_to_kv_pool

        rows = req.shape[0]
        page_indices = core.sparse_page_indices(ratio)
        page_indices.fill_(-1)
        if rows == 0:
            return

        block_size = int(getattr(indexer, "candidate_block_size", 0) or 0)
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None
        window = min(
            metadata.max_compressed_seq_len,
            self.req_to_token.shape[1] // ratio,
        )
        window = (window // 512) * 512
        # 0519 aligned window downward to a multiple of block_size here; the
        # migration dropped this line and replaced it with a modulo-window bail
        # condition. Measured window=262411 block_size=8 (remainder 3), so decode
        # bailed to the native torch path every time: the fused indexer op never
        # ran (slow), and the native path is normally not covered, so its
        # out-of-bounds access surfaced as status=700. Aligning back lets it take
        # the same fused path as 0519.
        if block_size >= 1:
            window = (window // block_size) * block_size
        k = min(int(indexer.index_topk), window)

        # Delegate to the native kernel when the fused op's static constraints
        # cannot be satisfied.
        _bail = (
            block_size < 1
            or 128 % block_size != 0
            or window < block_size
            or k < 1
            or k > 512
            or not hasattr(torch.ops.xspeedgate_ops, op_name)
        )
        if _bail:
            return self._low_ratio_index_topk_sm90_decode_native(
                layer, x, q_lora, req, pos
            )

        # Do not cast q to bf16 here: the fused op requires head_w to have the
        # same dtype as q (otherwise it raises dsv41_low_ratio_indexer: head_w
        # dtype must match q), and on this precision-validated 0519 path both
        # are the plugin-native fp16. Keep it consistent with 0519.
        q = indexer.queries(q_lora, layer.freqs_cis[pos]).contiguous()  # [R,32,128]
        head_w = indexer.head_weights(x).contiguous()  # [R,32]
        k_pool = pool.get_index_k_with_scale_buffer(layer.layer_id)
        layout_args = () if use_int8 else (self._low_ratio_fp4_lut(q.device),)

        req_i = req.to(torch.int32).contiguous()
        pos_i = pos.to(torch.int32).contiguous()
        req_to_token = self.req_to_token
        if req_to_token.dtype != torch.int32:
            req_to_token = req_to_token.to(torch.int32)
        req_to_token = req_to_token.contiguous()

        is_publish = bool(getattr(indexer, "is_candidate_source", False))
        consume_mask = None
        if getattr(indexer, "uses_candidates", False):
            masks = getattr(self, "candidate_masks", None)
            if isinstance(masks, dict):
                m = masks.get("rows")
                if torch.is_tensor(m) and m.shape[0] == rows:
                    consume_mask = m
        is_consume = consume_mask is not None

        if is_consume:
            cm = consume_mask.to(torch.bool)
            if cm.shape[1] < window:
                cm = torch.nn.functional.pad(cm, (0, window - cm.shape[1]))
            cand_mask = cm[:, :window].contiguous()
        else:
            # Ignored when `is_consume` is False, but the op still type-checks it.
            cand_mask = torch.zeros((rows, window), dtype=torch.bool, device=q.device)

        page_out, published = getattr(torch.ops.xspeedgate_ops, op_name)(
            q,
            head_w,
            k_pool,
            *layout_args,
            req_to_token,
            req_i,
            pos_i,
            cand_mask,
            int(ratio),
            int(k),
            int(window),
            max(int(getattr(indexer, "candidate_topk_blocks", 0) or 0), 1),
            int(block_size),
            is_publish,
            is_consume,
        )

        page_indices[:rows, :k] = page_out
        if is_publish:
            # Row-keyed mask shared with the native path's consume layers.
            self.candidate_masks = {"rows": published.view(torch.bool)}

    def _low_ratio_index_topk_sm90_decode_native(self, layer, x, q_lora, req, pos) -> None:
        """Capture-safe low-ratio decode/verify indexer top-k.

        Upstream routes non-Blackwell decode through the SM90 `fp4_index_logits_decode`
        Triton kernel (absent on XPU) and keeps `_low_ratio_index_topk_torch` as the
        eager oracle. That oracle cannot run under CUDA graph capture: it derives
        output shapes from data (`torch.unique_consecutive(req).tolist()`) and reads
        scalars back to the host (`lens.max().item()`), both illegal during capture --
        observed as `RuntimeError: Trying to create tensor with negative dimension
        -72340172838076673` at the first capture call.

        This is a row-wise reformulation with static shapes:

        * every row is scored against a fixed window of compressed positions
          `j in [0, window)` and masked by that row's own visible length, which is
          the reference's per-request `lc = max(lens)` set expressed per row (rows
          of a request share `req`, hence share the candidate slots for a given j);
        * `window` comes from the current paged metadata's reserved compressed
          length, bounded by the request table capacity; it never reads device
          lengths on the host or silently truncates them to a legacy env cap.
        * the window is chunked and a running top-k is kept across chunks, so the
          transient tensors stay bounded while the result is still the exact top-k
          over the whole window.

        Candidate publish/consume is row-indexed: within one forward the rows are
        the same for every layer, so a `[rows, window]` mask carries exactly the
        request-keyed masks the reference built.
        """
        ratio = layer.compress_ratio
        indexer = layer.indexer
        core = self.forward_metadata.core_metadata
        pool = self.token_to_kv_pool
        rows = x.shape[0]
        if rows == 0:
            return
        topk = indexer.index_topk
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1 else self.forward_metadata.c2_indexer_metadata
        )
        window = min(metadata.max_compressed_seq_len, self.req_to_token.shape[1] // ratio)
        k = min(topk, window)

        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if k == 0:
            return

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [R, H, index_head_dim]
        head_w = indexer.head_weights(x)  # [R, H]
        visible = ((pos + 1) // ratio).to(torch.int64).clamp(0, window)
        req_i = req.to(torch.int64).reshape(-1)
        # Under graph replay the req / pos of padding rows are leftovers from the
        # previous round; indexing req_to_token with them directly would go
        # out-of-bounds (Kunlun reports gather load/store exceed memory size,
        # crashing the scheduler with status=700 illegal memory access). Clamp
        # both the row index and the visible length into the legal range: padding
        # rows degrade to reading row 0, and the result is masked out by reach
        # afterward.
        req_i = req_i.clamp(0, self.req_to_token.shape[0] - 1)
        device = x.device
        neg = torch.finfo(torch.float32).min
        req_width = self.req_to_token.shape[1] - 1

        publish = bool(getattr(indexer, "is_candidate_source", False))
        consume = None
        if getattr(indexer, "uses_candidates", False) and isinstance(
            self.candidate_masks, dict
        ):
            consume = self.candidate_masks.get("rows")
        # published must use the full `window` width, not `effective_window`,
        # because consumer layers index it with [j0 : j0 + jc] up to `window`.
        published = (
            torch.zeros((rows, window), dtype=torch.bool, device=device)
            if publish
            else None
        )

        best_val = torch.full((rows, k), neg, dtype=torch.float32, device=device)
        # Unfilled slots sort last: the attention op reads a *leading* prefix of
        # `sparse_topk_lengths` entries, so the valid positions must come first
        # (the reference sorts too). A `window` sentinel is greater than every real
        # position and is filtered by the `reach` mask below.
        best_j = torch.full((rows, k), window, dtype=torch.int64, device=device)

        jc = int(os.environ.get("DSV4_LOW_RATIO_INDEX_CHUNK", "2048"))
        jc = max(jc, k)
        rc = int(os.environ.get("DSV4_LOW_RATIO_INDEX_ROWS", "64"))
        # Iterate the reserved bound, so capture never reads device lengths.
        effective_window = window
        # Select candidate blocks globally across the full visible range,
        # not independently per chunk (which degenerates to keeping all blocks
        # when topk_blocks exceeds the per-chunk block count).
        global_candidates = bool(publish)
        if global_candidates:
            block_size = indexer.candidate_block_size
            if jc % block_size:
                raise ValueError("candidate score chunks must align to blocks")
            block_scores = torch.full(
                (rows, ((effective_window + jc - 1) // jc) * (jc // block_size)),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            )
        for j0 in range(0, effective_window, jc):
            j = torch.arange(j0, j0 + jc, device=device)  # static width
            tok_pos = (j * ratio).clamp_max(req_width)
            valid = (j.unsqueeze(0) < visible.unsqueeze(1)) & (j[None, :] < window)
            for r0 in range(0, rows, rc):
                r1 = min(r0 + rc, rows)
                slots = self.req_to_token[req_i[r0:r1, None], tok_pos[None, :]] // ratio
                v = valid[r0:r1]
                slots = slots.masked_fill(~v, 0)
                index_k = pool.get_low_ratio_index_k_dequant(
                    layer.layer_id, slots.reshape(-1)
                ).reshape(r1 - r0, jc, -1)
                # `DeepseekV41Indexer.scores` with a per-row K: bf16 up to the
                # reduction, then relu, head-weight and head sum in fp32.
                #
                # The broadcast form materialises [R, H, jc, d] before
                # reducing, which is what makes this loop OOM: at the capture
                # bucket (rows=64, H=64, d=128) that is ~2 MiB per unit of
                # ``jc``, so even the default jc=2048 wants ~4 GiB on top of a
                # pool that is already ~89 GiB deep. ``matmul`` is the same
                # contraction without the 4-D intermediate (d times less
                # transient memory) and accumulates in fp32 rather than in the
                # input dtype. ``DSV4_LOW_RATIO_INDEX_BMM=0`` restores the old
                # form.
                if _low_ratio_index_bmm_enabled():
                    # [R,H,d] x [R,d,jc] -> [R,H,jc]. ``bmm`` rather than
                    # ``matmul``: the contract is strictly 3-D with a per-row
                    # K, so a rank surprise should raise instead of silently
                    # broadcasting back into the 4-D form this exists to avoid.
                    # (Upstream's ``einsum("bhd,nd->bhn")`` shares one K across
                    # rows; here ``index_k`` is gathered per row, so that form
                    # does not apply.)
                    dots = torch.bmm(q[r0:r1], index_k.to(q.dtype).transpose(1, 2))
                else:
                    dots = (q[r0:r1].unsqueeze(2) * index_k.unsqueeze(1)).sum(dim=-1)
                s = (dots.relu() * head_w[r0:r1].unsqueeze(-1)).sum(dim=1).float()
                s = s.masked_fill(~v, neg)
                if consume is not None:
                    mask = consume[r0:r1, j0 : j0 + jc]
                    mask = torch.nn.functional.pad(mask, (0, jc - mask.shape[1]))
                    s = s.masked_fill(~mask, neg)
                if global_candidates:
                    block_scores[
                        r0:r1, j0 // block_size : (j0 + jc) // block_size
                    ] = s.masked_fill(~v, -torch.inf).unflatten(
                        -1, (-1, block_size)
                    ).amax(dim=-1)
                cand_val, cand_j = s.topk(k, dim=-1, sorted=False)
                merged_val = torch.cat([best_val[r0:r1], cand_val], dim=1)
                merged_j = torch.cat([best_j[r0:r1], j0 + cand_j], dim=1)
                top = merged_val.topk(k, dim=-1, sorted=False)
                best_val[r0:r1] = top.values
                best_j[r0:r1] = merged_j.gather(
                    1, top.indices.clamp(0, merged_j.shape[1] - 1)
                )

        # Sort the (value, position) pairs by position so valid entries form the
        # leading prefix the attention op expects; `best_val` must follow.
        order = best_j.argsort(dim=-1).clamp(0, best_j.shape[1] - 1)
        best_j = best_j.gather(1, order)
        best_val = best_val.gather(1, order)
        reach = (best_j < visible.unsqueeze(1)) & (best_val > neg)
        sel_pos = (best_j * ratio).clamp_max(req_width)
        sel_slots = self.req_to_token[req_i[:, None], sel_pos] // ratio
        page_indices[:, :k] = torch.where(
            reach, sel_slots, torch.full_like(sel_slots, -1)
        ).to(torch.int32)
        if global_candidates:
            count = min(indexer.candidate_topk_blocks, block_scores.shape[1])
            chosen = block_scores.topk(count, dim=-1)
            cand_mask = torch.zeros_like(block_scores, dtype=torch.bool)
            cand_mask.scatter_(1, chosen.indices, torch.isfinite(chosen.values))
            cand_mask = cand_mask.repeat_interleave(block_size, dim=-1)
            # Pad to full window width so consumers can slice without F.pad
            # at consumption time (which would break cudagraph capture).
            if cand_mask.shape[1] < window:
                cand_mask = torch.nn.functional.pad(
                    cand_mask, (0, window - cand_mask.shape[1])
                )
            published = cand_mask[:, :window]
        if published is not None:
            self.candidate_masks = {"rows": published}

    def get_swa_page_indices(
        self, seq_lens_casual, req_pool_indices_repeated, swa_replay_start=None
    ):
        """Translate request token offsets into Kunlun SWA page indices.

        ``swa_replay_start`` (4.1) is the per-row window floor used by decoder
        SWA bounded replay: offsets below it are never written at the tail
        layers, so the window is clamped at the floor instead of at 0.

        Runs the XSpeedGate ``causal_swa_page_indices_kernel`` op -- the Kunlun
        replacement for upstream ``build_causal_swa_page_indices_triton``
        (``kernel_ops.build_causal_swa_page_indices_kunlun``). Upstream only
        reaches that triton symbol through ``BuildCausalSwaPageIndices.execute``,
        which this backend bypasses, so the op has to be called directly here.
        The op writes -1 for lanes outside the per-row window/floor and pads the
        width to ``PAGE_INDEX_ALIGNED_SIZE``; the Kunlun compressed-attention
        path wants dense non-negative page indices (slot 0 marks an
        invalid/tombstoned SWA slot) and masks the tail with ``swa_topk_lengths``,
        so the sentinels are folded back to 0.
        """
        if hasattr(torch.ops.xspeedgate_ops, "causal_swa_page_indices_kernel"):
            swa_page_indices = build_causal_swa_page_indices_kunlun(
                req_to_token=self.req_to_token,
                full_to_swa_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
                req_pool_indices_repeated=req_pool_indices_repeated,
                seq_lens_casual=seq_lens_casual,
                swa_window=upstream.SWA_WINDOW,
                page_index_aligned_size=upstream.PAGE_INDEX_ALIGNED_SIZE,
                swa_replay_start=swa_replay_start,
            )
            return swa_page_indices.clamp_min_(0)

        # Torch fallback for wheels that predate the op. Match the contract:
        # invalid history offsets are clamped to the first physical row and
        # remain valid indices; the length tensor masks those rows. Upstream
        # 0.5.14 writes -1 here, which changes graph replay inputs and is not
        # accepted by the Kunlun compressed-attention path.
        pos_causal = seq_lens_casual - 1
        lower = (
            torch.zeros_like(pos_causal).unsqueeze(1)
            if swa_replay_start is None
            else swa_replay_start.reshape(-1, 1).to(pos_causal.dtype)
        )
        offsets = torch.clamp(
            pos_causal.unsqueeze(1)
            - torch.arange(
                upstream.SWA_WINDOW, **self.cuda_int32_kwargs
            ).unsqueeze(0),
            min=lower,
        )
        raw_indices = self.req_to_token[
            req_pool_indices_repeated[:, None].clamp(
                min=0, max=self.req_to_token.shape[0] - 1
            ),
            offsets.clamp(min=0, max=self.req_to_token.shape[1] - 1),
        ]
        # Upstream ``_causal_swa_page_indices_kernel`` masks every load, so lanes
        # beyond the row's causal length never actually access memory; this is a
        # dense rewrite that reads all lanes, so the mapping lookup must be
        # clamped manually -- out-of-bounds lanes are masked by
        # ``swa_topk_lengths``.
        mapping = self.token_to_kv_pool.full_to_swa_index_mapping
        mapped = mapping[
            raw_indices.clamp(min=0, max=mapping.shape[0] - 1)
        ].to(torch.int32)
        return mapped

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        """Initialize out-of-graph metadata and refresh replay lengths."""
        # ``_try_flash_mla_hybrid`` has to tell the capture phase (both eager
        # warmup passes and the capture itself) apart from real eager decode:
        # during capture both sides of its length validation are placeholders.
        # This hook also runs once per replay with ``in_capture=False``, so the
        # flag cannot go stale.
        self._kunlun_capture_phase = in_capture
        if in_capture and _is_graph_extend_mode(forward_batch.forward_mode):
            _get_graph_extend_aux(
                self._attention_graph_extend_aux,
                forward_batch.batch_size,
                forward_batch.positions.numel(),
                forward_batch.seq_lens.device,
            )
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            and _hybrid_seq_lens_prerefresh_enabled()
        ):
            # Hoisted out of the decode layer loop: the hybrid adapter's int32
            # mirror is per (device, bucket), not per layer, so one refresh here
            # feeds all hybrid layers and the capture records zero copies.
            # Runs before the warmup/capture forward and before every replay,
            # which is what keeps the recorded launches reading fresh lengths.
            _refresh_hybrid_seq_lens_i32(self, forward_batch)
        _refresh_graph_host_lengths(
            forward_batch,
            self._attention_decode_aux,
            self._c4_decode_aux,
            self._attention_graph_extend_aux,
        )
        # The draft writes its block KV through ``core.swa_out_cache_loc``, a
        # pointer-stable buffer cached at metadata build. Only the *device* copy is
        # refreshed in-graph, so a replayed DSPARK step kept writing the capture-time
        # slots -- in PD that corrupts the target's history and the model degenerates
        # into a repetition loop (accept length 1.1 vs 1.8 on 0.5.19). Refresh it
        # from the live out_cache_loc, device-to-device, on every replay.
        try:
            core = getattr(self.forward_metadata, "core_attn_metadata", None)
            cached_loc = getattr(core, "swa_out_cache_loc", None)
            live_loc = getattr(forward_batch, "out_cache_loc", None)
            if (
                cached_loc is not None
                and live_loc is not None
                and cached_loc.shape[0] == live_loc.shape[0]
                and not torch.cuda.is_current_stream_capturing()
            ):
                mapped = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    live_loc
                ).to(torch.int32)
                cached_loc.copy_(mapped)
        except Exception:  # noqa: BLE001 - keep the metadata path alive
            pass
        # A device-wide wait is not legal inside capture on XPU: it surfaces as
        # ``error code=299, wait for noc idle timeout`` from capture_decode_graph
        # (seen while capturing the DSPARK verify graphs). The fence only orders
        # replay against the next metadata build, so skip it while capturing.
        if not in_capture and not torch.cuda.is_current_stream_capturing():
            _serialize_graph_replay()

    def forward_c4_indexer(
        self,
        x,
        q_lora,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream=False,
        q_lora_ready=None,
        skip_compressor=False,
    ):
        """Run the C4 indexer between graph segments instead of inside one.

        The C4 indexer is inherently data dependent: its prefill contract is
        built from the live ``extend_seq_lens``/``c4_seq_lens`` and the page
        selection follows the live page table.  Inside a breakable CUDA graph
        segment the ops are *recorded* at capture, so replay re-reads the
        capture-time buffers -- the indexer then picks the wrong pages and the
        sparse attention attends to the wrong KV.  Running it as an eager break
        (the same treatment ``cp/bcg.py`` gives the compressor) makes every
        replay recompute it from live state.  Transparent when no capture is
        active, so BCG=0 and tc_piecewise keep calling the body directly.
        """
        return _eager_forward_c4_indexer(
            self._forward_c4_indexer_inner,
            self,
            x,
            q_lora,
            c4_indexer,
            forward_batch,
            alt_streams,
            enable_multi_stream,
            q_lora_ready,
            skip_compressor,
        )

    def _forward_c4_indexer_inner(
        self,
        x,
        q_lora,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream=False,
        q_lora_ready=None,
        skip_compressor=False,
    ):
        """Run the version-pinned C4 flow with explicit Kunlun request state."""
        import torch.nn.functional as F

        from sglang.srt.layers.attention.dsv4 import indexer as upstream_indexer

        if forward_batch.forward_mode.is_idle():
            return

        token_to_kv_pool = self.token_to_kv_pool
        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata
        assert isinstance(indexer_metadata, upstream_indexer.PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        if enable_multi_stream:
            q_indexer, weights = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_indexer, weights = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                skip_compressor=skip_compressor,
            )

        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        use_fp4_indexer = c4_indexer.use_fp4_indexer
        head_dim_with_sf = 68 if use_fp4_indexer else 132

        if use_fp4_indexer:
            q_fp4, q_sf = q_indexer
            assert len(q_fp4.shape) == 3
            assert len(q_sf.shape) == 2
            q = (q_fp4.unsqueeze(1), q_sf.unsqueeze(1))
        else:
            assert len(q_indexer.shape) == 3
            q = q_indexer.unsqueeze(1)

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0],
            block_kv,
            num_heads_kv,
            head_dim_with_sf,
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if use_fp4_indexer:
            weights = weights.float()
        # Kunlun always dispatches to `_compute_c4_logits_kunlun` (XPU op),
        # which ignores `fn`. Skip the upstream `fn` selection to avoid
        # importing `deep_gemm.fp8_paged_mqa_logits` (does not exist on Kunlun)
        # or other GPU/backends' fallbacks.
        fn = None

        query_rows = (
            q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]
        )

        # BCG replay pads the query rows to the captured bucket; run the indexer on
        # the live rows only (eager shape) and leave the pad rows at -1.  CP keeps
        # its own padded-row contract.
        if (
            is_in_breakable_cuda_graph()
            and not torch.cuda.is_current_stream_capturing()
            and _dsa_cp_prefill_ranks(forward_batch) is None
        ):
            # Live rows are a prefix; pad rows end with compressed length 0.
            _lens_all = indexer_metadata.compressed_seq_lens.reshape(-1)
            _nz = torch.nonzero(_lens_all > 0)
            live_rows = int(_nz[-1].item()) + 1 if _nz.numel() else 0
            if 0 < live_rows < query_rows:
                query_rows = live_rows
                if use_fp4_indexer:
                    q = (q[0][:live_rows], q[1][:live_rows])
                else:
                    q = q[:live_rows]
                weights = weights[:live_rows]

        def match_num_queries(tensor, value):
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (
                0,
                query_rows - tensor.shape[0],
            )
            return F.pad(tensor, pad, value=value)

        # 0.5.22 renamed ``PagedIndexerMetadata.c4_seq_lens`` to
        # ``compressed_seq_lens`` (and added compress_ratio/row_chunk etc.).
        c4_seq_lens = match_num_queries(
            indexer_metadata.compressed_seq_lens, value=1
        )
        c4_seq_lens_for_logits = c4_seq_lens
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )
        use_tilelang = (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.get() and not use_fp4_indexer
        )
        use_aiter = envs.SGLANG_OPT_USE_AITER_INDEXER.get() and not use_fp4_indexer
        if (
            c4_seq_lens_for_logits.dim() == 1
            and not use_tilelang
            and not use_aiter
        ):
            c4_seq_lens_for_logits = c4_seq_lens_for_logits.unsqueeze(-1)

        logits = self._compute_c4_indexer_logits(
            fn=fn,
            q=q,
            c4_indexer_kv_cache=c4_indexer_kv_cache,
            weights=weights,
            c4_seq_lens=c4_seq_lens_for_logits,
            page_table=page_table,
            indexer_metadata=indexer_metadata,
            core_metadata=core_metadata,
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
            positions=positions,
        )

        # The identity only holds when the indexer runs inside the captured
        # segment; under BCG replay the capture-time metadata objects are
        # refreshed in place and the two sides can be distinct tensors holding
        # the same values.
        if indexer_metadata.page_table is not core_metadata.page_table:
            logger.warning(
                "[C4IDX] page_table identity differs under BCG replay: idx=%s core=%s",
                tuple(indexer_metadata.page_table.shape),
                tuple(core_metadata.page_table.shape),
            )
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = upstream_indexer.get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None
        hisparse_coordinator = self.hisparse_coordinator
        # DSPARK drives the target forward as TARGET_VERIFY, not DECODE, so the
        # swap-in has to fire there too.  A verify batch carries
        # `bs * num_draft_tokens` query rows, each selecting its own top-k.
        hisparse_swap = hisparse_coordinator is not None and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_swap:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            # Routed through the plugin's row-chunked wrapper: the upstream helper
            # clones the whole (rows, seq_len) FP32 tile, which is an 8 GiB
            # allocation under a breakable prefill graph captured at 8192 rows.
            _dsv4_topk_torch_fallback(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.compressed_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform = getattr(upstream_indexer, "topk_transform_paged_v2", None)
            if topk_transform is None:
                upstream_indexer.topk_transform_pytorch_vectorized(
                    logits,
                    c4_seq_lens,
                    page_table,
                    c4_sparse_page_indices,
                    indexer_metadata.compressed_page_size,
                    raw_indices,
                )
            else:
                topk_transform(
                    logits,
                    c4_seq_lens,
                    page_table,
                    c4_sparse_page_indices,
                    indexer_metadata.compressed_page_size,
                    indexer_metadata.topk_metadata,
                )
        else:
            topk_transform = getattr(
                upstream_indexer,
                "topk_transform_paged",
                upstream_indexer.topk_transform_pytorch_vectorized,
            )
            topk_transform(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.compressed_page_size,
                raw_indices,
            )

        if hisparse_coordinator is not None:
            if hisparse_swap:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                # The coordinator is per-request, but a verify batch has one
                # query row per draft token; expand the request-indexed inputs
                # to one entry per row so every row swaps in its own top-k.
                swap_req_indices = forward_batch.req_pool_indices
                swap_seq_lens = indexer_metadata.compressed_seq_lens
                rows = raw_indices.size(0)
                n_reqs = swap_req_indices.size(0)
                real_rows = n_reqs
                if rows > n_reqs and n_reqs > 0 and rows % n_reqs == 0:
                    per_req = rows // n_reqs
                    swap_req_indices = swap_req_indices.repeat_interleave(per_req)
                    if swap_seq_lens.size(0) == n_reqs:
                        swap_seq_lens = swap_seq_lens.repeat_interleave(per_req)
                    real_rows = n_reqs * per_req
                # ``rows`` is the captured tensor shape and includes graph
                # padding.  Only the expanded live request rows are real.
                # Keep this scalar stable across C4-layer re-entry.
                if not capture_enabled:
                    hisparse_coordinator.num_real_reqs.fill_(real_rows)
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=swap_req_indices,
                        compressed_seq_lens=swap_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    ).to(torch.int32)
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)

    def _compute_c4_indexer_logits(
        self,
        *,
        fn,
        q,
        c4_indexer_kv_cache,
        weights,
        c4_seq_lens,
        page_table,
        indexer_metadata,
        core_metadata,
        forward_batch,
        c4_indexer,
        positions=None,
    ):
        del fn, core_metadata
        # Under BCG the indexer metadata carries the capture-time graph width; the
        # vendor kernel needs the live compressed length (eager already passes it).
        live_max_seq_len = indexer_metadata.max_compressed_seq_len
        if (
            forward_batch is not None
            and forward_batch.forward_mode.is_extend()
            and c4_seq_lens.numel() > 0
        ):
            # At capture the compressed lens are placeholders (1..4), and taking
            # them dimensioned the whole vendor call for a one-column context:
            # width=1, num_pages=1, k_scale=(rows, 1). Replay then indexes far
            # past that row -- dmesg kl3_wait_for_noc_idle / kernel exception 721
            # on fa_xpu::c4a_mqa_logits. Keep the metadata width unless the lens
            # look like real data.
            _live = int(c4_seq_lens.max())
            if _live > 4:
                live_max_seq_len = _live
        return _compute_c4_logits_kunlun(
            backend=self,
            q_fp8=q,
            kvcache_fp8=c4_indexer_kv_cache,
            weight=weights,
            seq_lens=c4_seq_lens,
            page_table=page_table,
            max_seq_len=live_max_seq_len,
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
            positions=positions,
        )

    def on_after_cuda_graph_warmup(self):
        """Skip FlashMLA metadata refresh, matching the Kunlun backend."""
        metadata = self.forward_metadata
        if isinstance(metadata, upstream.DSV4Metadata) and isinstance(
            metadata.core_metadata, upstream.DSV4AttnMetadata
        ):
            core = metadata.core_metadata
            core.c0_flashmla_metadata = None
            core.c1_flashmla_metadata = None
            core.c4_flashmla_metadata = None
            core.c128_flashmla_metadata = None

        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def _build_page_table(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Build the per-row page table through the Kunlun ``page_table_positions`` op.

        Upstream only reaches the registered replacement for
        ``_page_table_positions_kernel`` through
        ``BuildPageTablePositions.execute`` (``dsv4_attn_metadata_kernels.py``),
        which this backend bypasses by building the metadata itself, so the op
        (``kernel_ops.page_table_positions_kernel``) has to be called directly --
        the same wiring ``get_swa_page_indices`` uses for the SWA indices. The op
        returns the page table already divided by ``page_size`` and samples the
        columns ``p * page_size`` exactly like the ``req_to_token[:, :span:
        page_size]`` slice it replaces, so the widened ``_dsv4_page_table_span``
        is preserved.
        """
        page_size = self.page_size
        span = _dsv4_page_table_span(max_seq_len, seq_lens_casual)
        if (
            span > 0
            and req_to_token.dtype == torch.int32
            and req_to_token.stride(-1) == 1
            and hasattr(torch.ops.xspeedgate_ops, "page_table_positions")
        ):
            # The op samples ``p * page_size`` for ``p < num_pages``; clamp the
            # span to the row width so the last sample stays inside the source
            # row, matching the slice's own truncation.
            span = min(span, req_to_token.shape[1])
            num_pages = (span + page_size - 1) // page_size
            num_q = seq_lens_casual.shape[0]
            kw = dict(dtype=torch.int32, device=seq_lens_casual.device)
            seq_lens_out = torch.empty(num_q, **kw)
            positions_out = torch.empty(num_q, **kw)
            page_table = torch.empty((num_q, num_pages), **kw)
            topk_out = torch.empty(num_q, **kw)
            page_table_positions_kernel(
                req_to_token,
                req_pool_indices_repeated,
                seq_lens_casual,
                seq_lens_out,
                positions_out,
                page_table,
                topk_out,
                req_to_token.stride(0),
                num_pages,
                page_size,
                upstream.SWA_WINDOW,
                256,
            )
            return page_table

        # Torch fallback for wheels that predate the op, or non-int32 rows.
        page_table = req_to_token[req_pool_indices_repeated, :span:page_size]
        return (page_table // page_size).to(torch.int32)

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
        dspark_block_size: Optional[int] = None,
        num_tokens: Optional[int] = None,
        # 4.1 additions: `swa_replay_start` is the per-row window floor used by
        # decoder SWA bounded replay, `num_groups` only feeds the request-window
        # layout and `dspark_swa_buffers` lets the caller pass precomputed draft
        # window indices.
        swa_replay_start: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        dspark_swa_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KunlunDSV4AttnMetadata:
        """Build Kunlun core attention metadata for the current request."""
        if dspark_block_size is not None:
            return self._make_dspark_draft_core_attn_metadata(
                req_to_token=req_to_token,
                req_pool_indices_repeated=req_pool_indices_repeated,
                seq_lens_casual=seq_lens_casual,
                max_seq_len=max_seq_len,
                out_loc=out_loc,
                dspark_block_size=dspark_block_size,
                dspark_swa_buffers=dspark_swa_buffers,
            )
        # 4.1 can build the SWA window through `window_layout()` when the encoder
        # bounded-replay feature is on. That path also needs `swa_out_cache_loc`
        # and write-loc plumbing the Kunlun op does not have, so fail loudly
        # instead of silently approximating it.
        assert getattr(self.token_to_kv_pool, "request_window", None) is None, (
            "4.1 request-window (encoder SWA bounded replay) is not supported by "
            "the Kunlun DSV4 backend"
        )
        # The 0.5.19 backend had a ``self.swa_page_size = 128`` constant that was
        # identical to ``SWA_WINDOW``, so the original assertion here was
        # tautologically true; 0.5.22 deleted this redundant field. The
        # ``swa_page_size`` on the pool is a **different quantity** (= the served
        # ``page_size``, 256 in this deployment), and upstream itself asserts
        # ``token_to_kv_pool.swa_page_size % SWA_WINDOW == 0``
        # (``deepseek_v4_backend.py:1154``), which was already checked before
        # this method runs.
        assert self.token_to_kv_pool.swa_page_size % upstream.SWA_WINDOW == 0
        seq_lens_casual = seq_lens_casual.to(torch.int32)
        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            swa_replay_start=swa_replay_start,
        )
        # Match the PyTorch metadata contract: mapping==0 denotes an
        # invalid/tombstoned SWA slot, and valid entries must be a leading
        # prefix. Preserve -1 sentinels for the Kunlun attention operator.
        effective_swa_len = (
            (swa_page_indices != 0)
            .int()
            .cumprod(dim=1)
            .sum(dim=1)
            .to(torch.int32)
        )
        swa_page_indices = upstream._pad_last_dim(
            swa_page_indices, multiples_of=upstream.PAGE_INDEX_ALIGNED_SIZE
        )
        swa_topk_lengths = torch.minimum(
            torch.clamp(seq_lens_casual, max=upstream.SWA_WINDOW),
            effective_swa_len,
        )
        if swa_replay_start is not None:
            # Slots below the floor are unreachable: upstream shrinks the valid
            # count to match the floored window start.
            floored = (seq_lens_casual - 1) - swa_replay_start.to(
                seq_lens_casual.dtype
            ) + 1
            swa_topk_lengths = torch.minimum(
                swa_topk_lengths, floored.clamp_min(0).to(swa_topk_lengths.dtype)
            )
        page_table = self._build_page_table(
            req_to_token, req_pool_indices_repeated, seq_lens_casual, max_seq_len
        )
        metadata = KunlunDSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=seq_lens_casual - 1,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=torch.clamp(seq_lens_casual, max=upstream.SWA_WINDOW),
            # 0.5.22 renamed the dataclass field ``c4_sparse_topk`` to
            # ``index_topk`` and promoted ``present_ratios`` / ``low_ratios``
            # into the constructor (``present_ratios`` has no default), so they
            # are passed here instead of being patched on afterwards.
            index_topk=self.index_topk,
            present_ratios=self.present_ratios,
            low_ratios=self.low_ratios,
        )
        # Apply the effective-length contract after construction as well;
        # this keeps the value correct if upstream metadata initialization resets it.
        metadata.swa_topk_lengths = swa_topk_lengths
        if need_compress:
            metadata.init_compression_metadata(num_tokens=num_tokens)
            metadata.init_flashmla_related(is_prefill=is_prefill)
        else:
            metadata.c4_sparse_topk_lengths = None
            metadata.c4_sparse_page_indices = None
            metadata.c4_sparse_raw_indices = None
            metadata.c0_flashmla_metadata = None
            metadata.c1_flashmla_metadata = None
            metadata.c4_flashmla_metadata = None
            metadata.c128_flashmla_metadata = None
        # Golden Prefill keeps -1 as the invalid-tail sentinel. Decode and
        # speculative verify/draft modes use the existing Kunlun page-0
        # normalization contract.
        if (
            not is_prefill
            and os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") != "1"
        ):
            for field_name in (
                "swa_page_indices",
                "c4_sparse_page_indices",
                "c128_page_indices",
            ):
                indices = getattr(metadata, field_name, None)
                if indices is not None:
                    indices.masked_fill_(indices == -1, 0)
        return metadata


    def _make_dspark_draft_core_attn_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        dspark_block_size: int,
        dspark_swa_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KunlunDSV4AttnMetadata:
        """Core metadata for the DSpark draft block forward (window only, no
        compressed branch).

        ``init_forward_metadata_dspark_draft_block`` runs prefill with
        ``need_compress=False``; the draft does not touch c4/c128 and only uses
        the sliding window. Upstream ``get_dspark_swa_page_indices`` already
        built per-row causal window indices for the gamma draft rows (the
        relevant kernels have Kunlun replacement implementations), so the only
        thing Kunlun has to change is the convention for invalid lanes: fill
        page 0 instead of -1, with the tail masked out by ``swa_topk_lengths``.
        """
        # The 0.5.19 backend had a ``self.swa_page_size = 128`` constant that was
        # identical to ``SWA_WINDOW``, so the original assertion here was
        # tautologically true; 0.5.22 deleted this redundant field. The
        # ``swa_page_size`` on the pool is a **different quantity** (= the served
        # ``page_size``, 256 in this deployment), and upstream itself asserts
        # ``token_to_kv_pool.swa_page_size % SWA_WINDOW == 0``
        # (``deepseek_v4_backend.py:1154``), which was already checked before
        # this method runs.
        assert self.token_to_kv_pool.swa_page_size % upstream.SWA_WINDOW == 0
        seq_lens_casual = seq_lens_casual.to(torch.int32)
        if dspark_swa_buffers is not None:
            swa_page_indices, swa_topk_lengths = dspark_swa_buffers
        else:
            swa_page_indices, swa_topk_lengths = self.get_dspark_swa_page_indices(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool_indices_repeated,
                out_loc=out_loc,
                block_size=dspark_block_size,
            )
        page_table = self._build_page_table(
            req_to_token, req_pool_indices_repeated, seq_lens_casual, max_seq_len
        )
        metadata = KunlunDSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=seq_lens_casual - 1,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            # See the prefill builder above: 0.5.22 constructor field names.
            index_topk=self.index_topk,
            present_ratios=self.present_ratios,
            low_ratios=self.low_ratios,
        )
        metadata.c4_sparse_topk_lengths = None
        metadata.c4_sparse_page_indices = None
        metadata.c4_sparse_raw_indices = None
        metadata.c0_flashmla_metadata = None
        metadata.c1_flashmla_metadata = None
        metadata.c4_flashmla_metadata = None
        metadata.c128_flashmla_metadata = None
        if os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") != "1":
            indices = metadata.swa_page_indices
            if indices is not None:
                indices.masked_fill_(indices == -1, 0)
        return metadata

    @staticmethod
    def _match_queries(tensor, size: int, value: int):
        if tensor is None or tensor.shape[0] == size:
            return tensor
        if tensor.shape[0] > size:
            return tensor[:size]
        return upstream._pad_tensor_to_size(tensor, size, value=value)

    def _make_verify_row_lod_graph_safe(
        self, core_metadata, num_queries: int, device: torch.device
    ):
        """Capture-safe per-row TARGET_VERIFY LoD.

        ``_make_cp_prefill_lod`` expresses the same contract but has to do a D2H
        and then build a new CPU tensor, which cannot be captured into the graph
        (on replay the recorded H2D would read a freed host buffer). So during
        capture it previously fell back to request-level LoD -- exactly the
        layout that lets verify row j read the draft KV of rows j+1.. (see
        ``_verify_row_lod_enabled``), which is why under CUDA graph it degrades
        to a single-token dead loop with accept len pinned at gamma+1 while
        eager is clean.

        Here ``q_lod`` is a constant ``arange`` (which can be baked in), and
        ``kv_lens`` is filled by a device-side op recorded into the graph from
        ``seq_lens_casual``, a buffer refreshed in-place by ``copy_`` before
        every replay; ``kv_lens_cpu`` stays a placeholder, consistent with the
        original assumption of the request-level capture path.
        ``DSV4_VERIFY_ROW_LOD_GRAPH=0`` A/Bs back to the old behavior.
        """
        aux = self._verify_row_lod_aux.get(num_queries)
        if aux is None:
            q_lod_cpu = torch.arange(num_queries + 1, dtype=torch.int32)
            aux = (
                q_lod_cpu,
                q_lod_cpu.to(device, non_blocking=False),
                torch.ones(num_queries, dtype=torch.int32),
                torch.ones(num_queries, dtype=torch.int32, device=device),
            )
            self._verify_row_lod_aux[num_queries] = aux
        q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = aux
        casual = core_metadata.seq_lens_casual.reshape(-1)
        count = min(casual.shape[0], num_queries)
        if count:
            kv_lens[:count].copy_(torch.clamp(casual[:count].to(torch.int32), min=1))
        if count < num_queries:
            kv_lens[count:].fill_(1)
        return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

    def _make_lod(self, forward_batch, num_queries: int, device: torch.device):
        if forward_batch.forward_mode.is_decode_or_idle():
            batch_size = num_queries
            aux = self._attention_decode_aux.get(batch_size)
            if aux is None:
                q_lod_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
                q_lod = q_lod_cpu.to(device, non_blocking=False)
                kv_lens_cpu = torch.ones(batch_size, dtype=torch.int32)
                kv_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
                aux = q_lod_cpu, q_lod, kv_lens_cpu, kv_lens
                self._attention_decode_aux[batch_size] = aux
            q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = aux
            kv_lens.copy_(forward_batch.seq_lens[:batch_size].to(torch.int32))
            if os.environ.get("DSV4_KUNLUN_DECODE_HOST_KVLEN", "1") == "1":
                # ``kv_lens_cpu`` previously kept its ``ones`` initial value (only
                # the device side was backfilled), so
                # ``compressed_topk = max(kv_lens_cpu.max() // ratio, 1)`` was
                # always 1: the compressed branch read only 1 entry, while at
                # 257 tokens c4 should be 64 entries. prefill's host length is
                # the true value so it is unaffected, which is why the divergence
                # from the CUDA reference only starts at the first c4 layer
                # (layer 2) of decode.
                _copy_host_lengths_(
                    kv_lens_cpu, _seq_lens_cpu_i32(forward_batch), fill_value=1
                )
            return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

        if _is_graph_extend_mode(forward_batch.forward_mode) and not getattr(
            forward_batch, "_kunlun_ragged_draft_extend", False
        ):
            if _verify_row_lod_enabled(forward_batch):
                return _make_cp_prefill_lod(
                    self.forward_metadata.core_metadata, num_queries, device
                )
            if (
                forward_batch.forward_mode.is_target_verify()
                and os.environ.get("DSV4_VERIFY_ROW_LOD_GRAPH", "1") == "1"
            ):
                return self._make_verify_row_lod_graph_safe(
                    self.forward_metadata.core_metadata, num_queries, device
                )
            batch_size = forward_batch.batch_size
            aux = _get_graph_extend_aux(
                self._attention_graph_extend_aux, batch_size, num_queries, device
            )
            q_lod_cpu, q_lod, kv_lens_cpu, kv_lens, query_lens = aux
            _copy_host_lengths_(
                kv_lens_cpu,
                _seq_lens_cpu_i32(forward_batch),
                fill_value=1,
            )
            kv_lens.copy_(forward_batch.seq_lens[:batch_size].to(torch.int32))
            torch.maximum(kv_lens, query_lens, out=kv_lens)
            return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

        if _dsa_cp_prefill_ranks(forward_batch) is not None:
            return _make_cp_prefill_lod(
                self.forward_metadata.core_metadata, num_queries, device
            )

        lengths = forward_batch.extend_seq_lens_cpu
        if lengths is None:
            batch_size = forward_batch.seq_lens.shape[0]
            if batch_size == 0 or num_queries % batch_size:
                raise ValueError("cannot derive Kunlun DSV4 extend lengths")
            lengths = [num_queries // batch_size] * batch_size
        batch_size = len(lengths)
        q_lod_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        if batch_size:
            torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0, out=q_lod_cpu[1:])
        kv_lens = forward_batch.seq_lens[:batch_size].to(torch.int32)
        kv_lens_cpu = (
            forward_batch.seq_lens_cpu[:batch_size].to(torch.int32)
            if forward_batch.seq_lens_cpu is not None
            else kv_lens.to("cpu", non_blocking=False)
        )
        # The token dimension can be padded past the scheduler's extend lengths
        # by two independent sources: CP alignment, and CUDA-graph capture (a
        # prefill bucket is an aggregate token count and the host-side
        # ``extend_seq_lens_cpu`` is a placeholder during capture). Either way,
        # give every padding row its own batch item with a dummy KV length so
        # the operator contract qlod[-1] == q rows still holds.
        pad_rows = num_queries - int(q_lod_cpu[-1].item())
        if pad_rows < 0:
            raise ValueError(
                "Kunlun DSV4 extend lengths exceed the query rows: "
                f"lengths={int(q_lod_cpu[-1].item())}, queries={num_queries}"
            )
        if pad_rows:
            q_lod_cpu = torch.cat(
                [
                    q_lod_cpu,
                    q_lod_cpu[-1]
                    + torch.arange(1, pad_rows + 1, dtype=torch.int32),
                ]
            )
            kv_lens_cpu = torch.cat(
                [kv_lens_cpu, torch.ones(pad_rows, dtype=torch.int32)]
            )
            kv_lens = torch.cat(
                [
                    kv_lens,
                    torch.ones(pad_rows, dtype=torch.int32, device=kv_lens.device),
                ]
            )
        # Upload through pinned memory without waiting: a blocking H2D drains the
        # device queue, and under DSPARK that queue still holds an unmatched
        # collective, so the copy never returns (py-spy: rank 7 in _make_lod while
        # the other ranks sit in the embedding all_reduce). Stream order already
        # guarantees the LoD is visible to the kernels that consume it.
        host_lod = q_lod_cpu if q_lod_cpu.is_pinned() else q_lod_cpu.pin_memory()
        q_lod = host_lod.to(device, non_blocking=True)
        return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch
    ) -> None:
        """Use the half-cache writer for the non-fused DSV4 path."""
        pool = self.token_to_kv_pool
        if (
            envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get()
            or pool.swa_kv_pool is None
            or pool.swa_kv_pool.store_dtype not in (torch.bfloat16, torch.float16)
        ):
            return super().store_cache(layer_id, swa_k, forward_batch)

        scheduler_raw_loc = forward_batch.out_cache_loc
        raw_loc = scheduler_raw_loc.to(dtype=torch.int32).contiguous()
        mapping = pool.full_to_swa_index_mapping
        assert mapping is not None
        swa_pool = pool.swa_kv_pool
        local_layer_id = pool._swa_local_layer_id(layer_id)
        pack = dsv4_quant_k_cache_kunlun(swa_k)
        dsv4_set_k_and_s_with_mapping_kunlun(
            swa_pool.kv_buffer[local_layer_id],
            raw_loc,
            mapping,
            pack,
            swa_pool.page_size,
        )

    def _low_ratio_prefill_reads_page_indices(self, forward_batch) -> bool:
        """Kunlun prefill always needs the paged index set.

        This upstream switch is the opposite of **its own** sparse-prefill
        decision: when it thinks ``flash_mla_sparse_fwd`` will run, it hands the
        indexer ``out_page_indices=None`` and writes only ``out_raw_indices``
        (the two branches of ``write_prefill``,
        ``v41_indexer/scoring.py:348-352``). But the Kunlun backend only takes
        that path for ``_KUNLUN_SPARSE_PREFILL_RATIOS`` (0 and 128); every
        low-ratio indexer layer (ratio 1/2/4) falls back to
        ``compressed_attention`` (or its torch reference implementation), and
        that path reads ``c{ratio}_sparse_page_indices``.

        After 0.5.22 split prefill's ``c{ratio}_sparse_raw_indices`` into
        separate -1 buffers, if nobody writes the page buffer it stays -1:
        within the window things still work, so generation looks fluent, but
        everything earlier than the SWA window disappears (measured effective
        retrieval range collapses to about 60 tokens, while the old stack still
        answers correctly on the same probe at 450 tokens). So this
        unconditionally returns True; when ratio 0/128 really takes sparse
        prefill, writing the page index once more has no semantic impact.
        """
        return True

    def _maybe_forward_prefill_sparse(
        self,
        q: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        attn_sink: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Run pure prefill on the sparse-attention op, or return None to fall
        back to ``compressed_attention``.

        Upstream ``_forward_prefill_sparse`` gathers the SWA window and the
        compressed cache into one flat workspace and hands it to
        ``flash_mla_sparse_fwd``, which ``flash_mla_sparse_fwd_kunlun`` replaces
        with ``kunlun_ops.sparse_prefill_fwd``. Nothing in that path is
        CUDA-specific once ``dequantize_k_cache_paged`` has its Kunlun gather
        (``kernel_ops.dsv4_dequantize_k_cache_paged_kunlun``), so the whole
        chunk-cache scaffolding is reused as-is -- its two hot kernels already
        run on XSpeedGate (``build_swa_token_ids``, ``combine_topk_swa_indices``).

        The gate mirrors upstream's (``deepseek_v4_backend.py``: extend without
        speculation, no late-layer tail, no request window) plus the Kunlun-only
        exclusions: CP prefill splits the rows per rank in a way the chunk cache
        does not model, graph capture cannot host the data-dependent workspace
        sizing, and the reference-attention debug paths must stay reachable.
        """
        if not _kunlun_sparse_prefill_enabled():
            return None
        if compress_ratio not in _KUNLUN_SPARSE_PREFILL_RATIOS:
            return None
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return None
        if self.is_dspark_draft:
            return None
        if self.forward_metadata.late_layer_tail is not None:
            return None
        if self.token_to_kv_pool.request_window is not None:
            return None
        if _dsa_cp_prefill_ranks(forward_batch) is not None:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        if (
            os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") == "1"
            or os.environ.get("DSV4_KUNLUN_REFERENCE_ATTN") == "1"
        ):
            return None
        if attn_sink is None:
            # Upstream asserts a sink on this path; without one the post-hoc
            # sink fold in flash_mla_sparse_fwd_kunlun has nothing to apply.
            return None

        # Upstream takes (b, 1, h_q, d_qk) and squeezes dim 1. Its workspace is
        # bf16, so q is cast to match; on a fp16 pool that rounds the gathered
        # KV from 10 to 7 mantissa bits, which `compressed_attention` does not
        # do. Worth measuring before this path is widened past c0/c128.
        q_4d = q if q.ndim == 4 else q.unsqueeze(1)
        original_dtype = q_4d.dtype
        out = super()._forward_prefill_sparse(
            q=q_4d.to(torch.bfloat16),
            layer_id=layer.layer_id,
            compress_ratio=compress_ratio,
            forward_batch=forward_batch,
            token_to_kv_pool=self.token_to_kv_pool,
            core_attn_metadata=self.forward_metadata.core_metadata,
            attn_sink=attn_sink,
        )
        return out.to(original_dtype) if out.dtype != original_dtype else out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Run Kunlun attention for the requested compression ratio."""
        # TEMP DEBUG: confirm the Kunlun backend forward is really on the call chain.
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)
        assert k is v, "DeepseekV4 shares k and v"
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)

        sparse_out = self._maybe_forward_prefill_sparse(
            q, layer, forward_batch, compress_ratio, attn_sink
        )
        if sparse_out is not None:
            return sparse_out

        core = self.forward_metadata.core_metadata
        pool = self.token_to_kv_pool
        cache_dim = pool.swa_kv_pool.kv_cache_total_dim
        # `swa_window_size` was deleted in the new `DeepSeekV4TokenToKVPool`; it
        # was always just an alias for `swa_page_size` (the per-request width of
        # the SWA ring).
        swa_size = pool.swa_page_size
        win_cache = pool.get_swa_key_buffer_radix(layer.layer_id)
        win_cache = win_cache.reshape(-1, cache_dim)

        win_indices = self._match_queries(core.swa_page_indices, q.shape[0], 0)
        win_indices = win_indices[:, :swa_size].contiguous()
        win_lengths = self._match_queries(core.swa_topk_lengths, q.shape[0], 0)
        if self.is_dspark_draft:
            # The draft window constructor marks lanes beyond
            # ``swa_topk_lengths`` as -1, and ``compressed_attention`` has no
            # window-length parameter. So -1 cannot simply be clamped to page 0
            # -- the op would treat it as real KV and include it in softmax
            # (measured 59 of 192 lanes invalid, about 31% of weight landing on
            # unrelated tokens). Its window loop upper bound is
            # ``min(kv_len, max_window_size)``, so the way to mask the tail is
            # to clamp kv_len to the window length, see the clamp below.
            if (
                win_lengths.numel()
                and os.environ.get("DSV4_KUNLUN_DRAFT_WINDOW_COMPACT", "0") == "1"
            ):
                # Old fallback when DSV4_KUNLUN_DRAFT_WINDOW_KVLEN=0: compress
                # the lane dimension so ``max_window_size`` itself becomes the
                # mask. But ``valid`` is a device-side value and the result shape
                # depends on data, which cannot be captured into a CUDA graph.
                valid = int(win_lengths.max().item())
                valid = max(1, min(valid, win_indices.shape[1]))
                win_indices = win_indices[:, :valid].contiguous()
            win_indices.masked_fill_(win_indices < 0, 0)
        extra_cache = None
        extra_indices = None
        extra_lengths = None
        if compress_ratio in (1, 2):
            # DeepSeek V4.1 low ratios: the compressed latent pool is the extra
            # cache and the indexer published the top-k slots into it. The
            # `page_width = page_size // ratio` split below matches the
            # `page_size // ratio` the compressed pool was built with.
            extra_cache = pool.get_extra_key_buffer(layer.layer_id)
            extra_indices = core.sparse_page_indices(compress_ratio)
            extra_lengths = core.sparse_topk_lengths(compress_ratio)
        elif compress_ratio == 4:
            extra_cache = pool.get_extra_key_buffer(layer.layer_id)
            extra_indices = core.c4_sparse_page_indices
            extra_lengths = core.c4_sparse_topk_lengths
        elif compress_ratio == 128:
            extra_cache = pool.get_extra_key_buffer(layer.layer_id)
            extra_indices = core.c128_page_indices
            extra_lengths = core.c128_topk_lengths_clamp1
        extra_indices = self._match_queries(extra_indices, q.shape[0], -1)
        extra_lengths = self._match_queries(extra_lengths, q.shape[0], 0)
        q_3d = q.squeeze(1) if q.ndim == 4 else q
        local_sink = attn_sink
        padded_q_heads = None
        original_dtype = q_3d.dtype

        if (
            os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") == "1"
            or os.environ.get("DSV4_KUNLUN_REFERENCE_ATTN") == "1"
            or dsv4_stable_dsa_enabled()
            or _cp_short_prompt_needs_reference(forward_batch)
            or not hasattr(torch.ops.xspeedgate_ops, "compressed_attention")
        ):

            if extra_cache is not None:
                page_width = pool.page_size // compress_ratio
                extra_cache = extra_cache[:, : page_width * cache_dim].reshape(
                    -1, cache_dim
                )
            for name, value in (
                ("compressed_attention.input.q", q_3d),
                ("compressed_attention.input.win_cache", win_cache),
                ("compressed_attention.input.win_indices", win_indices),
                ("compressed_attention.input.extra_cache", extra_cache),
                ("compressed_attention.input.extra_indices", extra_indices),
                ("compressed_attention.input.win_lengths", win_lengths),
                ("compressed_attention.input.extra_lengths", extra_lengths),
            ):
                _dsv4_dump_backend_tensor(self, name, value, layer)
            reference_out = dsv4_compressed_attention_torch(
                q=q_3d,
                win_cache=win_cache,
                win_indices=win_indices,
                win_lengths=win_lengths,
                softmax_scale=self.softmax_scale,
                attn_sink=local_sink,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_lengths=extra_lengths,
            )
            _dsv4_dump_backend_tensor(
                self, "compressed_attention.output.out", reference_out, layer
            )
            return reference_out

        # forward overrides upstream._forward_attention: the sgl_kernel hook
        # alone is not reachable from serving. Call it before legacy index
        # clamping, with actual decode lengths rather than padded capacities.
        # Verify/extend, draft, CP and c0 keep their old paths.  Decode graph
        # capture is supported by the hybrid adapter when its eager warmup has
        # installed fixed-address output and metadata buffers.
        if (
            forward_batch.forward_mode.is_decode()
            and not self.is_dspark_draft
            and _dsa_cp_prefill_ranks(forward_batch) is None
            and compress_ratio in (1, 2, 4, 128)
            and extra_cache is not None
            and win_cache.dtype in (torch.float16, torch.bfloat16)
        ):
            # Resolve the host mirror at most once for this backend forward.
            # The operator receives the authoritative device seq_lens below;
            # the CPU value is only its legacy host-side metadata argument.
            decode_lengths_cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if decode_lengths_cpu is None:
                decode_lengths_cpu = _seq_lens_cpu_i32(forward_batch)
            page_width = pool.page_size // compress_ratio
            hybrid_extra = extra_cache[:, : page_width * cache_dim].reshape(
                -1, cache_dim
            )
            hybrid_result = flash_mla_with_kvcache_kunlun(
                q=q_3d.to(win_cache.dtype).unsqueeze(1), k_cache=win_cache,
                indices=win_indices.unsqueeze(1), head_dim_v=layer.v_head_dim,
                softmax_scale=self.softmax_scale, attn_sink=local_sink,
                topk_length=win_lengths, extra_k_cache=hybrid_extra,
                extra_indices_in_kvcache=extra_indices.unsqueeze(1),
                extra_topk_length=extra_lengths,
                # The hybrid adapter consumes device seq_lens directly when
                # the scheduler did not publish a CPU mirror.  This keeps the
                # decode layer loop free of a per-layer D2H.
                kunlun_decode_lengths=decode_lengths_cpu,
                kunlun_compress_ratio=compress_ratio,
                kunlun_layer_id=layer.layer_id,
                kunlun_forward_batch=forward_batch,
                kunlun_backend=self,
                kunlun_fallback_to_reference=False,
            )
            if hybrid_result is not None:
                return hybrid_result[0].squeeze(1).to(original_dtype)

        if q_3d.dtype != win_cache.dtype:
            q_3d = q_3d.to(win_cache.dtype)
        q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = self._make_lod(
            forward_batch, q_3d.shape[0], q_3d.device
        )
        if (
            self.is_dspark_draft
            and win_lengths.numel()
            and os.environ.get("DSV4_KUNLUN_DRAFT_WINDOW_KVLEN", "1") == "1"
        ):
            # ``compressed_attention``'s window loop upper bound is ``min(kv_len,
            # max_window_size)``, while the draft's kv_len is the whole sequence
            # length, far beyond ``swa_topk_lengths``, so all invalid lanes would
            # be attended. Clamping kv_len to the window length is equivalent to
            # lane compression, but the shape does not depend on data and can be
            # captured. All draft rows share the same window length, so taking
            # ``max()`` here is exact.
            # Only clamp the device-side tensor: ``kv_lens_cpu`` is a placeholder
            # during capture. Measured clamping it too produces byte-identical
            # 836-token output.
            kv_lens = torch.minimum(kv_lens, win_lengths.max().to(kv_lens.dtype))
        cp_prefill = _dsa_cp_prefill_ranks(forward_batch) is not None
        if int(q_lod_cpu[-1].item()) != q_3d.shape[0]:
            raise ValueError(
                "Kunlun DSV4 attention LoD does not cover the query rows: "
                f"qlod_last={int(q_lod_cpu[-1].item())}, q_rows={q_3d.shape[0]}, "
                f"cp_prefill={cp_prefill}, "
                f"casual_rows={core.seq_lens_casual.shape[0]}, "
                f"page_table_rows={core.page_table.shape[0]}, "
                f"extend_lens={forward_batch.extend_seq_lens_cpu}, "
                f"batch_size={forward_batch.batch_size}, "
                f"mode={forward_batch.forward_mode}"
            )
        if extra_cache is None:
            extra_cache = win_cache.new_empty((0, cache_dim))
            extra_indices = torch.empty(
                (q_3d.shape[0], 0), dtype=torch.int32, device=q_3d.device
            )
            compressed_topk = 0
            effective_ratio = 1
        else:
            page_width = pool.page_size // compress_ratio
            extra_cache = extra_cache[:, : page_width * cache_dim].reshape(-1, cache_dim)
            if (
                (
                    forward_batch.forward_mode.is_decode_or_idle()
                    or _is_graph_extend_mode(forward_batch.forward_mode)
                )
                and torch.cuda.is_current_stream_capturing()
            ):
                # During graph capture (decode, TARGET_VERIFY, DRAFT_EXTEND_V2),
                # kv_lens_cpu holds dummy placeholder values. Use the full static
                # stride so replay can consume the actual indices produced in-graph,
                # matching the MAX_SEQ_LEN_FOR_CAPTURE behaviour.
                compressed_topk = extra_indices.shape[1]
            else:
                max_seq_len = int(kv_lens_cpu.max().item()) if kv_lens_cpu.numel() else 0
                if (
                    forward_batch.forward_mode.is_target_verify()
                    and kv_lens_cpu.numel()
                    and os.environ.get("DSV4_VERIFY_PREFIX_TOPK", "1") == "1"
                ):
                    # ``compressed_topk`` is a single scalar applied to all query
                    # rows, originally taken from the longest row. In
                    # TARGET_VERIFY the longest is the last draft position, so
                    # every row (including row 0, which is the only committed one)
                    # was allowed to read the compressed entry covering the draft
                    # block, and that entry is published in this very forward.
                    # Change it to bound by the shortest row (i.e. the committed
                    # prefix): entries 0..K-1 only cover positions < K*ratio <=
                    # prefix, so no row can reach an uncommitted position. No
                    # information is lost -- the excluded tokens are already
                    # inside the exact SWA window (>=128).
                    max_seq_len = int(kv_lens_cpu.min().item())
                compressed_topk = min(
                    max(max_seq_len // compress_ratio, 1), extra_indices.shape[1]
                )
                if (
                    compress_ratio == 128
                    and forward_batch.forward_mode.is_extend()
                    and not _is_graph_extend_mode(forward_batch.forward_mode)
                    and not cp_prefill
                ):
                    compressed_topk = _clamp_c128_prefill_topk(
                        compressed_topk,
                        kv_lens_cpu,
                        forward_batch.extend_seq_lens_cpu,
                        compress_ratio,
                    )
            extra_indices = extra_indices[:, :compressed_topk].clamp(
                min=0, max=extra_cache.shape[0] - 1
            ).contiguous()
            effective_ratio = compress_ratio

        out = torch.zeros_like(q_3d)
        max_logits = torch.zeros(
            (q_3d.shape[0], q_3d.shape[1]),
            dtype=torch.float32,
            device=q_3d.device,
        )
        lse = torch.zeros_like(max_logits)
        q_op = _as_packed(q_3d, "compressed_attention.q")
        win_cache_op = _as_packed(win_cache, "compressed_attention.win_cache")
        win_indices_op = _as_packed(win_indices, "compressed_attention.win_indices")
        extra_cache_op = _as_packed(extra_cache, "compressed_attention.extra_cache")
        extra_indices_op = _as_packed(
            extra_indices, "compressed_attention.extra_indices"
        )
        # Outputs: a plain ``.contiguous()`` here would send the kernel into a
        # throwaway copy and leave `out` at zeros.
        out_op, out_writeback = _as_packed_out(out, "compressed_attention.out")
        max_logits_op, max_logits_writeback = _as_packed_out(
            max_logits, "compressed_attention.max_logits"
        )
        lse_op, lse_writeback = _as_packed_out(lse, "compressed_attention.lse")
        q_lod_cpu_op = _as_packed(q_lod_cpu, "compressed_attention.q_lod_cpu")
        q_lod_op = _as_packed(q_lod, "compressed_attention.q_lod")
        kv_lens_cpu_op = _as_packed(kv_lens_cpu, "compressed_attention.kv_lens_cpu")
        kv_lens_op = _as_packed(kv_lens, "compressed_attention.kv_lens")
        attn_sink_op = (
            _as_packed(local_sink, "compressed_attention.attn_sink")
            if local_sink is not None
            else None
        )
        # Mirror the Golden 0.5.14 device probe: local heads only, the
        # index-selected cache rows actually consumed, and post-clamp indices.
        local_q_heads = getattr(layer, "tp_q_head_num", q_op.shape[1])
        local_q_heads = min(local_q_heads, q_op.shape[1])
        for name, value in () if not _dsv4_dump_enabled() else (
            ("compressed_attention.input.q", q_op[:, :local_q_heads]),
            ("compressed_attention.input.win_indices", win_indices_op),
            (
                "compressed_attention.input.win_cache",
                win_cache_op.index_select(
                    0, torch.unique(win_indices_op).to(torch.long)
                ),
            ),
            ("compressed_attention.input.extra_indices", extra_indices_op),
            (
                "compressed_attention.input.extra_cache",
                extra_cache_op.index_select(
                    0, torch.unique(extra_indices_op).to(torch.long)
                ),
            ),
            ("compressed_attention.input.attn_sink", attn_sink_op),
            ("compressed_attention.input.q_lod_cpu", q_lod_cpu_op),
            ("compressed_attention.input.q_lod", q_lod_op),
            ("compressed_attention.input.kv_lens_cpu", kv_lens_cpu_op),
            ("compressed_attention.input.kv_lens", kv_lens_op),
            ("compressed_attention.input.causal", torch.tensor(not cp_prefill)),
            (
                "compressed_attention.input.compress_ratio",
                torch.tensor(effective_ratio),
            ),
            (
                "compressed_attention.input.compressed_topk",
                torch.tensor(compressed_topk),
            ),
            (
                "compressed_attention.input.max_window_size",
                torch.tensor(win_indices_op.shape[1]),
            ),
            (
                "compressed_attention.input.softmax_scale",
                torch.tensor(self.softmax_scale, dtype=torch.float64),
            ),
            (
                "compressed_attention.input.win_cache_rows_total",
                torch.tensor(win_cache_op.shape[0]),
            ),
            (
                "compressed_attention.input.extra_cache_rows_total",
                torch.tensor(extra_cache_op.shape[0]),
            ),
        ):
            _dsv4_dump_backend_tensor(self, name, value, layer)
        if os.environ.get("DSV4_KUNLUN_COMPACT_CACHE") == "1":
            # Diagnostic: rebuild the caches so they contain only the indexed
            # rows. If the output then matches Golden, the kernel is sensitive
            # to buffer content outside the indexed rows.
            win_unique = torch.unique(win_indices_op)
            win_cache_op = win_cache_op.index_select(
                0, win_unique.to(torch.long)
            ).contiguous()
            win_indices_op = torch.searchsorted(
                win_unique, win_indices_op.reshape(-1)
            ).reshape(win_indices_op.shape).to(win_indices_op.dtype).contiguous()
            if extra_cache_op.shape[0] > 0 and extra_indices_op.numel() > 0:
                extra_unique = torch.unique(extra_indices_op)
                extra_cache_op = extra_cache_op.index_select(
                    0, extra_unique.to(torch.long)
                ).contiguous()
                extra_indices_op = torch.searchsorted(
                    extra_unique, extra_indices_op.reshape(-1)
                ).reshape(extra_indices_op.shape).to(
                    extra_indices_op.dtype
                ).contiguous()
        torch.ops.xspeedgate_ops.compressed_attention(
            q_op,
            win_cache_op,
            win_indices_op,
            extra_cache_op,
            extra_indices_op,
            out_op,
            max_logits_op,
            lse_op,
            q_lod_cpu_op,
            q_lod_op,
            kv_lens_cpu_op,
            kv_lens_op,
            self.softmax_scale,
            # CP round-robin batches one local token per item; the per-token KV
            # lengths already encode the causal prefix, so kernel-side causal
            # masking would truncate valid context.
            not cp_prefill,
            win_indices_op.shape[1],
            effective_ratio,
            compressed_topk,
            attn_sink_op,
            # # Keep the C4 producer on the caller stream so PyTorch owns the
            # # lifetime of the contiguous temporary inputs.
            # side_stream=torch.cuda.current_stream().cuda_stream,
        )
        if out_writeback is not None:
            out_writeback.copy_(out_op)
        if max_logits_writeback is not None:
            max_logits_writeback.copy_(max_logits_op)
        if lse_writeback is not None:
            lse_writeback.copy_(lse_op)
        for name, value in () if not _dsv4_dump_enabled() else (
            ("compressed_attention.output.out_pre_rope", out_op[:, :local_q_heads]),
            (
                "compressed_attention.output.max_logits",
                max_logits_op[:, :local_q_heads],
            ),
            ("compressed_attention.output.lse", lse_op[:, :local_q_heads]),
        ):
            _dsv4_dump_backend_tensor(self, name, value, layer)
        if padded_q_heads is not None:
            padded_out = out.new_zeros(out.shape[0], padded_q_heads, out.shape[2])
            padded_out[:, : out.shape[1]] = out
            out = padded_out
        return out.to(original_dtype) if out.dtype != original_dtype else out


# NOTE: 0.5.17 needed a REPLACE hook on ``Compressor.forward_cuda`` because
# ``MultiPlatformOp`` declared it as ``raise NotImplementedError``. In 0.5.19
# ``Compressor`` derives from ``BaseFusedOp``, which defines no platform
# forwards at all — an undefined ``forward_cuda`` now falls back to
# ``forward_native`` on its own, which is exactly what the hook restored.



def _build_c4_prefill_contract(
    forward_batch, c4_seq_lens, page_table, device, num_queries, positions=None
):
    """Build the request-level LoD contract used by the extend path."""
    global_extend_lens = [int(value) for value in forward_batch.extend_seq_lens_cpu]
    cp_ranks = _dsa_cp_prefill_ranks(forward_batch)

    if cp_ranks is None:
        # c4_flat is only used here; keeping it off the CP path avoids a dead per-layer D2H sync.
        c4_flat = c4_seq_lens.reshape(-1).to("cpu", dtype=torch.int32)
        extend_lens = list(global_extend_lens)
        # BCG replay reports the bucket-padded extend length; trim it so
        # ``last_rows`` addresses the request's real last row (k_lens != 0).
        _pad_excess = sum(extend_lens) - int(num_queries)
        while _pad_excess > 0 and extend_lens:
            _cut = min(_pad_excess, max(extend_lens[-1] - 1, 0))
            if _cut <= 0:
                break
            extend_lens[-1] -= _cut
            _pad_excess -= _cut
        qlod_cpu = torch.zeros(len(extend_lens) + 1, dtype=torch.int32)
        if extend_lens:
            qlod_cpu[1:] = torch.cumsum(
                torch.tensor(extend_lens, dtype=torch.int32), 0
            )
        last_rows = (qlod_cpu[1:] - 1).clamp(min=0, max=page_table.shape[0] - 1)
        per_req_k_lens = c4_flat.index_select(
            0, last_rows.clamp(max=c4_flat.numel() - 1).long()
        )
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        use_causal = prefix_lens is not None and all(
            int(value) < 3 for value in prefix_lens
        )
        # ``extend_prefix_lens_cpu`` is a capture-time placeholder under replay;
        # take the prefix from the first query row's position instead.
        if positions is not None and positions.numel() > 0:
            _starts = qlod_cpu[:-1].long()
            _starts = _starts.clamp(max=max(positions.numel() - 1, 0))
            use_causal = all(int(positions[int(_i)]) < 3 for _i in _starts)
    else:
        # CP round-robin: c4_seq_lens/page_table are per local token while
        # extend_seq_lens_cpu stays global, so rebuild the per-request query
        # windows from this rank's round-robin share.
        cp_rank, cp_size = cp_ranks
        extend_lens = _dsa_cp_local_extend_lens(
            global_extend_lens, cp_rank, cp_size
        )
        # [fix] The replay batch is padded to the capture bucket, so
        # ``extend_seq_lens_cpu`` only carries the *padded* token count (e.g.
        # 7168 for a 5603-token tail chunk) and the request's real local row
        # count is not available from it.  ``c4_seq_lens`` is per row and CP
        # padding rows sit at 1 (row 0 legitimately has 0), so it reveals the
        # real row count -- which matches the eager shape exactly.  Without this
        # the request's last row lands on a padding row, ``per_req_k_lens``
        # collapses to 1, the logits buffer is sized 4 columns and the C4
        # indexer selects almost no pages (nz=32 vs 4096).  The trailing padding
        # rows are covered by the existing pad_rows logic.
        if is_in_breakable_cuda_graph():
            # The captured graph owns the row geometry, so ``extend_lens`` must
            # span *all* captured rows -- including the CP padding rows a partial
            # tail chunk leaves at the end of the bucket.  Upstream pins that
            # geometry in ``attn_cp_metadata.per_rank_actual_token`` at replay
            # (cp/bcg.py restores it to the captured local token count).  Do NOT
            # shrink it to the "real" row count: the live ``c4_seq_lens`` is not
            # monotonic in the row index, so any such heuristic drops real rows.
            _md = getattr(forward_batch, "attn_cp_metadata", None)
            _prat = (
                getattr(_md, "per_rank_actual_token", None) if _md is not None else None
            )
            if _prat:
                _pinned = int(_prat[min(cp_rank, len(_prat) - 1)])
                if _pinned > 0 and len(extend_lens) == 1 and sum(extend_lens) != _pinned:
                    extend_lens = [_pinned]
        extend_lens_tensor = torch.tensor(extend_lens, dtype=torch.int32)
        qlod_cpu = torch.zeros(len(extend_lens) + 1, dtype=torch.int32)
        if extend_lens:
            qlod_cpu[1:] = torch.cumsum(extend_lens_tensor, 0)
        last_rows = (qlod_cpu[1:].long() - 1).clamp(min=0)
        # Ranks with no local token for a request must still address a valid
        # page_table row.
        last_rows[extend_lens_tensor <= 0] = 0
        last_rows = last_rows.clamp(max=max(page_table.shape[0] - 1, 0)).to(
            torch.int32
        )
        # The last LOCAL token is not the last GLOBAL token of the request, so
        # the compressed KV length must come from the full request context.
        # [fix] Take the compressed length from the *live* indexer metadata
        # (``c4_seq_lens`` -- the same source the non-CP branch uses) instead of
        # ``forward_batch.seq_lens_cpu``, which under BCG replay holds the
        # capture/padded value.  ``c4_seq_lens`` grows along the rows, so the
        # request's full compressed length is its last real row.
        if not is_in_breakable_cuda_graph():
            # Eager path is unchanged: the batch's own seq_lens are authoritative.
            per_req_k_lens = (
                _seq_lens_cpu_i32(forward_batch)[: len(extend_lens)] // 4
            ).to(torch.int32)
            per_req_k_lens = torch.clamp(per_req_k_lens, min=1)
        else:
            c4_flat = c4_seq_lens.reshape(-1).to("cpu", dtype=torch.int32)
            if c4_flat.numel() == 0:
                per_req_k_lens = torch.ones(len(extend_lens), dtype=torch.int32)
            else:
                per_req_k_lens = c4_flat.index_select(
                    0, last_rows.long().clamp(max=c4_flat.numel() - 1)
                )
            per_req_k_lens = torch.clamp(per_req_k_lens, min=1)
        # The logits width is ``per_req_k_lens.max()`` and the vendor topk scans
        # ``c4_seq_lens[row]`` columns of it, so the contract must cover the live
        # context maximum; ``c4_seq_lens`` is not strictly monotonic in the row
        # index, so the last-row value alone can undershoot.
        if is_in_breakable_cuda_graph() and len(extend_lens) == 1 and c4_flat.numel():
            # Bound by what the page table can address: during capture
            # ``c4_seq_lens`` is placeholder data and an unbounded floor would ask
            # for a ~1e9-wide logits buffer.
            _cap = max(1, int(page_table.shape[1]) * 64)  # c4 page size = 256 // 4
            per_req_k_lens = torch.clamp(
                per_req_k_lens, min=min(int(c4_flat.max().item()), _cap)
            )
        use_causal = False

    klod_cpu = torch.zeros(len(extend_lens) + 1, dtype=torch.int32)
    if extend_lens:
        klod_cpu[1:] = torch.cumsum(per_req_k_lens * 4, 0)
    com_k_start_cpu = torch.zeros(len(extend_lens), dtype=torch.int32)

    # CP alignment pads the token dimension past the scheduler's extend lengths;
    # each padding row becomes its own item with a single dummy compressed entry.
    pad_rows = num_queries - int(qlod_cpu[-1].item())
    if pad_rows < 0:
        raise ValueError(
            "Kunlun C4 extend lengths exceed the query rows: "
            f"lengths={int(qlod_cpu[-1].item())}, queries={num_queries}"
        )
    if pad_rows:
        pad_offsets = torch.arange(1, pad_rows + 1, dtype=torch.int32)
        pad_last_rows = int(qlod_cpu[-1].item()) - 1 + pad_offsets
        qlod_cpu = torch.cat([qlod_cpu, qlod_cpu[-1] + pad_offsets])
        last_rows = torch.cat(
            [
                last_rows.to(torch.int32),
                pad_last_rows.clamp(min=0, max=max(page_table.shape[0] - 1, 0)),
            ]
        )
        per_req_k_lens = torch.cat(
            [per_req_k_lens, torch.ones(pad_rows, dtype=torch.int32)]
        )
        klod_cpu = torch.cat([klod_cpu, klod_cpu[-1] + pad_offsets * 4])
        com_k_start_cpu = torch.cat(
            [com_k_start_cpu, torch.zeros(pad_rows, dtype=torch.int32)]
        )

    # One host->device copy instead of three (read-only LoD inputs; slices are safe).
    _nq, _nk, _nc = qlod_cpu.numel(), klod_cpu.numel(), com_k_start_cpu.numel()
    _packed_xpu = torch.cat([qlod_cpu, klod_cpu, com_k_start_cpu]).to(device)
    qlod_xpu = _packed_xpu[:_nq]
    klod_xpu = _packed_xpu[_nq : _nq + _nk]
    com_k_start_xpu = _packed_xpu[_nq + _nk : _nq + _nk + _nc]
    return {
        "qlod_cpu": qlod_cpu,
        "qlod_xpu": qlod_xpu,
        "klod_cpu": klod_cpu,
        "klod_xpu": klod_xpu,
        "com_k_start_cpu": com_k_start_cpu,
        "com_k_start_xpu": com_k_start_xpu,
        "per_req_k_lens": per_req_k_lens,
        "last_rows": last_rows,
        "cp_ranks": cp_ranks,
        "max_seq_q": max(extend_lens) if extend_lens else 0,
        "max_seq_k": int(per_req_k_lens.max().item()) * 4 if per_req_k_lens.numel() else 0,
        "max_seq_k_compressed": int(per_req_k_lens.max().item()) if per_req_k_lens.numel() else 0,
        "use_causal": use_causal,
    }


def _gather_c4_prefill_kv(cache, page_table, contract, device):
    """Gather packed C4 pages exactly as the extend path."""
    block_size = cache.shape[1]
    head_dim = cache.shape[-1] - 4
    cache_u8 = cache.contiguous().view(torch.uint8).reshape(
        cache.shape[0], block_size * (head_dim + 4)
    )
    k_pages = cache_u8[:, : block_size * head_dim].reshape(
        -1, block_size, head_dim
    )
    scale_pages = cache_u8[:, block_size * head_dim :].contiguous().view(
        torch.float32
    ).reshape(-1, block_size)
    lengths = contract["per_req_k_lens"]
    total_k = int(lengths.sum().item())
    k_contiguous = torch.empty((total_k, head_dim), dtype=torch.int8, device=device)
    k_scale_contiguous = torch.empty(total_k, dtype=torch.float32, device=device)
    offset = 0
    for request_index, k_len_value in enumerate(lengths.tolist()):
        k_len = int(k_len_value)
        if not k_len:
            continue
        num_pages = (k_len + block_size - 1) // block_size
        page_row = int(contract["last_rows"][request_index].item())
        pages = page_table[page_row, :num_pages].long()
        k_contiguous[offset : offset + k_len] = k_pages[pages].reshape(
            -1, head_dim
        )[:k_len].to(torch.int8)
        k_scale_contiguous[offset : offset + k_len] = scale_pages[pages].reshape(
            -1
        )[:k_len]
        offset += k_len
    return k_contiguous, k_scale_contiguous


def _c4_target_verify_logits_chunked(
    *, q_fp8, kvcache_fp8, weight, seq_lens, page_table, max_seq_len, num_requests
):
    """Per-row C4 verify logits, computed one draft column at a time.

    Each query row must select its own pages, but the reference implementation's
    intermediates scale with the row count, so doing all draft rows in one call
    exhausts device memory during graph capture. Rows of request r live at
    r * nd + j, hence the strided column slices.
    """
    num_queries = q_fp8.shape[0]
    if num_requests <= 0 or num_queries % num_requests:
        raise ValueError(
            "Kunlun C4 TARGET_VERIFY requires uniform queries per request: "
            f"num_requests={num_requests}, num_queries={num_queries}"
        )
    num_draft = num_queries // num_requests
    seq_lens, page_table = _expand_c4_target_verify_rows(
        q_fp8, seq_lens, page_table
    )
    if num_draft <= 1:
        return dsv4_c4_paged_mqa_logits_torch(
            q_int8=q_fp8,
            kvcache_int8=kvcache_fp8,
            weight=weight,
            seq_lens=seq_lens,
            page_table=page_table,
            max_seq_len=max_seq_len,
        )
    logits = q_fp8.new_empty((num_queries, max_seq_len), dtype=torch.float32)
    for column in range(num_draft):
        rows = slice(column, num_queries, num_draft)
        logits[rows] = dsv4_c4_paged_mqa_logits_torch(
            q_int8=q_fp8[rows],
            kvcache_int8=kvcache_fp8,
            weight=weight[rows],
            seq_lens=seq_lens[rows],
            page_table=page_table[rows],
            max_seq_len=max_seq_len,
        )
    return logits


def _expand_c4_target_verify_rows(q_fp8, seq_lens, page_table):
    """Give every TARGET_VERIFY query row its own C4 length and page-table row.

    Upstream computes the sparse selection per draft row; collapsing to one
    representative row makes row 0 attend to another row's pages, which is
    exactly where the MTP output diverges from the DECODE path.
    """
    num_queries = q_fp8.shape[0]
    seq_lens = seq_lens.reshape(-1)
    if seq_lens.shape[0] != num_queries:
        if num_queries % seq_lens.shape[0]:
            raise ValueError(
                "Kunlun C4 TARGET_VERIFY cannot expand lengths: "
                f"num_queries={num_queries}, lengths={seq_lens.shape[0]}"
            )
        seq_lens = seq_lens.repeat_interleave(num_queries // seq_lens.shape[0], dim=0)
    if page_table.shape[0] != num_queries:
        if num_queries % page_table.shape[0]:
            raise ValueError(
                "Kunlun C4 TARGET_VERIFY cannot expand the page table: "
                f"num_queries={num_queries}, page_rows={page_table.shape[0]}"
            )
        page_table = page_table.repeat_interleave(
            num_queries // page_table.shape[0], dim=0
        )
    return seq_lens, page_table


def _select_c4_target_verify_rows(
    q_fp8, weight, seq_lens, page_table, num_requests
):
    """Match the C4 TARGET_VERIFY representative-row contract."""
    num_queries = q_fp8.shape[0]
    if num_requests <= 0 or num_queries % num_requests:
        raise ValueError(
            "Kunlun C4 TARGET_VERIFY requires uniform queries per request: "
            f"num_requests={num_requests}, num_queries={num_queries}"
        )

    num_queries_per_request = num_queries // num_requests
    request_rows = torch.arange(
        num_requests, dtype=torch.long, device=q_fp8.device
    )
    page_table_is_per_request = page_table.shape[0] == num_requests
    if page_table_is_per_request:
        representative_rows = request_rows
    else:
        representative_rows = (
            request_rows * num_queries_per_request + num_queries_per_request - 1
        )

    q_fp8 = q_fp8.index_select(0, representative_rows)
    weight = weight.index_select(0, representative_rows)

    seq_lens = seq_lens.reshape(-1)
    if page_table_is_per_request:
        seq_lens = seq_lens[:num_requests]
    else:
        seq_lens = seq_lens.index_select(0, representative_rows)
        page_table = page_table.index_select(0, representative_rows)
    return q_fp8, weight, seq_lens, page_table, num_queries_per_request


def _has_xspeedgate_op(name: str) -> bool:
    """Whether the installed XSpeedGate build exposes ``name``.

    The paged c4 logits fast path below is optional: ``mqa_logits_paged`` is
    present only in newer vendor builds, and a missing op raises
    ``AttributeError: '_OpNamespace' 'xspeedgate_ops' object has no attribute
    ...`` on the first prefill rather than at import. Probe once and fall back to
    the gather + ``c4a_mqa_logits`` path, which computes the same logits.
    """
    cached = _XSPEEDGATE_OP_CACHE.get(name)
    if cached is None:
        cached = hasattr(torch.ops.xspeedgate_ops, name)
        _XSPEEDGATE_OP_CACHE[name] = cached
        if not cached:
            logger.warning(
                "xspeedgate_ops.%s is unavailable in this build; using the "
                "non-paged Kunlun fallback for the same computation",
                name,
            )
    return cached


_XSPEEDGATE_OP_CACHE: dict = {}


def _compute_c4_logits_kunlun(
    *,
    backend,
    q_fp8,
    kvcache_fp8,
    weight,
    seq_lens,
    page_table,
    max_seq_len,
    forward_batch,
    c4_indexer,
    positions=None,
):
    """Compute C4 logits with explicit Kunlun request ownership."""

    # Capture-time indexer metadata can carry an uninitialised
    # ``max_compressed_seq_len`` (placeholder lens), and DSPARK verify-graph
    # capture then asks for a 2134 GiB logits buffer. Bound the width by the
    # model context (c4 compresses 4:1); this only ever shrinks placeholder
    # capture-time widths, never a real one.
    _ctx_cap = int(getattr(backend, "MAX_SEQ_LEN_FOR_CAPTURE", 0) or 0)
    if _ctx_cap > 0:
        max_seq_len = min(int(max_seq_len), _ctx_cap // 4 + _page_slack())

    is_target_verify = (
        forward_batch is not None
        and forward_batch.forward_mode.is_target_verify()
    )
    if (
        os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") == "1"
        or os.environ.get("DSV4_KUNLUN_REFERENCE_C4") == "1"
        or dsv4_stable_dsa_enabled()
        or _cp_short_prompt_needs_reference(forward_batch)
    ):
        if is_target_verify:
            return _c4_target_verify_logits_chunked(
                q_fp8=q_fp8,
                kvcache_fp8=kvcache_fp8,
                weight=weight,
                seq_lens=seq_lens,
                page_table=page_table,
                max_seq_len=max_seq_len,
                num_requests=forward_batch.batch_size,
            )
        return dsv4_c4_paged_mqa_logits_torch(
            q_int8=q_fp8,
            kvcache_int8=kvcache_fp8,
            weight=weight,
            seq_lens=seq_lens,
            page_table=page_table,
            max_seq_len=max_seq_len,
        )

    import kunlun_ops

    layer_id = c4_indexer.layer_id

    use_contiguous_prefill = (
        forward_batch is not None
        and forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.extend_seq_lens_cpu is not None
    )
    if use_contiguous_prefill:
        q = q_fp8.squeeze(1).contiguous()
        if q.dtype == torch.uint8:
            q = q.view(torch.int8)
        weights = weight.float().contiguous()
        contract = _build_c4_prefill_contract(
            forward_batch,
            seq_lens,
            page_table,
            q.device,
            q.shape[0],
            positions=positions,
        )
        if (
            contract["cp_ranks"] is None
            and q.shape[1:] == (64, 128)
            and weights.shape[1] == 64
            and kvcache_fp8.shape[1:] == (64, 1, 132)
            and _has_xspeedgate_op("mqa_logits_paged")
        ):
            pt_rows = (
                contract["last_rows"]
                .clamp(min=0, max=page_table.shape[0] - 1)
                .to(torch.long)
            )
            logits = torch.ops.xspeedgate_ops.mqa_logits_paged(
                q,
                weights,
                kvcache_fp8,
                page_table[pt_rows].contiguous(),
                contract["per_req_k_lens"].to(q.device),
                contract["qlod_cpu"],
                contract["qlod_xpu"],
                max_seq_len,
                contract["use_causal"],
            )
            pad_width = (logits.shape[1] + 3) & ~3
            if logits.shape[1] < pad_width:
                padded = torch.full(
                    (q.shape[0], pad_width),
                    float("-inf"),
                    dtype=logits.dtype,
                    device=q.device,
                )
                padded[:, : logits.shape[1]] = logits
                logits = padded
            return logits
        k, k_scale = _gather_c4_prefill_kv(
            kvcache_fp8, page_table, contract, q.device
        )
        logits = torch.empty(
            (q.shape[0], contract["max_seq_k_compressed"]),
            dtype=torch.float32,
            device=q.device,
        )
        if k.shape[0]:
            kunlun_ops.c4a_mqa_logits(
                q=q,
                weight=weights,
                k=k,
                k_scale=k_scale,
                logits=logits,
                max_seq_q=contract["max_seq_q"],
                max_seq_k=contract["max_seq_k"],
                qlod_cpu=contract["qlod_cpu"],
                qlod_xpu=contract["qlod_xpu"],
                klod_cpu=contract["klod_cpu"],
                klod_xpu=contract["klod_xpu"],
                com_k_start_cpu=contract["com_k_start_cpu"],
                com_k_start_xpu=contract["com_k_start_xpu"],
                is_causal=contract["use_causal"],
                compress_ratio=4,
                clean_logits=True,
            )
        else:
            logits.zero_()
        # Pad only to a 4-aligned width: topk_transform_512_v2 requires
        # score_stride % 4 == 0 for vectorized loads, and every topk variant
        # reads exactly seq_lens[row] columns per row. Padding to the global
        # max_c4_seq_len (page-table width, e.g. 262144 at 1M context) instead
        # would allocate a batch-sized, context-width fp32 buffer on every
        # prefill chunk and easily OOM when the GPU is already tight.
        pad_width = (logits.shape[1] + 3) & ~3
        if logits.shape[1] < pad_width:
            padded = torch.full(
                (q.shape[0], pad_width),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            padded[:, : logits.shape[1]] = logits
            logits = padded
        return logits

    num_queries_per_request = 1
    per_row_verify = False
    if is_target_verify:
        if os.environ.get("DSV4_C4_VENDOR_PER_ROW", "1") != "0":
            # The vendor kernel indexes one page-table row per q row, so keeping
            # every draft row (with its own c4 length) makes the sparse selection
            # per row, exactly like DECODE. Broadcasting one representative row
            # instead makes draft rows attend to another row's pages.
            per_row_verify = True
            seq_lens, page_table = _expand_c4_target_verify_rows(
                q_fp8, seq_lens, page_table
            )
        else:
            q_fp8, weight, seq_lens, page_table, num_queries_per_request = (
                _select_c4_target_verify_rows(
                    q_fp8,
                    weight,
                    seq_lens,
                    page_table,
                    forward_batch.batch_size,
                )
            )

    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    seq_lens = seq_lens.view(-1)[:batch_size]
    q = q_fp8.view(torch.int8) if q_fp8.dtype == torch.uint8 else q_fp8
    q = _as_packed(q, "c4a_paged_mqa_logits.q")
    weights = _as_packed(
        weight.unsqueeze(1), "c4a_paged_mqa_logits.weight"
    )

    cache_flat = kvcache_fp8.view(
        kvcache_fp8.shape[0], block_size * (head_dim + 4)
    )
    scale_offset = block_size * head_dim
    k_cache = (
        cache_flat[:, :scale_offset]
        .contiguous()
        .view(-1, block_size, 1, head_dim)
        .view(torch.int8)
    )
    num_pages = min(
        (max_seq_len + block_size - 1) // block_size,
        page_table.shape[1],
    )
    # k_scale is built from ``num_pages`` pages, so the context bound has to
    # match it: the paged kernel asserts max_context_len / compress_ratio ==
    # k_scale.size(1), and the capture-time width can exceed it by a page.
    max_seq_len = min(int(max_seq_len), num_pages * block_size)
    # ``page_table`` is already int32, so ``.to(torch.int32)`` is a no-op and the
    # column slice keeps ``stride(0) == page_table.shape[1]``.  The vendor kernel
    # derives the row stride from ``size(1)``, not ``stride(0)`` (verified with
    # probe_block_table_stride.py: rows 1..N read another request's pages), so the
    # table has to be packed before the call.  At page_size=256 the slice is a
    # no-op and this costs nothing; it only bites at other page sizes.
    block_table = _as_packed(
        page_table[:, :num_pages].to(torch.int32), "c4a_paged_mqa_logits.block_table"
    )
    scale_pages = (
        cache_flat[:, scale_offset:]
        .contiguous()
        .view(torch.float32)
        .view(-1, block_size)
    )
    # ``k_scale`` is materialised from ``num_pages`` pages, so the kernel's
    # ``max_context_len / compress_ratio`` must equal ``k_scale.size(1)``. The
    # capture-time width (context bound) can exceed the page table's coverage at
    # this mem-fraction, which fails the vendor assert -> clamp to what the table
    # actually describes.
    max_seq_len = min(int(max_seq_len), num_pages * block_size)
    # ``context_lens`` must never exceed ``max_context_len`` (= max_seq_len * 4):
    # the vendor kernel (fa_xpu::c4a_mqa_logits) indexes the logits row with it, and
    # a longer length walks off the row -- dmesg kl3_wait_for_noc_idle, kernel
    # exception 721, then the TP collective behind it hangs. Widening the row to
    # the real length is safe (clipping the length is not: it would silently drop
    # context). The bound comes from the host mirror, so no device sync, and
    # capture-time placeholders are ignored by the ``<= MAX_SEQ_LEN_FOR_CAPTURE``
    # sanity check.
    _host_lens = _seq_lens_cpu_i32(forward_batch)
    _ctx_tokens = int(getattr(backend, "MAX_SEQ_LEN_FOR_CAPTURE", 0) or 0)
    if _host_lens.numel():
        _host_max = int(_host_lens.max())
        if _host_max > 0 and (_ctx_tokens <= 0 or _host_max <= _ctx_tokens):
            _needed_width = (_host_max + 3) // 4
            if _needed_width > int(max_seq_len):
                max_seq_len = min(_needed_width, num_pages * block_size)
    # Capture-time page ids are placeholder garbage; an out-of-range gather (or a
    # kernel reading them) stalls the device queue instead of raising.  Replay
    # refreshes the table in place, so clamping only affects warmup/capture.
    block_table = block_table.clamp_(
        min=0, max=max(int(scale_pages.shape[0]) - 1, 0)
    )
    page_indices = block_table.to(torch.long)
    k_scale = scale_pages[page_indices].reshape(batch_size, -1)[
        :, :max_seq_len
    ].contiguous()

    rows_per_req = (
        max(batch_size // forward_batch.batch_size, 1) if per_row_verify else 1
    )
    key = (batch_size, q_fp8.device, max_seq_len, rows_per_req)
    aux = backend._c4_decode_aux.get(key)
    if aux is None:
        qlod_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
        qlod_xpu = qlod_cpu.to(q_fp8.device)
        context_lens_cpu = torch.full(
            (batch_size,), 4, dtype=torch.int32
        )
        context_lens_xpu = torch.full(
            (batch_size,), 4, dtype=torch.int32, device=q_fp8.device
        )
        aux = qlod_cpu, qlod_xpu, context_lens_cpu, context_lens_xpu
        backend._c4_decode_aux[key] = aux
    qlod_cpu, qlod_xpu, context_lens_cpu, context_lens_xpu = aux
    # Capture-time placeholder lens can be garbage (~1.5e9); feeding them to the
    # kernel as context_lens overflows int32 and leaves the device queue stuck.
    # Replay refreshes these tensors in place, so clamping here only affects the
    # warmup/capture pass that would otherwise hang.
    _lens_cap = int(max_seq_len)
    seq_lens = seq_lens.clamp(min=1, max=_lens_cap)
    raw_seq_lens_cpu = _seq_lens_cpu_i32(forward_batch).clamp(max=_lens_cap * 4)
    if per_row_verify:
        # batch_size is now bs*num_draft; the host lengths must line up row-wise
        # with the expanded page table.
        requests = raw_seq_lens_cpu.numel()
        if requests and batch_size % requests == 0:
            raw_seq_lens_cpu = raw_seq_lens_cpu.repeat_interleave(
                batch_size // requests
            )
    _copy_host_lengths_(
        context_lens_cpu,
        (raw_seq_lens_cpu // 4) * 4,
        fill_value=4,
    )
    context_lens_xpu.copy_(seq_lens.to(torch.int32) * 4)

    # The vendor dispatch is `fa_xpu::c4a_mqa_logits<...>`; a shape mismatch there
    # does not raise, it wedges the device (dmesg: kl3_wait_for_noc_idle) and the
    # following TP collective hangs. Assert the contract first so a bad shape is
    # a Python error instead of a node reboot.
    _rows = (
        int(q.shape[0]),
        int(batch_size),
        int(seq_lens.numel()),
        int(block_table.shape[0]),
        int(k_scale.shape[0]),
        int(context_lens_cpu.numel()),
        int(context_lens_xpu.numel()),
        int(qlod_cpu.numel()) - 1,
    )
    if len(set(_rows)) != 1:
        raise RuntimeError(
            "Kunlun c4a_paged_mqa_logits row mismatch "
            "(q, batch_size, seq_lens, block_table, k_scale, ctx_cpu, ctx_xpu, qlod): "
            f"{_rows} max_seq_len={max_seq_len} per_row_verify={per_row_verify} "
            f"verify={is_target_verify}"
        )
    if int(logits_width := max_seq_len) <= 0:
        raise RuntimeError(f"Kunlun c4a_paged_mqa_logits bad width {logits_width}")
    logits = torch.empty(
        (batch_size, 1, max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    layer_id = c4_indexer.layer_id
    kunlun_ops.c4a_paged_mqa_logits(
        q=q,
        weight=weights,
        k_cache=k_cache,
        k_scale=k_scale,
        logits=logits,
        max_context_len=max_seq_len * 4,
        qlod_cpu=qlod_cpu,
        qlod_xpu=qlod_xpu,
        context_lens_cpu=context_lens_cpu,
        context_lens_xpu=context_lens_xpu,
        block_table=block_table,
        compress_ratio=4,
        clean_logits=True,
        use_xfa_boost=False,
    )
    logits = logits.squeeze(1)
    if is_target_verify and not per_row_verify:
        logits = (
            logits.unsqueeze(1)
            .expand(-1, num_queries_per_request, -1)
            .reshape(-1, max_seq_len)
            .contiguous()
        )
    return logits


class KunlunDeepseekV4MultiStepBackend(DeepseekV4MultiStepBackend):
    """0.5.14 multi-step control flow composed only of Kunlun backends."""

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        DeepseekV4AttnBackend.__init__(self, model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[KunlunDeepseekV4AttnBackend] = [
            KunlunDeepseekV4AttnBackend(
                model_runner,
                speculative_step_id=i,
                topk=topk,
                speculative_num_steps=speculative_num_steps,
            )
            for i in range(speculative_num_steps)
        ]
        self._mtp_probe_replay_calls = 0

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        # Seed replay from step 0, then copy its live metadata into every
        # later draft step that participates in this captured decode.
        from types import SimpleNamespace

        seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None and out_cache_loc.dtype != torch.int32:
            out_cache_loc = out_cache_loc.to(torch.int32)
        inner_fb = SimpleNamespace(
            batch_size=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            actual_forward_mode=getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            ),
            input_ids=getattr(forward_batch, "input_ids", None),
            positions=getattr(forward_batch, "positions", None),
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=seq_lens_i32,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=out_cache_loc,
            spec_info=forward_batch.spec_info,
        )
        if in_capture:
            for backend in self.attn_backends:
                backend.init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
        elif (
            self.speculative_num_steps > 1
            and os.environ.get("DSV4_MULTISTEP_PER_STEP_METADATA") == "1"
        ):
            # Each backend slices out_cache_loc by its own speculative_step_id, so
            # let every step build its own metadata rather than inherit step 0's.
            for backend in self.attn_backends:
                backend.init_forward_metadata_out_graph(inner_fb)
        elif self.speculative_num_steps > 1:
            self.attn_backends[0].init_forward_metadata_out_graph(inner_fb)
            temp_metadata = self.attn_backends[0].forward_metadata
            last_step = (
                self.speculative_num_steps
                if os.environ.get("DSV4_MULTISTEP_REFRESH_ALL_STEPS") == "1"
                else self.speculative_num_steps - 1
            )
            for i in range(1, last_step):
                backend = self.attn_backends[i]
                backend.replay_cuda_graph_metadata_from(
                    bs=forward_batch.batch_size,
                    temp_metadata=temp_metadata,
                    bucket=upstream._GraphBucket.DECODE_OR_IDLE,
                )
                _refresh_graph_host_lengths(
                    inner_fb,
                    backend._attention_decode_aux,
                    backend._c4_decode_aux,
                    backend._attention_graph_extend_aux,
                )


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor."
    "CompressorBackendMixin.forward_compress",
    type=HookType.REPLACE,
)
def _forward_compress_zeroed_kunlun(
    self,
    *,
    kv_score_buffer,
    kv_score_input,
    ape,
    head_dim,
    norm,
    freqs_cis_cache,
    rotate,
    forward_batch,
    compress_ratio,
    is_paged=False,
):
    """Zero sparse-plan rows that the kernel leaves untouched before cache store."""
    from sglang.kernels.ops.attention.dsv4.compress_old import (
        compress_forward,
        compress_fused_norm_rope_inplace,
    )
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
    from sglang.srt.layers.attention.dsv4.compressor import (
        is_overlap_compress,
        make_compressor_plan,
    )
    from sglang.srt.environ import envs

    assert compress_ratio in (4, 128)
    if is_paged:
        metadata = self.get_paged_compress_metadata(compress_ratio)
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
    else:
        plan = make_compressor_plan(compress_ratio, forward_batch)
        metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
    indices, extra_data, plan = metadata

    out = kv_score_input.new_zeros((kv_score_input.shape[0], head_dim))
    kv_compressed = compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        plan=plan,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        extra_data=extra_data,
        out=out,
    )
    compress_fused_norm_rope_inplace(
        kv_compressed,
        norm.weight,
        norm.variance_epsilon,
        freqs_cis_cache,
        plan,
    )
    return rotate_activation(kv_compressed) if rotate else kv_compressed


# ── CP prefill indexer: dense fp4 logits exist only in deep_gemm; route to
# upstream's pure torch implementation ──
#
# Symptom: with CP enabled on P, when a **real request** arrives all 8 ranks
# crash
#     ImportError: cannot import name 'fp8_fp4_mqa_logits' from 'deep_gemm'
# Call chain:
#     forward_low_ratio_sources(2832) -> _forward_low_ratio_sources_cp(2902)
#       -> _low_ratio_index_topk_dense(3313) -> _dense_fp4_mqa_logits(339)
#         -> from deep_gemm import fp8_fp4_mqa_logits
#
# Upstream already has a dispatch (`_low_ratio_index_topk`, 3200-3233):
#     elif self._use_dense_fp4_prefill_indexer(fb) and _is_sm100_or_newer():
#         self._low_ratio_index_topk_extend(...)   # dense fp4 (deep_gemm)
#     else:
#         self._low_ratio_index_topk_torch(...)    # pure torch fallback
# `_use_dense_fp4_prefill_indexer` reads `SGLANG_DSV41_TORCH_PREFILL_INDEXER`
# (set to 1 in the script). **But the CP branch bypasses this dispatch** and
# calls `_low_ratio_index_topk_dense` directly, -> the switch has no effect on
# CP, so requests inevitably hit deep_gemm.
#
# Why the 1-token warmup did not blow up: `_cp_short_prompt_needs_reference` in
# the plugin routes degenerate batches to the Torch reference path; only real
# prompts reach here.
#
# Approach: hook `_low_ratio_index_topk_dense` AROUND and, when "the switch
# requires torch" or "deep_gemm cannot get fp8_fp4_mqa_logits", redirect to
# upstream's own `_low_ratio_index_topk_torch`. **Do not copy any upstream
# method** -- directly reuse that pure torch implementation. It has its own
# tiling (`_TORCH_INDEXER_SCORE_BUDGET_BYTES` controls the `[rows, heads, lc]`
# intermediate tensor, added to cap memory for 16k prompts), and semantically
# it already processes per request group.
#
# How `req` is constructed: the dense version receives `q_lens` (tensor) and
# `q_lens_cpu` (list) as parameters; they are the **per-request local row
# counts** of this rank (under CP each rank only computes its own rows), from
# which `repeat_interleave` can exactly rebuild each row's request index. The
# torch version iterates per request via `torch.unique_consecutive(req)`, and
# this rank's rows are naturally contiguous per request.
#
# The switch reuses `SGLANG_DSV41_TORCH_PREFILL_INDEXER`; additionally, when
# deep_gemm is unavailable this path is taken automatically, so it does not
# blow up just because someone forgot to set the switch.


def _dense_fp4_indexer_usable() -> bool:
    """Whether the dense fp4 indexer is usable (depends on deep_gemm's fp8_fp4_mqa_logits)."""
    from sglang_kunlun.hooks.mem_cache.common import low_ratio_index_k_int8_enabled

    if low_ratio_index_k_int8_enabled():
        return False
    if envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get():
        return False
    try:
        from deep_gemm import fp8_fp4_mqa_logits  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True
