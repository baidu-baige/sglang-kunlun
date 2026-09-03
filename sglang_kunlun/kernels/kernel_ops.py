"""Fine-grained Kunlun replacements for SGLang Triton/JIT kernel symbols."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)
_ENABLE_DSV4_ACCURACY_DUMPS = False
_DSV4_PROBE_COUNTERS: Dict[str, int] = {}

import xspeedgate_ops


def _dsv4_probe(group: str, tensors: Mapping[str, object], **meta) -> None:
    """Persist opt-in DSV4 intermediate tensors without affecting the hot path."""
    output_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
    tensor_values = [value for value in tensors.values() if isinstance(value, torch.Tensor)]
    if not output_dir or not tensor_values:
        return
    expected_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
    if expected_tokens and tensor_values[0].shape[0] != expected_tokens:
        return
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank != int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
        return
    sample_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_SAMPLE_TOKENS", "8"))
    tensors = {
        name: value[:sample_tokens] if isinstance(value, torch.Tensor) and value.ndim else value
        for name, value in tensors.items()
    }
    key = f"{rank}:{group}"
    index = _DSV4_PROBE_COUNTERS.get(key, 0)
    _DSV4_PROBE_COUNTERS[key] = index + 1
    if index >= int(os.environ.get("DSV4_ACCURACY_DUMP_LIMIT", "16")):
        return
    payload = {
        "meta": {"rank": rank, "group": group, "call": index, **meta},
        "tensors": {
            name: value.detach().cpu()
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, os.path.join(output_dir, f"rank{rank}_{group}_{index:04d}.pt"))

def _debug_tensor_meta(name: str, value: object) -> str:
    if isinstance(value, torch.Tensor):
        return f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    return f"{name}: value={value!r}, type={type(value).__name__}"


KernelKey = Tuple[str, str]


@dataclass(frozen=True)
class KernelSpec:
    """Describe an upstream kernel symbol and its Kunlun replacement."""

    module_path: str
    kernel_name: str
    impl: Callable
    metadata: Mapping[str, object] = field(default_factory=dict)


class KernelLauncher:
    """Compatibility wrapper for Triton ``kernel[grid](*args)`` call sites."""

    def __init__(self, name: str, impl: Callable):
        """Initialize the launcher with an upstream name and replacement callable."""

        self.name = name
        self.impl = impl

    def __getitem__(self, grid):
        """Return a callable compatible with Triton ``kernel[grid]`` syntax."""

        def run(*args, **kwargs):
            """Invoke the Kunlun replacement implementation."""

            signature = inspect.signature(self.impl)
            if "grid" in signature.parameters:
                kwargs.setdefault("grid", grid)
            return self.impl(*args, **kwargs)

        return run

    def __call__(self, *args, **kwargs):
        """Invoke the replacement directly for non-indexed call sites."""

        return self.impl(*args, **kwargs)


_TRITON_OPS: Dict[KernelKey, KernelSpec] = {}
_JIT_OPS: Dict[KernelKey, KernelSpec] = {}


def register_triton_op(
    module_path: str,
    kernel_name: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Callable:
    """Register a Kunlun replacement for an upstream Triton kernel symbol."""

    def decorator(fn: Callable) -> Callable:
        """Store the decorated function as the replacement implementation."""
        key = (module_path, kernel_name)
        if key in _TRITON_OPS:
            raise ValueError(f"triton op already registered: {key}")
        _TRITON_OPS[key] = KernelSpec(module_path, kernel_name, fn, metadata or {})
        return fn

    return decorator


def register_jit_op(
    module_path: str,
    kernel_name: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Callable:
    """Register a Kunlun replacement for an upstream Python JIT helper."""

    def decorator(fn: Callable) -> Callable:
        """Store the decorated function as the replacement implementation."""
        key = (module_path, kernel_name)
        if key in _JIT_OPS:
            raise ValueError(f"jit op already registered: {key}")
        _JIT_OPS[key] = KernelSpec(module_path, kernel_name, fn, metadata or {})
        return fn

    return decorator


def registered_triton_ops() -> Dict[KernelKey, KernelSpec]:
    """Return registered Triton kernel replacement specifications."""

    return dict(_TRITON_OPS)


def registered_jit_ops() -> Dict[KernelKey, KernelSpec]:
    """Return registered Python JIT helper replacement specifications."""

    return dict(_JIT_OPS)


def _replace_imported_bindings(original: object, replacement: object) -> None:
    """Patch already-imported SGLang aliases that still point to the old symbol."""

    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("sglang."):
            continue
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for symbol_name, value in list(namespace.items()):
            if value is not original:
                continue
            setattr(module, symbol_name, replacement)
            logger.info("kernel_ops: patched imported binding %s.%s", module_name, symbol_name)


def _patch_symbol(spec: KernelSpec, replacement: object) -> None:
    """Replace one upstream module symbol with its Kunlun implementation."""

    key = (spec.module_path, spec.kernel_name)
    try:
        module = importlib.import_module(spec.module_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kernel_ops: failed to import %s.%s: %s", *key, exc)
        return
    if not hasattr(module, spec.kernel_name):
        logger.warning("kernel_ops: %s.%s not found", *key)
        return
    original = getattr(module, spec.kernel_name)
    _replace_imported_bindings(original, replacement)
    setattr(module, spec.kernel_name, replacement)
    logger.info("kernel_ops: patched %s.%s", *key)


def dsv4_mqa_wo_a_einsum_kunlun(
    o: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Run the DSV4 wo_a reduction with the Torch reference contraction."""
   
    return torch.ops.xspeedgate_ops.einsum_tgd_grd_tgr(
        o.contiguous(),
        weight.contiguous(),
    )

    # return torch.einsum("tgd,grd->tgr", o, weight)


def dsv4_mqa_forward_with_full_sink_kunlun(original_fn, self, *args, **kwargs):
    """Expose the full attention sink while MQALayer.forward runs."""

    original_local_sink = self._attn_sink_local
    if self.attn_tp_size > 1:
        self._attn_sink_local = self.attn_sink
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        self._attn_sink_local = original_local_sink


def install() -> None:
    """Install all registered Kunlun kernel replacements into upstream modules."""

    for spec in _TRITON_OPS.values():
        replacement = (
            spec.impl
            if spec.metadata.get("call_style") == "direct"
            else KernelLauncher(spec.kernel_name, spec.impl)
        )
        _patch_symbol(spec, replacement)
    for spec in _JIT_OPS.values():
        _patch_symbol(spec, spec.impl)


def _dsv4_plan_payload(plan):
    """Return the compressor-v1 payload expected by the Kunlun kernels."""

    if getattr(plan, "is_decode", False):
        return plan.seq_lens, None, 1
    return plan.compress_plan, plan.write_plan, 0


@register_jit_op("sglang.kernels.ops.attention.dsv4.compress_old", "compress_forward")
def dsv4_compress_forward_kunlun(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan=None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the compressor-v1 C4/C128 forward kernel on Kunlun."""

    if plan is None:
        from sglang.kernels.ops.attention.dsv4.compress_old import compress_plan

        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            kv_score_input.shape[0],
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert head_dim % 128 == 0
    assert plan.compress_ratio == compress_ratio
    if out is None:
        out = kv_score_input.new_empty((kv_score_input.shape[0], head_dim))
    payload, write_plan, _ = _dsv4_plan_payload(plan)
    torch.ops.xspeedgate_ops.compress_forward_fast(
        kv_score_buffer,
        kv_score_input,
        out,
        ape,
        indices,
        payload,
        write_plan,
        extra_data,
    )
    return out


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.compress_old", "compress_fused_norm_rope_inplace"
)
def dsv4_compress_fused_norm_rope_inplace_kunlun(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan,
) -> None:
    """Apply compressor-v1 RMSNorm and GPT-J RoPE in place on Kunlun."""

    import kunlun_ops

    payload, _, mode = _dsv4_plan_payload(plan)
    if payload.numel() == 0:
        return
    handle = payload.view(torch.int32) if not mode else payload
    kunlun_ops.dpsk_v4_norm_rope_gptj(
        kv,
        weight,
        handle,
        freq_cis,
        mode,
        plan.compress_ratio,
        eps,
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.attn", "triton_create_paged_compress_data")
def dsv4_create_paged_compress_data_kunlun(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
):
    """Create compressor-v1 paged indices using the Kunlun fused op."""

    del block
    return torch.ops.xspeedgate_ops.create_paged_compress_data(
        req_pool_indices=req_pool_indices.to(torch.int64),
        seq_lens=seq_lens.to(torch.int32).contiguous(),
        extend_seq_lens=extend_seq_lens.to(torch.int32).contiguous(),
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping.to(torch.int64),
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        compress_ratio=compress_ratio,
        is_overlap=is_overlap,
    )


def _dsv4_topk_torch_fallback(*args, **kwargs) -> None:
    from sglang.srt.layers.attention.dsv4.indexer import (
        topk_transform_512_pytorch_vectorized,
    )

    topk_transform_512_pytorch_vectorized(*args, **kwargs)


_C4_LOGITS_CHUNK_BYTES_ENV = "SGLANG_KUNLUN_C4_LOGITS_CHUNK_BYTES"
# Peak device bytes allowed for one row tile of the reference C4 logits path.
# The (rows, kv_len, num_heads) FP32 score tile dominates, so an unbounded row
# count OOMs on long context; 512 MiB keeps the tile small while still giving
# the BMM enough rows to stay efficient.
_DEFAULT_C4_LOGITS_CHUNK_BYTES = 512 * 1024 * 1024


def dsv4_c4_paged_mqa_logits_torch(
    q_int8: torch.Tensor,
    kvcache_int8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute C4 indexer logits from Kunlun's INT8 plus FP32-scale cache.

    Memory, not compute, is the binding constraint: the score tile is
    ``rows * kv_len * num_heads`` FP32 and the dequantised key tile is
    ``rows * kv_len * head_dim`` FP32, so a full-batch long-context call needs
    tens of GiB. Three things keep the peak bounded:

    * only the page columns the caller can consume are gathered (everything
      past ``max_seq_len`` used to be computed and then sliced off);
    * rows are processed in tiles sized by a byte budget
      (``SGLANG_KUNLUN_C4_LOGITS_CHUNK_BYTES``);
    * relu / weight / mask run in place, and each tile is written straight into
      the zero-filled output instead of building padded copies.

    Results are bit-identical to the unchunked formulation.
    """

    batch_size, _, num_heads, head_dim = q_int8.shape
    block_size = kvcache_int8.shape[1]
    if head_dim != 128 or block_size != 64:
        raise ValueError(
            f"unsupported C4 layout: head_dim={head_dim}, block_size={block_size}"
        )
    if kvcache_int8.shape[2:] != (1, head_dim + 4):
        raise ValueError(f"invalid C4 cache shape: {tuple(kvcache_int8.shape)}")

    seq_lens = seq_lens.reshape(-1)[:batch_size]
    page_table = page_table[:batch_size]
    if weight.shape != (batch_size, num_heads):
        raise ValueError(
            f"invalid C4 weight shape: expected {(batch_size, num_heads)}, "
            f"got {tuple(weight.shape)}"
        )

    out = q_int8.new_zeros((batch_size, max_seq_len), dtype=torch.float32)
    if batch_size == 0 or max_seq_len <= 0:
        return out

    # Columns beyond max_seq_len are dropped by the caller: never gather them.
    num_pages = min(page_table.shape[1], -(-max_seq_len // block_size))
    if num_pages == 0:
        return out
    kv_len = min(num_pages * block_size, max_seq_len)

    page_count = kvcache_int8.shape[0]
    page_bytes = block_size * (head_dim + 4)
    scale_offset = block_size * head_dim
    cache_flat = kvcache_int8.reshape(page_count, page_bytes).view(torch.int8)
    safe_pages = (
        page_table[:, :num_pages]
        .to(torch.int64)
        .clamp(min=0, max=max(page_count - 1, 0))
    )
    queries = q_int8[:, 0].view(torch.int8).float()
    weights = weight.float()
    positions = torch.arange(kv_len, device=out.device).unsqueeze(0)

    # FP32 keys + FP32 scores + INT8 gather + FP32 row logits + bool mask.
    row_bytes = kv_len * (head_dim * 4 + num_heads * 4 + (head_dim + 4) + 4 + 1)
    budget = int(
        os.environ.get(_C4_LOGITS_CHUNK_BYTES_ENV, _DEFAULT_C4_LOGITS_CHUNK_BYTES)
    )
    chunk = max(1, budget // max(row_bytes, 1))

    for lo in range(0, batch_size, chunk):
        hi = min(lo + chunk, batch_size)
        rows = hi - lo

        gathered = cache_flat.index_select(0, safe_pages[lo:hi].reshape(-1)).reshape(
            rows, num_pages, page_bytes
        )
        # ``.float()`` on the strided slice fuses the contiguous copy with the
        # cast, so the INT8 key tile is never materialised separately.
        keys = gathered[..., :scale_offset].float().reshape(rows, -1, head_dim)
        scales = (
            gathered[..., scale_offset:]
            .contiguous()
            .view(torch.float32)
            .reshape(rows, -1)
        )
        del gathered

        scores = torch.bmm(keys, queries[lo:hi].transpose(1, 2))
        del keys
        scores.relu_().mul_(weights[lo:hi].unsqueeze(1))
        row_logits = scores.sum(dim=2)
        del scores

        row_logits.mul_(scales)
        row_logits = row_logits[:, :kv_len]
        row_logits.masked_fill_(positions >= seq_lens[lo:hi].unsqueeze(1), 0.0)
        out[lo:hi, :kv_len] = row_logits

    return out


def dsv4_compressed_attention_torch(
    q: torch.Tensor,
    win_cache: torch.Tensor,
    win_indices: torch.Tensor,
    win_lengths: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    query_block_size: int = 256,
) -> torch.Tensor:
    """Reference DSV4 SWA plus compressed-K attention over half-precision caches."""

    if q.ndim != 3:
        raise ValueError(f"expected q to be rank 3, got shape {tuple(q.shape)}")
    num_queries, num_heads, head_dim = q.shape
    if win_cache.ndim != 2 or win_cache.shape[1] != head_dim:
        raise ValueError(f"invalid SWA cache shape: {tuple(win_cache.shape)}")
    if win_indices.shape[0] != num_queries or win_lengths.shape[0] != num_queries:
        raise ValueError("SWA metadata does not match the query count")
    has_extra = extra_cache is not None
    if has_extra != (extra_indices is not None and extra_lengths is not None):
        raise ValueError("extra cache, indices, and lengths must be provided together")
    if has_extra and (
        extra_cache.ndim != 2
        or extra_cache.shape[1] != head_dim
        or extra_indices.shape[0] != num_queries
        or extra_lengths.shape[0] != num_queries
    ):
        raise ValueError("compressed-K metadata does not match the query layout")
    if attn_sink is not None and attn_sink.numel() != num_heads:
        raise ValueError(
            f"attention sink has {attn_sink.numel()} heads, expected {num_heads}"
        )

    def gather_rows(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        safe = indices.to(torch.int64).clamp(min=0, max=cache.shape[0] - 1)
        return cache.index_select(0, safe.reshape(-1)).reshape(
            indices.shape[0], indices.shape[1], head_dim
        )

    output = torch.empty_like(q)
    for query_start in range(0, num_queries, query_block_size):
        query_end = min(query_start + query_block_size, num_queries)
        query = q[query_start:query_end].float()

        block_win_indices = win_indices[query_start:query_end]
        block_win_lengths = win_lengths[query_start:query_end].to(torch.int64)
        win_slots = torch.arange(
            block_win_indices.shape[1], device=q.device, dtype=torch.int64
        ).unsqueeze(0)
        win_valid = (
            (win_slots < block_win_lengths.unsqueeze(1))
            & (block_win_indices >= 0)
            & (block_win_indices < win_cache.shape[0])
        )
        keys = gather_rows(win_cache, block_win_indices).float()
        valid = win_valid

        if has_extra:
            block_extra_indices = extra_indices[query_start:query_end]
            block_extra_lengths = extra_lengths[query_start:query_end].to(torch.int64)
            extra_slots = torch.arange(
                block_extra_indices.shape[1], device=q.device, dtype=torch.int64
            ).unsqueeze(0)
            extra_valid = (
                (extra_slots < block_extra_lengths.unsqueeze(1))
                & (block_extra_indices >= 0)
                & (block_extra_indices < extra_cache.shape[0])
            )
            extra_keys = gather_rows(extra_cache, block_extra_indices).float()
            keys = torch.cat((keys, extra_keys), dim=1)
            valid = torch.cat((valid, extra_valid), dim=1)

        scores = torch.einsum("qhd,qkd->qhk", query, keys) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        if attn_sink is not None:
            sink_scores = attn_sink.float().reshape(1, num_heads, 1)
            sink_scores = sink_scores.expand(query_end - query_start, -1, -1)
            probabilities = torch.softmax(
                torch.cat((scores, sink_scores), dim=-1), dim=-1
            )[..., :-1]
        else:
            probabilities = torch.softmax(scores, dim=-1)
        block_output = torch.einsum("qhk,qkd->qhd", probabilities, keys)
        output[query_start:query_end].copy_(block_output.to(q.dtype))
    return output


def _dsv4_topk_transform_graph_safe(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Select the exact top-k set without graph-hostile advanced indexing."""
    batch_size, max_seq_len = scores.shape
    topk = out_page_indices.shape[1]
    actual_k = min(topk, max_seq_len)
    cache = getattr(_dsv4_topk_transform_graph_safe, "_arange_cache", None)
    if cache is None:
        cache = {}
        _dsv4_topk_transform_graph_safe._arange_cache = cache

    seq_key = (max_seq_len, scores.device, torch.int64)
    positions = cache.get(seq_key)
    if positions is None:
        positions = torch.arange(max_seq_len, device=scores.device)
        cache[seq_key] = positions
    valid_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
    raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    ).indices
    raw_indices = torch.sort(raw_indices, dim=1).values
    if actual_k < topk:
        raw_indices = torch.nn.functional.pad(
            raw_indices, (0, topk - actual_k), value=0
        )

    selected_scores = torch.gather(scores, 1, raw_indices.clamp(min=0))
    valid_topk = selected_scores != float("-inf")
    topk_key = (topk, scores.device, raw_indices.dtype)
    topk_positions = cache.get(topk_key)
    if topk_positions is None:
        topk_positions = torch.arange(
            topk, device=scores.device, dtype=raw_indices.dtype
        )
        cache[topk_key] = topk_positions
    if actual_k < topk:
        valid_topk &= topk_positions.unsqueeze(0) < actual_k

    sequential = topk_positions.unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential < seq_lens.unsqueeze(1)
    needs_sequential = (seq_lens <= topk).unsqueeze(1)
    raw_indices = torch.where(needs_sequential, sequential, raw_indices)
    valid_topk = torch.where(needs_sequential, sequential_valid, valid_topk)

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask
    physical_pages = torch.gather(page_tables, 1, page_idx.clamp(min=0).long())
    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices.masked_fill_(~valid_topk, -1)
    out_page_indices.copy_(page_indices.to(torch.int32))


@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_512")
def dsv4_topk_transform_512_kunlun(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Transform DSV4 top-k indices to paged locations on Kunlun (XPU fused).

    Uses ``xspeedgate_ops::topk_transform`` — the same fused kernel wired in
    the legacy 0.5.8 patch — so the top-K search, length masking and
    page-table transform run as one XPU op with no multi-GiB torch
    intermediates.  Falls back to the pure-torch path only when the caller
    requests ``out_raw_indices`` (the XPU kernel does not emit raw indices).
    """

    if out_raw_indices is not None:
        _dsv4_topk_torch_fallback(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
        )
        _dsv4_probe(
            "topk",
            {"scores": scores, "raw_indices": out_raw_indices, "page_indices": out_page_indices},
            page_size=page_size,
        )
        return
    if os.environ.get("DSV4_KUNLUN_REFERENCE_ONLY") == "1":
        _dsv4_topk_transform_graph_safe(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
        )
        _dsv4_probe(
            "topk",
            {"scores": scores, "page_indices": out_page_indices},
            page_size=page_size,
        )
        return

    topk = out_page_indices.shape[1]
    # Pre-fill -1: XPU kernel skips writes for seq_len=0 rows, so the sentinel
    # guards those slots and matches the contract downstream expects.
    dst_page_table = scores.new_full(
        (scores.size(0), topk), -1, dtype=torch.int32,
    )
    # xspeedgate_ops::topk_transform(score, lengths, src_page_table,
    #     dst_page_table, topk, cu_seqlens_q=None, block_size=page_size,
    #     fast_path=True)
    torch.ops.xspeedgate_ops.topk_transform(
        scores,
        seq_lens,
        page_tables,
        dst_page_table,
        topk,
        None,
        page_size,
        True,
    )
    out_page_indices.copy_(dst_page_table)
    _dsv4_probe(
        "topk",
        {"scores": scores, "page_indices": out_page_indices},
        page_size=page_size,
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_512_v2")
def dsv4_topk_transform_512_v2_kunlun(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    """Run the Kunlun top-k transform while retaining the v2 API contract."""

    del metadata
    dsv4_topk_transform_512_kunlun(
        scores, seq_lens, page_tables, out_page_indices, page_size
    )


def dsv4_fused_scale_kunlun(
    weight: torch.Tensor, out_scale: float, q_scale: torch.Tensor
) -> torch.Tensor:
    """Fuse DSV4 indexer weight and query scales on Kunlun."""

    weight_2d = weight.squeeze(-1) if weight.ndim > 2 else weight
    q_scale_2d = q_scale.squeeze(-1) if q_scale.ndim > 2 else q_scale
    out = torch.ops.xspeedgate_ops.fused_scale(
        weight=weight_2d, out_scale=out_scale, q_scale=q_scale_2d
    )
    return out.unsqueeze(-1) if out.ndim == 2 else out


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.metadata_kernel",
    "init_compression_metadata",
)
def dsv4_init_compression_metadata_kunlun(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
):
    return torch.ops.xspeedgate_ops.init_compressed_attn_metadata_v2(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )
    



@plugin_hook(
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels."
    "ExpandPrefillCausally.execute",
    type=HookType.REPLACE,
)
def dsv4_expand_prefill_causally_kunlun(
    cls,
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[List[int]],
    extend_seq_lens_cpu: Optional[List[int]],
    num_tokens: int,
    padded_num_tokens: Optional[int],
):
    """Expand target-verify metadata with graph-safe Torch tensor operations."""

    del extend_start_loc
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        ExpandPrefillCausallyResult,
    )

    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.cpu().tolist()

    if extend_seq_lens_cpu is None:
        extend_seq_lens_cpu = extend_seq_lens.cpu().tolist()

    if req_pool_indices.dtype != torch.int32:
        req_pool_indices_i32 = req_pool_indices.to(torch.int32)
    else:
        req_pool_indices_i32 = req_pool_indices

    if not req_pool_indices_i32.is_contiguous():
        req_pool_indices_i32 = req_pool_indices_i32.contiguous()

    padded_arg = None
    if padded_num_tokens is not None and padded_num_tokens > num_tokens:
        padded_arg = padded_num_tokens

    seq_lens_casual, req_pool_indices_repeated = (
        torch.ops.xspeedgate_ops.expand_prefill_casually(
            num_tokens,
            seq_lens_cpu,
            extend_seq_lens_cpu,
            req_pool_indices_i32,
            padded_arg,
        )
    )

    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )

import logging as _dsv4_logging

_KERNEL_OPS_LOGGER = _dsv4_logging.getLogger(__name__)

_DSV4_STATE_LOC_TRACE = {"lines": 0}


def _dsv4_trace_state_loc(pos, raw_loc, swa_loc, ring_size):
    """Count C4 state reads whose SWA translation is the invalid slot 0."""

    import os

    if os.environ.get("DSV4_C4_STATE_LOC_TRACE") != "1":
        return
    if _DSV4_STATE_LOC_TRACE["lines"] >= 200:
        return
    positive = pos > 0
    if not bool(positive.any()):
        return
    if int(pos.max()) < 1000:
        return
    bad = positive & (swa_loc == 0)
    num_bad = int(bad.sum())
    _DSV4_STATE_LOC_TRACE["lines"] += 1
    bad_pos = pos[bad][:8].tolist() if num_bad else []
    swa_min = int(swa_loc.min())
    _KERNEL_OPS_LOGGER.warning(
        "[DSV4_C4_STATE_LOC] rows=%s pos_max=%s zero_mapped=%s sample_pos=%s "
        "swa_min=%s swa_max=%s swa_min_mod_ring=%s raw_min=%s",
        int(pos.numel()),
        int(pos.max()),
        num_bad,
        bad_pos,
        swa_min,
        int(swa_loc.max()),
        swa_min % ring_size,
        int(raw_loc.min()),
    )


def _dsv4_state_loc_torch(
    compress_ratio: int,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_state: torch.Tensor,
    swa_page_size: int,
    ring_size: int,
) -> torch.Tensor:
    """Translate request positions to the v2 compressor state-slot address."""

    rid = req_pool_indices.to(torch.int64)
    pos = positions.to(torch.int64)
    if compress_ratio == 128:
        return rid * ring_size + pos.remainder(ring_size)
    if compress_ratio != 4:
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    safe_pos = pos.clamp(min=0, max=req_to_token.shape[1] - 1)
    raw_loc = req_to_token[rid, safe_pos].to(torch.int64)
    swa_loc = full_to_state[raw_loc].to(torch.int64)
    _dsv4_trace_state_loc(pos, raw_loc, swa_loc, ring_size)
    return (
        torch.div(swa_loc, swa_page_size, rounding_mode="floor") * ring_size
        + swa_loc.remainder(ring_size)
    )


def _dsv4_pack_plan_i32(columns: List[torch.Tensor], width: int) -> torch.Tensor:
    """Pack int32 plan fields into the byte ABI consumed by compressor v2."""

    if not columns:
        raise ValueError("at least one plan column is required")
    return torch.stack([column.to(torch.int32) for column in columns], dim=1).contiguous().view(
        torch.uint8
    ).reshape(columns[0].shape[0], width)


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.compress.CompressorDecodePlan.generate",
    type=HookType.REPLACE,
)
def dsv4_compressor_decode_plan_kunlun(
    compress_ratio: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_state: torch.Tensor,
    seq_lens: torch.Tensor,
    swa_page_size: int,
    ring_size: int,
):
    """dsv4_compressor_decode_plan_kunlun"""
    from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    plan_d = torch.ops.xspeedgate_ops.dsv4_compressor_decode_plan(
        req_pool_indices,
        req_to_token,
        full_to_state,
        seq_lens,
        compress_ratio,
        swa_page_size,
        ring_size,
    )
    return CompressorDecodePlan(compress_ratio, plan_d)


#: c4/c128 state ring 为「尚未提交的投机位置」预留的余量。XSpeedGate 的
#: ``plan_compress_prefill_v2`` 把它写死成 ``min(ring_size - cr, 4)``（对应上游
#: ``c_plan.cuh`` 老版本的 ``kMaxMTPDraftTokens = 4``），而 DSPARK 的 verify window 是 6，
#: 于是最后几个 query 位置的 state 没落进 ring，下一步读到的是别人的历史 —— 同一请求多次
#: 发送结果不一致。上游 H20 侧的修复是把常量改成 8；算子签名不暴露 mtp_pad，也没有本地
#: xtdk 工具链重编，所以这里在框架侧按 pad=8 补出算子少给的那几行 plan_w。
#: 0（默认）= 按 ring 几何推导，见 ``_dsv4_compress_state_write_pad``；非 0 = 强制覆盖，
#: 仅用于复验。原来这里写死 8：能盖住当前 6 的 verify window，但和上游语义不一致，
#: draft token 数一调大就会重新踩坑。
_DSV4_MTP_PAD_OVERRIDE = int(os.environ.get("DSV4_MTP_PAD", "0"))

_DSV4_MTP_PAD_LOGGED: set[tuple[int, int]] = set()


def _dsv4_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """state ring 能服务的最大 draft token 数，即 plan 的 ``mtp_pad``。

    与社区a1fe4e3一致：``ring_size > window_size ? ring_size - window_size + 2 : 0``
    （``c_plan.cuh`` 的 mtp_pad、``deepseek_v4_memory_pool.get_compress_state_write_pad``）。
    v0.5.17 kernel 里还是 ``min(ring_size - cr, kMaxMTPDraftTokens=4)``，4 小于
    DSPARK 的 6-token verify window，尾部位置进不了 ring，下一步读到的是上一个占用者的
    state —— 同一请求多次发送结果不一致。投机配置下本公式给出 c4=10、c128=130。

    升级到带这个修复的 sglang / 重编算子之后，本函数和
    ``_dsv4_augment_prefill_plan_w`` 一起删掉。
    """

    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    pad = ring_size - window_size + 2 if ring_size > window_size else 0
    if _DSV4_MTP_PAD_OVERRIDE:
        pad = _DSV4_MTP_PAD_OVERRIDE
    key = (compress_ratio, ring_size)
    if key not in _DSV4_MTP_PAD_LOGGED:
        _DSV4_MTP_PAD_LOGGED.add(key)
        num_draft = _dsv4_num_draft_tokens()
        logger.info(
            "dsv4 compress state ring: ratio=%d ring_size=%d window=%d mtp_pad=%d "
            "(vendor kernel uses %d, num_draft_tokens=%s)",
            compress_ratio,
            ring_size,
            window_size,
            pad,
            min(ring_size - compress_ratio, 4),
            num_draft,
        )
        if num_draft is not None and pad < num_draft:
            raise AssertionError(
                f"compress state ring cannot serve the verify window: mtp_pad={pad} < "
                f"num_draft_tokens={num_draft} (ratio={compress_ratio}, "
                f"ring_size={ring_size}). Rejected draft state would be read back from "
                f"whatever the ring held before, which shows up as non-reproducible "
                f"output rather than an error. Raise ring_size or lower the window."
            )
    return pad


def _dsv4_num_draft_tokens() -> int | None:
    """投机的 draft token 数；拿不到就返回 None（不阻塞启动）。"""

    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args().speculative_num_draft_tokens
    except Exception:  # pragma: no cover - 仅用于日志与断言
        return None


def _dsv4_augment_prefill_plan_w(
    plan_w: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_state: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    compress_ratio: int,
    swa_page_size: int,
    ring_size: int,
    mtp_pad: int,
) -> torch.Tensor:
    """按更大的 ``mtp_pad`` 补齐 plan_w 缺的行。

    算子用的是 ``first_w_pos = min(last_c_pos - (overlap ? cr : 0), seq_len - vendor_pad)``，
    我们要的是同一公式换成 ``new_pad``。两者只差 ``[fw_new, fw_old)`` 这一小段位置，
    每个请求最多 ``new_pad - vendor_pad`` 行，所以直接补行、不用重算整张表。

    plan_w 是 ``[num_q_tokens, 8] uint8`` = ``(uint32 ragged_id, int32 write_loc)``，
    有效行在前、尾部填 ``(-1, -1)`` 哨兵；有效行数上界是 ``sum(extend_lens) <= num_q_tokens``，
    而每个位置最多产生一行，所以补的行一定放得进哨兵区。全程只用张量算子（无 D2H），
    可以被 CUDA graph 捕获。
    """

    num_rows = int(plan_w.shape[0])
    cr = compress_ratio
    is_overlap = cr == 4
    vendor_pad = 0 if seq_lens.device.type == "cpu" else min(ring_size - cr, 4)
    new_pad = min(ring_size - cr, mtp_pad)
    if num_rows == 0 or new_pad <= vendor_pad:
        return plan_w

    device = req_to_token.device
    seq = seq_lens.to(device=device, dtype=torch.int64)
    ext = extend_lens.to(device=device, dtype=torch.int64)
    rid = req_pool_indices.to(device=device, dtype=torch.int64)
    prefix = seq - ext
    shift = seq.new_full((), cr if is_overlap else 0)
    last_w = (seq // cr) * cr - shift
    lo = torch.clamp(torch.maximum(torch.minimum(last_w, seq - new_pad), prefix), min=0)
    hi = torch.maximum(torch.minimum(last_w, seq - vendor_pad), prefix)

    offsets = torch.arange(new_pad - vendor_pad, device=device, dtype=torch.int64)
    position = lo.unsqueeze(1) + offsets.unsqueeze(0)
    valid = position < hi.unsqueeze(1)
    ragged = (torch.cumsum(ext, 0) - ext).unsqueeze(1) + (position - prefix.unsqueeze(1))
    valid &= (ragged >= 0) & (ragged < num_rows)

    if cr == 128:
        write_loc = rid.unsqueeze(1) * ring_size + position % ring_size
    else:
        raw = req_to_token[rid.unsqueeze(1), position].to(torch.int64)
        swa = full_to_state[torch.clamp(raw, min=0)].to(torch.int64)
        # r2t / f2s 里的空洞是负数，这些位置还没有 state，补出来只会写坏别人的行
        valid &= (raw >= 0) & (swa >= 0)
        write_loc = (swa // swa_page_size) * ring_size + swa % ring_size

    invalid = torch.full_like(ragged, -1)
    extra = torch.stack(
        [torch.where(valid, ragged, invalid), torch.where(valid, write_loc, invalid)],
        dim=-1,
    ).reshape(-1, 2)

    rows = plan_w.contiguous().view(torch.int32).reshape(num_rows, 2)
    combined = torch.cat([rows, extra.to(torch.int32)], dim=0)
    keep = combined[:, 0] >= 0
    total = int(combined.shape[0])
    # 有效行按原序压到前面；无效行全丢到最后一行（一定在 num_rows 之外，且此时它必然空闲）
    dest = torch.cumsum(keep.to(torch.int64), 0) - 1
    dest = torch.where(keep, dest, torch.full_like(dest, total - 1))
    packed = torch.full((total, 2), -1, dtype=torch.int32, device=combined.device)
    packed.index_copy_(0, dest, combined)
    return packed[:num_rows].contiguous().view(torch.uint8).reshape(num_rows, 8)


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.compress.CompressorPrefillPlan.generate",
    type=HookType.REPLACE,
)
def dsv4_compressor_prefill_plan_kunlun(
    compress_ratio: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_state: torch.Tensor,
    swa_page_size: int,
    ring_size: int,
    num_q_tokens: int,
    use_cuda_graph: bool = False,
):
    """dsv4_compressor_prefill_plan_kunlun"""
    from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    if num_q_tokens < req_pool_indices.shape[0]:
        raise ValueError("num_q_tokens must be at least the batch size")
    pin_buffer = torch.empty(
            0,
            dtype=torch.uint8,
        ) # plan_compress_prefill_v2 only use its dtype
    device = req_pool_indices.device
    plan_c, plan_w = torch.ops.xspeedgate_ops.plan_compress_prefill_v2(
        req_pool_indices,
        req_to_token,
        full_to_state,
        seq_lens,
        extend_lens,
        pin_buffer,
        num_q_tokens,
        compress_ratio,
        swa_page_size,
        ring_size,
        use_cuda_graph,
    )
    plan_w = _dsv4_augment_prefill_plan_w(
        plan_w,
        req_pool_indices,
        req_to_token,
        full_to_state,
        seq_lens,
        extend_lens,
        compress_ratio,
        swa_page_size,
        ring_size,
        _dsv4_compress_state_write_pad(compress_ratio, ring_size),
    )
    return CompressorPrefillPlan(compress_ratio, plan_c, plan_w, None)


def _dsv4_compress_rows_torch(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan,
    head_dim: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Execute the v2 C4/C128 weighted-pooling contract with Torch ops."""

    plan_c = plan[1].contiguous().view(torch.int32)
    num_rows = plan_c.shape[0]
    if num_rows == 0:
        return kv_score_input.new_empty((0, head_dim))
    is_decode = plan.is_decode
    valid = (
        plan_c[:, 0].remainder(compress_ratio) == 0
        if is_decode
        else plan_c[:, 0] != -1
    )
    ragged_ids = (
        torch.arange(num_rows, device=plan_c.device, dtype=torch.int64)
        if is_decode
        else plan_c[:, 1].bitwise_and(0xFFFF).to(torch.int64)
    )
    ragged_ids_safe = ragged_ids.clamp(min=0, max=max(kv_score_input.shape[0] - 1, 0))
    buffer_len = (
        torch.full((num_rows,), compress_ratio * (2 if compress_ratio == 4 else 1),
                device=plan_c.device, dtype=torch.int64)
        if is_decode
        else plan_c[:, 1].bitwise_right_shift(16).bitwise_and(0xFFFF).to(torch.int64)
    )

    width = 8 if compress_ratio == 4 else 128
    offsets = torch.arange(width, device=plan_c.device, dtype=torch.int64)
    input_rows = ragged_ids_safe.unsqueeze(1) - (width - 1 - offsets).unsqueeze(0)
    input_rows_safe = input_rows.clamp(min=0, max=max(kv_score_input.shape[0] - 1, 0))
    input_values = kv_score_input.index_select(0, input_rows_safe.reshape(-1)).reshape(
        num_rows, width, -1
    )

    max_page = max(kv_score_buffer.shape[0] - 1, 0)
    if compress_ratio == 128:
        page_1 = plan_c[:, 3].to(torch.int64).clamp(min=0, max=max_page)
        buffer_values = kv_score_buffer.index_select(0, page_1)
        buffer_kv = buffer_values[:, :, :head_dim]
        buffer_score = buffer_values[:, :, head_dim : 2 * head_dim]
        input_kv = input_values[:, :, :head_dim]
        input_score = input_values[:, :, head_dim : 2 * head_dim]
    else:
        page_0 = plan_c[:, 2].to(torch.int64).clamp(min=0, max=max_page)
        page_1 = plan_c[:, 3].to(torch.int64).clamp(min=0, max=max_page)
        buffer_0 = kv_score_buffer.index_select(0, page_0)
        buffer_1 = kv_score_buffer.index_select(0, page_1)
        buffer_kv = torch.cat(
            [buffer_0[:, :, :head_dim], buffer_1[:, :, head_dim : 2 * head_dim]],
            dim=1,
        )
        buffer_score = torch.cat(
            [
                buffer_0[:, :, 2 * head_dim : 3 * head_dim],
                buffer_1[:, :, 3 * head_dim : 4 * head_dim],
            ],
            dim=1,
        )
        input_kv = torch.cat(
            [input_values[:, :4, :head_dim], input_values[:, 4:, head_dim : 2 * head_dim]],
            dim=1,
        )
        input_score = torch.cat(
            [
                input_values[:, :4, 2 * head_dim : 3 * head_dim],
                input_values[:, 4:, 3 * head_dim : 4 * head_dim],
            ],
            dim=1,
        )
        need_overlap = plan_c[:, 0].to(torch.int64) > compress_ratio
        overlap_rows = offsets.unsqueeze(0) < compress_ratio
        input_kv = torch.where(
            overlap_rows.unsqueeze(-1) & ~need_overlap[:, None, None],
            torch.zeros_like(input_kv),
            input_kv,
        )
        input_score = torch.where(
            overlap_rows.unsqueeze(-1) & ~need_overlap[:, None, None],
            torch.full_like(input_score, -torch.inf),
            input_score,
        )
        buffer_kv = torch.where(
            overlap_rows.unsqueeze(-1) & ~need_overlap[:, None, None],
            torch.zeros_like(buffer_kv),
            buffer_kv,
        )
        buffer_score = torch.where(
            overlap_rows.unsqueeze(-1) & ~need_overlap[:, None, None],
            torch.full_like(buffer_score, -torch.inf),
            buffer_score,
        )

    from_buffer = offsets.unsqueeze(0) < buffer_len.unsqueeze(1)
    values = torch.where(from_buffer.unsqueeze(-1), buffer_kv, input_kv).float()
    scores = torch.where(from_buffer.unsqueeze(-1), buffer_score, input_score).float()
    weights = torch.softmax(scores + ape[:width].float().unsqueeze(0), dim=1)
    output = (weights * values).sum(dim=1).to(kv_score_input.dtype)
    return torch.where(valid.unsqueeze(1), output, torch.zeros_like(output))


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.compress", "compress_forward"
)
def dsv4_compress_forward_v2_kunlun(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan,
    *,
    head_dim: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
    is_online: bool = False,
) -> torch.Tensor:
    """dsv4_compress_forward_v2_kunlun"""
    if is_online:
        raise NotImplementedError("online C128 is not enabled for the Torch reference")
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    plan_c = plan[1].contiguous()
    if plan.is_decode:
        plan_w = None
    else:
        plan_w = plan[2].contiguous()
    if out is None:
        num_q_tokens = plan[1].shape[0]
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    torch.ops.xspeedgate_ops.dsv4_compress_forward_v2(
        kv_score_buffer,
        kv_score_input,
        out,
        ape,
        plan_c,
        plan_w,
        compress_ratio,
    )
    return out


def _dsv4_hadamard_torch(value: torch.Tensor) -> torch.Tensor:
    """Apply the normalized Walsh-Hadamard transform along the last axis."""

    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    output = value.float()
    block = 1
    while block < width:
        pairs = output.reshape(*output.shape[:-1], -1, 2, block)
        low, high = pairs.unbind(dim=-2)
        output = torch.cat((low + high, low - high), dim=-1).reshape_as(output)
        block *= 2
    return output * (width ** -0.5)


def _dsv4_norm_rope_torch(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply the fused RMSNorm and complex RoPE math in Torch."""

    value = kv.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + norm_eps)
    value = value * norm_weight.float()
    freq_real = (
        torch.view_as_real(freq_cis).flatten(-2)
        if freq_cis.is_complex()
        else freq_cis
    ).float()
    rope_dim = freq_real.shape[-1]
    if rope_dim:
        positions_safe = positions.to(torch.int64).clamp(
            min=0, max=max(freq_real.shape[0] - 1, 0)
        )
        freq = freq_real.index_select(0, positions_safe).reshape(value.shape[0], -1, 2)
        rope = value[:, -rope_dim:].reshape(value.shape[0], -1, 2)
        real = rope[..., 0] * freq[..., 0] - rope[..., 1] * freq[..., 1]
        imag = rope[..., 0] * freq[..., 1] + rope[..., 1] * freq[..., 0]
        value = torch.cat(
            [value[:, :-rope_dim], torch.stack((real, imag), dim=-1).flatten(-2)],
            dim=-1,
        )
    return value


@register_triton_op(
    "sglang.kernels.ops.speculative.ragged_verify_kernels",
    "_qo_indptr_kernel",
)
def qo_indptr_kernel(
    verify_lens_ptr: torch.Tensor,
    qo_indptr_ptr: torch.Tensor,
    extend_start_loc_ptr: torch.Tensor,
    bs: int,
    BLOCK: int,
) -> None:
    """Build inclusive and exclusive verify-length prefix sums."""

    verify_lens = verify_lens_ptr[:bs].to(torch.int32)
    inclusive = torch.cumsum(verify_lens, dim=0, dtype=torch.int32)
    exclusive = inclusive - verify_lens

    qo_indptr_ptr[0] = 0
    qo_indptr_ptr[1 : bs + 1].copy_(inclusive.to(qo_indptr_ptr.dtype))
    extend_start_loc_ptr[:bs].copy_(exclusive.to(extend_start_loc_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.ragged_verify_kernels",
    "_padded_to_bucket_kernel",
)
def padded_to_bucket_kernel(
    verify_lens_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    bs: int,
    padded_bs: int,
    graph_num_tokens: int,
    BLOCK: int,
) -> None:
    """Pad verify lengths to a CUDA-graph bucket using Torch operations."""

    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold "
        "this batch's requests"
    )
    verify_lens = verify_lens_ptr[:bs].to(torch.int64)
    leftover = graph_num_tokens - verify_lens.sum()
    num_pad = padded_bs - bs

    if num_pad > 0:
        base = leftover // num_pad
        rem = leftover - base * num_pad
        pad_idx = torch.arange(
            num_pad, device=verify_lens.device, dtype=torch.int64
        )
        pad_lens = base + (pad_idx < rem).to(torch.int64)
        final = torch.cat((verify_lens, pad_lens))
    else:
        final = verify_lens.clone()
        if bs > 0:
            final[-1] += leftover

    out_ptr[:padded_bs].copy_(final.to(out_ptr.dtype))



@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.compress", "compress_norm_rope_store"
)
def dsv4_compress_norm_rope_store_v2_kunlun(
    kv: torch.Tensor,
    plan,
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
) -> None:
    """Apply v2 post-compression transforms and store into Kunlun cache pages."""

    if use_fp4:
        raise NotImplementedError("FP4 indexer is not enabled for the Torch reference")
    plan_raw = plan[1].contiguous()
    if plan_raw.shape[0] == 0:
        return
    freq_real = (
        torch.view_as_real(freq_cis).flatten(-2)
        if freq_cis.is_complex()
        else freq_cis
    ).float()

    torch.ops.xspeedgate_ops.dsv4_compress_norm_rope_store_v2(
        kv,
        plan_raw,
        norm_weight,
        norm_eps,
        freq_real,
        out_loc,
        kvcache,
        plan.is_decode,
        plan.compress_ratio,
        page_size,
        bf16_store,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.quant_k_cache",
    "quant_to_nope_fp8_rope_bf16_pack_triton",
)
def dsv4_quant_k_cache_kunlun(k_bf16: torch.Tensor):
    """Wrap a BF16 DSV4 key for the Kunlun fused cache-store operator."""

    from types import SimpleNamespace

    assert k_bf16.shape[-1] == 512
    if (
        os.environ.get("DSV4_MTP_PROBE") == "1"
        and os.environ.get("RANK", "0") == "0"
        and not getattr(dsv4_quant_k_cache_kunlun, "_probe_logged", False)
    ):
        logger.warning(
            "[DSV4_CALLSTACK] quant_k_cache replacement input dtype=%s shape=%s",
            k_bf16.dtype,
            tuple(k_bf16.shape),
        )
        dsv4_quant_k_cache_kunlun._probe_logged = True
    # Kunlun performs the cache conversion while storing. The upstream pack
    # validates the CUDA FP8 representation in __post_init__, so use the same
    # attribute contract without constructing that CUDA-only representation.
    return SimpleNamespace(
        # Match the patched compressor contract: its FP32 compressor
        # output is rounded to BF16 before conversion to the FP16 cache dtype.
        k_nope_fp8=k_bf16.to(torch.bfloat16).to(torch.float16).contiguous(),
        k_rope_bf16=None,
        scale_k_nope_ue8m0=None,
    )


@register_triton_op(
    "sglang.kernels.ops.attention.dsv4.index_buf_accessor",
    "_set_k_and_s_triton",
    metadata={"call_style": "direct"},
)
def dsv4_set_k_and_s_kunlun(
    buf: torch.Tensor,
    loc: torch.Tensor,
    nope_fp8_rope_bf16_pack,
    page_size: int,
):
    """Store an unquantized DSV4 key through the Kunlun cache wrapper."""

    max_valid_loc = buf.shape[0] * page_size - 1
    loc_safe = loc.clamp(min=0, max=max_valid_loc) if loc.numel() else loc
    if (
        os.environ.get("DSV4_MTP_PROBE") == "1"
        and os.environ.get("RANK", "0") == "0"
        and not getattr(dsv4_set_k_and_s_kunlun, "_probe_logged", False)
    ):
        logger.warning(
            "[DSV4_CALLSTACK] set_k_and_s_v4 replacement buf_dtype=%s "
            "buf_shape=%s loc_dtype=%s k_dtype=%s k_shape=%s page_size=%d",
            buf.dtype,
            tuple(buf.shape),
            loc_safe.dtype,
            nope_fp8_rope_bf16_pack.k_nope_fp8.dtype,
            tuple(nope_fp8_rope_bf16_pack.k_nope_fp8.shape),
            page_size,
        )
        dsv4_set_k_and_s_kunlun._probe_logged = True
    torch.ops.xspeedgate_ops.set_k_and_s_v4(
        buf, loc_safe, nope_fp8_rope_bf16_pack.k_nope_fp8, page_size
    )


def dsv4_set_k_and_s_with_mapping_kunlun(
    buf: torch.Tensor,
    raw_loc: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    nope_fp8_rope_bf16_pack,
    page_size: int,
) -> None:
    """Match the raw-location DSV4 cache writer contract.

    The XSpeedGate mapping variant owns full-to-SWA translation. Keeping the
    raw location and mapping as separate inputs is important for multi-step
    MTP, where the allocator's full-pool location is the source of truth.
    """
    # Golden passes allocator locations as contiguous int32. Keep the
    # scheduler's int64 buffer untouched and normalize only the operator input.
    raw_loc = raw_loc.to(dtype=torch.int32).contiguous()
    if full_to_swa_index_mapping.device != raw_loc.device:
        full_to_swa_index_mapping = full_to_swa_index_mapping.to(raw_loc.device)
    torch.ops.xspeedgate_ops.set_k_and_s_v4_with_mapping(
        buf,
        raw_loc,
        full_to_swa_index_mapping,
        nope_fp8_rope_bf16_pack.k_nope_fp8,
        page_size,
    )


@register_triton_op("sglang.kernels.ops.memory.common", "write_req_to_token_pool_triton")
def write_req_to_token_pool_triton(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_tensors: torch.Tensor,
    pre_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token_ptr_stride: int,
) -> None:
    """Write request token mappings into the request-to-token pool."""

    _dsv4_probe(
        "write_req_to_token_input",
        {
            "req_pool_indices": req_pool_indices,
            "prefix_tensors": prefix_tensors,
            "pre_lens": pre_lens,
            "seq_lens": seq_lens,
            "extend_lens": extend_lens,
            "out_cache_loc": out_cache_loc,
        },
        req_to_token_stride=tuple(req_to_token_ptr.stride()),
        req_to_token_is_contiguous=req_to_token_ptr.is_contiguous(),
        req_pool_indices_stride=tuple(req_pool_indices.stride()),
        req_pool_indices_is_contiguous=req_pool_indices.is_contiguous(),
        prefix_tensors_stride=tuple(prefix_tensors.stride()),
        prefix_tensors_is_contiguous=prefix_tensors.is_contiguous(),
        pre_lens_stride=tuple(pre_lens.stride()),
        pre_lens_is_contiguous=pre_lens.is_contiguous(),
        seq_lens_stride=tuple(seq_lens.stride()),
        seq_lens_is_contiguous=seq_lens.is_contiguous(),
        extend_lens_stride=tuple(extend_lens.stride()),
        extend_lens_is_contiguous=extend_lens.is_contiguous(),
        out_cache_loc_stride=tuple(out_cache_loc.stride()),
        out_cache_loc_is_contiguous=out_cache_loc.is_contiguous(),
    )
    torch.ops.xspeedgate_ops.write_req_to_token_pool(
        req_to_token_ptr,
        req_pool_indices.to(torch.int32),
        prefix_tensors,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc.to(torch.int64),
    )
    if os.environ.get("DSV4_ACCURACY_DUMP_DIR"):
        written_rows = req_to_token_ptr.index_select(
            0, req_pool_indices.to(torch.long)
        )[:, :35328]
        _dsv4_probe(
            "write_req_to_token_output",
            {"written_rows": written_rows},
        )


@register_triton_op("sglang.kernels.ops.memory.common", "get_last_loc_kernel")
def get_last_loc_kernel(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    result: torch.Tensor,
    num_tokens: int,
    req_to_token_stride: int,
    BLOCK_SIZE: int,
) -> None:
    """Return the last cache location for each request prefix."""

    if prefix_lens_tensor.dtype != torch.int64:
        prefix_lens_tensor = prefix_lens_tensor.to(torch.int64)
    loc = torch.ops.xspeedgate_ops.get_last_loc(
        req_to_token, req_pool_indices_tensor[:num_tokens], prefix_lens_tensor[:num_tokens]
    )
    result[:num_tokens].copy_(loc.to(result.dtype))


@register_triton_op(
    "sglang.kernels.ops.kvcache.cache_ops",
    "concat_and_cast_mha_k_kernel",
)
def concat_and_cast_mha_k_kernel(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    head_cnt: int,
    k_stride0: int,
    k_stride1: int,
    nope_stride0: int,
    nope_stride1: int,
    rope_stride0: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Concatenate and cast MHA key tensors into the destination buffer."""

    torch.ops.xspeedgate_ops.concat_and_cast_mha_k(k, k_nope, k_rope)


@register_triton_op(
    "sglang.kernels.ops.attention.pad",
    "seqlens_expand_kernel",
)
def seqlens_expand_kernel(
    extend_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    offsets: torch.Tensor,
    output: torch.Tensor,
    N: int,
    **kwargs,
) -> None:
    """Expand per-request sequence lengths into token positions."""

    extend_seq_lens = extend_seq_lens[:N]
    seq_lens = seq_lens[:N]
    offsets = offsets[:N]
    token_offsets = torch.arange(output.numel(), device=output.device, dtype=output.dtype)
    request_offsets = torch.repeat_interleave(offsets.to(output.dtype), extend_seq_lens)
    starts = (seq_lens - extend_seq_lens + 1).to(output.dtype)
    request_starts = torch.repeat_interleave(starts, extend_seq_lens)
    output.copy_(request_starts + token_offsets - request_offsets)


@register_triton_op(
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels",
    "_page_table_positions_kernel",
)
def page_table_positions_kernel(
    req_to_token_ptr: torch.Tensor,
    req_pool_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    seq_lens_out_ptr: torch.Tensor,
    positions_out_ptr: torch.Tensor,
    page_table_ptr: torch.Tensor,
    topk_out_ptr: torch.Tensor,
    rt_stride: int,
    num_pages: int,
    page_size: int,
    swa_window: int,
    BLOCK_P: int,
) -> None:
    """Build DeepSeek-V4 page-table metadata without Triton."""

    del BLOCK_P
    num_q = seq_lens_ptr.shape[0]
    op = "page_table_positions"
    _dspark_require(num_pages > 0 and page_size > 0, op, "num_pages/page_size must be > 0")
    _dspark_require(
        req_to_token_ptr.dim() == 2 and req_to_token_ptr.dtype == torch.int32,
        op,
        f"req_to_token must be 2D int32, got dim={req_to_token_ptr.dim()} "
        f"dtype={req_to_token_ptr.dtype}",
    )
    # 算子内部用 stride(0)/ceil(max_seq_len/page_size) 推导，必须与入参一致
    _dspark_require(
        req_to_token_ptr.stride(0) == rt_stride and req_to_token_ptr.stride(1) == 1,
        op,
        f"req_to_token strides {req_to_token_ptr.stride()} do not match "
        f"rt_stride={rt_stride} with row-contiguous rows",
    )
    _dspark_require(
        (num_pages - 1) * page_size < req_to_token_ptr.shape[1],
        op,
        f"last sampled column {(num_pages - 1) * page_size} is out of range for "
        f"req_to_token width {req_to_token_ptr.shape[1]}",
    )
    _dspark_require(
        req_pool_ptr.dim() == 1
        and seq_lens_ptr.dim() == 1
        and req_pool_ptr.shape[0] == num_q,
        op,
        "req_pool/seq_lens must be 1D with the same length",
    )
    _dspark_require(
        tuple(page_table_ptr.shape) == (num_q, num_pages),
        op,
        f"page_table shape {tuple(page_table_ptr.shape)} != {(num_q, num_pages)}",
    )

    seq_lens, positions, page_table, topk = (
        torch.ops.xspeedgate_ops.page_table_positions(
            req_to_token_ptr,
            req_pool_ptr.contiguous(),
            seq_lens_ptr.to(torch.int32).contiguous(),
            (num_pages - 1) * page_size + 1,
            page_size,
            swa_window,
        )
    )
    seq_lens_out_ptr.copy_(seq_lens)
    positions_out_ptr.copy_(positions)
    page_table_ptr.copy_(page_table)
    topk_out_ptr.copy_(topk)


def _dspark_kernel_out(
    tensor: torch.Tensor, dtype: torch.dtype, shape=None
) -> torch.Tensor:
    """Return the buffer an XSpeedGate DSpark kernel should write ``tensor`` into.

    The kernels pin the dtype/layout of every output they write, while the
    upstream Triton call sites allocate a few of them differently (e.g.
    ``offsets`` as int64, or a buffer longer than the rows the kernel touches).
    In that case a scratch buffer is returned and the caller must copy it back --
    check with ``buf is not tensor``. Otherwise ``tensor`` itself is returned and
    the kernel writes into it directly.

    The scratch is never left uninitialized. Several XPU kernels skip writes for
    rows they consider empty (see the ``new_full(-1)`` prefill in
    ``dsv4_topk_transform_kunlun``), and a ``torch.empty`` scratch would then copy
    the previous tenant of that allocation back into the caller's buffer --
    reproducible for a fixed request order, but different for every request, which
    is exactly the non-determinism signature we are chasing. Seeding the scratch
    from ``tensor`` keeps untouched rows at the value the call site allocated.
    """

    shape = tuple(tensor.shape) if shape is None else tuple(shape)
    if (
        tensor.dtype == dtype
        and tensor.is_contiguous()
        and tuple(tensor.shape) == shape
    ):
        return tensor
    if tuple(tensor.shape) == shape:
        return tensor.to(dtype=dtype).contiguous()
    numel = 1
    for dim in shape:
        numel *= dim
    flat = tensor.reshape(-1)
    if flat.numel() >= numel:
        return flat[:numel].to(dtype=dtype).contiguous().view(shape)
    return torch.zeros(shape, dtype=dtype, device=tensor.device)


def _dspark_require(condition: bool, op: str, message: str) -> None:
    """Fail loudly when a DSpark call site breaks an XSpeedGate input contract.

    Used for the dtype/shape/stride expectations that every upstream Triton call
    site already satisfies. Degrading to the torch path there would silently
    give up the vendor kernel, so a broken contract must surface instead.
    Capacity ceilings the upstream genuinely can exceed (e.g. the bs<=64 limit of
    ``mixed_accept_select_kernel``) keep their torch fallback.
    """

    if not condition:
        raise AssertionError(f"xspeedgate_ops.{op}: {message}")


_DSPARK_FALLBACK_WARNED: set = set()


def _dspark_warn_fallback(op: str, reason: str) -> None:
    """Log the first time an op degrades to torch so it is never silent."""

    if op in _DSPARK_FALLBACK_WARNED:
        return
    _DSPARK_FALLBACK_WARNED.add(op)
    _KERNEL_OPS_LOGGER.warning(
        "sglang-kunlun: %s runs the torch path instead of the XSpeedGate "
        "kernel (%s)",
        op,
        reason,
    )


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_attn_metadata",
    "_window_gather_kernel",
)
def window_gather_kernel(
    seq_lens_casual_ptr: torch.Tensor,
    req_pool_rep_ptr: torch.Tensor,
    context_lens_ptr: torch.Tensor,
    req_pool_out_ptr: torch.Tensor,
    offsets_ptr: torch.Tensor,
    invalid_ptr: torch.Tensor,
    block_size: int,
    swa_window: int,
    W_BLOCK: int,
) -> None:
    """Build DSpark sliding-window metadata via xspeedgate_ops.

    The XPU kernel writes int32 offsets while the Triton call site allocates
    them as int64, so that output goes through a scratch buffer.
    """

    bs = context_lens_ptr.numel()
    if bs == 0:
        return

    context_lens = _dspark_kernel_out(context_lens_ptr, torch.int32)
    req_pool_out = _dspark_kernel_out(req_pool_out_ptr, torch.int32)
    offsets = _dspark_kernel_out(offsets_ptr, torch.int32)
    invalid = _dspark_kernel_out(invalid_ptr, torch.bool)

    torch.ops.xspeedgate_ops.window_gather_kernel(
        seq_lens_casual_ptr.to(torch.int32).contiguous(),
        req_pool_rep_ptr.to(torch.int32).contiguous(),
        context_lens,
        req_pool_out,
        offsets,
        invalid,
        block_size,
        swa_window,
        W_BLOCK,
    )

    if context_lens is not context_lens_ptr:
        context_lens_ptr.copy_(context_lens)
    if req_pool_out is not req_pool_out_ptr:
        req_pool_out_ptr.copy_(req_pool_out)
    if offsets is not offsets_ptr:
        offsets_ptr.copy_(offsets)
    if invalid is not invalid_ptr:
        invalid_ptr.copy_(invalid)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_attn_metadata",
    "_swa_page_indices_kernel",
)
def swa_page_indices_kernel(
    req_to_token_ptr: torch.Tensor,
    full_to_swa_ptr: torch.Tensor,
    req_pool_ptr: torch.Tensor,
    offsets_ptr: torch.Tensor,
    out_loc_ptr: torch.Tensor,
    context_lens_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    topk_ptr: torch.Tensor,
    rt_stride: int,
    swa_window: int,
    block_size: int,
    target_width: int,
    TW_BLOCK: int,
) -> None:
    """Build DSpark SWA page indices via xspeedgate_ops.

    The XPU op allocates its own ``[n_q, target_width]`` / ``[n_q]`` outputs and
    dispatches on a single int32 dtype, so the int64 ``offsets`` produced upstream
    is cast and the results are copied into the caller's buffers.

    ``out_loc`` is zero-padded up to ``n_q`` when needed: under CUDA graph bs
    padding the sliced ``out_loc`` can be shorter than the padded batch demands,
    which would trip the op's ``out_loc.numel() >= bs * block_size`` check. Those
    rows are discarded downstream, so pointing them at slot 0 is as harmless as
    the index clamping the previous dense torch path did.
    """

    del TW_BLOCK, rt_stride
    n_q = out_ptr.shape[0]
    if n_q == 0:
        return

    out_loc = out_loc_ptr.to(torch.int32).contiguous()
    if out_loc.numel() < n_q:
        padded = out_loc.new_zeros(n_q)
        padded[: out_loc.numel()] = out_loc
        out_loc = padded

    swa_page_indices, swa_topk_lengths = (
        torch.ops.xspeedgate_ops.swa_page_indices_kernel(
            req_to_token_ptr.to(torch.int32),
            full_to_swa_ptr.to(torch.int32).contiguous(),
            req_pool_ptr.to(torch.int32).contiguous(),
            offsets_ptr.to(torch.int32).contiguous(),
            out_loc,
            context_lens_ptr.to(torch.int32).contiguous(),
            swa_window,
            block_size,
            target_width,
        )
    )
    out_ptr.copy_(swa_page_indices)
    topk_ptr.copy_(swa_topk_lengths)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_attn_metadata",
    "_block_seq_lens_casual_kernel",
)
def block_seq_lens_casual_kernel(
    seq_lens_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    block_size: int,
    n_out: int,
    BLOCK: int,
) -> None:
    """Build causal sequence lengths for each DSpark block."""

    del BLOCK
    op = "block_seq_lens_casual"
    _dspark_require(block_size > 0, op, f"block_size must be > 0, got {block_size}")
    n_rows = n_out // block_size
    _dspark_require(
        n_out == n_rows * block_size,
        op,
        f"n_out={n_out} is not a multiple of block_size={block_size}",
    )
    _dspark_require(
        seq_lens_ptr.dim() == 1 and n_rows <= seq_lens_ptr.numel(),
        op,
        f"seq_lens must be 1D with at least {n_rows} rows, got "
        f"shape {tuple(seq_lens_ptr.shape)}",
    )

    values = torch.ops.xspeedgate_ops.block_seq_lens_casual(
        seq_lens_ptr[:n_rows], block_size, str(out_ptr.device)
    )
    out_ptr[:n_out].copy_(values.to(out_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_softmax_temp_kernel",
)
def softmax_temp_kernel(
    logits_ptr: torch.Tensor,
    temp_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    vocab: int,
    rows_per_request: int,
    logits_row_stride: int,
    BLOCK_V: int,
) -> None:
    """Apply per-request temperatures and compute row-wise softmax."""

    num_rows = out_ptr.shape[0]
    if num_rows == 0:
        return
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"

    # The op reads rows with logits.stride(0), so only the last dim has to be
    # packed; a row-padded logits view is fine.
    logits = logits_ptr if logits_ptr.stride(-1) == 1 else logits_ptr.contiguous()
    out = _dspark_kernel_out(out_ptr, out_ptr.dtype)

    torch.ops.xspeedgate_ops.softmax_temp(
        logits,
        temp_ptr.reshape(-1)[:bs].to(torch.float32).contiguous(),
        out,
        vocab,
        rows_per_request,
        logits_row_stride,
        BLOCK_V,
    )
    if out is not out_ptr:
        out_ptr.copy_(out)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_gather_two_level_bonus_kernel",
)
def gather_two_level_bonus_kernel(
    accept_index_ptr: torch.Tensor,
    predicts_ptr: torch.Tensor,
    correct_len_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    cols: int,
    n: int,
    BLOCK: int,
) -> None:
    """Gather accepted bonus tokens from the two-level accept index."""
    if n == 0:
        return
    accept_index = accept_index_ptr.reshape(-1)
    if accept_index.dtype not in (torch.int32, torch.int64):
        accept_index = accept_index.to(torch.int64)
    predicts = predicts_ptr.reshape(-1)
    if predicts.dtype not in (torch.int32, torch.int64):
        predicts = predicts.to(torch.int64)
    out = _dspark_kernel_out(out_ptr, torch.int64, (n,))
    torch.ops.xspeedgate_ops.gather_two_level_bonus_kernel(
        accept_index.contiguous(),
        predicts.contiguous(),
        correct_len_ptr.reshape(-1)[:n].to(torch.int64).contiguous(),
        out,
        cols,
        n,
        max(int(BLOCK), 1),
    )
    if out is not out_ptr:
        out_ptr[:n].copy_(out)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_gather_row_bonus_kernel",
)
def gather_row_bonus_kernel(
    table_ptr: torch.Tensor,
    idx_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    cols: int,
    n: int,
    BLOCK: int,
) -> None:
    """Gather one bonus token from each row using per-row column indices."""

    if n == 0:
        return
    table = (
        table_ptr
        if table_ptr.dtype in (torch.int32, torch.int64)
        else table_ptr.to(torch.int64)
    )
    out = _dspark_kernel_out(out_ptr, torch.int64, shape=(n,))

    torch.ops.xspeedgate_ops.gather_row_bonus_kernel(
        table,
        idx_ptr.reshape(-1)[:n].to(torch.int64).contiguous(),
        out,
        cols,
        n,
        max(int(BLOCK), 1),
    )
    if out is not out_ptr:
        out_ptr[:n].copy_(out)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_mixed_accept_select_kernel",
)
def mixed_accept_select_kernel(
    greedy_mask_ptr: torch.Tensor,
    greedy_len_ptr: torch.Tensor,
    greedy_bonus_ptr: torch.Tensor,
    greedy_trim_ptr: torch.Tensor,
    sampling_len_ptr: torch.Tensor,
    sampling_bonus_ptr: torch.Tensor,
    sampling_trim_ptr: torch.Tensor,
    correct_len_ptr: torch.Tensor,
    bonus_ptr: torch.Tensor,
    cap_trim_ptr: torch.Tensor,
    bs: int,
    BLOCK: int,
) -> None:
    """Select greedy or sampling accept results row by row."""
    op = "mixed_accept_select_kernel"
    bonus_dtype = greedy_bonus_ptr.dtype
    _dspark_require(
        greedy_len_ptr.dtype == greedy_trim_ptr.dtype
        and greedy_len_ptr.dtype in (torch.int32, torch.int64),
        op,
        f"greedy len/trim must share one int32/int64 dtype, got "
        f"{greedy_len_ptr.dtype}/{greedy_trim_ptr.dtype}",
    )
    _dspark_require(
        sampling_len_ptr.dtype == torch.int32
        and sampling_trim_ptr.dtype == torch.int32,
        op,
        f"sampling len/trim must be int32, got "
        f"{sampling_len_ptr.dtype}/{sampling_trim_ptr.dtype}",
    )
    _dspark_require(
        sampling_bonus_ptr.dtype == bonus_dtype,
        op,
        f"bonus dtypes differ: greedy={bonus_dtype} sampling="
        f"{sampling_bonus_ptr.dtype}",
    )

    # vendor 算子把 bs 行整表缓存到 SM，硬性 bs <= 64；running bs 可以超过它。
    if bs > 64:
        _dspark_warn_fallback(op, f"bs={bs} exceeds the vendor limit of 64")
        is_greedy = greedy_mask_ptr[:bs].to(torch.bool)
        correct_len_ptr[:bs].copy_(
            torch.where(
                is_greedy,
                greedy_len_ptr[:bs].to(correct_len_ptr.dtype),
                sampling_len_ptr[:bs],
            )
        )
        bonus_ptr[:bs].copy_(
            torch.where(is_greedy, greedy_bonus_ptr[:bs], sampling_bonus_ptr[:bs])
        )
        cap_trim_ptr[:bs].copy_(
            torch.where(
                is_greedy,
                greedy_trim_ptr[:bs].to(cap_trim_ptr.dtype),
                sampling_trim_ptr[:bs],
            )
        )
        return

    correct_len = _dspark_kernel_out(correct_len_ptr, torch.int32, (bs,))
    bonus = _dspark_kernel_out(bonus_ptr, bonus_dtype, (bs,))
    cap_trim = _dspark_kernel_out(cap_trim_ptr, torch.int32, (bs,))
    torch.ops.xspeedgate_ops.mixed_accept_select_kernel(
        greedy_mask_ptr[:bs].to(torch.bool).contiguous(),
        greedy_len_ptr[:bs].contiguous(),
        greedy_bonus_ptr[:bs].contiguous(),
        greedy_trim_ptr[:bs].contiguous(),
        sampling_len_ptr[:bs].contiguous(),
        sampling_bonus_ptr[:bs].contiguous(),
        sampling_trim_ptr[:bs].contiguous(),
        correct_len,
        bonus,
        cap_trim,
        bs,
        max(int(BLOCK), 1),
    )
    if correct_len is not correct_len_ptr:
        correct_len_ptr[:bs].copy_(correct_len)
    if bonus is not bonus_ptr:
        bonus_ptr[:bs].copy_(bonus)
    if cap_trim is not cap_trim_ptr:
        cap_trim_ptr[:bs].copy_(cap_trim)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "_online_partial_kernel",
)
def online_partial_kernel(
    logits_ptr: torch.Tensor,
    temperatures_ptr: torch.Tensor,
    greedy_mask_ptr: torch.Tensor,
    exp_noise_ptr: torch.Tensor,
    tile_max_ptr: torch.Tensor,
    partial_key_ptr: torch.Tensor,
    partial_idx_ptr: torch.Tensor,
    V: int,
    stride_row: int,
    n_tiles: int,
    BLOCK_V: int,
) -> None:
    """Compute per-tile sampling maxima and token candidates."""

    del stride_row
    bs = logits_ptr.shape[0]
    tile_ids = torch.arange(n_tiles, device=logits_ptr.device, dtype=torch.int64)
    columns = torch.arange(BLOCK_V, device=logits_ptr.device, dtype=torch.int64)
    indices = tile_ids[:, None] * BLOCK_V + columns[None, :]
    mask = indices < V
    safe_indices = indices.clamp(max=max(V - 1, 0))
    logits = logits_ptr[:, safe_indices].to(torch.float32)
    logits = logits.masked_fill(~mask[None, :, :], float("-inf"))
    temperatures = temperatures_ptr[:bs, None].to(torch.float32)
    scaled = logits / temperatures[:, None, :]
    tile_max = scaled.amax(dim=-1)
    greedy = greedy_mask_ptr[:bs].to(torch.bool)[:, None, None]
    noise = exp_noise_ptr[:bs, safe_indices].to(torch.float32)
    denominator = torch.where(greedy, torch.ones_like(noise), noise)
    keys = torch.exp(scaled - tile_max[:, :, None]) / denominator
    keys = keys.masked_fill(~mask[None, :, :], -1.0)
    best = keys.amax(dim=-1)
    candidates = torch.where(
        keys == best[:, :, None],
        indices[None, :, :],
        torch.iinfo(torch.int32).max,
    )
    tile_max_ptr.copy_(tile_max.to(tile_max_ptr.dtype))
    partial_key_ptr.copy_(best.to(partial_key_ptr.dtype))
    partial_idx_ptr.copy_(candidates.amin(dim=-1).to(partial_idx_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "_online_combine_kernel",
)
def online_combine_kernel(
    tile_max_ptr: torch.Tensor,
    partial_key_ptr: torch.Tensor,
    partial_idx_ptr: torch.Tensor,
    next_tokens_ptr: torch.Tensor,
    n_tiles: int,
    BLOCK_TILES: int,
) -> None:
    """Combine online sampling tile results into next-token indices."""

    del BLOCK_TILES
    tile_max = tile_max_ptr[:, :n_tiles].to(torch.float32)
    keys = partial_key_ptr[:, :n_tiles].to(torch.float32)
    indices = partial_idx_ptr[:, :n_tiles]
    global_max = tile_max.amax(dim=-1, keepdim=True)
    rescaled = keys * torch.exp(tile_max - global_max)
    best = rescaled.amax(dim=-1, keepdim=True)
    sentinel = torch.iinfo(torch.int32).max
    candidates = torch.where(
        rescaled == best,
        indices,
        torch.full_like(indices, sentinel),
    )
    next_tokens = candidates.amin(dim=-1).masked_fill(
        candidates.amin(dim=-1) == sentinel, 0
    )
    next_tokens_ptr.copy_(next_tokens.to(next_tokens_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "_build_step_local_kernel",
)
def build_step_local_kernel(
    bias_ptr: torch.Tensor,
    base_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    org_width: int,
    per_partition: int,
    BLOCK: int,
) -> None:
    """Add padded DSpark step-local bias to the base logits."""

    rows = bias_ptr.shape[0]
    op = "build_step_local_kernel"
    _dspark_require(
        bias_ptr.dim() == 2 and base_ptr.dim() == 2 and out_ptr.dim() == 2,
        op,
        "bias/base/out must all be 2D",
    )
    _dspark_require(
        bias_ptr.shape[1] == org_width
        and base_ptr.shape == (rows, per_partition)
        and out_ptr.shape == (rows, per_partition),
        op,
        f"shapes {tuple(bias_ptr.shape)}/{tuple(base_ptr.shape)}/"
        f"{tuple(out_ptr.shape)} do not match org_width={org_width} "
        f"per_partition={per_partition}",
    )
    _dspark_require(
        out_ptr.dtype == torch.float32 and out_ptr.is_contiguous(),
        op,
        f"out must be contiguous float32, got {out_ptr.dtype} "
        f"contiguous={out_ptr.is_contiguous()}",
    )

    if org_width > per_partition:
        # 上游 triton 用 mask=offs<org_width 截断，允许 bias 宽于本地分区；
        # vendor 算子 TORCH_CHECK(org_width <= per_partition)，只能走 torch。
        _dspark_warn_fallback(op, f"org_width={org_width} > per_partition={per_partition}")
        bias = bias_ptr[:, :per_partition].to(torch.float32)
        base = base_ptr.to(torch.float32)
        out_ptr.copy_((base + bias).to(out_ptr.dtype))
        return

    bias = bias_ptr.to(torch.float32).contiguous()
    base = base_ptr.to(torch.float32).contiguous()
    torch.ops.xspeedgate_ops.build_step_local_kernel(
        bias, base, out_ptr, org_width, per_partition, BLOCK
    )


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_finalize_accept_lens_kernel",
)
def finalize_accept_lens_kernel(
    correct_len_ptr: torch.Tensor,
    cap_trim_ptr: torch.Tensor,
    prefix_lens_ptr: torch.Tensor,
    commit_lens_ptr: torch.Tensor,
    new_seq_lens_ptr: torch.Tensor,
    cap_trim_out_ptr: torch.Tensor,
    bs: int,
    BLOCK: int,
) -> None:
    """Finalize DSpark accepted lengths via xspeedgate_ops."""

    if bs == 0:
        return
    commit_lens = _dspark_kernel_out(commit_lens_ptr, torch.int32, shape=(bs,))
    new_seq_lens = _dspark_kernel_out(new_seq_lens_ptr, torch.int64, shape=(bs,))
    cap_trim_out = _dspark_kernel_out(cap_trim_out_ptr, torch.int32, shape=(bs,))

    torch.ops.xspeedgate_ops.finalize_accept_lens_kernel(
        correct_len_ptr.reshape(-1)[:bs].to(torch.int32).contiguous(),
        cap_trim_ptr.reshape(-1)[:bs].to(torch.int32).contiguous(),
        prefix_lens_ptr.reshape(-1)[:bs].to(torch.int64).contiguous(),
        commit_lens,
        new_seq_lens,
        cap_trim_out,
        bs,
        max(int(BLOCK), 1),
    )
    if commit_lens is not commit_lens_ptr:
        commit_lens_ptr[:bs].copy_(commit_lens)
    if new_seq_lens is not new_seq_lens_ptr:
        new_seq_lens_ptr[:bs].copy_(new_seq_lens)
    if cap_trim_out is not cap_trim_out_ptr:
        cap_trim_out_ptr[:bs].copy_(cap_trim_out)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_accept",
    "_cap_correct_len_kernel",
)
def cap_correct_len_kernel(
    correct_len_ptr: torch.Tensor,
    verify_lens_ptr: torch.Tensor,
    capped_ptr: torch.Tensor,
    trim_ptr: torch.Tensor,
    n: int,
    BLOCK: int,
) -> None:
    """Cap accepted lengths to the available verification window."""
    del BLOCK
    op = "cap_correct_len"
    correct_len = correct_len_ptr[:n]
    _dspark_require(
        correct_len.dtype == torch.int32
        and capped_ptr.dtype == torch.int32
        and trim_ptr.dtype == torch.int32,
        op,
        f"correct_len/capped/trim must be int32, got {correct_len.dtype}/"
        f"{capped_ptr.dtype}/{trim_ptr.dtype}",
    )
    if n == 0:
        return

    capped, trim = torch.ops.xspeedgate_ops.cap_correct_len(
        correct_len.contiguous(),
        verify_lens_ptr[:n]
        .to(device=correct_len.device, dtype=torch.int32)
        .contiguous(),
    )
    capped_ptr[:n].copy_(capped)
    trim_ptr[:n].copy_(trim)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_ragged_finalize_kernel",
)
def ragged_finalize_kernel(
    req_ptr: torch.Tensor,
    within_ptr: torch.Tensor,
    prefix_ptr: torch.Tensor,
    cache_ptr: torch.Tensor,
    pos_out_ptr: torch.Tensor,
    cache_out_ptr: torch.Tensor,
    bs: int,
    n: int,
    real_len: int,
    BLOCK: int,
) -> None:
    """Finalize padded ragged verification positions and cache locations."""
    op = "ragged_finalize"
    _dspark_require(bs > 0, op, f"bs must be > 0, got {bs}")
    _dspark_require(
        0 <= real_len <= n, op, f"real_len={real_len} must be within [0, n={n}]"
    )
    _dspark_require(
        bs <= 8192, op, f"bs={bs} exceeds the vendor prefix-cache limit of 8192"
    )

    req = req_ptr[:n].to(torch.int64).contiguous()
    within = within_ptr[:n].to(torch.int64).contiguous()
    prefix = prefix_ptr[:bs].to(torch.int64).contiguous()
    cache = cache_ptr[:real_len].to(torch.int64).contiguous()
    pos_out = _dspark_kernel_out(pos_out_ptr, torch.int64, (n,))
    cache_out = _dspark_kernel_out(cache_out_ptr, torch.int64, (n,))
    torch.ops.xspeedgate_ops.ragged_finalize(
        req,
        within,
        prefix,
        cache,
        pos_out,
        cache_out,
        bs,
        n,
        real_len,
        max(int(BLOCK), 1),
    )
    if pos_out is not pos_out_ptr:
        pos_out_ptr[:n].copy_(pos_out)
    if cache_out is not cache_out_ptr:
        cache_out_ptr[:n].copy_(cache_out)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_compact_row_index_kernel",
)
def compact_row_index_kernel(
    incl_ptr: torch.Tensor,
    req_out_ptr: torch.Tensor,
    within_out_ptr: torch.Tensor,
    valid_out_ptr: torch.Tensor,
    bs: int,
    n: int,
    BLOCK: int,
    NBITS: int,
) -> None:
    """Map padded rows to compact request/within-row indices."""
    del BLOCK, NBITS
    # 上游 triton 也是 tl.load(incl_ptr + bs - 1)，bs==0 本来就不是合法输入
    _dspark_require(bs > 0, "compact_row_index", f"bs must be > 0, got {bs}")
    incl = incl_ptr[:bs].to(torch.int64)
    # vendor 算子收 verify_lens 并在内部做 cumsum，这里从 incl 还原
    verify_lens = torch.diff(incl, prepend=incl.new_zeros(1))
    req, within, valid = torch.ops.xspeedgate_ops.compact_row_index(
        verify_lens, n, str(req_out_ptr.device)
    )
    req_out_ptr[:n].copy_(req.to(req_out_ptr.dtype))
    within_out_ptr[:n].copy_(within.to(within_out_ptr.dtype))
    valid_out_ptr[:n].copy_(valid.to(valid_out_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_compact_verify_ids_gather_kernel",
)
def compact_verify_ids_gather_kernel(
    req_ptr: torch.Tensor,
    within_ptr: torch.Tensor,
    draft_block_ids_ptr: torch.Tensor,
    draft_tokens_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    bs: int,
    gamma: int,
    n: int,
    BLOCK: int,
) -> None:
    """Gather compact verification token ids from ragged row indices."""
    del BLOCK
    req = req_ptr[:n].to(torch.int64)
    within = within_ptr[:n].to(torch.int64)
    valid = req < bs
    safe_req = req.clamp(min=0, max=max(bs - 1, 0))
    draft_block_ids = draft_block_ids_ptr.reshape(-1, gamma)
    draft_tokens = draft_tokens_ptr.reshape(-1, gamma)
    anchor = draft_block_ids[safe_req, 0]
    draft_col = (within - 1).clamp(min=0, max=max(gamma - 1, 0))
    draft = draft_tokens[safe_req, draft_col]
    values = torch.where(within == 0, anchor, draft)
    values = torch.where(valid, values, torch.zeros_like(values))
    out_ptr[:n].copy_(values.to(out_ptr.dtype))


@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "compact_verify_ids_triton",
)
def compact_verify_ids_triton(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout,
    device,
) -> torch.Tensor:
    """Build the compact verify-token ids for a ragged verify layout."""

    # vendor 算子对齐上层 python 函数（行索引 + gather 一次算完），
    # 因此这里替换 host 函数而不是 _compact_verify_ids_gather_kernel。
    op = "compact_verify_ids"
    verify_lens = layout.verify_lens.to(device=device)
    if verify_lens.dtype not in (torch.int32, torch.int64):
        verify_lens = verify_lens.to(torch.int32)
    block_ids = draft_block_ids.to(device=device, dtype=torch.int64).contiguous()
    tokens = draft_tokens.to(device=device, dtype=torch.int64).contiguous()
    bs = verify_lens.shape[0]
    _dspark_require(bs > 0, op, f"bs must be > 0, got {bs}")
    _dspark_require(
        tokens.dim() == 2 and tokens.shape[1] >= 1,
        op,
        f"draft_tokens must be [bs, gamma>=1], got {tuple(tokens.shape)}",
    )
    _dspark_require(
        block_ids.shape == tokens.shape,
        op,
        f"draft_block_ids {tuple(block_ids.shape)} != draft_tokens "
        f"{tuple(tokens.shape)}",
    )

    if bs <= 1024:  # vendor 硬性上限
        return torch.ops.xspeedgate_ops.compact_verify_ids(
            block_ids,
            tokens,
            verify_lens.contiguous(),
            layout.graph_num_tokens,
            str(tokens.device),
        )

    _dspark_warn_fallback(op, f"bs={bs} exceeds the vendor limit of 1024")
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        compact_verify_ids,
    )

    return compact_verify_ids(
        draft_block_ids=block_ids,
        draft_tokens=tokens,
        layout=layout,
        device=device,
    )


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_scatter_compact_to_strided_kernel",
)
def scatter_compact_to_strided_kernel(
    compact_ptr: torch.Tensor,
    verify_lens_ptr: torch.Tensor,
    start_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    stride: int,
    dim: int,
    fill_value,
    BLOCK_D: int,
) -> None:
    """Scatter compact ragged rows into fixed-stride output rows."""
    del dim, BLOCK_D
    n_out = out_ptr.shape[0]
    rows = torch.arange(n_out, device=out_ptr.device, dtype=torch.int64)
    bs = verify_lens_ptr.numel()
    request = rows // stride
    column = rows % stride
    valid_request = request < bs
    safe_request = request.clamp(min=0, max=max(bs - 1, 0))
    verify_lens = verify_lens_ptr[safe_request].to(torch.int64)
    valid = valid_request & (column < verify_lens)
    starts = start_ptr[safe_request].to(torch.int64)
    source = (starts + column).clamp(min=0, max=max(compact_ptr.shape[0] - 1, 0))
    values = compact_ptr[source]
    values = torch.where(
        valid[:, None],
        values,
        torch.full_like(values, fill_value),
    )
    out_ptr[:n_out].copy_(values.to(out_ptr.dtype))


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_commit_inject_layout_kernel",
)
def commit_inject_layout_kernel(
    req_pool_ptr: torch.Tensor,
    req_to_token_ptr: torch.Tensor,
    prefix_lens_ptr: torch.Tensor,
    block_pos_offsets_ptr: torch.Tensor,
    full_to_swa_ptr: torch.Tensor,
    commit_lens_ptr: torch.Tensor,
    swa_loc_ptr: torch.Tensor,
    positions_ptr: torch.Tensor,
    rt_stride: int,
    stride: int,
    n: int,
    BLOCK: int,
) -> None:
    """Build committed SWA locations and absolute positions."""
    del BLOCK
    # vendor 算子要求 req_to_token 为 int64，而线上 req_to_token 是 int32；
    # 这里只把本次用到的那几行取出来转成 int64（整表 cast 太贵），
    # 并把 req_pool 重映射成 0..n_req-1。
    n_req = (n + stride - 1) // stride
    request_rows = req_pool_ptr[:n_req].to(torch.int64)
    token_pool = req_to_token_ptr.as_strided(req_to_token_ptr.shape, (rt_stride, 1))
    token_pool_i64 = token_pool.index_select(0, request_rows).to(torch.int64)
    swa_loc = _dspark_kernel_out(swa_loc_ptr, torch.int32, (n_req * stride,))
    positions = _dspark_kernel_out(positions_ptr, torch.int64, (n_req * stride,))
    torch.ops.xspeedgate_ops.commit_inject_layout_kernel(
        torch.arange(n_req, dtype=torch.int64, device=swa_loc_ptr.device),
        token_pool_i64,
        prefix_lens_ptr[:n_req].to(torch.int64).contiguous(),
        block_pos_offsets_ptr[:stride].contiguous(),
        full_to_swa_ptr.to(torch.int64).contiguous(),
        commit_lens_ptr[:n_req].to(torch.int32).contiguous(),
        swa_loc,
        positions,
        token_pool_i64.stride(0),
        stride,
        n,
        256,
    )
    if swa_loc is not swa_loc_ptr:
        swa_loc_ptr[:n].copy_(swa_loc[:n])
    if positions is not positions_ptr:
        positions_ptr[:n].copy_(positions[:n])


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window",
    "_build_out_tokens_kernel",
)
def build_out_tokens_kernel(
    draft_tokens_ptr: torch.Tensor,
    correct_len_ptr: torch.Tensor,
    bonus_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    gamma: int,
    T: int,
    n_out: int,
    BLOCK: int,
) -> None:
    """Build draft tokens plus the accepted-token bonus."""

    if n_out == 0:
        return
    draft_tokens = draft_tokens_ptr.to(torch.int64).contiguous()
    if draft_tokens.dim() != 2:
        draft_tokens = draft_tokens.reshape(-1, gamma)
    out = _dspark_kernel_out(
        out_ptr, torch.int64, shape=(out_ptr.numel() // T, T)
    )

    torch.ops.xspeedgate_ops.build_out_tokens_kernel(
        draft_tokens,
        correct_len_ptr.reshape(-1).to(torch.int64).contiguous(),
        bonus_ptr.reshape(-1).to(torch.int64).contiguous(),
        out,
        gamma,
        T,
        n_out,
        max(int(BLOCK), 1),
    )
    if out is not out_ptr:
        out_ptr.reshape(-1)[:n_out].copy_(out.reshape(-1)[:n_out])


@register_triton_op(
    "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin",
    "create_chunked_prefix_cache_kv_indices",
)
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices_ptr: torch.Tensor,
    chunk_start_idx_ptr: torch.Tensor,
    chunk_seq_lens_ptr: torch.Tensor,
    chunk_cu_seq_lens_ptr: torch.Tensor,
    chunk_kv_indices_ptr: torch.Tensor,
    req_to_token_ptr_stride: int,
) -> None:
    """Create flattened KV indices for chunked prefix cache on Kunlun."""

    batch_size = req_pool_indices_ptr.numel()
    for batch_index in range(batch_size):
        req_pool_index = int(req_pool_indices_ptr[batch_index].item())
        chunk_start = int(chunk_start_idx_ptr[batch_index].item())
        chunk_seq_len = int(chunk_seq_lens_ptr[batch_index].item())
        if chunk_seq_len <= 0:
            continue
        chunk_kv_offset = int(chunk_cu_seq_lens_ptr[batch_index].item())
        chunk_kv_indices_ptr[
            chunk_kv_offset : chunk_kv_offset + chunk_seq_len
        ].copy_(
            req_to_token_ptr[
                req_pool_index, chunk_start : chunk_start + chunk_seq_len
            ].to(chunk_kv_indices_ptr.dtype)
        )


@register_triton_op(
    "sglang.kernels.ops.memory.memcpy_triton",
    "memcpy_triton_kernel",
)
def memcpy_triton_kernel(
    dst: torch.Tensor,
    src: torch.Tensor,
    offset: torch.Tensor,
    sz: torch.Tensor,
    offset_src: bool,
    chunk_size: int,
    BLOCK_SIZE: int,
) -> None:
    """Copy token ranges between tensors for data-parallel attention."""

    torch.ops.xspeedgate_ops.memcpy_token(dst, src, 0, offset, sz, offset_src)


# [Dropped for 0.5.14] ``overlap_utils._resolve_future_token_ids`` no longer
# exists; future-token-id resolution moved to ``FutureMap._resolve_spec_extras``
# (pure torch). This deployment also runs with --disable-overlap-schedule, so
# the path is not exercised. No Triton kernel to replace.


@register_jit_op("sglang.srt.model_executor.forward_batch_info", "clamp_position")
def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute non-negative last-token positions from sequence lengths."""

    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


@register_jit_op("sglang.kernels.ops.quantization.hadamard", "hadamard_transform")
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply the Kunlun Hadamard matmul contract without dtype changes."""

    import kunlun_ops

    hidden_size = x.shape[-1]
    matrix = torch.empty(
        (hidden_size, hidden_size), dtype=x.dtype, device=x.device
    )
    kunlun_ops.gen_hadamard_matrix(matrix, scale)
    original_shape = x.shape
    x_2d = x.view(-1, hidden_size) if x.ndim > 2 else x
    out = torch.empty_like(x_2d)
    kunlun_ops.matmul(x_2d, matrix, out, False, True, 1.0, 0.0)
    return out.view(original_shape) if x.ndim > 2 else out


@register_jit_op("sglang.kernels.ops.activation.activation", "silu_and_mul")
def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """Kunlun SwiGLU replacement for 0.5.14's JIT silu_and_mul.

    0.5.14 routes ``activation.SiluAndMul.forward_cuda`` through the nvcc-JIT
    ``sglang.jit_kernel.activation.silu_and_mul`` (which fails on Kunlun: nvcc
    11.7 rejects ``-std=c++20``). Replace it with ``kunlun_ops.swiglu``.

    The 0.5.14 signature adds optional ``expert_ids`` / ``expert_step`` for
    filtered MoE activation; the Kunlun SwiGLU op does not support per-expert
    filtering, so assert it is unused on this path.
    """
    import kunlun_ops

    assert expert_ids is None, (
        "Kunlun silu_and_mul does not support expert_ids filtering"
    )
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    kunlun_ops.swiglu(x=input, y=out)
    return out


def dsv4_moe_fused_gate_kunlun(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    scoring_func: str = "sqrtsoftplus",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
):
    """Compute the DSV4 fused-gate routing contract with Torch operators."""

    if scoring_func != "sqrtsoftplus":
        raise ValueError(f"unsupported DSV4 fused-gate scoring: {scoring_func}")
    if num_expert_group != 1 or topk_group != 1:
        raise ValueError("grouped routing is not supported by the DSV4 reference gate")
    if moe_softcapping != 0.0:
        raise ValueError("MoE softcapping is not supported by the DSV4 reference gate")
    if topk <= num_fused_shared_experts:
        raise ValueError("topk must exceed the number of fused shared experts")

    scores_fp32 = torch.nn.functional.softplus(scores.float()).sqrt()
    ranking_scores = scores_fp32
    if bias is not None:
        ranking_scores = ranking_scores + bias.float().unsqueeze(0)
    ranking_scores = torch.nan_to_num(ranking_scores, nan=-1e30)

    routed_topk = topk - num_fused_shared_experts
    _, topk_ids = torch.topk(
        ranking_scores,
        k=routed_topk,
        dim=-1,
        sorted=True,
    )
    topk_weights = torch.gather(scores_fp32, 1, topk_ids)
    routed_sum = topk_weights.sum(dim=-1, keepdim=True)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)

    if num_fused_shared_experts:
        num_tokens, num_experts = scores.shape
        shared_ids = torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            device=scores.device,
            dtype=topk_ids.dtype,
        ).reshape(1, -1).expand(num_tokens, -1)
        shared_weights = (routed_sum / scale).expand(-1, num_fused_shared_experts)
        topk_ids = torch.cat((topk_ids, shared_ids), dim=-1)
        topk_weights = torch.cat((topk_weights, shared_weights), dim=-1)

    if renormalize:
        normalizer = torch.where(routed_sum > 0.0, routed_sum, 1.0)
        topk_weights = topk_weights / normalizer
    if apply_routed_scaling_factor_on_output:
        topk_weights = topk_weights * scale

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


_DSV4_HASH_TOPK_GUARD_COUNTERS: dict = {}
_DSV4_HASH_TOPK_GUARD_REPORTED: dict = {}


def _dsv4_guard_hash_topk_input_ids(
    input_ids: torch.Tensor, tid2eid: torch.Tensor
) -> torch.Tensor:
    """Return token ids clamped into the tid2eid domain, counting the clamps.

    ``moe_hash_topk_fused`` looks the routing table up by token id, so a padded
    row carrying -1 (the mask_topk_ids / draft buffer fill) or a stale id reads
    outside the table and faults the device. Real token ids are untouched.
    """
    ids = input_ids.to(torch.int64)
    table_size = tid2eid.shape[0]
    invalid = (ids < 0) | (ids >= table_size)
    counter = _DSV4_HASH_TOPK_GUARD_COUNTERS.get(ids.device)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int32, device=ids.device)
        _DSV4_HASH_TOPK_GUARD_COUNTERS[ids.device] = counter
    # Captured into the graph so replay accumulates too; never read here, that
    # would sync the host on the hot path.
    counter.add_(invalid.sum().to(torch.int32))
    return ids.masked_fill(invalid, 0)


def _dsv4_report_hash_topk_guard(device) -> None:
    """Log the clamp counter when it grows; env-gated because reading it syncs."""
    import os

    if os.getenv("DSV4_HASH_TOPK_GUARD_DEBUG", "0") != "1":
        return
    counter = _DSV4_HASH_TOPK_GUARD_COUNTERS.get(device)
    if counter is None:
        return
    total = int(counter)
    if total <= _DSV4_HASH_TOPK_GUARD_REPORTED.get(device, 0):
        return
    _DSV4_HASH_TOPK_GUARD_REPORTED[device] = total
    logger.warning(
        "[DSV4_HASH_TOPK_GUARD] device=%s clamped_token_ids_total=%s", device, total
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.moe", "hash_topk")
def dsv4_hash_topk_kunlun(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
):
    """Compute the DSV4 hash top-k contract with graph-safe Torch operators."""

    if scoring_func != "sqrtsoftplus":
        raise ValueError(f"unsupported DSV4 hash top-k scoring: {scoring_func}")

    # compared with v0.5.8, _dsv4_guard_hash_topk_input_ids can be moved.
    safe_input_ids = input_ids.to(torch.int64)
    _dsv4_report_hash_topk_guard(safe_input_ids.device)
    topk_ids, topk_weights = torch.ops.xspeedgate_ops.moe_hash_topk_fused(
        router_logits,
        safe_input_ids,
        tid2eid,
        num_fused_shared_experts,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


@register_jit_op("sglang.kernels.ops.attention.dsv4.gemm", "linear_bf16_fp32")
def dsv4_linear_bf16_fp32_kunlun(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Preserve the shared DSV4 FP32-output GEMM contract."""

    return torch.nn.functional.linear(x.float(), y.float())


@register_jit_op("sglang.kernels.ops.attention.dsv4.moe", "silu_and_mul_clamp")
def dsv4_silu_and_mul_clamp_kunlun(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    output.copy_(
        torch.ops.xspeedgate_ops.silu_and_mul_with_swiglu_limit(input, swiglu_limit)
    )


@register_jit_op("sglang.srt.utils.common", "fast_topk")
def fast_topk(values: torch.Tensor, topk: int, dim: int):
    """Patch speculative fast_topk aliases onto a torch implementation on Kunlun."""

    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    return torch.topk(values, topk, dim=dim)


# Set by mask_topk_ids so the MoE apply hook can drop the padded rows: the gate
# runs immediately before it in the same forward, and the tensor is a stable
# buffer under CUDA graph replay.
_DSV4_LAST_NUM_TOKEN_NON_PADDED: dict = {}


@register_jit_op("sglang.kernels.ops.attention.dsv4.moe", "mask_topk_ids")
def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor) -> None:
    """Mask padded token rows in MoE top-k ids on Kunlun.

    Boolean-mask assignment is shape-dependent and therefore not capturable, so
    the padded rows would keep stale gate output during graph replay. masked_fill_
    over a static-shaped mask is capturable. The fill value is 0, not upstream's
    -1: every Kunlun MoE kernel indexes per-expert buffers by id, and -1 is the
    out-of-range value that faults them. Routing weights for padded rows are
    irrelevant because their output rows are discarded; what matters is that the
    ids are constant, so real rows always get the same expert grouping.
    """

    _DSV4_LAST_NUM_TOKEN_NON_PADDED[topk_ids.device] = num_token_non_padded
    indices = torch.arange(
        0, topk_ids.shape[0], device=topk_ids.device, dtype=torch.int64
    )
    padded = (indices >= num_token_non_padded.reshape(()).to(torch.int64)).unsqueeze(1)
    topk_ids.masked_fill_(padded, 0)


@register_jit_op("sglang.kernels.ops.moe.moe_fused_gate", "moe_fused_gate")
def moe_fused_gate_dsv4_kunlun(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
):
    """Match the monkey-patched DeepSeek-V4 fused MoE gate."""
    import kunlun_ops

    if num_expert_group != 1 or topk_group != 1:
        raise ValueError("grouped routing is not supported by the DSV4 fused gate")
    if moe_softcapping != 0.0:
        raise ValueError("MoE softcapping is not supported by the DSV4 fused gate")

    num_rows = input.shape[0]
    topk_weights = torch.empty(
        num_rows, topk, dtype=torch.float32, device=input.device
    )
    topk_indices = torch.empty(
        num_rows, topk, dtype=torch.int32, device=input.device
    )
    kunlun_ops.moe_fused_gate_dsv4(
        input.to(torch.float32),
        bias,
        topk_weights,
        topk_indices,
        topk,
        scoring_func,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_indices


@register_jit_op("sglang.srt.layers.moe.topk", "grouped_topk")
def grouped_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    scaling_factor: float,
):
    """Kunlun fused sigmoid+grouped-topk+norm (replaces 0.5.14's nvcc-JIT kernel).

    0.5.14's ``select_experts`` routes the CUDA single-group path through
    ``sglang.jit_kernel.grouped_topk.grouped_topk`` which nvcc-JIT compiles
    ``moe/grouped_topk.cuh`` with ``-std=c++20`` (rejected by Kunlun nvcc 11.7).
    Replace it with ``kunlun_ops.moe_sigmoid_group_topk_norm``, returning the
    same ``(topk_values, topk_indices)`` tuple. The block-statistic tensor it
    also computes is discarded here and regenerated in the MoE method.
    """
    import kunlun_ops

    m_, n_ = scores.shape
    block_statistic = torch.empty(
        12, n_ + 1, dtype=torch.int32, device=scores.device
    )
    topk_values = torch.empty(m_, topk, dtype=torch.float32, device=scores.device)
    topk_indices = torch.empty(m_, topk, dtype=torch.int32, device=scores.device)
    if m_ == 0:
        return topk_values, topk_indices
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=scores,
        topk_index=topk_indices,
        norm_score=topk_values,
        block_statistic=block_statistic,
        bias=bias.float(),
        scale=scaling_factor,
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_values, topk_indices


@register_triton_op("sglang.srt.mem_cache.utils", "set_mla_kv_buffer_kernel")
def set_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    buffer_stride: int,
    nope_stride: int,
    rope_stride: int,
    nope_dim: int,
    rope_dim: int,
    *,
    BLOCK: int,
) -> None:
    """Store MLA KV tensors into the paged KV buffer."""

    from kunlun_ops import set_mla_kv_buffer_triton as impl

    impl(
        kv_buffer,
        loc,
        cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
        cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
    )


@register_triton_op("sglang.srt.mem_cache.utils", "get_mla_kv_buffer_kernel")
def get_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    buffer_stride: int,
    nope_stride: int,
    rope_stride: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Load MLA KV tensors from the paged KV buffer."""

    torch.ops.xspeedgate_ops.get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)


def _dsv4_alloc_extend_torch(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Torch port of upstream ``alloc_extend_kernel``.

    Slots come from three regions, in this order: the tail of the page the prefix
    already occupies, whole freshly taken pages, and the head of one last fresh
    page. Requests consume ``free_pages`` in batch order.
    """
    prefix_list = prefix_lens.to(torch.int64).tolist()
    seq_list = seq_lens.to(torch.int64).tolist()
    last_list = last_loc.to(torch.int64).tolist()
    pages = free_pages.to(torch.int64)
    device = out_indices.device
    out_pos = 0
    page_pos = 0

    for index in range(len(seq_list)):
        prefix_len = prefix_list[index]
        seq_len = seq_list[index]
        extend_len = seq_len - prefix_len
        prefix_page_end = -(-prefix_len // page_size) * page_size
        num_new_pages = -(-seq_len // page_size) * page_size // page_size - (
            prefix_page_end // page_size
        )
        if extend_len <= 0:
            page_pos += max(num_new_pages, 0)
            continue

        chunks = []
        num_part1 = min(seq_len, prefix_page_end) - prefix_len
        if num_part1 > 0:
            chunks.append(
                last_list[index]
                + 1
                + torch.arange(num_part1, dtype=torch.int64, device=device)
            )
        num_part2 = (seq_len // page_size) * page_size - prefix_page_end
        if num_part2 > 0:
            offsets = torch.arange(num_part2, dtype=torch.int64, device=device)
            page_ids = pages.index_select(
                0,
                (page_pos + torch.div(offsets, page_size, rounding_mode="floor")),
            )
            chunks.append(page_ids * page_size + offsets.remainder(page_size))
        num_part3 = 0
        if num_part1 + max(num_part2, 0) < extend_len:
            num_part3 = seq_len - (seq_len // page_size) * page_size
            start_page = pages[page_pos + num_new_pages - 1]
            chunks.append(
                start_page * page_size
                + torch.arange(num_part3, dtype=torch.int64, device=device)
            )

        allocated = torch.cat(chunks) if chunks else out_indices.new_empty((0,))
        assert allocated.numel() == extend_len, (
            f"alloc_extend torch mismatch: req={index} extend={extend_len} "
            f"got={allocated.numel()} parts={(num_part1, num_part2, num_part3)}"
        )
        out_indices[out_pos : out_pos + extend_len] = allocated.to(out_indices.dtype)
        out_pos += extend_len
        page_pos += max(num_new_pages, 0)


@register_triton_op("sglang.kernels.ops.memory.allocator", "alloc_extend_kernel")
def alloc_extend_kernel(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Allocate KV cache pages for extend batches."""

    _dsv4_probe(
        "alloc_extend_input",
        {
            "prefix_lens": prefix_lens,
            "seq_lens": seq_lens,
            "last_loc": last_loc,
            "free_pages": free_pages,
        },
        bs_upper=bs_upper,
        page_size=page_size,
        prefix_lens_dtype=str(prefix_lens.dtype),
        seq_lens_dtype=str(seq_lens.dtype),
        last_loc_dtype=str(last_loc.dtype),
        prefix_lens_stride=tuple(prefix_lens.stride()),
        seq_lens_stride=tuple(seq_lens.stride()),
        last_loc_stride=tuple(last_loc.stride()),
        last_loc_is_contiguous=last_loc.is_contiguous(),
    )
    if prefix_lens.dtype != torch.int64:
        prefix_lens = prefix_lens.to(torch.int64)
    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    if last_loc.dtype != torch.int64:
        last_loc = last_loc.to(torch.int64)

    # try:
    #     from sglang.srt.environ import envs

    #     use_fast_alloc_extend = (
    #         getattr(envs, "USE_FAST_ALLOC_EXTEND_KUNLUN", None)
    #         and envs.USE_FAST_ALLOC_EXTEND_KUNLUN.get()
    #     )
    # except Exception:
    #     use_fast_alloc_extend = False

    if 0:
        from torch_xmlir.nn.alloc_extend import Alloc_extend

        alloc_extend_op = Alloc_extend()
        xdnn_out_indices, _ = alloc_extend_op(
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            page_size,
            out_indices.shape[0],
        )
        out_indices.copy_(xdnn_out_indices.to(out_indices.dtype))
        return

    import os

    if os.getenv("DSV4_TORCH_ALLOC_EXTEND", "0") == "1":
        _dsv4_alloc_extend_torch(
            prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size
        )
        return

    ret_value = torch.zeros(1, dtype=torch.int64, device=out_indices.device)
    torch.ops.xspeedgate_ops.alloc_extend(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        prefix_lens.shape[0],
        page_size,
        out_indices.shape[0],
        out_indices,
        ret_value,
    )
    _dsv4_probe(
        "alloc_extend_output",
        {
            "out_indices": out_indices,
            "ret_value": ret_value,
        },
        bs_upper=bs_upper,
        page_size=page_size,
    )


@register_triton_op("sglang.kernels.ops.memory.allocator", "alloc_decode_kernel")
def alloc_decode_kernel(
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Allocate KV cache pages for decode batches."""

    torch.ops.xspeedgate_ops.alloc_decode_kernel(
        seq_lens,
        last_loc.to(torch.int32).contiguous(),
        free_pages,
        out_indices,
        bs_upper,
        page_size,
        seq_lens.shape[0],
    )


@register_jit_op("sglang.kernels.ops.kvcache.kvcache", "store_cache")
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,
) -> None:
    """Store key and value tensors into paged KV cache."""

    from kunlun_ops import reshape_and_cache

    reshape_and_cache(
        k.contiguous(), v.contiguous(), k_cache, v_cache, indices.to(torch.int32), None, None, 0
    )


def _copy_fused_metadata_fallback(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: torch.Tensor | None,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: torch.Tensor | None,
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: torch.Tensor | None,
    flashmla_num_splits_dst: torch.Tensor | None,
    flashmla_metadata_dst: torch.Tensor | None,
    forward_mode: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """Copy NSA metadata without invoking CUDA JIT fused kernels."""

    cache_seqlens_dst.copy_(cache_seqlens_src)
    cu_seqlens_k_dst[1:].copy_(cu_seqlens_k_src[1:])

    if forward_mode == 0:
        page_table_1_dst[:, :max_len].copy_(page_indices_src)
        nsa_cache_seqlens_dst.copy_(nsa_cache_seqlens_src)
    elif forward_mode == 1:
        page_table_1_dst[:, :max_seqlen_k].copy_(page_indices_src)
        if seqlens_expanded_dst is not None and seqlens_expanded_src is not None:
            seqlens_expanded_dst.copy_(seqlens_expanded_src)
        nsa_cache_seqlens_dst.copy_(nsa_cache_seqlens_src)
    elif forward_mode == 2:
        rows = page_indices_src.shape[0]
        page_table_1_dst[:rows, :max_seqlen_k].copy_(page_indices_src)
        if seqlens_expanded_dst is not None and seqlens_expanded_src is not None:
            seqlens_expanded_dst[:seqlens_expanded_size].copy_(seqlens_expanded_src)
        nsa_cache_seqlens_dst[:seqlens_expanded_size].copy_(nsa_cache_seqlens_src)

    nsa_cu_seqlens_k_dst[1 : 1 + seqlens_expanded_size].copy_(
        nsa_cu_seqlens_k_src[1 : 1 + seqlens_expanded_size]
    )

    if real_page_table_src is not None and real_page_table_dst is not None:
        rows, cols = real_page_table_src.shape
        real_page_table_dst[:rows, :cols].copy_(real_page_table_src)

    if flashmla_metadata_src is not None and flashmla_metadata_dst is not None:
        flashmla_metadata_dst[: seqlens_expanded_size + 1].copy_(
            flashmla_metadata_src[: seqlens_expanded_size + 1]
        )
    if flashmla_num_splits_src is not None and flashmla_num_splits_dst is not None:
        flashmla_num_splits_dst[: seqlens_expanded_size + 1].copy_(
            flashmla_num_splits_src[: seqlens_expanded_size + 1]
        )


@register_jit_op("sglang.kernels.ops.attention.fused_metadata_copy", "fused_metadata_copy_cuda")
def fused_metadata_copy_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: torch.Tensor | None,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: torch.Tensor | None,
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: torch.Tensor | None,
    flashmla_num_splits_dst: torch.Tensor | None,
    flashmla_metadata_dst: torch.Tensor | None,
    forward_mode: int,
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """Silently fall back from CUDA fused metadata copy on Kunlun."""

    _copy_fused_metadata_fallback(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        nsa_cache_seqlens_dst,
        seqlens_expanded_dst,
        nsa_cu_seqlens_k_dst,
        real_page_table_dst,
        flashmla_num_splits_dst,
        flashmla_metadata_dst,
        forward_mode,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )


@register_jit_op("sglang.kernels.ops.attention.fused_metadata_copy", "fused_metadata_copy_multi_cuda")
def fused_metadata_copy_multi_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst0: torch.Tensor,
    cu_seqlens_k_dst0: torch.Tensor,
    page_table_1_dst0: torch.Tensor,
    nsa_cache_seqlens_dst0: torch.Tensor,
    nsa_cu_seqlens_k_dst0: torch.Tensor,
    real_page_table_dst0: torch.Tensor | None,
    flashmla_num_splits_dst0: torch.Tensor | None,
    flashmla_metadata_dst0: torch.Tensor | None,
    cache_seqlens_dst1: torch.Tensor,
    cu_seqlens_k_dst1: torch.Tensor,
    page_table_1_dst1: torch.Tensor,
    nsa_cache_seqlens_dst1: torch.Tensor,
    nsa_cu_seqlens_k_dst1: torch.Tensor,
    real_page_table_dst1: torch.Tensor | None,
    flashmla_num_splits_dst1: torch.Tensor | None,
    flashmla_metadata_dst1: torch.Tensor | None,
    cache_seqlens_dst2: torch.Tensor,
    cu_seqlens_k_dst2: torch.Tensor,
    page_table_1_dst2: torch.Tensor,
    nsa_cache_seqlens_dst2: torch.Tensor,
    nsa_cu_seqlens_k_dst2: torch.Tensor,
    real_page_table_dst2: torch.Tensor | None,
    flashmla_num_splits_dst2: torch.Tensor | None,
    flashmla_metadata_dst2: torch.Tensor | None,
    bs: int,
    max_len: int,
    seqlens_expanded_size: int,
) -> None:
    """Silently copy multi-backend NSA metadata without CUDA JIT."""

    for dst in (
        (
            cache_seqlens_dst0,
            cu_seqlens_k_dst0,
            page_table_1_dst0,
            nsa_cache_seqlens_dst0,
            nsa_cu_seqlens_k_dst0,
            real_page_table_dst0,
            flashmla_num_splits_dst0,
            flashmla_metadata_dst0,
        ),
        (
            cache_seqlens_dst1,
            cu_seqlens_k_dst1,
            page_table_1_dst1,
            nsa_cache_seqlens_dst1,
            nsa_cu_seqlens_k_dst1,
            real_page_table_dst1,
            flashmla_num_splits_dst1,
            flashmla_metadata_dst1,
        ),
        (
            cache_seqlens_dst2,
            cu_seqlens_k_dst2,
            page_table_1_dst2,
            nsa_cache_seqlens_dst2,
            nsa_cu_seqlens_k_dst2,
            real_page_table_dst2,
            flashmla_num_splits_dst2,
            flashmla_metadata_dst2,
        ),
    ):
        _copy_fused_metadata_fallback(
            cache_seqlens_src,
            cu_seqlens_k_src,
            page_indices_src,
            nsa_cache_seqlens_src,
            None,
            nsa_cu_seqlens_k_src,
            real_page_table_src,
            flashmla_num_splits_src,
            flashmla_metadata_src,
            dst[0],
            dst[1],
            dst[2],
            dst[3],
            None,
            dst[4],
            dst[5],
            dst[6],
            dst[7],
            0,
            max_len,
            max_len,
            seqlens_expanded_size,
        )


@register_triton_op(
    "sglang.srt.speculative.multi_layer_eagle_utils",
    "rotate_input_ids_kernel",
)
def rotate_input_ids_kernel(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index: torch.Tensor | None,
    *,
    BLOCK_SIZE: int,
) -> None:
    """Rotate Eagle draft input ids according to selected top-k indices."""

    if input_ids.dtype != torch.int64:
        input_ids = input_ids.to(torch.int64)
    if extend_seq_lens.dtype != torch.int64:
        extend_seq_lens = extend_seq_lens.to(torch.int64)
    if topk_index.dtype != torch.int64:
        topk_index = topk_index.to(torch.int64)
    torch.ops.xspeedgate_ops.rotate_input_ids_triton(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index=select_index,
    )


def assign_hidden_states_pool_triton(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
) -> None:
    """Assign draft hidden states into the request hidden-state pool."""

    # pool_capacity = req_to_hidden_states_pool.shape[0]
    # for req in range(num_seqs):
    #     pool_idx = req_pool_indices[req]
    #     if pool_idx < 0 or pool_idx >= pool_capacity:
    #         continue
    #     extend_len = extend_seq_lens[req]
    #     start_loc = extend_start_loc[req]
    #     end_loc = start_loc + extend_len
    #     req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
    #         hidden_states[end_loc - pool_size : end_loc, :]
    #     )

    pool_capacity = req_to_hidden_states_pool.shape[0]
    active_req_pool_indices = req_pool_indices[:num_seqs].to(torch.int64)
    write_req_pool_indices = active_req_pool_indices
    valid_mask = (write_req_pool_indices >= 0) & (write_req_pool_indices < pool_capacity)
    if not bool(valid_mask.all().item()):
        valid_req_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        logger.debug(
            "assign_hidden_states_pool filters out-of-range req_pool_indices: "
            "pool_capacity=%s, num_seqs=%s, "
            "invalid_req_indices=%s, invalid_pool_indices=%s",
            pool_capacity,
            num_seqs,
            torch.nonzero(~valid_mask, as_tuple=False).flatten().detach().cpu().tolist(),
            active_req_pool_indices[~valid_mask].detach().cpu().tolist(),
        )
        active_req_pool_indices = active_req_pool_indices[valid_req_indices].contiguous()
        write_req_pool_indices = write_req_pool_indices[valid_req_indices].contiguous()
        extend_seq_lens = extend_seq_lens[:num_seqs][valid_req_indices].contiguous()
        extend_start_loc = extend_start_loc[:num_seqs][valid_req_indices].contiguous()
        num_seqs = active_req_pool_indices.numel()

    if num_seqs == 0:
        return

    for req in range(num_seqs):
        pool_idx = write_req_pool_indices[req]
        extend_len = extend_seq_lens[req]
        start_loc = extend_start_loc[req]
        end_loc = start_loc + extend_len
        req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
            hidden_states[end_loc - pool_size : end_loc, :]
        )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.c128_cleanup",
    "clear_unaccepted_c128_draft_states",
)
def clear_unaccepted_c128_draft_states_torch(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    """Reset rejected offline C128 draft slots without launching Triton."""

    batch_size = req_pool_indices.numel()
    if batch_size == 0 or num_draft_tokens == 0:
        return

    draft_offsets = torch.arange(
        num_draft_tokens,
        dtype=torch.int64,
        device=state.device,
    ).unsqueeze(0)
    row_indices = (
        req_pool_indices.to(torch.int64).unsqueeze(1) * ring_size
        + (seq_lens.to(torch.int64).unsqueeze(1) + draft_offsets).remainder(ring_size)
    ).reshape(-1)
    rows = state.index_select(0, row_indices)
    half = state.shape[-1] // 2
    reset_rows = torch.cat(
        (
            torch.zeros_like(rows[:, :half]),
            torch.full_like(rows[:, half:], float("-inf")),
        ),
        dim=-1,
    )
    rejected = draft_offsets >= accept_lens.to(torch.int64).unsqueeze(1)
    state[row_indices] = torch.where(rejected.reshape(-1, 1), reset_rows, rows)


@register_jit_op(
    "sglang.kernels.ops.speculative.cache_locs",
    "assign_extend_cache_locs_uniform_func",
)
def assign_extend_cache_locs_uniform_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    """Gather uniform target-verify cache slots without launching Triton."""

    del device
    req_indices = req_pool_indices[:batch_size].to(torch.int64)
    token_rows = req_to_token.index_select(0, req_indices)
    token_offsets = torch.arange(
        draft_token_num,
        dtype=torch.int64,
        device=req_to_token.device,
    ).unsqueeze(0)
    token_offsets = token_offsets + start_offset[:batch_size].to(torch.int64).unsqueeze(1)
    return torch.gather(token_rows, 1, token_offsets).reshape(-1).to(torch.int64)


@register_jit_op(
    "sglang.kernels.ops.speculative.multi_layer_eagle",
    "rotate_input_ids",
)
def rotate_input_ids_triton(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index: torch.Tensor | None = None,
):
    """Rotate draft input ids without launching the upstream Triton kernel."""

    batch_size = extend_seq_lens.shape[0]
    token_ids = topk_index.reshape(-1)[:batch_size]
    for pid in range(batch_size):
        start = int(extend_start_loc[pid].item())
        seq_len = int(extend_seq_lens[pid].item())
        if seq_len <= 0:
            continue
        if seq_len > 1:
            input_ids[start : start + seq_len - 1].copy_(
                input_ids[start + 1 : start + seq_len].clone()
            )
        if select_index is not None:
            last_pos = int(select_index[pid].item())
        else:
            last_pos = start + seq_len - 1
        input_ids[last_pos] = token_ids[pid].to(input_ids.dtype)
    return input_ids


def assign_new_state_kernel(
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    next_token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    step: int,
    stride_hidden_seq: int,
    stride_hidden_dim: int,
    stride_pool_req: int,
    stride_pool_step: int,
    stride_pool_dim: int,
    stride_req_token_0: int,
    stride_req_token_1: int,
    *,
    HIDDEN_DIM: int,
    BLOCK_SEQ: int,
    BLOCK_HID: int,
    grid: tuple[int, ...] | None = None,
) -> None:
    """Build the next Eagle draft state tensors from the previous state."""

    num_seqs = grid[0] if grid is not None else old_extend_seq_lens.numel()
    num_seqs = int(num_seqs)
    max_num_seqs = old_extend_seq_lens.shape[0]
    old_extend_len = old_input_ids.shape[0] // max_num_seqs
    new_extend_len = old_extend_len + 1
    hidden_dim = old_hidden_states.shape[1]
    device = old_input_ids.device

    extend_seq_lens[:num_seqs].fill_(new_extend_len)
    extend_start_loc[:num_seqs].copy_(
        old_extend_start_loc[:num_seqs]
        + torch.arange(num_seqs, dtype=old_extend_start_loc.dtype, device=device)
    )

    old_ids = old_input_ids[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_pos = old_positions[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_cache = old_out_cache_loc[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_hidden = old_hidden_states[: num_seqs * old_extend_len].view(
        num_seqs, old_extend_len, hidden_dim
    )

    new_ids = input_ids[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_pos = positions[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_cache = out_cache_loc[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_hidden = hidden_states[: num_seqs * new_extend_len].view(
        num_seqs, new_extend_len, hidden_dim
    )

    new_ids[:, :old_extend_len].copy_(old_ids)
    token_ids = next_token_ids[:num_seqs, 0] if next_token_ids.dim() > 1 else next_token_ids[:num_seqs]
    insert_pos = (old_extend_len - padding_lens[:num_seqs]).long().unsqueeze(1)
    new_ids.scatter_(1, insert_pos, token_ids.unsqueeze(1).to(new_ids.dtype))

    new_pos[:, 1:].copy_(old_pos)
    new_pos[:, 0] = (old_pos[:, 0] - 1).clamp(min=0)

    new_cache[:, 1:].copy_(old_cache)
    req_indices = req_pool_indices[:num_seqs].long()
    token_cols = (seq_lens[:num_seqs] - old_extend_len - 1).long()
    valid = token_cols >= 0
    first_locs = req_to_token[req_indices, token_cols.clamp(min=0)]
    new_cache[:, 0] = torch.where(valid, first_locs, new_cache[:, 0])

    new_hidden[:, 1:, :].copy_(old_hidden)
    # Upstream's flattened pointer expression uses ``req_idx + 1`` only to
    # address a negative step within row ``req_idx``. Tensor indexing must use
    # the raw request row directly.
    new_hidden[:, 0, :].copy_(req_to_hidden_states_pool[req_indices, -(step + 1)])


def assign_new_state_triton(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    num_seqs: int,
    step: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
) -> None:
    """Direct replacement for the v0.5.14 draft-extend graph state updater."""

    assign_new_state_kernel(
        old_input_ids,
        old_positions,
        old_hidden_states,
        old_out_cache_loc,
        old_extend_seq_lens,
        old_extend_start_loc,
        input_ids,
        positions,
        hidden_states,
        out_cache_loc,
        extend_seq_lens,
        extend_start_loc,
        next_token_ids,
        seq_lens,
        padding_lens,
        req_pool_indices,
        req_to_token,
        req_to_hidden_states_pool,
        step,
        old_hidden_states.stride(0),
        old_hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        HIDDEN_DIM=hidden_states.shape[1],
        BLOCK_SEQ=8,
        BLOCK_HID=64,
        grid=(num_seqs,),
    )


@register_triton_op(
    "sglang.kernels.ops.speculative.cache_locs",
    "assign_draft_cache_locs_contiguous",
)
def assign_draft_cache_locs_page_size_1(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
) -> None:
    """Assign draft cache locations for the page_size==1 / topk==1 contiguous path.

    0.5.14 renamed the upstream Triton kernel to
    ``assign_draft_cache_locs_contiguous`` (in
    ``sglang.kernels.ops.speculative.cache_locs``), called from
    ``base_spec_worker.prepare_for_draft``. The Kunlun replacement copies
    ``topk * speculative_num_steps`` slots per request via xspeedgate_ops.
    """

    out_cache_loc_i32 = torch.empty_like(out_cache_loc, dtype=torch.int32)
    torch.ops.xspeedgate_ops.assign_draft_cache_locs_page_size_1(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc_i32,
        pool_len,
        topk,
        speculative_num_steps,
    )
    out_cache_loc.copy_(out_cache_loc_i32)


# [Dropped for 0.5.14] ``fill_new_verified_id`` no longer exists upstream; the
# verified-id flow moved to ``FutureMap._resolve_spec_extras`` (pure torch) in
# ``sglang.srt.managers.overlap_utils`` and is not a Triton kernel. No XPU
# replacement needed.


@register_triton_op("sglang.kernels.ops.speculative.eagle", "fill_bonus_tokens")
def fill_bonus_tokens(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    accept_stride: int,
) -> None:
    """Fill speculative bonus tokens without launching the upstream Triton kernel."""

    num_tokens = accept_lens.numel()
    if num_tokens == 0:
        return
    row_ids = torch.arange(num_tokens, device=accept_tokens.device, dtype=torch.int64)
    col_ids = accept_lens.to(torch.int64) - 1
    bonus_tokens[:num_tokens].copy_(
        accept_tokens[row_ids, col_ids].to(bonus_tokens.dtype)
    )


@register_triton_op("sglang.kernels.ops.speculative.eagle", "fill_accept_out_cache_loc")
def fill_accepted_out_cache_loc(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accepted_out_cache_loc: torch.Tensor,
    size_upper: int,
) -> None:
    """Collect accepted output cache locations from draft indices.

    0.5.14 renamed the upstream Triton kernel to ``fill_accept_out_cache_loc``
    (in ``sglang.kernels.ops.speculative.eagle``).
    """

    valid_indices = accept_index[accept_index != -1]
    if valid_indices.numel() > 0:
        accepted_out_cache_loc[: valid_indices.shape[0]] = out_cache_loc[valid_indices]


@register_triton_op(
    "sglang.kernels.ops.speculative.cache_locs",
    "generate_draft_decode_kv_indices",
)
def generate_draft_decode_kv_indices(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    positions: torch.Tensor,
    pool_len: int,
    kv_indices_stride: int,
    kv_indptr_stride: int,
    bs_upper: int,
    iter_upper: int,
    num_tokens_upper: int,
    page_size: int,
    *,
    grid: tuple[int, ...] | None = None,
) -> None:
    """Build draft-decode KV indices for the split v0.5.14 cache-locs op."""

    if grid is None:
        raise ValueError("generate_draft_decode_kv_indices requires Triton-style grid")
    num_steps, num_seqs, topk = int(grid[0]), int(grid[1]), int(grid[2])
    req_indices = req_pool_indices[:num_seqs].long()
    seq_lens = paged_kernel_lens[:num_seqs].long()
    pos = positions[: num_seqs * topk].long()

    for iters0 in range(num_steps):
        iters = iters0 + 1
        kv_indices_row = kv_indices[iters0]
        kv_indptr_row = kv_indptr[iters0]
        cum_seq_len = 0
        for bid in range(num_seqs):
            seq_len = int(seq_lens[bid].item())
            token_pool = req_to_token[req_indices[bid]]
            for topk_id in range(topk):
                kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (
                    seq_len + iters
                )
                if seq_len > 0:
                    kv_indices_row[kv_offset : kv_offset + seq_len].copy_(
                        token_pool[:seq_len].to(kv_indices_row.dtype)
                    )
                extend_offset = torch.arange(iters, device=kv_indices.device)
                if page_size == 1 or topk == 1:
                    start = seq_len + topk_id * num_steps
                else:
                    last_page_len = seq_len % page_size
                    num_new_pages_per_topk = (
                        last_page_len + num_steps + page_size - 1
                    ) // page_size
                    prefix_base = seq_len // page_size * page_size
                    start = (
                        prefix_base
                        + topk_id * num_new_pages_per_topk * page_size
                        + last_page_len
                    )
                kv_indices_row[
                    kv_offset + seq_len : kv_offset + seq_len + iters
                ].copy_(token_pool[start + extend_offset].to(kv_indices_row.dtype))
            cum_seq_len += seq_len

        kv_indptr_row[0] = 0
        for zid in range(1, num_seqs * topk + 1):
            base = int(pos[:zid].sum().item())
            kv_indptr_row[zid] = base + zid * iters


@register_triton_op("sglang.kernels.ops.speculative.cache_locs", "assign_extend_cache_locs")
def assign_extend_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs_upper: int,
) -> None:
    """Assign cache locations for accepted extended tokens.

    0.5.14 moved this Triton kernel to
    ``sglang.kernels.ops.speculative.cache_locs`` (wrapped by
    ``assign_extend_cache_locs_func``).
    """

    batch_size = req_pool_indices.shape[0]
    draft_token_num = out_cache_loc.numel() // batch_size if batch_size > 0 else 0
    result = torch.ops.xspeedgate_ops.assign_extend_cache_locs(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        batch_size,
        draft_token_num,
    )
    out_cache_loc.copy_(result.to(out_cache_loc.dtype))


# [Dropped for 0.5.14] ``spec_utils.create_extend_after_decode_spec_info`` was
# removed upstream; the extend-after-decode setup is inlined with torch ops in
# the modern EAGLE v2 pipeline. No Triton kernel to replace.


def _dsv4_assign_req_to_token_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Torch port of upstream ``assign_req_to_token_pool``.

    Request i receives ``out_cache_loc`` slots for positions
    [start_offset[i], end_offset[i]); the source offset is the running sum of the
    preceding spans.
    """
    starts = start_offset.to(torch.int64).tolist()
    ends = end_offset.to(torch.int64).tolist()
    indices = req_pool_indices.to(torch.int64).tolist()
    values = out_cache_loc.to(req_to_token.dtype)
    source = 0
    for row, (start, end) in enumerate(zip(starts, ends)):
        span = end - start
        if span <= 0:
            continue
        req_to_token[indices[row], start:end] = values[source : source + span]
        source += span


@register_triton_op("sglang.srt.mem_cache.allocation", "assign_req_to_token_pool")
def assign_req_to_token_pool(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs_upper: int,
) -> None:
    """Write accepted cache locations into the request token pool."""

    import os

    batch_size = req_pool_indices.shape[0]
    if out_cache_loc.dtype != req_to_token.dtype:
        out_cache_loc = out_cache_loc.to(req_to_token.dtype)
    if os.getenv("DSV4_TORCH_ASSIGN_REQ_TO_TOKEN", "0") == "1":
        _dsv4_assign_req_to_token_torch(
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc
        )
        return
    torch.ops.xspeedgate_ops.assign_req_to_token_pool(
        req_pool_indices.to(torch.int32),
        req_to_token,
        start_offset.to(torch.int32),
        end_offset.to(torch.int32),
        out_cache_loc,
        pool_len,
        batch_size,
    )


@register_triton_op("sglang.kernels.ops.speculative.cache_locs", "align_evict_mask_to_page_size")
def align_evict_mask_to_page_size(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
    BLOCK_SIZE: int,
) -> None:
    """Keep partial pages from being evicted in Eagle verification."""

    torch.ops.xspeedgate_ops.align_evict_mask_to_page_size(
        seq_lens,
        evict_mask,
        page_size,
        num_draft_tokens,
        BLOCK_SIZE,
    )


@register_jit_op("sglang.kernels.ops.speculative.topk1", "draft_topk1_postprocess")
def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: Optional[torch.Tensor] = None,
    draft_token_column: int = 0,
):
    """Torch greedy draft argmax replacement for the upstream Triton kernels.

    The upstream helper splits the vocab reduction across two Triton kernels,
    neither of which compiles on Kunlun, so EAGLE graph capture aborts. This
    keeps every observable effect of that contract: constant 1.0 ``topk_p``, an
    int64 argmax index, the in-place ``positions`` advance, and the optional
    write into ``draft_tokens[:, draft_token_column]``. Shapes are static and no
    value is read back to the host, so it stays CUDA-graph capturable.
    """

    assert next_token_logits.ndim == 2
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    # Upstream returns before advancing positions when the batch is empty.
    if bs == 0:
        return topk_p, topk_index

    logits = next_token_logits.float()
    # The Triton kernel demotes NaN lanes so they can never win the reduction.
    logits = torch.where(logits == logits, logits, logits.new_full((), -1e30))
    # Match the kernel's tie-break: the lowest vocab index among the maxima.
    max_val = logits.max(dim=-1, keepdim=True).values
    lane = torch.arange(vocab_size, device=logits.device).expand_as(logits)
    index = torch.where(logits == max_val, lane, lane.new_full((), vocab_size))
    index = index.min(dim=-1, keepdim=True).values.to(torch.int64)

    topk_index.copy_(index)
    topk_p.fill_(1.0)
    if write_draft_token:
        draft_tokens[:, draft_token_column : draft_token_column + 1].copy_(index)
    positions.add_(1)
    return topk_p, topk_index


@register_jit_op("sglang.kernels.ops.speculative.gather_spec_extras", "gather_spec_extras")
def gather_spec_extras(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: torch.Tensor | None,
):
    """Torch row-gather replacement for speculative extras on Kunlun."""

    indices = indices.to(torch.long).contiguous()
    topk_p = topk_p_buf[indices].contiguous()
    topk_index = topk_index_buf[indices].contiguous()
    bonus_tokens = output_tokens_buf[indices].contiguous()
    hidden_states = (
        hidden_states_buf[indices].contiguous()
        if hidden_states_buf is not None
        else None
    )
    return topk_p, topk_index, bonus_tokens, hidden_states


@register_jit_op("sglang.kernels.ops.speculative.reject_sampling", "chain_speculative_sampling_triton")
def chain_speculative_sampling_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Torch replacement for chain speculative rejection sampling on Kunlun."""

    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    for bid in range(batch_size):
        root_global_idx = int(retrive_index[bid, 0].item())
        accept_index[bid, 0] = root_global_idx
        last_accepted_global_idx = root_global_idx
        num_accept = 0
        cur_prob_row = 0
        continue_verifying = True

        step = 1
        while step < num_slots and continue_verifying:
            draft_token = int(candidates[bid, step].item())
            p = target_probs[bid, cur_prob_row, draft_token]
            q = draft_probs[bid, cur_prob_row, draft_token]
            coin = uniform_samples[bid, step - 1]
            if bool((coin * q < p).item()):
                num_accept += 1
                cur_prob_row = step
                predicts[last_accepted_global_idx] = draft_token
                curr_global_idx = int(retrive_index[bid, step].item())
                accept_index[bid, num_accept] = curr_global_idx
                last_accepted_global_idx = curr_global_idx
                step += 1
            else:
                continue_verifying = False

        accept_token_num[bid] = num_accept

        target_row = target_probs[bid, cur_prob_row]
        if continue_verifying:
            residual = target_row
        else:
            draft_row = draft_probs[bid, cur_prob_row]
            # A degenerate draft row can carry NaN. clamp() propagates NaN, which
            # would make norm_sum NaN, every cumsum comparison False, and the
            # fallback emit vocab_size - 1 (a reserved id) instead of a token
            # drawn from the target. Upstream's kernel treats NaN q as 0 so the
            # residual falls back to p; match that.
            # 对应triton算子的q_val = tl.where(q_val == q_val, q_val, 0.0)
            draft_row = torch.where(torch.isnan(draft_row), 0.0, draft_row)
            residual = torch.clamp(target_row - draft_row, min=0)

        norm_sum = residual.sum()
        if bool((norm_sum <= 0).item()):
            final_token = vocab_size - 1
        else:
            threshold = uniform_samples_for_final_sampling[bid] * norm_sum
            above = torch.cumsum(residual, dim=0) > threshold
            if bool(above.any().item()):
                final_token = int(torch.argmax(above.to(torch.int32)).item())
            else:
                final_token = vocab_size - 1
        predicts[last_accepted_global_idx] = final_token


@register_triton_op(
    "sglang.kernels.ops.attention.position",
    "compute_position_kernel",
)
def compute_position_kernel(
    positions: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    has_prefix: bool,
) -> None:
    """Compute token positions and extend start locations for a batch."""

    prefix_lens = extend_prefix_lens if has_prefix else torch.empty(0, device=extend_seq_lens.device, \
            dtype=extend_seq_lens.dtype)
    out_positions, out_start_loc = torch.ops.xspeedgate_ops.compute_position_kernel(
        prefix_lens, extend_seq_lens, positions.numel()
    )
    positions.copy_(out_positions.to(positions.dtype))
    extend_start_loc.copy_(out_start_loc.to(extend_start_loc.dtype))


_DSV4_FREQS_REAL_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_DSV4_IDENTITY_MAPPING_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _dsv4_cos_sin_cache(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Return the pointer-stable GPT-J cache expected by the Kunlun RoPE op."""

    key = (
        freqs_cis.data_ptr(),
        tuple(freqs_cis.shape),
        freqs_cis.dtype,
        freqs_cis.device,
    )
    cached = _DSV4_FREQS_REAL_CACHE.get(key)
    if cached is None:
        cached = torch.cat(
            (torch.real(freqs_cis), torch.imag(freqs_cis)), dim=-1
        ).contiguous()
        _DSV4_FREQS_REAL_CACHE[key] = cached
    return cached

def _dsv4_rotate_gptj_tail(
    value: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if value.ndim == 3 and value.shape[1:] == (64, 128):
        return torch.ops.xspeedgate_ops.dsv4_rotate_gptj_tail(
            value=value.contiguous(),
            freqs_cis=freqs_cis.contiguous(),
            positions=positions.flatten().to(dtype=torch.int32).contiguous(),
            inverse=False,
        )

    freqs_real = _dsv4_cos_sin_cache(freqs_cis)
    rope_dim = freqs_real.shape[-1]
    rope_tail = value[..., -rope_dim:].contiguous()
    rope_shape = rope_tail.shape
    rope_tail_3d = rope_tail.reshape(rope_shape[0], -1, rope_dim)
    rotated, _ = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions.flatten(),
        rotary_dim=rope_dim,
        head_size=rope_dim,
        cos_sin_cache=freqs_real,
        is_neox_style=False,
        query=rope_tail_3d,
        key=None,
        offsets=None,
        inverse=False,
    )
    result = value.clone()
    result[..., -rope_dim:].copy_(rotated.reshape(rope_shape))
    return result

@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.elementwise", "fused_rope_inplace"
)
def dsv4_fused_rope_inplace_kunlun(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply the Kunlun GPT-J RoPE path with 0.5.14 semantics."""

    if q.shape[0] == 0:
        return

    freqs_real = _dsv4_cos_sin_cache(freqs_cis)
    rotary_dim = q.shape[-1]
    q_shape = q.shape
    q_input = q.contiguous().reshape(q_shape[0], -1, rotary_dim)

    k_shape = k.shape if k is not None else None
    k_input = (
        k.contiguous().reshape(k_shape[0], -1, rotary_dim)
        if k is not None
        else None
    )
    q_rotated, k_rotated = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions.flatten(),
        rotary_dim=rotary_dim,
        head_size=rotary_dim,
        cos_sin_cache=freqs_real,
        is_neox_style=False,
        query=q_input,
        key=k_input,
        offsets=None,
        inverse=inverse,
    )
    q.copy_(q_rotated.reshape(q_shape))
    if k is not None:
        k.copy_(k_rotated.reshape(k_shape))


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.elementwise",
    "fused_q_indexer_rope_hadamard_quant",
)
def dsv4_fused_q_indexer_rope_hadamard_quant_kunlun(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    """Compose RoPE, Hadamard and row-wise INT8 quantization in Torch."""

    import kunlun_ops

    q_wq_b = q_input
    q_rope = _dsv4_rotate_gptj_tail(q_wq_b, freqs_cis, positions)
    hidden_size = q_rope.shape[-1]
    hadamard_matrix = torch.empty(
        (hidden_size, hidden_size), dtype=q_rope.dtype, device=q_rope.device
    )
    kunlun_ops.gen_hadamard_matrix(hadamard_matrix, hidden_size ** -0.5)
    q_hadamard = torch.empty_like(q_rope)
    q_rope_2d = q_rope.view(-1, hidden_size) if q_rope.ndim > 2 else q_rope
    q_hadamard_2d = (
        q_hadamard.view(-1, hidden_size) if q_hadamard.ndim > 2 else q_hadamard
    )
    kunlun_ops.matmul(
        q_rope_2d, hadamard_matrix, q_hadamard_2d, False, True, 1.0, 0.0
    )
    _dsv4_probe(
        "q_fused",
        {"q_wq_b": q_wq_b, "q_rope": q_rope, "q_hadamard": q_hadamard},
        positions_shape=tuple(positions.shape),
    )
    q = q_hadamard
    q_shape = q.shape
    q_2d = q.contiguous().view(-1, q_shape[-1])
    q_int8 = torch.empty_like(q_2d, dtype=torch.int8)
    q_scale = torch.empty(
        (q_2d.shape[0], 1), dtype=torch.float32, device=q.device
    )
    kunlun_ops.quant2d(q_2d, q_int8, q_scale, force_sdnn=True)
    q_int8 = q_int8.view(q_shape)
    q_scale = q_scale.view(q_shape[0], -1)
    weights = dsv4_fused_scale_kunlun(weight, weight_scale, q_scale)
    _dsv4_probe(
        "q_fused_quant",
        {
            "q_int8": q_int8,
            "q_scale": q_scale,
            "weights_raw": weight,
            "weights_scaled": weights,
        },
        weight_scale=float(weight_scale),
    )
    return q_int8, weights


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.elementwise", "fused_q_norm_rope"
)

def dsv4_fused_q_norm_rope_kunlun(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Match the 0.5.14 fused full-head RMSNorm plus GPT-J RoPE contract."""

    if q_input.shape[0] == 0:
        return

    import kunlun_ops

    normalized = torch.empty_like(q_input)
    kunlun_ops.rmsnorm(
        q_input,
        None,
        normalized,
        eps,
        False,
        True,
        None,
        None,
        None,
    )
    if (
        _ENABLE_DSV4_ACCURACY_DUMPS
        and q_input.shape[0] == 8192
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and not getattr(dsv4_fused_q_norm_rope_kunlun, "_accuracy_dumped", False)
    ):
        torch._dsv4_accuracy_q_stages = {
            "q.wq_b_reshaped": q_input.detach().cpu(),
            "q.rmsnorm_output": normalized.detach().cpu(),
        }
        dsv4_fused_q_norm_rope_kunlun._accuracy_dumped = True

    q_output.copy_(normalized)
    rope_dim = freqs_cis.shape[-1] * 2
    rope = q_output[..., -rope_dim:]
    rope_input = rope.contiguous()
    rotated, _ = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions,
        rotary_dim=rope.shape[-1],
        head_size=rope.shape[-1],
        cos_sin_cache=_dsv4_cos_sin_cache(freqs_cis),
        is_neox_style=False,
        query=rope_input,
        key=None,
        offsets=None,
        inverse=False,
    )
    rope.copy_(rotated)
    if (
        _ENABLE_DSV4_ACCURACY_DUMPS
        and q_input.shape[0] == 8192
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and not getattr(dsv4_fused_q_norm_rope_kunlun, "_rope_dumped", False)
    ):
        stages = getattr(torch, "_dsv4_accuracy_q_stages", {})
        stages.update(
            {
                "q.rope_positions": positions.detach().cpu(),
                "q.rope_freqs": _dsv4_cos_sin_cache(freqs_cis).detach().cpu(),
                "q.rope_output": rope.detach().cpu(),
            }
        )
        torch._dsv4_accuracy_q_stages = stages
        dsv4_fused_q_norm_rope_kunlun._rope_dumped = True


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.elementwise", "fused_k_norm_rope_flashmla"
)
def dsv4_fused_k_norm_rope_flashmla_kunlun(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    """Use the normalize/rotate/store sequence for the 0.5.14 K path."""

    if kv.shape[0] == 0:
        return

    import kunlun_ops

    normalized = torch.empty_like(kv)
    kunlun_ops.rmsnorm(
        kv.reshape(-1, kv.shape[-1]),
        kv_weight,
        normalized.reshape(-1, normalized.shape[-1]),
        eps,
    )
    rotated = _dsv4_rotate_gptj_tail(normalized, freqs_cis, positions)
    max_valid_loc = kvcache.shape[0] * page_size - 1
    # A negative out_loc means "this row is not committed" (DSpark's verify commit
    # passes -1 for the rejected tail, routinely 5 of 6 rows). Those rows are
    # neutralised by writing zeros into the padded slot 0, which the paged
    # allocator reserves for dummy writes and never hands to a request
    # (allocator/paged.py::clear starts free pages at 1), so page 0 stays zero
    # just like before anything wrote to it.
    #
    # Selecting the committed rows instead -- ``.any()`` plus ``.nonzero()`` --
    # needs a host sync and a data-dependent output shape. Inside a CUDA graph
    # capture both read unsynchronized memory: the reported number of negatives
    # varies between calls of the same forward, so the capture either asks the
    # allocator for a garbage-sized buffer ("Tried to allocate more than 1EB")
    # or, worse, silently freezes an arbitrary row count into the graph and
    # replays it forever. Keep this path fixed-shape and sync-free.
    loc_flat = out_loc.reshape(-1)
    committed = (loc_flat >= 0).unsqueeze(-1)
    loc = loc_flat.contiguous().clamp(min=0, max=max_valid_loc)
    dump_layer0_k = (
        _ENABLE_DSV4_ACCURACY_DUMPS
        and kv.shape[0] == 8192
        and positions.numel() == 8192
        and int(positions[0]) == 8192
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and not getattr(dsv4_fused_k_norm_rope_flashmla_kunlun, "_dumped", False)
    )
    rotated_rows = rotated.reshape(rotated.shape[0], -1)
    if rotated_rows.dtype != kvcache.dtype:
        # ``set_k_and_s_v4_with_mapping`` copies 2-byte elements without
        # converting, so handing it bf16 rows for an fp16 cache
        # (--kv-cache-dtype fp16) reinterprets the bits: the sign survives and
        # the magnitude is scrambled. Convert explicitly.
        rotated_rows = rotated_rows.to(kvcache.dtype)
    rotated_rows = torch.where(
        committed,
        rotated_rows,
        torch.zeros((), dtype=rotated_rows.dtype, device=rotated_rows.device),
    )
    identity_key = (kvcache.device, max_valid_loc + 1)
    identity_mapping = _DSV4_IDENTITY_MAPPING_CACHE.get(identity_key)
    if identity_mapping is None:
        identity_mapping = torch.arange(
            max_valid_loc + 1, dtype=loc.dtype, device=loc.device
        )
        _DSV4_IDENTITY_MAPPING_CACHE[identity_key] = identity_mapping
    torch.ops.xspeedgate_ops.set_k_and_s_v4_with_mapping(
        kvcache,
        loc,
        identity_mapping,
        rotated_rows.contiguous(),
        page_size,
    )
    if dump_layer0_k:
        cache_rows = kvcache.view(-1, page_size, rotated_rows.shape[1]).reshape(
            -1, rotated_rows.shape[1]
        )[loc.long()]
        torch.save(
            {
                "kv.raw": kv.detach().cpu(),
                "kv.rmsnorm": normalized.detach().cpu(),
                "kv.rope": rotated.detach().cpu(),
                "positions": positions.detach().cpu(),
                "swa_loc": out_loc.detach().cpu(),
                "effective_loc": loc.detach().cpu(),
                "cache_rows": cache_rows.detach().cpu(),
                "page_size": page_size,
                "cache_shape": tuple(kvcache.shape),
            },
            f"/home/zx/debug_dumps/layer0_k_stages_0514_pid{os.getpid()}.pt",
        )
        dsv4_fused_k_norm_rope_flashmla_kunlun._dumped = True


@register_jit_op("sglang.kernels.ops.attention.rope", "apply_rope_with_cos_sin_cache_inplace")
def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: object | None = None,
) -> None:
    """Apply RoPE in-place using Kunlun's flashinfer-compatible op."""

    if fused_args is not None:
        raise NotImplementedError("fused_args is not supported on Kunlun RoPE")

    rope_dim = rope_dim or cos_sin_cache.shape[-1]
    head_size = q.shape[-1]
    q_rot, k_rot = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions.flatten(),
        rotary_dim=rope_dim,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache.to(q.dtype),
        is_neox_style=is_neox,
        query=q,
        key=k,
        offsets=None,
    )
    q.copy_(q_rot.reshape_as(q))
    k.copy_(k_rot.reshape_as(k))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "_get_k_triton_kernel",
)
def _get_k_triton_kernel(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    page_size: int,
    buf_numel_per_page: int,
    index_head_dim: int,
    *,
    BLOCK_SIZE: int,
) -> None:
    """_get_k_triton_kernel"""
    result = torch.ops.xspeedgate_ops.get_k_kernel(
        buf, page_indices, seq_len, page_size, buf_numel_per_page, index_head_dim
    )
    out.copy_(result.to(out.dtype))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "_get_s_triton_kernel",
)
def _get_s_triton_kernel(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    page_size: int,
    buf_numel_per_page: int,
    s_offset_in_page: int,
) -> None:
    """_get_s_triton_kernel"""
    result = torch.ops.xspeedgate_ops.get_s_kernel(
        buf, page_indices, seq_len, page_size, buf_numel_per_page, s_offset_in_page
    )
    out.copy_(result.to(out.dtype))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "_set_k_and_s_triton_kernel",
)
def _set_k_and_s_triton_kernel(
    buf_fp8: torch.Tensor,
    buf_fp32: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    index_k_ptr_stride_0: int,
    *,
    PAGE_SIZE: int,
    BUF_NUMEL_PER_PAGE: int,
    NUM_K_ELEMS_PER_TOKEN: int,
    S_OFFSET_NBYTES_IN_PAGE: int,
) -> None:
    """_set_k_and_s_triton_kernel"""
    from kunlun_ops import set_k_and_s_triton

    set_k_and_s_triton(
        buf=buf_fp8.view(torch.uint8).contiguous(),
        loc=loc.to(torch.int64),
        index_k=index_k.contiguous(),
        index_k_scale=index_k_scale.contiguous(),
        page_size=PAGE_SIZE,
    )


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.triton_kernel",
    "_act_quant_kernel",
)
def _act_quant_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    M: int,
    N: int,
    *,
    group_size: int,
    round_scale: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    **kwargs,
) -> None:
    """act_quant kernel replacement."""

    x_blocks = x.view(-1, group_size)
    scale = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    y.copy_((x_blocks / scale * 448.0).to(y.dtype).view_as(y))
    s.copy_(scale.view_as(s).to(s.dtype))


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
    "_per_token_quant_int8",
)
def _per_token_quant_int8(
    x: torch.Tensor,
    x_q: torch.Tensor,
    scales: torch.Tensor,
    x_sum: torch.Tensor | None,
    *,
    stride_x: int,
    stride_xq: int,
    N: int,
    CAL_SUM: bool,
    BLOCK: int,
    **kwargs,
) -> None:
    """per-token int8 quantization kernel replacement."""

    assert not CAL_SUM, "per_token_quant_int8: cal_sum is not supported on Kunlun."
    import kunlun_ops

    tmp_scales = torch.empty(x.shape[:-1], dtype=scales.dtype, device=x.device)
    kunlun_ops.quant2d(x, x_q, tmp_scales, force_sdnn=True)
    scales.copy_(tmp_scales.view_as(scales))
    if x_sum is not None:
        raise NotImplementedError("per_token_quant_int8: cal_sum is not supported on Kunlun.")


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
    "_per_token_group_quant_int8",
)
def _per_token_group_quant_int8(
    y: torch.Tensor,
    y_q: torch.Tensor,
    y_s: torch.Tensor,
    y_stride: int,
    N: int,
    eps: float,
    *,
    int8_min: int,
    int8_max: int,
    BLOCK: int,
    **kwargs,
) -> None:
    """per-token-group int8 quantization kernel replacement."""

    raise NotImplementedError("per_token_group_quant_int8 is not supported on Kunlun.")


@register_jit_op(
    "sglang.kernels.ops.quantization.int8_kernel",
    "sglang_per_token_group_quant_int8",
)
def sglang_per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    enable_v2: Optional[bool] = None,
):
    """sglang_per_token_group_quant_int8"""
    raise NotImplementedError("sglang_per_token_group_quant_int8 is not supported on Kunlun.")


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
    "_w8a8_block_int8_matmul",
)
def _w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_n: int,
    group_k: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_As_m: int,
    stride_As_k: int,
    stride_Bs_k: int,
    stride_Bs_n: int,
    *,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    **kwargs,
) -> None:
    """w8a8 block int8 matmul kernel replacement."""

    raise NotImplementedError("w8a8_block_int8_matmul is not supported on Kunlun.")
