"""Fine-grained Kunlun replacements for SGLang Triton/JIT kernel symbols."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)
_ENABLE_DSV4_ACCURACY_DUMPS = False

import xspeedgate_ops

# cocopod registers operators such as einsum_tgd_grd_tgr into the torch.ops.xspeedgate_ops
# namespace as an "umbrella". The xspeedgate_ops/__init__.py at +9228079 has `import cocopod`, but the one
# currently in use (+c9bac3b) does not, so import it explicitly here; otherwise dsv4_mqa_wo_a_einsum_kunlun would
# AttributeError: '_OpNamespace' 'xspeedgate_ops' object has no attribute 'einsum_tgd_grd_tgr'.
try:
    import cocopod  # noqa: F401  # side effect: register the operators under the xspeedgate_ops namespace
except ImportError:
    cocopod = None


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


def _op_trace_enabled() -> bool:
    """Migrated diagnostics are disabled; operator dispatch is unchanged."""
    return False


def _traced(spec: KernelSpec, impl: Callable) -> Callable:
    """Preserve the installer seam without wrapping or counting calls."""
    return impl


def op_trace_counts() -> Dict[KernelKey, int]:
    """Compatibility accessor after removal of diagnostic counters."""
    return {}


def note_op_path(name: str, variant: str) -> None:
    """Compatibility no-op: callers retain their existing dispatch decisions."""
    return None


def op_probe(op: str, stage: str, detail: str = "") -> None:
    """Compatibility no-op after removal of enter/exit probes."""
    return None


def op_probe_counts() -> Dict[str, int]:
    """Compatibility accessor after removal of diagnostic counters."""
    return {}


def op_probe_rank() -> int:
    """This process's global rank; 0 when ``torch.distributed`` is not up."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def op_probe_shape(
    op: str,
    stage: str,
    tensors: Mapping[str, object],
    scalars: Optional[Mapping[str, object]] = None,
    key=None,
) -> None:
    """Compatibility no-op after removal of shape probes."""
    return None


def dump_op_io(op: str, inputs=None, outputs=None, **scalars) -> None:
    """Compatibility no-op; no manifests or atexit writers are installed."""
    return None


def shape_dump_records() -> List[dict]:
    """Compatibility accessor after removal of diagnostic manifests."""
    return []


def install() -> None:
    """Install all registered Kunlun kernel replacements into upstream modules."""

    trace = _op_trace_enabled()
    for spec in _TRITON_OPS.values():
        replacement = (
            spec.impl
            if spec.metadata.get("call_style") == "direct"
            else KernelLauncher(spec.kernel_name, spec.impl)
        )
        _patch_symbol(spec, replacement)
    for spec in _JIT_OPS.values():
        _patch_symbol(spec, _traced(spec, spec.impl) if trace else spec.impl)
    _bind_dsv4_topk_explicitly(trace)


def _dspark_warn_fallback(op: str, reason: str) -> None:
    """Record fallback routes only when the existing operator trace is enabled."""
    note_op_path(op, f"torch ({reason})")


# ``dsv4/indexer.py`` binds the top-k transforms with ``from
# sglang.kernels.ops.attention.dsv4 import topk_transform_paged`` at its own
# import time, so whether ``_patch_symbol`` reaches that binding depends on
# whether the module is already in ``sys.modules`` when ``install()`` runs.
# When it is not, the alias keeps pointing at the upstream implementation:
# ``topk.py:113`` then dispatches XPU to ``torch.ops.sgl_kernel.topk_transform``
# and ``topk_transform_paged_v2`` JIT-builds a CUDA kernel, which dies on this
# host with ``fatal error: concepts`` (no C++20 host compiler). Bind the
# consumer module explicitly and report what the name actually resolves to, so
# "is the fused XPU op wired in?" is answerable from the log instead of by
# inference.
_DSV4_TOPK_BINDINGS = (
    ("topk_transform_paged", "dsv4_topk_transform_512_kunlun"),
    ("topk_transform_paged_v2", "dsv4_topk_transform_512_v2_kunlun"),
)


def _bind_dsv4_topk_explicitly(trace: bool) -> None:
    """Force ``dsv4.indexer``'s top-k aliases onto the Kunlun implementations."""

    try:
        import sglang.srt.layers.attention.dsv4.indexer as dsv4_indexer
    except Exception as exc:  # noqa: BLE001
        logger.warning("kernel_ops: dsv4.indexer unavailable, top-k not bound: %s", exc)
        return
    for name, impl_name in _DSV4_TOPK_BINDINGS:
        impl = globals().get(impl_name)
        if impl is None:
            logger.warning("kernel_ops: %s missing, %s left as-is", impl_name, name)
            continue
        spec = _JIT_OPS.get(("sglang.kernels.ops.attention.dsv4.topk", name))
        if spec is not None and trace:
            impl = _traced(spec, impl)
        setattr(dsv4_indexer, name, impl)


def _dsv4_plan_payload(plan):
    """Return the compressor-v1 payload expected by the Kunlun kernels."""

    if getattr(plan, "is_decode", False):
        return plan.seq_lens, None, 1
    return plan.compress_plan, plan.write_plan, 0


def _assert_v4_kv_layout(layout, where: str) -> None:
    """Reject V4.1's packed KV layouts.

    DSV4.1 attaches a `KVLayout` to every paged KV write path (V4=584B fp8+bf16, V41=528B,
    V41_FP4=288B). The Kunlun pool stores plain fp16/bf16 latent, corresponding to the upstream V4 slot; the other packed
    formats have no XSpeedGate write implementation, so we must fail explicitly instead of silently writing a layout the reader cannot parse
    -- this kind of mismatch shows up as NaN attention output rather than an error.
    """
    if layout is None:
        return
    name = getattr(layout, "value", layout)
    if name not in ("v4", "V4"):
        raise NotImplementedError(
            f"{where}: KV layout {name!r} has no Kunlun implementation; "
            "the half-cache pool only speaks the V4 layout"
        )
    if layout is None:
        return
    name = getattr(layout, "value", layout)
    if name not in ("v4", "V4"):
        raise NotImplementedError(
            f"{where}: KV layout {name!r} has no Kunlun implementation; "
            "the half-cache pool only speaks the V4 layout"
        )


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
    use_req_ring: bool = False,
    block: int = 128,
):
    """Create compressor-v1 paged indices using the Kunlun fused op."""

    del block
    if use_req_ring:
        raise NotImplementedError(
            "use_req_ring is not supported by the Kunlun compress plan builder"
        )
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


# Peak device bytes for one row tile of the reference top-k transform. The
# upstream helper clones the whole (rows, seq_len) FP32 score tile before the
# selection, so an unbounded call costs rows * seq_len * 4 * 2 bytes: under a
# breakable prefill graph captured at 8192 rows x 262144 columns that is an
# 8 GiB allocation and the scheduler dies with CUDA OOM. Row slices are
# independent, so the tile is bounded here the same way the C4 logits path
# bounds its own score tile.
_DEFAULT_TOPK_TILE_BYTES = 512 * 1024 * 1024


_TOPK_PAGE_CAP_WARNED: set = set()


def _warn_topk_page_cap(page_cap: int, width: int) -> None:
    """Warn once when the page table cannot cover the whole sequence (deduped by (page_cap, width)).

    A hit means the long context was truncated to page_cap tokens participating in top-k -- better than an out-of-range crash,
    but lossy in precision, so it must be visible from the logs.
    """
    key = (page_cap, width)
    if key in _TOPK_PAGE_CAP_WARNED:
        return
    _TOPK_PAGE_CAP_WARNED.add(key)
    logger.warning(
        "DSV4 topk fallback: page table covers %d tokens but score width is %d; "
        "clamping seq_lens to the page table (long-context rows get truncated)",
        page_cap,
        width,
    )


def _dsv4_topk_torch_fallback(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Row-chunked wrapper around ``topk_transform_pytorch_vectorized``.

    Writes land in ``out_*`` views, so a chunked call produces exactly the same
    result as one call over every row.
    """
    from sglang.srt.layers.attention.dsv4.indexer import (
        topk_transform_pytorch_vectorized,
    )

    rows = int(scores.shape[0])
    if rows == 0:
        return
    width = int(scores.shape[1])
    # Upstream `_topk_transform_vectorized` only clamps page_idx with clamp(min=0), without setting
    # an upper bound based on page_tables' column count: after selecting a position past the page table,
    # `torch.gather(page_tables, 1, page_idx)` reads out of bounds directly, which on Kunlun shows up as
    # gather_2d's "load/store operation exceed memory size" (status=700
    # illegal memory access) and kills the scheduler on the spot. During decode graph replay the captured
    # page table may be narrower than the live seq_lens, so clamp to the length the page table can actually address -- the same
    # protection as the seq_lens clamp already present in the fused `topk_transform` branch.
    _page_cap = int(page_tables.shape[1]) * int(page_size)
    if _page_cap < width:
        _warn_topk_page_cap(_page_cap, width)
    seq_lens = seq_lens.clamp(max=min(width, _page_cap))
    row_bytes = max(1, width * 4) * 2
    chunk = max(1, min(rows, _DEFAULT_TOPK_TILE_BYTES // row_bytes))
    if chunk >= rows:
        topk_transform_pytorch_vectorized(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
        )
        return
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        topk_transform_pytorch_vectorized(
            scores[lo:hi],
            seq_lens[lo:hi],
            page_tables[lo:hi],
            out_page_indices[lo:hi],
            page_size,
            None if out_raw_indices is None else out_raw_indices[lo:hi],
        )


_FUSED_TOPK_ENTRY_SEEN: set = set()


def _report_fused_topk_entry(out_raw_indices) -> None:
    """Report, once per outcome, that the fused XPU top-k was actually reached.

    Binding the symbol is not the same as executing it: the caller may live in
    a dispatch branch that never runs.  ``[DSV4 topk binding]`` answers "which
    function does the name point at"; this answers "was it called, and did it
    take the fused path or the ``out_raw_indices`` torch fallback".
    """

    key = out_raw_indices is None
    if key in _FUSED_TOPK_ENTRY_SEEN:
        return
    _FUSED_TOPK_ENTRY_SEEN.add(key)
    logger.warning(
        "DSV4 fused topk entered: out_raw_indices=%s path=%s",
        "None" if out_raw_indices is None else tuple(out_raw_indices.shape),
        "xspeedgate_fused" if out_raw_indices is None else "torch_fallback",
    )


_C4_LOGITS_CHUNK_BYTES_ENV = "SGLANG_KUNLUN_C4_LOGITS_CHUNK_BYTES"
# Peak device bytes allowed for one row tile of the reference C4 logits path.
# The (rows, kv_len, num_heads) FP32 score tile dominates, so an unbounded row
# count OOMs on long context; 512 MiB keeps the tile small while still giving
# the BMM enough rows to stay efficient.
_DEFAULT_C4_LOGITS_CHUNK_BYTES = 512 * 1024 * 1024


def _trunc_div(value: torch.Tensor, divisor: int) -> torch.Tensor:
    """Integer division that truncates toward zero, like Triton's ``//``."""
    return torch.div(value, divisor, rounding_mode="trunc")


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

        # Masked lanes have to contribute exactly nothing.  ``gather_rows``
        # clamps the -1 / out-of-range lanes onto cache slot 0, and slot 0 is
        # the dummy target that breakable-prefill-cuda-graph padding writes to
        # (``out_cache_loc`` pads with PaddingPolicy.ZERO), so it can hold NaN
        # -- graph capture runs the compressor on an all-zero batch and leaves
        # NaN in the C4 pool (measured: xknan=2068480, xkmax=nan).  The
        # ``masked_fill`` below only sanitises ``scores``; the value einsum
        # still consumes the raw gathered keys, where 0 * NaN = NaN poisons
        # every query row that owns an invalid lane (measured: a whole 1024-row
        # prefill chunk going NaN on all 8 CP ranks).  Zeroing the masked keys
        # keeps both einsums exact and is a no-op for finite caches.
        # Done in place with no mask: ``masked_fill_`` would broadcast the
        # (q, k, 1) mask up to ``keys``' (q, k, head_dim) and materialise it as
        # a full-size bool, which is ~1 GiB next to the ~4 GiB fp32 ``keys``
        # transient and OOMs the decode graph-capture pool at mem-fraction 0.72.
        torch.nan_to_num_(keys)
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


@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged")
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

    _report_fused_topk_entry(out_raw_indices)
    if out_raw_indices is not None:
        _dsv4_topk_torch_fallback(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
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
    # The vendor kernel scans ``seq_lens[row]`` columns of ``scores``.  Under the
    # breakable prefill CUDA graph the logits buffer is sized from the captured
    # contract while ``c4_seq_lens`` is the live metadata, so the live value can
    # exceed the buffer and the kernel reads out of bounds (observed
    # score_stride=128 vs seq_len=304, and garbage 6.3e8 during capture) ->
    # topk_transform_decode_kernel illegal access.  Bound it unconditionally so
    # the recorded graph always carries the clamp.
    seq_lens = seq_lens.clamp(max=int(scores.shape[1]))
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


@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged_v2")
def dsv4_topk_transform_512_v2_kunlun(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Run the Kunlun top-k transform while retaining the v2 API contract.

    ``out_raw_indices`` is the v2 kernel's third output mode (page-table
    transform *and* the raw selected indices). The DSV4 indexer passes it
    **positionally** as the 7th argument (``indexer.py`` extend and decode
    paths), so omitting it here is a TypeError on the first indexer call, not a
    silently dropped option. The Kunlun entry point already implements the mode:
    it routes to the torch fallback, because the XPU fused kernel emits only the
    paged indices.
    """

    del metadata
    dsv4_topk_transform_512_kunlun(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
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
    """Bridge the 0.5.19 CP-v2 length contract.

    Since 0.5.19 ``raw_out_loc`` is an **unpadded** array with "one row per logical token"
    (``init_compression_metadata`` asserts ``raw_out_loc.shape[0] == num_tokens``),
    while ``seq_lens/positions/page_table`` are per-rank-partition-padded metadata lengths;
    the vendor op requires the two to be equal. Here we pad ``raw_out_loc`` up to the metadata length before calling the op --
    the padded rows are filled with 0 (the dummy slot the SWA pool reserves, matching 0.5.17's pad rows) -- then trim the two
    out_loc outputs back to ``raw_out_loc``'s original row count: ``c4_out_loc``/``c128_out_loc``
    belong to core's ``_CP_GLOBAL_FIELDS`` (unpadded), while the remaining per-row outputs keep the padded length,
    aligned with ``seq_lens_casual`` (``_CP_REINDEX_FIELDS``).
    """
    meta_len = int(seq_lens.shape[0])
    loc_len = int(raw_out_loc.shape[0])
    if loc_len < meta_len:
        raw_out_loc = torch.cat(
            [raw_out_loc, raw_out_loc.new_zeros(meta_len - loc_len)]
        )
    outs = torch.ops.xspeedgate_ops.init_compressed_attn_metadata_v2(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )
    if loc_len < meta_len:
        outs = list(outs)
        outs[0] = outs[0][:loc_len]   # c4_out_loc
        outs[4] = outs[4][:loc_len]   # c128_out_loc
        outs = tuple(outs)
    return outs
    



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

    # ``seq_lens_cpu`` cannot be trusted under CUDA-graph replay: the decode graph
    # runner fills ``buffers.seq_lens_cpu`` once with ``seq_len_fill_value`` and only
    # the *eagle* draft runners copy the live lengths into it per replay, so a DSPARK
    # TARGET_VERIFY replay hands us the capture-time placeholder. The vendor op reads
    # the host list, so ``seq_lens_casual`` -- and therefore ``c4_topk_lengths_raw``
    # and every sparse page selection downstream -- came out of uninitialised memory
    # (the C4 probe logged compressed lens of 570687470 on a 25K prompt). The device
    # tensor *is* refreshed before replay, so always derive the host view from it.
    #
    # The device buffer is only refreshed for the *live* rows, though. On the DSPARK
    # draft-block path the graph's ``seq_lens`` tail keeps whatever the capture left
    # there, and those entries can exceed int32 -- the vendor op then raises
    # "expand_prefill_casually: seq_lens values must be in int32 range" and takes all
    # eight schedulers down (observed 2026-10-31 03:34:53 on 144). Sanitise per row:
    # keep a plausible device length, otherwise fall back to the host mirror entry,
    # which is exactly the value this path used before deriving from the device.
    if seq_lens_cpu is None or len(seq_lens_cpu) == seq_lens.shape[0]:
        _live_lens = seq_lens.to("cpu", non_blocking=False).tolist()
        _host_lens = list(seq_lens_cpu) if seq_lens_cpu is not None else None
        _int32_max = 2**31 - 1

        def _plausible(value) -> bool:
            return isinstance(value, int) and 0 <= value <= _int32_max

        _clean_lens = []
        _repaired = []
        for _row, _value in enumerate(_live_lens):
            if _plausible(_value):
                _clean_lens.append(_value)
                continue
            _fallback = (
                _host_lens[_row]
                if _host_lens is not None and _plausible(_host_lens[_row])
                else 0
            )
            _clean_lens.append(_fallback)
            _repaired.append((_row, _value, _fallback))
        if _repaired and not getattr(
            dsv4_expand_prefill_causally_kunlun, "_oob_logged", False
        ):
            dsv4_expand_prefill_causally_kunlun._oob_logged = True
            _KERNEL_OPS_LOGGER.warning(
                "[DSV4] out-of-int32 seq_lens rows repaired: rows=%s "
                "(row, device, used) device_len=%s host=%s extend=%s",
                _repaired[:8],
                len(_live_lens),
                None if _host_lens is None else _host_lens[:8],
                extend_seq_lens.shape[0],
            )
        if (
            _host_lens is not None
            and _host_lens != _clean_lens
            and not getattr(dsv4_expand_prefill_causally_kunlun, "_stale_logged", False)
        ):
            dsv4_expand_prefill_causally_kunlun._stale_logged = True
            _KERNEL_OPS_LOGGER.warning(
                "[DSV4] stale seq_lens_cpu mirror replaced: host=%s device=%s",
                _host_lens[:8],
                _clean_lens[:8],
            )
        seq_lens_cpu = _clean_lens

    if extend_seq_lens_cpu is None or len(extend_seq_lens_cpu) == extend_seq_lens.shape[0]:
        extend_seq_lens_cpu = extend_seq_lens.to("cpu", non_blocking=False).tolist()

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
    use_req_ring: bool = False,
):
    """dsv4_compressor_decode_plan_kunlun"""
    from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    # Consistent with upstream's own XPU branch: the request-ring plan layout (unified KV, ratio 4) has no
    # XSpeedGate implementation. The Kunlun pool is not `_unified_kv`, so callers always pass False.
    if use_req_ring:
        raise NotImplementedError(
            "use_req_ring is not supported by the Kunlun compress plan builder"
        )
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


#: The slack the c4/c128 state ring reserves for "not-yet-committed speculative positions". XSpeedGate's
#: ``plan_compress_prefill_v2`` hardcodes it as ``min(ring_size - cr, 4)`` (matching the old upstream
#: ``c_plan.cuh`` ``kMaxMTPDraftTokens = 4``), while DSPARK's verify window is 6,
#: so the last few query positions' state does not land in the ring and the next step reads someone else's history -- the same request sent
#: multiple times yields inconsistent results. The upstream H20-side fix changes the constant to 8; the op signature does not expose mtp_pad, and there is no local
#: xtdk toolchain rebuild, so here we pad out, on the framework side, the few plan_w rows the op omits (pad=8).
#: 0 (default) = derive from the ring geometry, see ``_dsv4_compress_state_write_pad``; non-zero = force override,
#: for re-verification only. Previously this was hardcoded to 8: it covers the current 6-wide verify window, but is inconsistent with upstream semantics,
#: and it would hit the same pitfall again as soon as the draft token count is raised.
_DSV4_MTP_PAD_OVERRIDE = int(os.environ.get("DSV4_MTP_PAD", "0"))

def _dsv4_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """The maximum number of draft tokens the state ring can serve, i.e. the plan's ``mtp_pad``.

    Consistent with community commit a1fe4e3: ``ring_size > window_size ? ring_size - window_size + 2 : 0``
    (the mtp_pad in ``c_plan.cuh``, and ``deepseek_v4_memory_pool.get_compress_state_write_pad``).
    The v0.5.17 kernel still uses ``min(ring_size - cr, kMaxMTPDraftTokens=4)``; 4 is less than
    DSPARK's 6-token verify window, so the tail positions cannot enter the ring and the next step reads the previous occupant's
    state -- the same request sent multiple times yields inconsistent results. Under speculative config this formula gives c4=10, c128=130.

    Once we upgrade to an sglang carrying this fix / rebuild the ops, delete this function together with
    ``_dsv4_augment_prefill_plan_w``.
    """

    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    pad = ring_size - window_size + 2 if ring_size > window_size else 0
    if _DSV4_MTP_PAD_OVERRIDE:
        pad = _DSV4_MTP_PAD_OVERRIDE
    return pad


def _dsv4_num_draft_tokens() -> int | None:
    """The speculative draft token count; return None when unavailable (do not block startup)."""

    try:
        from sglang_kunlun.compat import server_arg

        return server_arg("speculative_num_draft_tokens")
    except Exception:  # pragma: no cover - only used for logging and assertions
        return None


#: Print the compress plan's row count / valid row count (diagnose the fixed-shape plan under BCG).
_DEBUG_COMPRESS_PLAN = os.environ.get("SGLANG_KUNLUN_DEBUG_COMPRESS_PLAN") == "1"


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
    """Fill in the missing plan_w rows for a larger ``mtp_pad``.

    The op uses ``first_w_pos = min(last_c_pos - (overlap ? cr : 0), seq_len - vendor_pad)``,
    and we want the same formula with ``new_pad``. The two differ only in the small ``[fw_new, fw_old)`` range,
    at most ``new_pad - vendor_pad`` rows per request, so just append rows instead of recomputing the whole table.

    plan_w is ``[num_q_tokens, 8] uint8`` = ``(uint32 ragged_id, int32 write_loc)``,
    with valid rows first and ``(-1, -1)`` sentinels filling the tail; the valid row count is bounded by ``sum(extend_lens) <= num_q_tokens``,
    and each position produces at most one row, so the appended rows are guaranteed to fit in the sentinel region. The whole path uses only tensor ops (no D2H),
    so it can be captured by the CUDA graph.
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
        # The holes in r2t / f2s are negative; these positions have no state yet, and writing them would corrupt someone else's row
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
    # Pack valid rows to the front in original order; dump all invalid rows into the last row (guaranteed beyond num_rows, and necessarily free at this point)
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
    use_req_ring: bool = False,
):
    """dsv4_compressor_prefill_plan_kunlun"""
    from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan

    try:
        from sglang_kunlun.hooks.layers.cp.bcg import _compressor_runs_eager
    except ImportError:
        # The prefill-CP BCG hooks are only installed on nodes that run the
        # breakable prefill graph; the decode node has no cp package, and
        # without the eager compressor break the compact-plan fallback below
        # must stay off.
        def _compressor_runs_eager():
            return False

    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio: {compress_ratio}")
    if use_req_ring:
        raise NotImplementedError(
            "use_req_ring is not supported by the Kunlun compress plan builder"
        )
    if num_q_tokens < req_pool_indices.shape[0]:
        raise ValueError("num_q_tokens must be at least the batch size")
    if use_cuda_graph and _compressor_runs_eager():
        # The compressor is already an eager breakpoint (the forward_core_compressor hook in ``cp/bcg.py``),
        # and the op's workload equals the plan's row count directly: a fixed-shape plan would pad to ``num_q_tokens`` rows,
        # and under c128 every row still runs a full compression -- measured 20 calls at 387ms/step (compact plan 5ms).
        # Nothing captured by the graph reads the plan after this breakpoint, so falling back to a compact plan here is safe.
        use_cuda_graph = False
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
    if _DEBUG_COMPRESS_PLAN:
        raw = plan_c.view(torch.int32)
        logger.info(
            "compress plan(base): cr=%d use_cuda_graph=%s num_q=%d num_reqs=%d "
            "plan_c=%s valid_rows=%d plan_w=%s",
            compress_ratio,
            use_cuda_graph,
            num_q_tokens,
            req_pool_indices.shape[0],
            tuple(plan_c.shape),
            int((raw[:, 0] != -1).sum().item()),
            tuple(plan_w.shape),
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
    if _DEBUG_COMPRESS_PLAN:
        raw = plan_c.view(torch.int32)
        logger.info(
            "compress op: cr=%d is_decode=%s plan_c=%s valid_rows=%d out=%s kv_in=%s",
            compress_ratio,
            plan.is_decode,
            tuple(plan_c.shape),
            int((raw[:, 0] != -1).sum().item()),
            tuple(out.shape),
            tuple(kv_score_input.shape),
        )
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
    # 4.1 added these three for the HIP FP4 writer; the XPU branch of upstream
    # `compress_norm_rope_store` does not consume them either.
    kvcache_scale: Optional[torch.Tensor] = None,
    rope_cache=None,
    fp4_k_write_metadata=None,
    # 4.1 attaches a layout marker to every paged KV write; the Kunlun pool is a plain fp16/bf16 half cache,
    # corresponding to the upstream V4 layout.
    layout=None,
    # 0.5.22 adds fp8_2buff / kvcache_rope: unified-KV FP8 splits the latent into
    # two buffers, "nope fp8 payload + a separate rope pool" (only when `is_unified_kv_fp8()`
    # does compressor_v2 pass kv_cache_rope in and take module.forward_fp8_2buff).
    # XSpeedGate's dsv4_compress_norm_rope_store_v2 only writes a single plain half cache,
    # with no write implementation for the second rope pool, so here we fail explicitly instead of dropping the argument -- silently ignoring
    # it would write rope into the nope pool, showing up as NaN attention output.
    fp8_2buff: bool = False,
    kvcache_rope: Optional[torch.Tensor] = None,
) -> None:
    """Apply v2 post-compression transforms and store into Kunlun cache pages."""

    if use_fp4:
        raise NotImplementedError("FP4 indexer is not enabled for the Torch reference")
    # 0.5.22: honour fp8_2buff / kvcache_rope by refusing them explicitly.
    if fp8_2buff:
        raise NotImplementedError(
            "compress_norm_rope_store: the 0.5.22 fp8 two-pool store (fp8_2buff) "
            "has no Kunlun implementation; the XSpeedGate writer only fills a "
            "single plain half-precision latent cache"
        )
    if kvcache_rope is not None:
        raise NotImplementedError(
            "compress_norm_rope_store: the 0.5.22 separate rope pool "
            "(kvcache_rope) has no Kunlun implementation; it is only consumed by "
            "the fp8 two-pool store"
        )
    _assert_v4_kv_layout(layout, "compress_norm_rope_store")
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
    # The vendor kernel indexes the mapping with ``raw_loc`` directly, so an
    # out-of-range location is an out-of-bounds device read, not an error return:
    # on 2026-10-31 all eight 144 devices took
    # ``KL_XID_KERNEL_EXCEPTION`` inside
    # ``set_k_and_s_v4_with_mapping_kernel<int, long>`` (the ``<int, long>``
    # instantiation is this call site -- the other one passes an int32 identity
    # mapping) and the schedulers died with ``cudaErrorLaunchTimeout``.
    #
    # DSpark's verify commit passes -1 for the rejected tail (routinely 5 of 6
    # rows) and a CUDA-graph replay can leave capture-time sentinels in the
    # padded rows, so clamp into the mapping's index range exactly as
    # ``dsv4_set_k_and_s_kunlun`` already does for the non-mapping variant.
    # Location 0 is the paged allocator's reserved dummy page (free pages start
    # at 1), so the neutralised rows land somewhere no request ever reads; there
    # is no need to also zero the payload. Pure device ops, fixed shape, no host
    # sync -- safe inside a graph capture.
    if raw_loc.numel() and full_to_swa_index_mapping.numel():
        raw_loc = raw_loc.clamp(
            min=0, max=int(full_to_swa_index_mapping.shape[0]) - 1
        )
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
    # The op internally derives stride(0)/ceil(max_seq_len/page_size), which must match the arguments
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


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels",
    "build_causal_swa_page_indices_triton",
)
def build_causal_swa_page_indices_kunlun(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
    swa_replay_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the causal SWA window page-index table on the XSpeedGate kernel.

    Replaces the whole launcher (``dsv4_attn_metadata_kernels.py:544``) instead
    of just ``_causal_swa_page_indices_kernel``: the op derives
    ``padded_width`` from the same ceil-div and allocates its own
    ``[num_qo, padded_width]`` int32 output, so the upstream ``torch.empty`` and
    the copy back into it both disappear.

    Semantics match ``build_causal_swa_page_indices`` (the torch baseline)
    element for element, including the ``full < 0 -> -1`` slots: the op checks
    the sign explicitly, where the torch path only lands on ``-1`` because
    ``full_to_swa_mapping[-1]`` is the sentinel and the Triton path reads out of
    bounds there.

    ``full_to_swa_mapping`` is int64 upstream (the SWA allocator builds it that
    way) and the op dispatches on int32, so it is cast per call -- same contract
    as the DSpark sibling ``swa_page_indices_kernel`` above.

    Args:
        req_to_token: Request-to-full-cache token mapping.
        full_to_swa_mapping: Full-cache slot to SWA slot mapping.
        req_pool_indices_repeated: Request pool index for each query row.
        seq_lens_casual: Causal sequence length for each query row.
        swa_window: Maximum number of tokens in each sliding window.
        page_index_aligned_size: Alignment used to pad the output width.
        swa_replay_start: Optional per-row lower bound for replay windows.

    Returns:
        An int32 SWA page-index table with invalid slots set to -1.
    """

    op = "causal_swa_page_indices_kernel"
    _dspark_require(
        req_to_token.dim() == 2
        and req_to_token.dtype == torch.int32
        and req_to_token.stride(1) == 1,
        op,
        f"req_to_token must be 2D int32 with row-contiguous rows, got "
        f"dim={req_to_token.dim()} dtype={req_to_token.dtype} "
        f"strides={req_to_token.stride()}",
    )
    _dspark_require(
        req_pool_indices_repeated.shape[0] == seq_lens_casual.shape[0],
        op,
        f"req_pool_indices_repeated length {req_pool_indices_repeated.shape[0]} "
        f"!= seq_lens_casual length {seq_lens_casual.shape[0]}",
    )
    _dspark_require(
        swa_window > 0 and page_index_aligned_size > 0,
        op,
        f"swa_window={swa_window} / "
        f"page_index_aligned_size={page_index_aligned_size} must be > 0",
    )

    # int32/int64 are the two pool-index template instances; anything else
    # (upstream builds this from req_pool_indices) is normalized to int32.
    if req_pool_indices_repeated.dtype not in (torch.int32, torch.int64):
        req_pool_indices_repeated = req_pool_indices_repeated.to(torch.int32)

    out = torch.ops.xspeedgate_ops.causal_swa_page_indices_kernel(
        req_to_token,
        full_to_swa_mapping.to(torch.int32).contiguous(),
        req_pool_indices_repeated.contiguous(),
        seq_lens_casual.to(torch.int32).contiguous(),
        swa_window,
        page_index_aligned_size,
        None
        if swa_replay_start is None
        else swa_replay_start.to(torch.int32).contiguous(),
    )
    return out


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.attn",
    "get_paged_mqa_logits_metadata",
)
def get_paged_mqa_logits_metadata_kunlun(
    seq_lens: torch.Tensor, page_size: int, num_sm: int
) -> torch.Tensor:
    """Build the paged-MQA-logits SM schedule on the XSpeedGate kernel.

    Replaces the whole launcher (``dsv4/attn.py:56``): the op flattens and casts
    ``seq_lens`` itself, allocates the ``[num_sm + 1, 2]`` metadata, and sizes
    its own multi-block workspace with the same ``kMBTileSize = 4096`` split, so
    the upstream ``new_empty`` pair and the ``bs > 2048`` branch disappear.

    ``page_size`` is passed through only to keep the upstream 64/128 assert --
    the schedule is a 256-token split of the sequence lengths and does not
    depend on it.

    Args:
        seq_lens: Sequence lengths to partition into scheduling work.
        page_size: KV page size, restricted to 64 or 128.
        num_sm: Number of workers for the paged-MQA schedule.

    Returns:
        Scheduling metadata with shape [num_sm + 1, 2].
    """

    op = "get_paged_mqa_logits_metadata"
    _dspark_require(
        page_size in (64, 128), op, f"page_size must be 64 or 128, got {page_size}"
    )
    _dspark_require(1 <= num_sm <= 1024, op, f"num_sm must be in [1, 1024], got {num_sm}")

    metadata = torch.ops.xspeedgate_ops.get_paged_mqa_logits_metadata(
        seq_lens, page_size, num_sm
    )
    return metadata


@register_jit_op(
    "sglang.kernels.ops.moe.moe_align_small_numel",
    "moe_align_small_numel",
)
def moe_align_small_numel_kunlun(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Single-launch MoE align for tiny batches on the XSpeedGate kernel.

    Replaces the launcher (``moe/moe_align_small_numel.py:114``). The op writes
    the three buffers in place under the same "+1 offset" convention
    (``num_experts`` is ``E + 1``, bucket 0 is the EP-filtered ``-1`` expert),
    and like the triton kernel it leaves ``sorted_token_ids`` beyond
    ``num_tokens_post_pad`` unwritten.

    Intra-bucket order may differ from the triton kernel's pair-index order --
    both are valid permutations within a bucket and ``fused_moe`` only reads the
    block-padded layout, which is identical.

    The op caps ``numel`` at 1024 (the kernel's pairwise tensors), above the
    upstream ``SMALL_NUMEL_LIMIT = 64`` gate, so every call the call site routes
    here fits.

    Args:
        topk_ids: Expert assignments for the token/top-k pairs.
        num_experts: Expert bucket count including the filtered-expert bucket.
        block_size: Alignment for each expert's token count.
        sorted_token_ids: Output buffer receiving the padded token ordering.
        expert_ids: Output buffer receiving the expert for each block.
        num_tokens_post_pad: Output buffer receiving the total padded token count.
    """

    op = "moe_align_small_numel"
    _dspark_require(
        topk_ids.dtype == torch.int32,
        op,
        f"topk_ids must be int32, got {topk_ids.dtype}",
    )
    _dspark_require(
        sorted_token_ids.dtype == torch.int32
        and expert_ids.dtype == torch.int32
        and num_tokens_post_pad.dtype == torch.int32,
        op,
        "sorted_token_ids/expert_ids/num_tokens_post_pad must be int32, got "
        f"{sorted_token_ids.dtype}/{expert_ids.dtype}/{num_tokens_post_pad.dtype}",
    )
    _dspark_require(
        topk_ids.numel() <= 1024,
        op,
        f"numel {topk_ids.numel()} exceeds the kernel's 1024 capacity",
    )

    torch.ops.xspeedgate_ops.moe_align_small_numel(
        topk_ids.contiguous(),
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


@register_jit_op(
    "sglang.kernels.ops.embeddings.vocab_parallel_embedding",
    "vocab_parallel_embedding",
)
def vocab_parallel_embedding_kunlun(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    """Fused vocab-parallel embedding lookup on the XSpeedGate kernel.

    Replaces the launcher (``embeddings/vocab_parallel_embedding.py:56``): the
    op allocates the ``input_.shape + [hidden]`` output itself and zeroes the
    rows whose token falls outside this shard's two vocab windows, matching the
    upstream masked-load semantics.

    Both int32/int64 ids and fp32/fp16/bf16 weights are native template
    instances, so nothing is cast here.

    Args:
        input_: Token IDs to look up.
        weight: This shard's embedding table.
        org_vocab_start_index: Inclusive start of the original vocabulary shard.
        org_vocab_end_index: Exclusive end of the original vocabulary shard.
        num_org_vocab_padding: Padding between original and added vocabulary rows.
        added_vocab_start_index: Inclusive start of the added vocabulary shard.
        added_vocab_end_index: Exclusive end of the added vocabulary shard.

    Returns:
        Embeddings with out-of-shard token rows zeroed.
    """

    op = "vocab_parallel_embedding"
    _dspark_require(
        input_.dtype in (torch.int32, torch.int64),
        op,
        f"input must be int32 or int64, got {input_.dtype}",
    )
    _dspark_require(
        weight.dim() == 2 and weight.stride(1) == 1,
        op,
        f"weight must be 2D with contiguous rows, got dim={weight.dim()} "
        f"strides={weight.stride()}",
    )

    return torch.ops.xspeedgate_ops.vocab_parallel_embedding(
        input_.contiguous(),
        weight.contiguous(),
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )


# 0.5.22 folds Q RoPE into fused_k_norm_rope_flashmla; no deleted-module
# registration. Keep this callable for the Kunlun fused boundary.
def q_rope_store_kunlun(
    q: torch.Tensor,
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Rotate Q's 64-wide tail into ``output`` on the XSpeedGate kernel.

    Replaces the launcher (``dsv4/q_rope_store.py:69``), which picks between a
    decode and a prefill triton kernel by shape; the op covers both from one
    entry point.

    ``output`` keeps whatever Q padding the call site allocated: only lanes
    ``[448, 512)`` are rotated, lanes ``[0, 448)`` are copied verbatim, and rows
    /heads outside the passed views are never touched -- verified against the
    torch reference on the strided ``heads < padded_heads`` layout the real call
    site uses.

    Args:
        q: Query tensor with 512 lanes per head.
        output: Destination matching q's shape and dtype, updated in place.
        freqs_cis: Contiguous complex64 rotary-frequency table.
        positions: Position index for each query row.
    """

    op = "q_rope_store"
    _dspark_require(
        q.dim() == 3 and q.shape[2] == 512,
        op,
        f"q must be [rows, heads, 512], got {tuple(q.shape)}",
    )
    _dspark_require(
        q.shape == output.shape and q.dtype == output.dtype,
        op,
        f"output must match q shape/dtype, got {tuple(output.shape)}/{output.dtype} "
        f"vs {tuple(q.shape)}/{q.dtype}",
    )
    _dspark_require(
        q.stride(1) == 512 and output.stride(1) == 512,
        op,
        f"q/output head stride must be 512, got {q.stride(1)}/{output.stride(1)}",
    )
    _dspark_require(
        freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous(),
        op,
        f"freqs_cis must be contiguous complex64, got {freqs_cis.dtype}",
    )
    _dspark_require(
        positions.dtype in (torch.int32, torch.int64),
        op,
        f"positions must be int32 or int64, got {positions.dtype}",
    )

    torch.ops.xspeedgate_ops.q_rope_store(
        q, output, freqs_cis, positions.contiguous()
    )


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


# Consistent with DFLASH_CHUNK in `compute_dflash_correct_drafts_and_bonus.cpp`: a single core must hold
# one input row at a time, so block_size has an upper bound; bs is covered by the kernel's grid-stride loop and has no upper bound.
_DFLASH_CORRECT_DRAFTS_MAX_BLOCK = 320
_DFLASH_CORRECT_DRAFTS_DTYPES = (torch.int32, torch.int64)


def _dflash_correct_drafts_supported(
    candidates: torch.Tensor, target_predict: torch.Tensor, block_size: int
) -> Optional[str]:
    """Return why ``compute_dflash_correct_drafts_and_bonus`` cannot run, else None.

    Mirrors the TORCH_CHECKs in
    ``XSpeedGate/wrapper/src/attention/compute_dflash_correct_drafts_and_bonus.cpp``.
    The ``hasattr`` comes first because the op is newer than the deployed wheel
    (``xspeedgate_ops-1.6.0`` does not export it), and without the guard a DFlash
    run would die with ``AttributeError: '_OpNamespace' ...`` instead of keeping
    the torch path. Same guard style as ``_engram_gate_v2_usable``.
    """

    if not hasattr(torch.ops.xspeedgate_ops, "compute_dflash_correct_drafts_and_bonus"):
        return "xspeedgate_ops has no compute_dflash_correct_drafts_and_bonus"
    if candidates.dtype is not target_predict.dtype:
        return f"dtypes differ: {candidates.dtype} vs {target_predict.dtype}"
    if candidates.dtype not in _DFLASH_CORRECT_DRAFTS_DTYPES:
        return f"dtype {candidates.dtype} is not int32/int64"
    if not (candidates.is_contiguous() and target_predict.is_contiguous()):
        return "candidates/target_predict must be contiguous"
    if candidates.device != target_predict.device:
        return "candidates/target_predict on different devices"
    if not 1 < block_size <= _DFLASH_CORRECT_DRAFTS_MAX_BLOCK:
        return (
            f"block_size={block_size} outside the per-core LM chunk "
            f"(1, {_DFLASH_CORRECT_DRAFTS_MAX_BLOCK}]"
        )
    return None


@register_triton_op(
    "sglang.srt.speculative.dflash_utils",
    "_fused_correct_drafts_and_bonus_kernel",
)
def fused_correct_drafts_and_bonus_kernel(
    candidates_ptr: torch.Tensor,
    target_predict_ptr: torch.Tensor,
    num_correct_drafts_ptr: torch.Tensor,
    bonus_tokens_ptr: torch.Tensor,
    block_size: int,
    BLOCK: int = 0,
) -> None:
    """Greedy DFlash/DSpark accept lengths + bonus tokens.

    0.5.19 added a Triton fast path to
    ``dflash_utils.compute_dflash_correct_drafts_and_bonus`` that is taken for
    any ``candidates.is_cuda`` tensor, which includes Kunlun XPU (dispatch key
    ``cuda``); Triton has no Kunlun backend, so the launch fails with
    ``0 compatible backends for target (cuda)``. The XSpeedGate op of the same
    name implements the kernel contract directly; the torch reduction below is
    kept for the inputs it does not cover (wide Domino blocks, older wheels) and
    is the same rule the upstream CPU branch uses: accept while
    ``candidates[:, 1:] == target_predict[:, :-1]`` consecutively (leading match
    count, capped at ``block_size - 1``), then read the bonus token at that index.
    """
    candidates = candidates_ptr.reshape(-1, block_size)
    target_predict = target_predict_ptr.reshape(-1, block_size)
    bs = candidates.shape[0]
    if bs == 0:
        return

    op = "compute_dflash_correct_drafts_and_bonus"
    # Record the full kernel inputs (candidates/target_predict/two outputs + block_size), dedupe by shape
    # signature and log only on rank 0; `path` separates the xspeedgate and torch-fallback routes.
    probe_tensors = {
        "candidates": candidates,
        "target_predict": target_predict,
        "num_correct_drafts": num_correct_drafts_ptr,
        "bonus_tokens": bonus_tokens_ptr,
    }
    reason = _dflash_correct_drafts_supported(candidates, target_predict, block_size)
    if reason is None:
        correct_len, bonus = (
            torch.ops.xspeedgate_ops.compute_dflash_correct_drafts_and_bonus(
                candidates, target_predict
            )
        )
    else:
        _dspark_warn_fallback(op, reason)
        matches = candidates[:, 1:] == target_predict[:, :-1]
        correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        rows = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[rows, correct_len.to(torch.int64)]

    num_correct_drafts_ptr.reshape(-1)[:bs].copy_(
        correct_len.to(num_correct_drafts_ptr.dtype)
    )
    bonus_tokens_ptr.reshape(-1)[:bs].copy_(bonus.to(bonus_tokens_ptr.dtype))


@register_jit_op(
    "sglang.kernels.ops.speculative.row_argmax",
    "row_argmax",
)
def row_argmax_kunlun(x: torch.Tensor) -> torch.Tensor:
    """``x.argmax(dim=-1)`` for the DSpark verify logits, without Triton.

    ``dspark_accept._row_argmax`` opts into ``row_argmax`` for every DSV4.1
    target (``dspark_verify.py:171`` passes ``fused_argmax=True``) as soon as the
    logits are ``is_cuda`` -- which Kunlun XPU is, dispatch key ``cuda``. The
    upstream implementation is a two-stage split reduction written as two
    ``@triton.jit`` kernels, and it is the only thing left on the DSpark accept
    path that launches Triton directly instead of going through a kernel_ops
    replacement.

    That launch cannot work here. ``xpu_fla/python/triton_mock.py`` turns
    ``triton.jit`` into a no-op whenever the real Triton backend fails to
    initialise, so the decorated kernels stay plain Python functions and
    ``kernel[grid](...)`` raises ``'function' object is not subscriptable`` while
    the decode cuda graph is captured -- all 8 ranks die inside
    ``init_cuda_graphs`` and the server never comes up.

    Keeping the genuine ``triton.jit`` is not the alternative it looks like. The
    probe only fails because ``launch_dsv41_int4.sh`` drops ``/usr/local/xre/so``
    (the sole provider of ``libcuda.so.1``) to dodge the libxpurt ABI mismatch;
    put the library back and Triton does compile, but the driver then rejects the
    sm86 binary at load time with ``CUDA_ERROR_NOT_SUPPORTED``. There is no
    configuration of this container in which a Triton kernel runs on XPU.

    ``torch.argmax`` is what upstream itself uses whenever ``fused_argmax`` is
    off, and it is the contract the fused kernel approximates -- the fused one
    deviates only in NaN selection (``dspark_accept.py:670-672``), so this is if
    anything the stricter of the two.
    """

    return torch.argmax(x, dim=-1)


@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.commit_swa",
    "committed_swa_locations",
)
def committed_swa_locations_kunlun(
    cache_loc: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """XSpeedGate ``committed_swa_locations``: committed full slots -> draft SWA slots.

    ``dspark_kv_inject._inject_mla`` takes this Triton helper for any
    ``cache_loc.is_cuda`` tensor (``dspark_kv_inject.py:124``) -- true on Kunlun
    XPU, where no Triton kernel can launch (see ``row_argmax_kunlun`` for why the
    ``triton.jit`` no-op is the only workable regime).

    The op is the same computation with the result handed back through an
    ``out`` argument instead of a return value, so the only adaptation is
    allocating ``out`` as int32 with ``cache_loc``'s shape; its three inputs
    match the hook's one-to-one in order. The inputs go in contiguous and
    flattened because the op takes no strides and TORCH_CHECKs ``cache_loc``
    for 1D -- rows come from ``width`` alone, so flattening is a no-op in
    meaning and keeps a ``[bs, width]`` caller on the op instead of the
    fallback. Verified element-identical against the torch body below on
    ``bs in {1, 4, 8, 32}``, ``width=6``, mapping sizes up to 1737985, both 1D
    and ``[bs, width]`` inputs, including the negative-``cache_loc`` wrap -- so
    the op implements Python's negative indexing too, not just the
    non-negative case.

    Contract copied from ``_committed_swa_locations``, and what the torch
    fallback below reproduces: rows are ``width`` wide, a slot counts as
    committed when its column is below the row's commit length, committed slots
    read ``full_to_swa_mapping[loc]`` with Python-style negative index wrap, and
    everything else is ``-1``. The fallback is kept free of host syncs so it
    stays capturable in the decode cuda graph.

    Args:
        cache_loc: Full-cache locations arranged in rows of width slots.
        full_to_swa_mapping: Mapping from full-cache locations to SWA locations.
        commit_lens: Number of committed slots in each row.
        width: Number of candidate slots per request.

    Returns:
        Int32 SWA locations with cache_loc's shape and -1 for uncommitted slots.
    """

    out_shape = cache_loc.shape
    n = cache_loc.numel()
    if n == 0:
        return torch.empty_like(cache_loc, dtype=torch.int32)

    op = "committed_swa_locations"
    xsg = torch.ops.xspeedgate_ops
    if hasattr(xsg, "committed_swa_locations"):
        out = torch.empty(n, dtype=torch.int32, device=cache_loc.device)
        try:
            xsg.committed_swa_locations(
                cache_loc.reshape(-1).contiguous(),
                full_to_swa_mapping.contiguous(),
                commit_lens.contiguous(),
                out,
                width,
            )
        except RuntimeError as exc:
            # The op's dtype/layout constraints are undocumented, so its own
            # TORCH_CHECKs are the only guard. Those throw before the launch,
            # with `out` still untouched, which makes falling through to the
            # torch body safe. An out-of-bounds write inside the kernel hangs
            # the device instead and never surfaces in Python, so that case is
            # not something `except` can cover.
            _dspark_warn_fallback(op, f"{type(exc).__name__}: {exc}")
        else:
            note_op_path(op, "xspeedgate")
            return out.reshape(out_shape)
    else:
        _dspark_warn_fallback(
            op, "xspeedgate_ops.committed_swa_locations is unavailable"
        )

    note_op_path(op, "torch")
    rows = commit_lens.numel()
    loc = cache_loc.reshape(rows, width).to(torch.long)
    col = torch.arange(width, device=cache_loc.device).view(1, -1)
    committed = col < commit_lens.to(torch.long).view(-1, 1)

    # Uncommitted slots are never read, so park them on a valid index rather
    # than letting a stale/negative location index the mapping out of bounds.
    loc = torch.where(committed, loc, torch.zeros_like(loc))
    loc = torch.where(loc < 0, loc + full_to_swa_mapping.numel(), loc)

    swa = full_to_swa_mapping[loc].to(torch.int32)
    swa = torch.where(committed, swa, torch.full_like(swa, -1))
    return swa.reshape(out_shape)


def _sharded_greedy_reduce(
    value: torch.Tensor,
    index: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared (value, lowest-index) argmax reduction over the last dim.

    Both halves of ``sharded_greedy_step`` run the same reduction, and both give
    NaN priority the way ``torch.argmax`` does: if any live lane is NaN the
    winners are the NaN lanes, otherwise the lanes holding the maximum. Ties --
    including ties between NaNs -- resolve to the lowest index, and a reduction
    with no live lane at all yields the ``2147483647`` sentinel.
    """

    nan = valid & value.isnan()
    has_nan = nan.any(dim=-1, keepdim=True)
    live = valid & ~nan
    maximum = torch.where(live, value, float("-inf")).amax(dim=-1, keepdim=True)
    wins = torch.where(has_nan, nan, live & (value == maximum))
    sentinel = torch.full_like(index, 2147483647)
    winner = torch.where(wins, index, sentinel).amin(dim=-1)
    best = torch.where(
        has_nan.squeeze(-1),
        torch.full_like(maximum.squeeze(-1), float("nan")),
        maximum.squeeze(-1),
    )
    return best, winner


# The 0.5.22 sampler uses the markov greedy entry points registered above.
# Keep the implementation available without registering the removed module.
def sharded_greedy_step_kunlun(
    bias: torch.Tensor,
    base_local: torch.Tensor,
    *,
    group,
    vocab_start: int,
    gather=None,
) -> torch.Tensor:
    """Exact greedy pick over the TP-sharded vocab.

    ``DSparkDraftSampler.__call__`` takes this fast path for every DSV4.1 draft
    head that reports ``supports_sharded_greedy``
    (``deepseek_v4_dspark.py:511-517``), so the DSpark draft sampler cannot be
    captured at all while the two upstream ``@triton.jit`` kernels are inert
    Python functions -- see ``row_argmax_kunlun`` for why they are, and why
    reviving real Triton is not on the table here.

    XSpeedGate ``sharded_greedy_partial`` / ``sharded_greedy_finish`` stand in
    for those two kernels one for one. The torch branch below stays as the
    fallback for builds that ship neither op, and for operands they refuse: the
    ops pin ``bias`` to bf16 and ``base_local`` to fp16, which is what this head
    produces today (the markov W2 GEMM is bf16 under
    ``SGLANG_DSPARK_OPT_MARKOV_W2_BF16``, itself a precondition of
    ``supports_sharded_greedy``, and the block logits carry the lm_head dtype),
    but casting to reach them would move the result.

    The transport is untouched in either branch: one
    ``(value, global-index-bits)`` pair per 4096-wide block of the shard per row,
    the same padded partition width on every rank, indices carried as int32 bits
    through a float32 buffer and never converted numerically. That matters
    because the collective runs across all 8 ranks -- any change to the shape or
    the dtype of ``partial`` would have to land on every rank at once.

    Both branches match the two kernels lane for lane, including the details
    that are easy to lose: the ``WIDTH`` mask (``base_local`` is padded wider
    than ``bias``, and the final shard can be entirely padding), NaN priority
    with lowest-index tie-breaking, and stripping the sign bit the NVLink push
    transport sets on ``+0`` so that token id 0 survives the round trip.

    Args:
        bias: Per-row bias for the valid local vocabulary.
        base_local: Local logits, possibly padded beyond bias's width.
        group: Tensor-parallel group providing world_size and all_gather.
        vocab_start: Global token ID corresponding to local column zero.
        gather: Optional collective callable accepting flattened partial winners.

    Returns:
        One int64 global winning token ID per row, breaking ties by lowest ID.
    """

    assert bias.ndim == base_local.ndim == 2
    assert bias.shape[0] == base_local.shape[0]
    assert bias.shape[1] <= base_local.shape[1]
    assert bias.stride(1) == base_local.stride(1) == 1
    rows, width = bias.shape
    block = 4096
    parts = (base_local.shape[1] + block - 1) // block
    assert parts > 0
    padded = parts * block

    op = "sharded_greedy_step"
    xsg = torch.ops.xspeedgate_ops
    if (
        bias.dtype == torch.bfloat16
        and base_local.dtype == torch.float16
        and hasattr(xsg, "sharded_greedy_partial")
        and hasattr(xsg, "sharded_greedy_finish")
    ):
        try:
            op_probe(
                op,
                "enter",
                f"rows={rows} width={width} parts={parts} "
                f"world={group.world_size} vocab_start={vocab_start}",
            )
            partial = torch.empty(
                (rows, parts, 2), dtype=torch.float32, device=bias.device
            )
            xsg.sharded_greedy_partial(
                bias, base_local, partial, width, vocab_start, parts, block
            )
            if gather is not None:
                gathered = gather(partial.reshape(rows, parts * 2))
            else:
                gathered = (
                    group.all_gather(partial, dim=0)
                    if group.world_size > 1
                    else partial
                )
            out = torch.empty(rows, dtype=torch.int64, device=bias.device)
            # ``finish`` only checks the flat capacity, and the layout it walks
            # -- ((rank * rows + row) * parts + part) * 2 -- is the same one the
            # torch branch below reshapes by hand.
            xsg.sharded_greedy_finish(
                gathered.contiguous(), out, rows, parts, group.world_size
            )
            op_probe(op, "exit", f"rows={rows}")
        except RuntimeError as exc:
            # The TORCH_CHECKs throw before any launch, and both ops only write
            # buffers allocated right here, so the fallback still sees pristine
            # ``bias`` / ``base_local`` and recomputes the whole step.
            _dspark_warn_fallback(op, f"{type(exc).__name__}: {exc}")
        else:
            note_op_path(op, "xspeedgate")
            return out
    else:
        _dspark_warn_fallback(
            op,
            "xspeedgate_ops.sharded_greedy_partial/finish is unavailable or "
            f"bias dtype is {bias.dtype} and base_local dtype is "
            f"{base_local.dtype} (the ops require bfloat16 and float16)",
        )

    note_op_path(op, "torch")
    value = base_local[:, :width].to(torch.float32) + bias.to(torch.float32)
    if padded > width:
        value = torch.nn.functional.pad(value, (0, padded - width), value=0.0)
    value = value.view(rows, parts, block)

    columns = torch.arange(padded, device=bias.device, dtype=torch.int32)
    valid = (columns < width).view(1, parts, block).expand(rows, parts, block)
    index = (columns + vocab_start).view(1, parts, block).expand(rows, parts, block)

    best, winner = _sharded_greedy_reduce(value, index, valid)
    partial = torch.stack((best, winner.view(torch.float32)), dim=-1)

    if gather is not None:
        gathered = gather(partial.reshape(rows, parts * 2))
    else:
        gathered = (
            group.all_gather(partial, dim=0) if group.world_size > 1 else partial
        )

    # Flat layout of the transport, identical for both gather styles:
    # ((rank * rows + row) * parts + part) * 2.
    gathered = gathered.reshape(group.world_size, rows, parts, 2)
    value = gathered[..., 0].permute(1, 0, 2).reshape(rows, -1)
    index = gathered[..., 1].reshape(-1).view(torch.int32).view(
        group.world_size, rows, parts
    ).permute(1, 0, 2).reshape(rows, -1)
    index = index & 2147483647
    valid = index != 2147483647

    _, winner = _sharded_greedy_reduce(value, index, valid)
    return winner.to(torch.int64)


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

    # The vendor op caches the whole bs-row table into SM, hard-requiring bs <= 64; the running bs may exceed it.
    if bs > 64:
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


@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "markov_greedy_step_triton",
)
def markov_greedy_step_triton(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    """Markov draft head greedy step in plain torch.

    ``MarkovGreedyStep.execute`` routes every ``base_logits.is_cuda`` tensor to
    the Triton tiled argmax (new in 0.5.19), which cannot compile on Kunlun.
    Replace the whole triton entry point instead of its two kernels: the upstream
    torch reference (``markov_greedy_step``) is the same math, and the kernel
    already accumulates in fp32 with first-index tie-break.
    """
    step_logits = base_logits.to(torch.float32) + torch.nn.functional.linear(
        prev_embeds.to(torch.float32), w2_weight.to(torch.float32)
    )
    return torch.argmax(step_logits, dim=-1).to(torch.int64)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.markov_sample.MarkovSampleStep.execute",
    type=HookType.REPLACE,
)
def markov_sample_step_execute_kunlun(cls, **kwargs):
    """Keep the DSpark draft sampler on its own torch reference.

    0.5.22 replaced the 0.5.19 online sampler (``sample_step_tokens`` plus the
    ``_online_partial_kernel`` / ``_online_combine_kernel`` /
    ``_build_step_local_kernel`` triton kernels this plugin used to replace one by
    one) with ``MarkovSampleStep``, a single Philox Gumbel-max kernel. Its
    ``execute`` dispatches on ``inputs_on_cuda(**kwargs)``, which is true on
    Kunlun because the platform reuses the CUDA dispatch key, so every draft step
    would try to compile ``_markov_sample_partial_kernel``.

    ``MarkovSampleStep.torch`` is upstream's own reference for the same math
    (same Philox stream, same rounding through ``step_dtype``), so routing there
    is an upstream-sanctioned degradation rather than a Kunlun-only code path.
    """

    return cls.torch(**kwargs)


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
    # Upstream triton also does tl.load(incl_ptr + bs - 1); bs==0 is not a legal input in the first place
    _dspark_require(bs > 0, "compact_row_index", f"bs must be > 0, got {bs}")
    incl = incl_ptr[:bs].to(torch.int64)
    # The vendor op takes verify_lens and does cumsum internally; here we recover it from incl
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

    # The vendor op aligns with the upper-level python function (row index + gather computed in one go),
    # so here we replace the host function instead of _compact_verify_ids_gather_kernel.
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

    if bs <= 1024:  # vendor hard limit
        return torch.ops.xspeedgate_ops.compact_verify_ids(
            block_ids,
            tokens,
            verify_lens.contiguous(),
            layout.graph_num_tokens,
            str(tokens.device),
        )

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
    # The vendor op requires req_to_token to be int64, while the live req_to_token is int32;
    # here we only pull out the few rows used this time and cast them to int64 (casting the whole table is too expensive),
    # and remap req_pool to 0..n_req-1.
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


@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.dspark_schedule",
    "schedule_verify_lens_topk_triton",
)
def schedule_verify_lens_topk_kunlun(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg,
) -> torch.Tensor:
    """Confidence-driven per-request verify-length schedule on XSpeedGate.

    Mirrors ``dspark_schedule.schedule_verify_lens_topk_triton`` (survival
    cumprod -> global top-k budget select -> clamp finalize) but runs the three
    ``schedule_topk_*`` XSpeedGate kernels instead of Triton. It is only reached
    when the DSpark budget planner is active (``speculative_adaptive`` or an SPS
    table); in the default static-block config the planner returns early and this
    replacement is never called, so wiring it cannot perturb that config.

    The vendor ``schedule_topk_selected_extra_kernel`` is specialised to
    ``cols == 5`` and ``n <= 256`` (bs up to ~51). Anything outside that window
    falls back to the torch reference so correctness never depends on the fast
    path being eligible.

    Args:
        confidence: Per-request confidence for each draft position.
        budget: Global budget for selecting additional verification positions.
        cfg: Scheduling limits and survival-probability threshold.

    Returns:
        Per-request verification lengths bounded by the configured limits.
    """

    op = "schedule_verify_lens_topk"
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols

    xsg = torch.ops.xspeedgate_ops
    have_ops = all(
        hasattr(xsg, name)
        for name in (
            "schedule_topk_prep_kernel",
            "schedule_topk_selected_extra_kernel",
            "schedule_topk_finalize_kernel",
        )
    )
    # The selected_extra kernel is skipped when budget <= 0, so its
    # cols==5 / n<=256 contract only has to hold when a budget is actually spent.
    vendor_ok = (
        have_ops
        and num_requests > 0
        and (budget <= 0 or (cols == 5 and 0 <= n <= 256))
    )
    if not vendor_ok:
        _dspark_warn_fallback(
            op,
            f"cols={cols}, n={n}, budget={budget} outside vendor support "
            "(needs cols==5, n<=256); using torch reference",
        )
        from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
            schedule_verify_lens_topk,
        )

        return schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)

    note_op_path(op, "xspeedgate")
    confidence_c = confidence.contiguous()
    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    # prep: survival[:, :cols] = cumprod(confidence[:, :cols]); selected_extra = 0
    xsg.schedule_topk_prep_kernel(
        confidence_c, survival, selected_extra, int(gamma), int(cols), 0
    )
    if budget > 0 and n > 0:
        # global top-k budget select, accumulated per request into selected_extra
        xsg.schedule_topk_selected_extra_kernel(
            survival,
            selected_extra,
            int(budget),
            int(cols),
            int(n),
            float(cfg.survival_eps),
            0,
            0,
        )
    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    # finalize: clamp(min_verify_len + selected_extra, lower_bound, max_len)
    xsg.schedule_topk_finalize_kernel(
        selected_extra,
        verify_lens,
        int(cfg.min_verify_len),
        max(int(cfg.min_verify_len), 1),
        int(max_len),
        int(num_requests),
        0,
    )
    return verify_lens


@register_triton_op(
    "sglang.kernels.ops.kvcache.kv_indices",
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


@register_jit_op("sglang.kernels.ops.attention.clamp_position", "clamp_position")
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
    *,
    # 0.5.22 adds the keyword-only sorted/block-padded row filter
    # (``sorted_token_ids`` / ``num_tokens_post_padded`` / ``num_valid_tokens``,
    # see ``run_activation`` -> ``_run_activation_sorted_filtered_inplace``).
    # ``fused_moe._activation_row_filter`` only emits them together with
    # ``expert_ids`` on the TMA down path, which Kunlun already rejects below;
    # they are accepted here so the signature matches and refused explicitly
    # instead of being dropped.
    sorted_token_ids: Optional[torch.Tensor] = None,
    num_tokens_post_padded: Optional[torch.Tensor] = None,
    num_valid_tokens: int = 0,
) -> torch.Tensor:
    """Kunlun SwiGLU replacement for 0.5.14's JIT silu_and_mul.

    0.5.14 routes ``activation.SiluAndMul.forward_cuda`` through the nvcc-JIT
    ``sglang.jit_kernel.activation.silu_and_mul`` (which fails on Kunlun: nvcc
    11.7 rejects ``-std=c++20``). Replace it with ``kunlun_ops.swiglu``.

    The 0.5.14 signature adds optional ``expert_ids`` / ``expert_step`` for
    filtered MoE activation; the Kunlun SwiGLU op does not support per-expert
    filtering, so assert it is unused on this path.

    0.5.22 additionally allows the sorted, block-padded layout to be filtered
    row-wise. ``kunlun_ops.swiglu`` writes every row of ``y`` and exposes no row
    index list, so that contract (leave the padding rows of ``out`` unwritten)
    cannot be honoured; raise instead of ignoring the arguments.
    """
    import kunlun_ops

    assert expert_ids is None, (
        "Kunlun silu_and_mul does not support expert_ids filtering"
    )
    # 0.5.22: honour the new row filter by refusing it loudly.
    if sorted_token_ids is not None or num_tokens_post_padded is not None:
        raise NotImplementedError(
            "Kunlun silu_and_mul cannot honour the 0.5.22 sorted row filter "
            "(sorted_token_ids/num_tokens_post_padded/num_valid_tokens): "
            "kunlun_ops.swiglu writes every row and cannot skip the "
            "block-padding rows the down GEMM never reads"
        )
    if num_valid_tokens:
        raise NotImplementedError(
            "Kunlun silu_and_mul cannot honour the 0.5.22 num_valid_tokens row "
            "limit; it computes every row of the input buffer"
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
    return ids.masked_fill(invalid, 0)


@register_jit_op("sglang.kernels.ops.moe.dsv4", "hash_topk")
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
    topk_ids, topk_weights = torch.ops.xspeedgate_ops.moe_hash_topk_fused(
        router_logits,
        safe_input_ids,
        tid2eid,
        num_fused_shared_experts,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


# kunlun_ops.matmul writes an fp32 result directly for these input dtype pairs.
_DSV4_MATMUL_FP32_OUT_INPUTS = frozenset(
    (
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
    )
)
# fp16 has no fp32-output kernel ("matmul unimplemented"). Its fp16-output
# kernel still accumulates in fp32, so only the result is rounded, and both
# input casts are still skipped.
_DSV4_MATMUL_FP16_OUT_INPUTS = frozenset(((torch.float16, torch.float16),))


def _dsv4_kunlun_matmul_out_dtype(
    x: torch.Tensor, y: torch.Tensor
) -> Optional[torch.dtype]:
    """Return the kunlun_ops.matmul output dtype for x @ y.T, or None to fall back."""
    inputs = (x.dtype, y.dtype)
    if inputs in _DSV4_MATMUL_FP32_OUT_INPUTS:
        return torch.float32
    if inputs in _DSV4_MATMUL_FP16_OUT_INPUTS:
        return torch.float16
    return None


@register_jit_op("sglang.kernels.ops.gemm.bf16_fp32", "linear_bf16_fp32")
def dsv4_linear_bf16_fp32_kunlun(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    """Preserve the shared DSV4 FP32-output GEMM contract.

    kunlun_ops.matmul consumes the half-precision inputs as they are, so the two
    .float() copies the torch fallback needs disappear. The weight copy is the
    expensive one: the fallback re-materialises it on every forward.

    ``hpc_kernel_min_m`` only selects between upstream's HPC GEMM and cuBLAS for
    rows below the threshold (the LongCat-Flash router is the only caller that
    passes it); both are the same math, and neither exists on Kunlun. Accept and
    ignore it rather than raising -- but accept it by name, because omitting the
    parameter turns that call into a TypeError.
    """
    del hpc_kernel_min_m
    out_dtype = _dsv4_kunlun_matmul_out_dtype(x, y)
    if out_dtype is None:
        result = torch.nn.functional.linear(x.float(), y.float())
    else:
        import kunlun_ops

        out = torch.empty(
            (x.shape[0], y.shape[0]), dtype=out_dtype, device=x.device
        )
        kunlun_ops.matmul(x, y, out, False, True)
        result = out if out_dtype is torch.float32 else out.float()
    return result


def _resolve_silu_and_mul_with_swiglu_limit_out():
    """Return the ``_out`` variant of the swiglu-limit activation, else None."""
    return getattr(
        torch.ops.xspeedgate_ops, "silu_and_mul_with_swiglu_limit_out", None
    )


@register_jit_op("sglang.kernels.ops.moe.dsv4", "silu_and_mul_clamp")
def dsv4_silu_and_mul_clamp_kunlun(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Apply clamped SiLU-and-mul into ``output``.

    The ``_out`` variant writes the caller's buffer directly, so the allocation
    of the kernel's own return tensor and the following ``output.copy_`` are
    gone. Falls back to the allocating variant when unavailable.
    """

    op = _resolve_silu_and_mul_with_swiglu_limit_out()
    if op is not None:
        op(input, output, swiglu_limit)
        return

    output.copy_(
        torch.ops.xspeedgate_ops.silu_and_mul_with_swiglu_limit(input, swiglu_limit)
    )


@register_jit_op(
    "sglang.srt.model_executor.runner_utils.shared_read_event",
    "make_external_event",
)
def make_external_event(device_module):
    """No external CUDA-graph events on Kunlun.

    0.5.19's decode graph runner records an ``Event(external=True)`` inside the
    replayed graph and hands it to the scheduler's WAR barrier
    (``Scheduler._apply_war_barrier`` -> ``schedule_stream.wait_event``). The
    Kunlun runtime accepts the creation but fails the cross-stream wait with
    ``CUDA error: invalid argument`` (seen on PD decode with overlap schedule).

    Upstream already handles the "no external-event support" case: returning None
    keeps ``in_graph_metadata_prep_done`` unset, so ``_resolve_shared_read_ends``
    downgrades IN_REPLAY to PRE_REPLAY and ``_publish_read_done`` records a plain
    device event instead.
    """
    del device_module
    return None


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


@register_jit_op("sglang.kernels.ops.moe.dsv4", "mask_topk_ids")
def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor) -> None:
    """XSpeedGate ``mask_topk_ids_padded_region``: mask padded rows in MoE top-k ids.

    Boolean-mask assignment is shape-dependent and therefore not capturable, so
    the padded rows would keep stale gate output during graph replay. Both
    branches below are capturable: the op writes in place over a static shape,
    and the torch fallback uses masked_fill_ over a static-shaped mask.

    The op writes upstream's **-1** into the padded rows, while the torch body
    kept 0 out of caution. On this line -1 is in fact the better value, and the
    whole chain behind the gate handles it:

    * ``index_have_neg`` is ``moe_ep_size > 1`` in both compressed-tensors MoE
      schemes, true under EP8, so ``moe_pre_sorted`` treats a negative id the
      same as a lane routed to another EP rank: it goes into the ``num_experts``
      bucket and gets ``sorted_tokens_idx = -1``;
    * ``moe_fc`` never indexes by expert id, only by that sorted index;
    * ``moe_post`` masks exactly those lanes -- weight 0, index clamped to 0 --
      so nothing addresses a negative row and the padded output rows come out 0.

    So a padded row stops consuming an expert row instead of being routed to
    expert 0 to produce output that is discarded afterwards. The two branches
    therefore differ on the padded rows (-1 vs 0) and agree on every real row;
    the fallback stays at 0 because that is the value that is also safe when
    ``index_have_neg`` is false (``ep_size == 1``), which this hook cannot see.

    The op only accepts int32 ids (``topk_ids must be int32``) -- the DSV4 gate
    hands out int32, so the dtype guard here is for other callers, not this
    line. ``_dspark_warn_fallback`` logs any degradation once.
    """

    _DSV4_LAST_NUM_TOKEN_NON_PADDED[topk_ids.device] = num_token_non_padded

    op = "mask_topk_ids"
    xsg = torch.ops.xspeedgate_ops
    if topk_ids.dtype == torch.int32 and hasattr(xsg, "mask_topk_ids_padded_region"):
        try:
            xsg.mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        except RuntimeError as exc:
            # TORCH_CHECKs throw before the launch, and a partial in-place write
            # is harmless here because the fallback masks the same rows again.
            _dspark_warn_fallback(op, f"{type(exc).__name__}: {exc}")
        else:
            note_op_path(op, "xspeedgate")
            return
    else:
        _dspark_warn_fallback(
            op,
            "xspeedgate_ops.mask_topk_ids_padded_region is unavailable or "
            f"topk_ids dtype is {topk_ids.dtype} (the op requires int32)",
        )

    note_op_path(op, "torch")
    indices = torch.arange(
        0, topk_ids.shape[0], device=topk_ids.device, dtype=torch.int64
    )
    padded = (indices >= num_token_non_padded.reshape(()).to(torch.int64)).unsqueeze(1)
    topk_ids.masked_fill_(padded, 0)


@register_jit_op("sglang.kernels.ops.moe.moe_route_radix", "route_radix")
def moe_route_radix_kunlun(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    sorted: bool = False,
):
    """K3 radix-select router (896 experts, top-16) on the XSpeedGate operator.

    Upstream ``route_radix`` is a CUDA JIT kernel (``moe/route_radix.cuh``) that
    ``moe_fused_gate`` dispatches to for covered inputs instead of running the 16
    dependent argmax rounds of ``_router_triton_kernel``.
    ``xspeedgate_ops.moe_route_radix`` is parameter-for-parameter identical to it
    (``XSpeedGate/wrapper/src/moe/moe_route_radix.cpp``) and its TORCH_CHECKs are
    exactly upstream ``covered()``, so the caller keeps using that predicate and
    nothing needs a dtype/layout fixup here. ``sorted=False`` (the production
    dispatch) returns the winners in expert-id order.

    Args:
        scores: Routing scores for each token and expert.
        bias: Expert-selection bias.
        topk: Number of selected experts per token.
        renormalize: Whether to normalize the selected routing weights.
        routed_scaling_factor: Scale for the selected weights when enabled.
        apply_scale: Whether to apply routed_scaling_factor.
        sorted: Whether to request score-sorted rather than expert-ID ordering.

    Returns:
        Selected routing weights and expert IDs.
    """

    weights, ids = torch.ops.xspeedgate_ops.moe_route_radix(
        scores,
        bias,
        int(topk),
        bool(renormalize),
        float(routed_scaling_factor),
        bool(apply_scale),
        bool(sorted),
    )
    return weights, ids


def _moe_route_radix_usable(
    scores: torch.Tensor, bias: Optional[torch.Tensor], topk: int
) -> bool:
    """Whether the K3 radix router covers this gate call.

    ``covered()`` is pure Python (shape/dtype/stride only) so it is reused
    verbatim; the ``hasattr`` guards wheels older than the op.
    """

    if bias is None or not hasattr(torch.ops.xspeedgate_ops, "moe_route_radix"):
        return False
    from sglang.kernels.ops.moe.moe_route_radix import covered

    return bool(covered(scores, bias, topk))


def _moe_gate_sqrtsoftplus_log1p(
    scores, bias, topk, num_fused_shared_experts, renormalize,
    routed_scaling_factor, apply_routed_scaling_factor_on_output,
    bias_alt, input_ids, bias_alt_token_id, num_token_non_padded,
    renormalize_epsilon,
):
    """Device-only torch fallback for the ungrouped V4.1 Triton contract."""
    assert scores.ndim == 2
    assert scores.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert bias is not None and bias.ndim == 1
    assert bias.dtype in (torch.float32, torch.float16, torch.bfloat16)
    rows, experts = scores.shape
    assert bias.shape == (experts,)
    routed = topk - num_fused_shared_experts
    assert num_fused_shared_experts >= 0 and 0 < routed <= experts
    if input_ids is not None:
        assert bias_alt is not None and bias_alt_token_id is not None
        assert bias_alt.shape == bias.shape and input_ids.shape == (rows,)
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    x = scores.float()
    # Do not use log(1 + exp(x)): it loses small positive scores.
    activated = torch.where(x > 20.0, x, torch.log1p(torch.exp(x))).sqrt()
    row_bias = bias.float()
    if input_ids is not None:
        row_bias = torch.where(
            (input_ids == bias_alt_token_id)[:, None], bias_alt.float(), row_bias
        )
    ranking = activated + row_bias
    ranking = torch.where(torch.isnan(ranking), float("inf"), ranking)
    ids = torch.arange(experts, device=scores.device)[None, :]
    remaining = torch.ones_like(ranking, dtype=torch.bool)
    weights = torch.empty((rows, topk), dtype=torch.float32, device=scores.device)
    indices = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
    # Explicit argmax/min-id avoids topk's unspecified tie order, including
    # all -inf rows. Keep remaining separately so no expert is selected twice.
    for slot in range(routed):
        maximum = ranking.max(dim=1, keepdim=True).values
        winner = torch.where(remaining & (ranking == maximum), ids, experts).min(
            dim=1, keepdim=True
        ).values
        weights[:, slot:slot + 1] = activated.gather(1, winner)
        indices[:, slot:slot + 1] = winner.to(torch.int32)
        remaining = remaining & (ids != winner)
        ranking = ranking.masked_fill(~remaining, -float("inf"))
    routed_sum = weights[:, :routed].sum(dim=1, keepdim=True)
    if num_fused_shared_experts:
        weights[:, routed:] = routed_sum / scale
        indices[:, routed:] = torch.arange(
            experts, experts + num_fused_shared_experts,
            dtype=torch.int32, device=scores.device,
        )
    if renormalize:
        norm = (
            routed_sum + renormalize_epsilon
            if renormalize_epsilon > 0.0
            else torch.where(routed_sum > 0.0, routed_sum, 1.0)
        )
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * scale
    if num_token_non_padded is not None:
        live = (
            torch.arange(rows, device=scores.device)
            < num_token_non_padded.to(device=scores.device).reshape(())
        )[:, None]
        weights = weights.masked_fill(~live, 0.0)
        indices = indices.masked_fill(~live, -1)
    return weights, indices


@register_jit_op("sglang.kernels.ops.moe.moe_fused_gate", "moe_fused_gate")
def moe_fused_gate_dsv4_kunlun(
    scores: torch.Tensor,
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
    *,
    bias_alt: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    bias_alt_token_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize_epsilon: float = 0.0,
    packed_out: Optional[torch.Tensor] = None,
    # 0.5.22 added keyword-only ``sqrtsoftplus_log1p``: evaluate sqrtsoftplus as
    # ``sqrt(log1p(exp(x)))`` (with a >20 passthrough) and rank NaN logits FIRST
    # instead of last (DeepSeek-V4.1 routing; topk.py sets it only for
    # model_type == "deepseek_v41", together with fused_gate_packed_ids).
    sqrtsoftplus_log1p: bool = False,
):
    """Match the patched DeepSeek-V4 / V4.1 fused MoE gate.

    V4.1 log1p routing uses the adjacent torch fallback, including epsilon
    normalization and NaN-first/min-id selection. Legacy native routing below
    retains its existing behavior.

    The vision-language routing contract: ``bias_alt`` replaces
    ``bias`` on rows whose ``input_ids == bias_alt_token_id``, rows at or past
    the device scalar ``num_token_non_padded`` become zero weight / id -1, and
    ``renormalize_epsilon`` selects ``sum + eps`` over the zero-sum guard. The
    Kunlun op takes one bias vector, so a per-row alternative bias is handled by
    running the gate once per bias and selecting per row -- only the VL routing
    path (``vl_routing.vision_topk``) passes ``bias_alt``, so text-only routing
    pays nothing.
    """
    if num_expert_group != 1 or topk_group != 1:
        raise ValueError("grouped routing is not supported by the DSV4 fused gate")
    if moe_softcapping != 0.0:
        raise ValueError("MoE softcapping is not supported by the DSV4 fused gate")
    if packed_out is not None:
        raise NotImplementedError(
            "packed_out (FlashInfer routed-MoE form) is not supported by the "
            "Kunlun fused gate"
        )
    if sqrtsoftplus_log1p and scoring_func.lower() == "sqrtsoftplus":
        return _moe_gate_sqrtsoftplus_log1p(
            scores, bias, topk, num_fused_shared_experts, renormalize,
            routed_scaling_factor, apply_routed_scaling_factor_on_output,
            bias_alt, input_ids, bias_alt_token_id, num_token_non_padded,
            renormalize_epsilon,
        )

    import kunlun_ops

    if scoring_func == "sigmoid" and apply_routed_scaling_factor_on_output:
        # MiniMax-M3 contract (its only caller is the MiniMax sigmoid branch in
        # hooks/layers/moe/topk.py); DSV4 keeps the dsv4 op below. The MiniMax
        # implementation lives in the MiniMax package on purpose.
        from sglang_kunlun.hooks.models.minimax.kernels import (
            moe_fused_gate_minimax,
        )

        return moe_fused_gate_minimax(
            scores,
            bias,
            topk,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )

    num_rows = scores.shape[0]

    # K3 radix-select fast path, same dispatch condition as upstream
    # `moe_fused_gate` (moe_fused_gate.py:420): sigmoid scoring, no fused shared
    # expert, no per-token bias / padding / renorm-epsilon, and an input the
    # radix kernel covers ([M, 896] fp32|bf16 rows, fp32 bias, top-16). The DSV4
    # routers (384 experts, bf16 bias) never match, so they keep the
    # `moe_fused_gate_dsv4` path below.
    if (
        scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and input_ids is None
        and bias_alt is None
        and num_token_non_padded is None
        and renormalize_epsilon == 0.0
        and bias is not None
        and bias.stride(0) == 1
        and _moe_route_radix_usable(scores, bias, topk)
    ):
        # Call the override directly rather than through the patched upstream
        # symbol: the fast path must not depend on install() ordering.
        return moe_route_radix_kunlun(
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
            sorted=False,
        )


    def run(inp: torch.Tensor, bias_vec: torch.Tensor):
        weights = torch.empty(num_rows, topk, dtype=torch.float32, device=inp.device)
        indices = torch.empty(num_rows, topk, dtype=torch.int32, device=inp.device)
        kunlun_ops.moe_fused_gate_dsv4(
            inp.to(torch.float32),
            bias_vec,
            weights,
            indices,
            topk,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        return weights, indices

    use_alt = (
        bias_alt is not None and input_ids is not None and bias_alt_token_id is not None
    )
    if use_alt:
        alt = (input_ids.to(torch.int64) == int(bias_alt_token_id)).unsqueeze(-1)
        w_main, i_main = run(scores, bias)
        w_alt, i_alt = run(scores, bias_alt)
        topk_weights = torch.where(alt, w_alt, w_main)
        topk_indices = torch.where(alt, i_alt, i_main)
    else:
        topk_weights, topk_indices = run(scores, bias)

    if num_token_non_padded is not None:
        # Device scalar; keep this tensor-only so it stays graph-capturable.
        limit = num_token_non_padded.to(device=scores.device).reshape(())
        padded = (torch.arange(num_rows, device=scores.device) >= limit).unsqueeze(-1)
        topk_weights = topk_weights.masked_fill(padded, 0.0)
        topk_indices = topk_indices.masked_fill(padded, -1)
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


@register_triton_op("sglang.kernels.ops.kvcache.mla_buffer", "set_mla_kv_buffer_kernel")
def set_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    reserved_skip_index: int = -1,
    buffer_stride: int = 0,
    nope_stride: int = 0,
    rope_stride: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
    *,
    BLOCK: int = 0,
    DCP_RANK: int = 0,
    DCP_WORLD_SIZE: int = 1,
    USE_GDC: bool = False,
) -> None:
    """Store MLA KV tensors into the paged KV buffer.

    0.5.19 added ``reserved_skip_index`` (slot-0 padding sink) plus the decode
    context-parallel ``DCP_RANK`` / ``DCP_WORLD_SIZE`` constexprs. The Kunlun op
    writes every ``loc`` unconditionally: the reserved slot is a padding sink that
    is never read back, so clobbering it is harmless, but a real DCP split cannot
    be emulated and is rejected.
    """
    if DCP_WORLD_SIZE != 1:
        raise NotImplementedError(
            "set_mla_kv_buffer_kernel: decode context parallel (DCP_WORLD_SIZE="
            f"{DCP_WORLD_SIZE}) is not supported on Kunlun"
        )

    from kunlun_ops import set_mla_kv_buffer_triton as impl

    status = impl(
        kv_buffer,
        loc,
        cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
        cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
    )
    if status != 0:
        raise RuntimeError(
            f"Kunlun MLA cache set returned non-zero status {status}"
        )


@register_triton_op("sglang.kernels.ops.kvcache.mla_buffer", "get_mla_kv_buffer_kernel")
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
    "sglang.kernels.ops.speculative.multi_layer_eagle",
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
    # 0.5.22 fuses the draft-decode position write into this kernel: every one of
    # a request's ``topk`` columns gets ``kv_start = seq_lens[pid]``, and with
    # WRITE_MROPE the same value is broadcast to all 3 mrope axes (flat layout,
    # axis stride ``BS * topk``). eagle_worker_common.prepare_for_draft allocates
    # ``positions`` whenever ``seq_lens.is_cuda and topk == 1`` -- true on Kunlun
    # -- so this write is load-bearing and is implemented, not refused.
    positions: Optional[torch.Tensor] = None,
    mrope: Optional[torch.Tensor] = None,
    BS: int = 0,
    WRITE_POSITIONS: bool = False,
    WRITE_MROPE: bool = False,
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

    # 0.5.22: WRITE_POSITIONS / WRITE_MROPE part of the same launch.
    if not WRITE_POSITIONS:
        return
    if positions is None:
        raise ValueError(
            "assign_draft_cache_locs_contiguous: WRITE_POSITIONS requires positions"
        )
    rows = int(BS) if BS else int(req_pool_indices.shape[0])
    kv_start = seq_lens[:rows].reshape(rows, 1).expand(rows, topk).reshape(-1)
    positions.view(-1)[: rows * topk].copy_(kv_start)
    if not WRITE_MROPE:
        return
    if mrope is None:
        raise ValueError(
            "assign_draft_cache_locs_contiguous: WRITE_MROPE requires mrope"
        )
    if not BS:
        raise ValueError(
            "assign_draft_cache_locs_contiguous: WRITE_MROPE needs BS for the "
            "per-axis stride (axis * BS * topk)"
        )
    mrope_flat = mrope.view(-1)
    stride = int(BS) * topk
    for axis in range(3):
        mrope_flat[axis * stride : axis * stride + rows * topk].copy_(kv_start)


# [Dropped for 0.5.14] ``fill_new_verified_id`` no longer exists upstream; the
# verified-id flow moved to ``FutureMap._resolve_spec_extras`` (pure torch) in
# ``sglang.srt.managers.overlap_utils`` and is not a Triton kernel. No XPU
# replacement needed.


@register_jit_op(
    "sglang.kernels.ops.speculative.multi_layer_eagle",
    "fill_draft_extend_prepare_buffers_triton",
)
def fill_draft_extend_prepare_buffers_triton(*args, **kwargs):
    """Route the draft-extend buffer fill to upstream's native torch variant.

    The upstream kernel is a ``@triton.jit`` kernel compiled for the CUDA target and
    XPU has no backend for it ("0 compatible backends for target (cuda)"). Upstream
    ships ``fill_draft_extend_prepare_buffers_native`` with an identical signature and
    uses it on NPU, so reuse that instead of re-deriving the buffer layout here.
    """
    from sglang.kernels.ops.speculative.multi_layer_eagle import (
        fill_draft_extend_prepare_buffers_native,
    )

    return fill_draft_extend_prepare_buffers_native(*args, **kwargs)


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


def _draft_kv_row_view(
    table: torch.Tensor, row_id: int, row_stride: int
) -> torch.Tensor:
    """One read-source row, mirroring the kernel's ``ids + row_id * row_stride``.

    0.5.22 passes the read source as ``(ids, row_ids, row_stride)`` (see
    ``KVIndexTable``), and ``ids`` may be a *slice* of a wider table, so the row
    has to be taken by stride rather than by ``table[row_id]``.
    """
    base = int(row_id) * int(row_stride)
    return torch.as_strided(table, (base + int(row_stride),), (1,))[base:]


def _draft_load_token_ids(
    row: torch.Tensor,
    pos: torch.Tensor,
    page_size: int,
    entry_page_size: int,
    v2p: Optional[torch.Tensor],
    translate: bool,
) -> torch.Tensor:
    """Torch port of 0.5.22's ``_load_token_ids`` / ``_translate_token_ids``.

    ``ENTRY_PAGE_SIZE > 1`` means one entry covers a page, so the token id is
    ``entry * ENTRY_PAGE_SIZE + pos % ENTRY_PAGE_SIZE``; ``TRANSLATE`` then maps
    the virtual id through the page-level ``v2p`` table (negative ids go to the
    page-0 sink), exactly as ``translate_kv_loc`` does.
    """
    pos = pos.to(torch.int64)
    if int(entry_page_size) == 1:
        ids = row[pos]
    else:
        eps = int(entry_page_size)
        ids = row[pos // eps].to(torch.int64) * eps + pos % eps
    if not translate:
        return ids
    i64 = ids.to(torch.int64)
    zero = torch.zeros_like(i64)
    vpage = torch.where(i64 < 0, zero, i64 // int(page_size))
    phys = v2p[vpage].to(torch.int64)
    return torch.where(
        i64 < 0, zero, torch.clamp(phys * int(page_size) + i64 % int(page_size), min=0)
    )


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
    # 0.5.22 renamed ``pool_len`` -> ``row_stride`` and made it runtime (the read
    # source may be a slice of a wider table, so the row is taken by stride).
    row_stride: int,
    kv_indices_stride: int,
    kv_indptr_stride: int,
    bs_upper: int,
    iter_upper: int,
    num_tokens_upper: int,
    page_size: int,
    # 0.5.22 additions:
    #  - window_size / sink_size: restrict the DRAFT read to ``sink_size``
    #    prefix tokens + the most recent ``window_size`` (0 = identity);
    #    triton_backend / flashinfer_backend pass them POSITIONALLY.
    #  - NUM_STEPS: token-block launch specialization. grid.x becomes
    #    ``NUM_STEPS * num_token_blocks`` and the blocks only split the copy
    #    loop, so the step count comes from NUM_STEPS when it is non-zero.
    #  - ENTRY_PAGE_SIZE / v2p / TRANSLATE: the read source may be page-granular
    #    and/or hold virtual ids to be mapped through the page table.
    window_size: int = 0,
    sink_size: int = 0,
    NUM_STEPS: int = 0,
    ENTRY_PAGE_SIZE: int = 1,
    v2p: Optional[torch.Tensor] = None,
    TRANSLATE: bool = False,
    *,
    grid: tuple[int, ...] | None = None,
) -> None:
    """Build draft-decode KV indices for the split v0.5.14 cache-locs op."""

    if grid is None:
        raise ValueError("generate_draft_decode_kv_indices requires Triton-style grid")
    # grid.x is NUM_STEPS * num_token_blocks under the 0.5.22 block launch.
    num_steps = int(NUM_STEPS) if NUM_STEPS else int(grid[0])
    num_seqs, topk = int(grid[1]), int(grid[2])
    if TRANSLATE and v2p is None:
        raise ValueError(
            "generate_draft_decode_kv_indices: TRANSLATE requires the v2p table"
        )
    device = kv_indices.device
    req_indices = req_pool_indices[:num_seqs].long()
    seq_lens = paged_kernel_lens[:num_seqs].long()
    pos = positions[: num_seqs * topk].long()
    window_size = int(window_size)
    sink_size = int(sink_size)
    if window_size > 0:
        # Sliding-window draft: the row read (and the position prefix sums the
        # kv_indptr is built from) is capped at sink + window.
        cap = window_size + sink_size
        read_lens = torch.clamp(seq_lens, max=cap)
        pos = torch.clamp(pos, max=cap)
    else:
        read_lens = seq_lens

    for iters0 in range(num_steps):
        iters = iters0 + 1
        kv_indices_row = kv_indices[iters0]
        kv_indptr_row = kv_indptr[iters0]
        cum_seq_len = 0
        for bid in range(num_seqs):
            seq_len = int(seq_lens[bid].item())
            seq_len_w = int(read_lens[bid].item())
            if window_size > 0:
                s_eff = min(sink_size, seq_len)
                recent_start = seq_len - (seq_len_w - s_eff)
            else:
                s_eff = 0
                recent_start = 0
            row = _draft_kv_row_view(req_to_token, int(req_indices[bid]), row_stride)
            copy_offset = torch.arange(seq_len_w, device=device)
            src = (
                torch.where(copy_offset < s_eff, copy_offset, recent_start + copy_offset - s_eff)
                if window_size > 0
                else copy_offset
            )
            for topk_id in range(topk):
                kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (
                    seq_len_w + iters
                )
                if seq_len_w > 0:
                    kv_indices_row[kv_offset : kv_offset + seq_len_w].copy_(
                        _draft_load_token_ids(
                            row, src, page_size, ENTRY_PAGE_SIZE, v2p, TRANSLATE
                        ).to(kv_indices_row.dtype)
                    )
                extend_offset = torch.arange(iters, device=device)
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
                    kv_offset + seq_len_w : kv_offset + seq_len_w + iters
                ].copy_(
                    _draft_load_token_ids(
                        row,
                        start + extend_offset,
                        page_size,
                        ENTRY_PAGE_SIZE,
                        v2p,
                        TRANSLATE,
                    ).to(kv_indices_row.dtype)
                )
            cum_seq_len += seq_len_w

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


def _chain_block_verification_select(
    candidates: torch.Tensor,
    uniform_samples: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """0.5.22 block verification (Algorithm 2, arXiv:2403.10444), batched.

    Returns ``(num_accept, residual_scale)``. Unlike classic per-step rejection,
    this accepts the LONGEST prefix whose joint ratio passes its coin, and the
    correction distribution is then scaled by that prefix's probability (the
    kernel's ``residual_scale``). Mirrors
    ``speculative_sampling_classic_kernel``'s ``BLOCK_VERIFICATION`` branch:
    the same coins, the same per-step ``prefix_prob`` recurrence, the same
    residual-mass acceptance for a partial prefix, and the same
    highest-step-wins selection.
    """
    bs, num_slots = candidates.shape
    num_steps = num_slots - 1
    device = candidates.device
    if num_steps <= 0:
        return (
            torch.zeros(bs, dtype=torch.int64, device=device),
            torch.ones(bs, dtype=torch.float32, device=device),
        )

    tok = candidates[:, 1:num_slots].to(torch.int64).unsqueeze(-1)
    p = target_probs[:, :num_steps, :].gather(2, tok).squeeze(-1).float()
    q = draft_probs[:, :num_steps, :].gather(2, tok).squeeze(-1).float()
    q_is_prob = (q > 0.0) & (q <= 1.0)
    safe_q = torch.where(q_is_prob, q, torch.ones_like(q))

    # prefix_prob[:, s] = where(q is a prob, min(prefix * p / q, 1), 0)
    prefix = torch.ones(bs, dtype=torch.float32, device=device)
    prefix_cols = []
    for s in range(num_steps):
        prefix = torch.where(
            q_is_prob[:, s],
            torch.clamp(prefix * p[:, s] / safe_q[:, s], max=1.0),
            torch.zeros_like(prefix),
        )
        prefix_cols.append(prefix)
    prefix_probs = torch.stack(prefix_cols, dim=1)

    coin = uniform_samples[:, :num_steps].float()
    accept_prob = prefix_probs.clone()
    # A partial prefix before the last step is accepted against the residual
    # mass of the NEXT target row: mass / (mass + 1 - prefix).
    for s in range(num_steps - 1):
        partial = (prefix_probs[:, s] > 0.0) & (prefix_probs[:, s] < 1.0)
        mass = (
            torch.clamp(
                prefix_probs[:, s].unsqueeze(-1) * target_probs[:, s + 1, :].float()
                - draft_probs[:, s + 1, :].float(),
                min=0.0,
            )
            .sum(dim=-1)
            .float()
        )
        accept_prob[:, s] = torch.where(
            partial,
            mass / (mass + 1.0 - prefix_probs[:, s]),
            prefix_probs[:, s],
        )

    eligible = (coin < prefix_probs) & (coin < accept_prob)
    step_ids = torch.arange(1, num_slots, device=device).expand(bs, num_steps)
    num_accept = (
        torch.where(eligible, step_ids, torch.zeros_like(step_ids))
        .amax(dim=1)
        .to(torch.int64)
    )
    picked = prefix_probs.gather(1, (num_accept - 1).clamp(min=0).unsqueeze(1)).squeeze(
        1
    )
    residual_scale = torch.where(num_accept > 0, picked, torch.ones_like(picked))
    return num_accept, residual_scale


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
    *,
    # 0.5.22 added keyword-only ``block_verification``
    # (--speculative-use-block-verification): the kernel's BLOCK_VERIFICATION
    # branch. Implemented here by ``_chain_block_verification_select``, which
    # also returns the ``residual_scale`` the correction distribution needs.
    block_verification: bool = False,
) -> None:
    """XSpeedGate ``speculative_sampling_classic_kernel``: chain rejection sampling.

    Upstream ``chain_speculative_sampling_triton`` is a wrapper that launches a
    triton kernel of the same name, and the XSpeedGate op was built against that
    *wrapper* boundary: all 14 arguments match one-to-one in name and order,
    including the two the wrapper itself marks "not used"
    (``retrive_next_sibling`` / ``deterministic``, which upstream never forwards
    to the kernel). They pass straight through with no shape or stride fixup on
    the Kunlun side. ``predicts`` / ``accept_index`` / ``accept_token_num`` are
    in-place outputs.

    DSpark reaches this call site (op-trace hits
    ``speculative.reject_sampling.chain_speculative_sampling_triton``), and the
    torch body below is a ``for bid in range(batch_size)`` loop with a per-step
    ``.item()``, so it synchronises host and device once per token. That is where
    the replacement pays off.

    DSpark's caller (``dspark_accept.py:134``) passes ``threshold_single=1.0``,
    ``threshold_acc=1.0`` and ``deterministic=True``, which are the op's own
    defaults, so wiring it changes no numerics through those three. They are
    still forwarded as given rather than hardcoded here.

    Fall back to the torch body when the op is missing or rejects this call;
    ``_dspark_warn_fallback`` logs one line so the degradation is never silent.
    """

    op = "speculative_sampling_classic_kernel"
    xsg = torch.ops.xspeedgate_ops
    if not block_verification and hasattr(xsg, "speculative_sampling_classic_kernel"):
        try:
            xsg.speculative_sampling_classic_kernel(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                uniform_samples,
                uniform_samples_for_final_sampling,
                target_probs,
                draft_probs,
                float(threshold_single),
                float(threshold_acc),
                bool(deterministic),
            )
        except RuntimeError as exc:
            # The op's dtype and layout constraints are undocumented, so its own
            # TORCH_CHECKs are the only guard. Those throw before the launch,
            # while the three outputs are still untouched, which makes falling
            # back here safe. An out-of-bounds write inside the kernel hangs the
            # device instead and never surfaces in Python, so that case is not
            # something `except` should try to cover.
            _dspark_warn_fallback(op, f"{type(exc).__name__}: {exc}")
        else:
            note_op_path(op, "xspeedgate")
            return
    else:
        _dspark_warn_fallback(
            op, "xspeedgate_ops.speculative_sampling_classic_kernel is unavailable"
        )

    note_op_path(op, "torch")
    """Batched, sync-free chain rejection sampling.

    Semantically identical to the row-by-row reference below, but every decision
    stays on device. The reference walked ``bs`` rows in Python and called
    ``.item()`` about ten times per row (candidate id, coin comparison, retrieve
    index, ``norm_sum <= 0``, ``any()``, ``argmax``), so a batch of 32 issued
    ~330 blocking D2H copies per step and 32 separate 1-D ``cumsum`` launches
    over the whole vocab. Here the same work is 1 batched cumsum and 0 syncs.

    Chain (not tree) topology is what the DSPARK buffers describe:
    ``retrive_next_token`` is ``[1, 2, ..., -1]`` and ``retrive_next_sibling`` is
    all ``-1``, so acceptance is a prefix property and the probability row used
    at step ``s`` is always ``s - 1``.
    """
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    device = candidates.device
    num_steps = num_slots - 1
    rows = torch.arange(batch_size, device=device)
    slot_index = retrive_index[:, :num_slots].to(torch.int64)

    if block_verification:
        # 0.5.22 BLOCK_VERIFICATION: longest-passing-prefix acceptance plus the
        # prefix-scaled residual, instead of the per-step loop below.
        num_accept, residual_scale = _chain_block_verification_select(
            candidates, uniform_samples, target_probs, draft_probs
        )
    elif num_steps > 0:
        token = candidates[:, 1:num_slots].to(torch.int64).unsqueeze(-1)
        p = target_probs[:, :num_steps, :].gather(2, token).squeeze(-1)
        q = draft_probs[:, :num_steps, :].gather(2, token).squeeze(-1)
        coin = uniform_samples[:, :num_steps].to(p.dtype)
        accept = coin * q < p
        step_ids = torch.arange(1, num_slots, device=device).expand(batch_size, num_steps)
        # first rejected step id, or num_slots when the whole chain is accepted
        first_reject = torch.where(
            accept, step_ids.new_full((), num_slots), step_ids
        ).amin(dim=1)
        num_accept = first_reject - 1
        residual_scale = None
    else:
        num_accept = torch.zeros(batch_size, dtype=torch.int64, device=device)
        residual_scale = None

    all_accept = num_accept >= num_steps
    target_row = target_probs[rows, num_accept]
    draft_row = draft_probs[rows, num_accept.clamp(max=draft_probs.shape[1] - 1)]
    # A degenerate draft row can carry NaN; the reference treats NaN q as 0 so the
    # residual falls back to p instead of poisoning norm_sum.
    draft_row = torch.where(torch.isnan(draft_row), draft_row.new_zeros(()), draft_row)
    # residual_scale is 1.0 for classic verification and the accepted prefix's
    # probability under block verification (kernel: ``residual_scale * p - q``).
    scaled_target = (
        target_row
        if residual_scale is None
        else residual_scale.unsqueeze(1).to(target_row.dtype) * target_row
    )
    residual = torch.where(
        all_accept.unsqueeze(1), target_row, torch.clamp(scaled_target - draft_row, min=0)
    )

    norm_sum = residual.sum(dim=1)
    threshold = uniform_samples_for_final_sampling.to(norm_sum.dtype) * norm_sum
    above = residual.cumsum(dim=1) > threshold.unsqueeze(1)
    first_above = above.to(torch.int32).argmax(dim=1).to(torch.int64)
    valid = (norm_sum > 0) & above.any(dim=1)
    final_token = torch.where(
        valid, first_above, torch.full_like(first_above, vocab_size - 1)
    )

    accept_token_num.copy_(num_accept.to(accept_token_num.dtype))
    # Only columns <= num_accept are ever read (``accept_index[row, correct_len]``);
    # writing the full row is well-defined there and leaves nothing uninitialised.
    accept_index.copy_(slot_index.to(accept_index.dtype))

    slot_ids = torch.arange(num_slots, device=device).unsqueeze(0)
    flat_index = slot_index.reshape(-1)
    current = predicts[flat_index].reshape(batch_size, num_slots)
    if num_steps > 0:
        next_token = torch.cat([candidates[:, 1:num_slots], candidates[:, -1:]], dim=1)
    else:
        next_token = candidates
    limit = num_accept.unsqueeze(1)
    value = torch.where(
        slot_ids < limit,
        next_token.to(current.dtype),
        torch.where(
            slot_ids == limit,
            final_token.unsqueeze(1).to(current.dtype),
            current,
        ),
    )
    predicts[flat_index] = value.reshape(-1)


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


def _plain_copy(t: torch.Tensor) -> torch.Tensor:
    """Materialise ``t`` as a plain contiguous tensor.

    Inference tensors (the DSPARK draft runs under ``inference_mode``) reject
    ``contiguous()`` / ``to()`` / ``view_as_real`` with "Cannot set version counter
    for inference tensor". Allocating a fresh tensor with ``torch.empty`` and
    ``copy_`` into it only touches the new tensor's counter, which PyTorch allows.
    """
    if not t.is_inference():
        return t if t.is_contiguous() else t.contiguous()
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


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
        # The op reads this as ``cat((real, imag), dim=-1)``: the real parts of
        # every frequency first, then the imaginary parts. Build exactly that.
        #
        # ``torch.cat`` on ``torch.real``/``torch.imag`` views bumps the version
        # counter of an *inference* tensor (the DSPARK draft runs under
        # inference_mode) and raises "Cannot set version_counter for inference
        # tensor", so write into a fresh tensor we own instead -- only its counter
        # moves. Do NOT use ``view_as_real(...).flatten(-2, -1)``: that is the
        # *interleaved* layout (r0, i0, r1, i1, ...), not the concatenated halves
        # (r0, r1, ..., i0, i1, ...). Feeding the interleaved form to the op pairs
        # every rotation with the wrong cos/sin, which left greedy decoding
        # fluent for the first few high-confidence tokens and then collapsed into
        # repeated punctuation.
        if freqs_cis.is_complex():
            parts = torch.view_as_real(_plain_copy(freqs_cis))
            half = parts.shape[-2]
            cached = torch.empty(
                parts.shape[:-2] + (half * 2,),
                dtype=parts.dtype,
                device=parts.device,
            )
            cached[..., :half].copy_(parts[..., 0])
            cached[..., half:].copy_(parts[..., 1])
        else:
            # Already real-valued: the real/imag split is an identity.
            cached = _plain_copy(freqs_cis)
        _DSV4_FREQS_REAL_CACHE[key] = cached
    return cached

def _dsv4_rotate_gptj_tail(
    value: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if value.ndim == 3 and value.shape[1:] == (64, 128):
        pos_i32 = _plain_copy(
            positions.flatten().to(torch.int64).to(torch.int32)
        )
        return torch.ops.xspeedgate_ops.dsv4_rotate_gptj_tail(
            value=_plain_copy(value),
            freqs_cis=_plain_copy(freqs_cis),
            positions=pos_i32,
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

    # Since 0.5.22 `_compute_q_b` calls this entry with `eps=None` when `q_head_norm=False`
    # (the DSV4.1 config default), with the contract "only do RoPE, no norm"
    # (upstream `elementwise.py:194` treats `eps is not None` as the JIT norm switch).
    # Previously this ran RMSNorm unconditionally, so q was normalized to an RMS constantly equal to 1,
    # the attention logits shrank overall and wq_b's magnitude information was lost layer by layer -- measured: the new stack has
    # q_rms=1.000000 (same for every layer), the old stack 1.75~2.12 (different per layer),
    # so under long prompts the model cannot focus precisely and just repeats nearby content.
    if eps is None:
        normalized = q_input
    else:
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
    layout=None,
    # 0.5.22 adds q: apply RoPE in-place to the same batch of tokens' query heads ([B, H, head_dim]) within this same
    # launch (upstream only ROCm's forward_with_q takes this path; the XPU branch
    # does `assert q is None`, and deepseek_v4.py:1461 passes q only on HIP).
    # The Kunlun K store here is assembled from three steps (rmsnorm + rotate + store) and has no fused-q
    # implementation, so we fail explicitly instead of dropping the argument -- silently ignoring it would leave q completely unroped.
    q: Optional[torch.Tensor] = None,
) -> None:
    """Use the normalize/rotate/store sequence for the 0.5.14 K path."""

    _assert_v4_kv_layout(layout, "fused_k_norm_rope_flashmla")
    # 0.5.22: mirror the upstream XPU branch's `assert q is None`.
    if q is not None:
        raise NotImplementedError(
            "fused_k_norm_rope_flashmla: the 0.5.22 fused query RoPE (q=) is a "
            "ROCm-only launch; the Kunlun K store does not rope q, so the "
            "caller must rope the query separately"
        )
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
    PRESHUFFLE_TILE: int = 0,
) -> None:
    """_set_k_and_s_triton_kernel

    0.5.19 added ``PRESHUFFLE_TILE`` for the AITER preshuffled indexer-K layout.
    The Kunlun op writes the plain layout, so only the disabled value is accepted.
    """
    if PRESHUFFLE_TILE:
        raise NotImplementedError(
            "_set_k_and_s_triton_kernel: preshuffled indexer-K cache "
            f"(PRESHUFFLE_TILE={PRESHUFFLE_TILE}) is not supported on Kunlun"
        )
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
    "sglang.kernels.ops.gemm.int8_kernel",
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


@register_jit_op("sglang.kernels.ops.layernorm.fused_eh_norm", "fused_eh_norm")
def fused_eh_norm(
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """MTP embed/hidden norm + concat, without the nvcc-JIT CUDA kernel.

    ``deepseek_nextn`` takes the ``_is_cuda`` branch on Kunlun and calls the JIT
    ``fused_eh_norm``, whose source is compiled with ``-std=c++20`` (rejected by
    the Kunlun nvcc 11.7 toolchain). The two RMSNorms run through
    ``kunlun_ops.rmsnorm`` and are concatenated, which is what upstream's non-CUDA
    branch does.
    """
    from sglang_kunlun.kernels.sgl_kernel_kunlun.layernorm import rmsnorm

    if inputs_embeds.shape[0] == 0:
        return torch.empty(
            (0, inputs_embeds.shape[1] * 2),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    return torch.cat(
        (
            rmsnorm(inputs_embeds.contiguous(), enorm_weight, eps),
            rmsnorm(previous_hidden.contiguous(), hnorm_weight, eps),
        ),
        dim=-1,
    )


# --------------------------------------------------------------------------- #
# DSA graph metadata (0.5.19 fused Triton builders)
#
# 0.5.19 replaced the per-field Python metadata construction with three fused
# Triton kernels in ``sglang.kernels.ops.attention.dsa_metadata``. Triton has no
# Kunlun backend, so the *wrapper functions* are replaced with Torch equivalents
# (replacing wrappers instead of kernels keeps stride/grid plumbing out of the
# plugin). Semantics follow the kernels exactly, including the "a page-table row
# is written only over its live prefix" contract: the tail keeps stale values
# across graph replays and consumers bound their reads by ``cache_seqlens``.
# --------------------------------------------------------------------------- #


def _dsa_write_page_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_row: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    real_page_table: Optional[torch.Tensor],
    max_cols: int,
    real_page_size: int,
) -> None:
    """Scatter ``req_to_token`` rows into the page-1 / real page tables.

    ``req_row[i]`` is the request owning output row ``i`` (output rows are always
    the contiguous range ``[0, len(req_row))``) and ``kv_lens[i]`` is that row's
    live kv length.
    """
    rows = req_row.numel()
    if max_cols <= 0 or rows == 0:
        return

    device = req_to_token.device
    req_idx = req_pool_indices.index_select(0, req_row).long()
    vals = req_to_token.index_select(0, req_idx)[:, :max_cols].to(torch.int32)
    cols = torch.arange(max_cols, device=device, dtype=torch.int32)
    live = cols.unsqueeze(0) < kv_lens.unsqueeze(1)

    if page_table_1 is not None:
        view = page_table_1[:rows, :max_cols]
        view.copy_(torch.where(live, vals, view))

    if real_page_size > 1 and real_page_table is not None:
        sub_vals = vals[:, ::real_page_size] // real_page_size
        sub_live = live[:, ::real_page_size]
        view = real_page_table[:rows, : sub_vals.shape[1]]
        view.copy_(torch.where(sub_live, sub_vals, view))


def _dsa_store_lens_and_cu(
    lengths: torch.Tensor, out_lengths: torch.Tensor, out_cu: torch.Tensor
) -> None:
    """Write ``lengths`` plus its cumulative sum prefixed with a zero."""
    n = lengths.numel()
    out_lengths[:n].copy_(lengths)
    out_cu[:1].zero_()
    out_cu[1 : n + 1].copy_(torch.cumsum(lengths, dim=0, dtype=torch.int32))


@register_jit_op(
    "sglang.kernels.ops.attention.dsa_metadata", "fused_dsa_decode_metadata"
)
def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    # 0.5.22 renamed the trailing ``real_page_size`` to ``physical_page_size``
    # (all upstream call sites pass it by keyword, so the name is load-bearing).
    physical_page_size: int,
) -> None:
    """Torch port of ``_fused_dsa_decode_metadata_kernel``."""
    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    seq = seq_lens[:bs].to(torch.int32)
    _dsa_store_lens_and_cu(seq, cache_seqlens, cu_seqlens_k)
    _dsa_store_lens_and_cu(
        torch.clamp(seq, max=dsa_index_topk), dsa_cache_seqlens, dsa_cu_seqlens_k
    )

    if physical_page_size <= 1:
        real_page_table = None
    _dsa_write_page_tables(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        req_row=torch.arange(bs, device=seq.device),
        kv_lens=seq,
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        max_cols=max_len,
        real_page_size=physical_page_size,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsa_metadata", "fused_dsa_target_verify_metadata"
)
def fused_dsa_target_verify_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    # 0.5.22 renamed ``real_page_size`` -> ``physical_page_size`` here too; the
    # upstream verify call sites pass every argument by keyword.
    physical_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None,
) -> None:
    """Torch port of ``_fused_dsa_target_verify_metadata_kernel``."""
    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert next_n > 0

    device = seq_lens.device
    seq = seq_lens[:bs].to(torch.int32)
    cache_seq = seq + next_n
    _dsa_store_lens_and_cu(cache_seq, cache_seqlens, cu_seqlens_k)

    expanded_size = bs * next_n
    offs_e = torch.arange(expanded_size, device=device)
    req_row = offs_e // next_n
    draft_off = (offs_e - req_row * next_n).to(torch.int32)
    base_seq = seq.index_select(0, req_row)
    expanded_seq = base_seq + draft_off + 1
    seqlens_expanded[:expanded_size].copy_(expanded_seq)
    _dsa_store_lens_and_cu(
        torch.clamp(expanded_seq, max=dsa_index_topk),
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
    )
    if paged_mqa_ctx_lens_2d is not None:
        paged_mqa_ctx_lens_2d.copy_(cache_seq.unsqueeze(1).expand(bs, next_n))

    if physical_page_size <= 1:
        real_page_table = None
    _dsa_write_page_tables(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        req_row=req_row,
        kv_lens=cache_seq.index_select(0, req_row),
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        max_cols=max_seqlen_k,
        real_page_size=physical_page_size,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsa_metadata", "fused_dsa_draft_extend_metadata"
)
def fused_dsa_draft_extend_metadata(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    total_len: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    # 0.5.22 renamed ``real_page_size`` -> ``physical_page_size`` on the
    # draft-extend entry point as well (keyword-only in practice upstream).
    physical_page_size: int,
    max_extend_len: int,
    max_total_len: int,
    static_extend_len: bool = False,
) -> None:
    """Torch port of ``_fused_dsa_draft_extend_metadata_kernel``."""
    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    seq = seq_lens[:bs].to(torch.int32)
    if total_len == 0:
        _dsa_store_lens_and_cu(seq, cache_seqlens, cu_seqlens_k)
        dsa_cu_seqlens_k[:1].zero_()
        return

    device = seq_lens.device
    _dsa_store_lens_and_cu(seq, cache_seqlens, cu_seqlens_k)

    ext = extend_seq_lens[:bs].to(torch.int32)
    # ``output_size`` keeps repeat_interleave from reading the lengths back to
    # the host, which would sync in the replay hot path.
    req_row = torch.repeat_interleave(
        torch.arange(bs, device=device), ext.long(), output_size=total_len
    )
    prefix = torch.cumsum(ext, dim=0, dtype=torch.int32) - ext
    local_off = torch.arange(total_len, device=device, dtype=torch.int32) - (
        prefix.index_select(0, req_row)
    )
    # Padded / idle rows carry the graph seq_len fill value, which can make the
    # visible kv length negative; consumers read these lengths as unsigned, so
    # clamp at 0 to keep such rows on the trivial all-(-1) top-k path.
    expanded_seq = torch.clamp(
        seq.index_select(0, req_row) - ext.index_select(0, req_row) + local_off + 1,
        min=0,
    )
    seqlens_expanded[:total_len].copy_(expanded_seq)
    _dsa_store_lens_and_cu(
        torch.clamp(expanded_seq, max=dsa_index_topk),
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
    )

    if physical_page_size <= 1:
        real_page_table = None
    _dsa_write_page_tables(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        req_row=req_row,
        kv_lens=seq.index_select(0, req_row),
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        max_cols=max_seqlen_k,
        real_page_size=physical_page_size,
    )


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.cp_split",
    "dsa_cp_interleave_q_seqs_kernel",
)
def dsa_cp_interleave_q_seqs_kernel(
    in_seqs: torch.Tensor,
    out_seqs: torch.Tensor,
    bs_idx: torch.Tensor,
    tokens: int,
    cp_size: int,
    cp_rank: int,
    **kwargs,
) -> None:
    """Round-robin CP q-sequence split, without the sequential Triton loop."""
    if out_seqs.numel() == 0:
        return

    lengths = in_seqs[:tokens].to(torch.int64)
    total = torch.cumsum(lengths, dim=0)
    taken = torch.div(total, cp_size, rounding_mode="floor") + (
        total % cp_size > cp_rank
    ).to(total.dtype)
    per_seq = taken - torch.cat([taken.new_zeros(1), taken[:-1]])

    selected = torch.nonzero(per_seq > 0, as_tuple=False).flatten()
    if selected.numel() != out_seqs.numel():
        raise RuntimeError(
            "dsa_cp_round_robin_split_q_seqs: selected rows do not match the "
            f"preallocated output ({selected.numel()} vs {out_seqs.numel()}; "
            f"tokens={tokens}, cp_size={cp_size}, cp_rank={cp_rank})"
        )
    out_seqs.copy_(per_seq.index_select(0, selected).to(out_seqs.dtype))
    bs_idx.copy_(selected.to(bs_idx.dtype))


@register_jit_op(
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
    "swiglu_no_interleaved_with_alpha_and_limit",
)
def swiglu_no_interleaved_with_alpha_and_limit_kunlun(
    x: torch.Tensor,
    gemm1_alpha: float,
    gemm1_limit: float,
) -> torch.Tensor:
    """Apply the Kunlun fused non-interleaved SwiGLU-OAI replacement.

    MiniMax-M3 configures gemm1_alpha / gemm1_clamp_limit (SwiGLU-OAI), which
    upstream routes through this helper; the W8A8 MoE hook below bypasses the
    MoeRunner, so the helper is replaced here and called explicitly.
    """
    input_2d = x.contiguous().view(-1, x.shape[-1])
    output = torch.empty(
        (input_2d.shape[0], input_2d.shape[1] // 2),
        dtype=x.dtype,
        device=x.device,
    )

    import kunlun_ops

    kunlun_ops.swiglu_bias(
        input_2d,
        output,
        None,
        float(gemm1_alpha),
        1.0,
        float(gemm1_limit),
        True,
    )
    return output.view(x.shape[:-1] + (x.shape[-1] // 2,))

# ===========================================================================
# HiSparse (hierarchical sparse attention) swap-in / prefetch.
#
# Upstream drives the decode-side KV swap-in with JIT CUDA kernels
# (``sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh``) that cannot be built on
# Kunlun.  The three symbols below are the complete runtime hot path of the
# DeepSeek-V4 hisparse decode: the fused top-k/hit/miss/LRU swap-in, the
# copy-only replay used by the shared-index prefetch, and (for completeness)
# the generic MLA variant.  Each is reimplemented with plain torch ops that
# reproduce the CUDA kernel's semantics:
#
#   * per request, the ``top_k`` selected token positions are matched against
#     the LRU-ordered device-buffer slots (hit / newest-token / miss);
#   * misses evict the LRU-most non-hit slots, the miss plan is recorded for
#     the skip-layer prefetch and the LRU order is rewritten as
#     [miss-evicted evictables][remaining evictables][hits];
#   * the missed host->device items are copied with the layout-specific
#     addressing (DSv4 page-padded C4 row, or a flat byte stride).
#
# All shapes are static (batch, top_k, hot_buffer_size) and no host
# synchronisation is issued, so the ops stay usable inside a captured decode
# CUDA graph.  ``SGLANG_KUNLUN_HISPARSE_TRACE=1`` prints the first few calls.
# ===========================================================================

from sglang_kunlun.kernels.sgl_kernel_kunlun.dsv4_hisparse import (
    DSV4_PAGE_SIZE as _HISPARSE_DSV4_PAGE_SIZE,
    copy_items as _hisparse_copy_items,
)

def _hisparse_load_cache_to_device_buffer(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    hot_buffer_size: int,
    num_real_reqs: Optional[torch.Tensor],
    miss_src: Optional[torch.Tensor],
    miss_dst: Optional[torch.Tensor],
    miss_count: Optional[torch.Tensor],
    skip_io: bool,
    backed_up_len: Optional[torch.Tensor] = None,
) -> None:
    """Torch-native replacement for ``load_cache_to_device_buffer{,_dsv4}_mla``."""
    batch_size, top_k = top_k_tokens.shape
    if batch_size == 0:
        return
    device = top_k_tokens.device
    hot = int(hot_buffer_size)
    max_slots = device_buffer_tokens.shape[0]
    buffer_len = device_buffer_tokens.shape[1]
    host_stride = host_cache_locs.shape[1]
    # Slot `hot` is the reserved newest-token slot: it sits outside the LRU
    # window ([0, hot)) and its token entry is pinned to -1, so it doubles as
    # the scratch destination for the no-op writes of padded/irrelevant rows.
    scratch_slot = hot

    if num_real_reqs is None:
        num_real = torch.full((1,), batch_size, dtype=torch.int32, device=device)
    else:
        num_real = num_real_reqs.reshape(-1)[:1]
    num_real_scalar = num_real.to(torch.int64).reshape(())

    k_ids = torch.arange(top_k, dtype=torch.int64, device=device)
    lru_pos = torch.arange(hot, dtype=torch.int64, device=device)
    out_sentinel = k_ids.new_full((), top_k)
    scratch_idx = lru_pos.new_full((), scratch_slot)
    neg_one = torch.zeros((), dtype=torch.int32, device=device) - 1

    # lru_slots is int16 and XPU's advanced indexing rejects int16, so stage the
    # whole table through int32 for the gather/scatter.
    lru_all = lru_slots.to(torch.int32)


    # DSV4 TARGET_VERIFY has multiple rows for the same request.  Their
    # newest tokens and top-k sets are different, so the request-level
    # reserved slot and the per-column LRU update cannot represent them
    # safely.  Use a stable, row-owned slot range for the whole call.
    Kd = top_k
    rids = req_pool_indices.to(torch.int64).clamp(0, max_slots - 1)
    seq_l = seq_lens.to(torch.int64).reshape(batch_size, 1)
    real = (
        torch.arange(batch_size, device=device) < num_real_scalar
    ).reshape(batch_size, 1)
    k_row = torch.arange(Kd, dtype=torch.int64, device=device).reshape(1, Kd)
    tok = top_k_tokens.to(torch.int64)
    dev_loc = device_buffer_locs[rids]
    host_row = host_cache_locs[rids]
    # Host slot backing each selected position (-1 == never backed up).
    src_host_early = torch.gather(host_row, 1, tok.clamp(0, host_stride - 1))

    if is_dsv4_layout:
        # Rows belonging to one request are contiguous after
        # repeat_interleave in the backend.  Each row gets Kd stable slots.
        row_id = torch.arange(batch_size, dtype=torch.int64, device=device)
        boundary = torch.ones(batch_size, dtype=torch.bool, device=device)
        if batch_size > 1:
            boundary[1:] = rids[1:] != rids[:-1]
        run_start = torch.cummax(
            torch.where(boundary, row_id, torch.zeros_like(row_id)), dim=0
        ).values
        row_rank = row_id - run_start
        max_rows = max(1, hot // Kd)
        row_overflow = row_rank >= max_rows
        safe_rank = row_rank.clamp(0, max_rows - 1)
        row_slot = safe_rank.reshape(batch_size, 1) * Kd + k_row
        row_slot = row_slot.clamp(0, hot - 1)

        # Row-owned slot ranges need hot >= rows_per_req * Kd.  The production
        # config has hot == Kd == 512, so max_rows == 1 and every DSPARK verify
        # row but the first was dropped (locs = -1), i.e. 11 of 12 rows attended
        # to nothing.  Whenever the whole compressed sequence fits the hot
        # window, a token's slot is simply its own id (alloc_device_buffer seeds
        # tokens[j] = j), and that mapping is request-level: all rows of the
        # request can share it, exactly like the generic planner's fast path
        # below.  Fall back to the row-owned ranges only past the hot window.
        fits = seq_l <= hot
        row_slot = torch.where(fits, tok.clamp(0, hot - 1), row_slot)

        newest = seq_l - 1
        valid = (
            real
            & (row_overflow.reshape(batch_size, 1).logical_not() | fits)
            & (tok >= 0)
            & (k_row < seq_l)
        )
        # Causal bound on the SELECTED POSITION (tok is the compressed position,
        # seq_l the row's compressed length).  The sparse attention consumes raw
        # device locs with no per-row causal mask, so without this a verify row
        # attends to tokens it has not produced yet.
        valid = valid & (tok < seq_l)
        # Non-fits only: drop attended entries with NO host backing.  Past the
        # hot window a token's device slot is its top-k RANK, not its position,
        # so an un-backed-up entry would read a slot still holding a recycled
        # position's KV.  The fits regime must NOT be gated (there slot ==
        # position, so the untouched slot is already correct).
        valid = valid & ((src_host_early >= 0) | fits)
        # The newest compressed position has no host copy yet (advance registers
        # it at the next step's prepare) and its device-buffer slot is one of
        # alloc_device_buffer's spare slots, i.e. recycled residue.  Route it to
        # the reserved slot instead, which the coordinator points at the real
        # mapped location.
        is_newest = valid & (tok == newest)

        cold_valid = valid & ~is_newest
        # Per-request watermark of compressed positions actually backed up to
        # host.  An attended entry with tok >= wm has no host refresh this step,
        # so its device-buffer slot may still hold a recycled request's KV.
        wm = (
            backed_up_len.to(torch.int64)[rids].reshape(batch_size, 1)
            if backed_up_len is not None
            else None
        )
        tok_host = tok.clamp(0, host_stride - 1)
        row_locs = torch.gather(dev_loc, 1, row_slot).to(torch.int32)
        newest_locs = dev_loc[:, hot].reshape(batch_size, 1).expand(
            batch_size, Kd
        ).to(torch.int32)
        top_k_device_locs.copy_(
            torch.where(
                is_newest,
                newest_locs,
                torch.where(valid, row_locs, neg_one),
            )
        )

        # Mark a slot resident only where its data was actually copied in.
        # The host-loc guard above suppresses the copy for positions that have
        # no host backing, but the token table was still being stamped for them
        # -- so the next step saw "slot holds token tok" while the slot still
        # held whatever the PREVIOUS owner of that recycled pool slot left.
        # Benign on the first pass after a boot (slots are zero) and poison on
        # every later pass, which is exactly the measured collapse pattern.
        src_host_mark = torch.gather(host_row, 1, tok_host)
        cold_loaded = cold_valid & (src_host_mark >= 0)
        write_slot = torch.where(cold_loaded, row_slot, scratch_idx)
        write_val = torch.where(cold_loaded, tok, neg_one.to(torch.int64)).to(
            device_buffer_tokens.dtype
        )
        device_buffer_tokens[
            rids.reshape(batch_size, 1).expand(batch_size, Kd), write_slot
        ] = write_val

        if miss_src is not None:
            miss_src.zero_()
            miss_dst.zero_()
            miss_count.zero_()

        if not skip_io:
            src_host = torch.gather(host_row, 1, tok_host)
            src_locs = src_host.reshape(-1)
            dst_locs = row_locs.to(torch.int64).reshape(-1)
            # A compressed position only has a host copy once it has been backed
            # up; ``req_to_host_pool`` holds -1 until then.  Copying from loc -1
            # reads out of bounds and OVERWRITES the device-buffer slot that is
            # about to be attended to with garbage (observed sporadically on
            # real decodes, ~20-60 per 100 steps per DP rank -- exactly the
            # "occasional repeat collapses" shape of the ifeval regression).
            # Entries without a host copy are already correct in the device
            # buffer (alloc_device_buffer seeds slot j with token j and staging
            # filled it), so the right action is to skip the copy, not to clamp
            # the address.
            copy_mask = cold_valid & (src_host >= 0)
            if wm is not None:
                # Refresh a slot from host ONLY when that position was actually
                # backed up.  req_to_host_pool is page-granular, so an
                # un-backed-up position still carries a >=0 loc whose page holds
                # the previous owner's KV; copying it overwrites the correctly
                # preloaded device-buffer slot with foreign data.  The entry is
                # still attended (from the device buffer), unlike the refuted
                # "drop the entry" arm.
                copy_mask = copy_mask & (tok < wm)
            valid_flat = copy_mask.reshape(-1)
            dev_scratch = dst_locs.new_full(
                (), device_buffer.shape[0] * _HISPARSE_DSV4_PAGE_SIZE - 1
            )
            host_scratch = src_locs.new_full(
                (), host_cache.shape[0] * _HISPARSE_DSV4_PAGE_SIZE - 1
            )
            _hisparse_copy_items(
                host_cache,
                device_buffer,
                torch.where(valid_flat, src_locs, host_scratch),
                torch.where(valid_flat, dst_locs, dev_scratch),
                item_size_bytes,
                is_dsv4_layout,
            )

        lru_all[rids] = lru_pos
        lru_slots.copy_(lru_all.to(lru_slots.dtype))
        return

    # ================= vectorized planner (batch folded into dim 0) =================
    # The row loop above launched batch x ~15 ops over hot-sized tensors every
    # decode step; folding the batch into dim 0 turns that into one op per stage
    # (~6x fewer XPU kernel launches).  Bit-exact vs the loop on distinct-rid
    # inputs; rows sharing a rid (DSPARK verify) write identical values so the
    # duplicate index_put is order-independent.  See doc/hisparse_perf_opt.md.
    # The common path below handles non-DSV4 requests with distinct request
    # ids and keeps the original LRU planner semantics.
    use_fast = torch.logical_or(torch.logical_not(real), seq_l <= hot)        # (B,1)
    long_active = torch.logical_not(use_fast)                                 # (B,1)
    k_row = k_ids.reshape(1, Kd)                                              # (1,K)
    sent = out_sentinel

    tok = top_k_tokens.to(torch.int64)                                        # (B,K)
    dev_tok = device_buffer_tokens[rids]                                      # (B,L)
    dev_loc = device_buffer_locs[rids]                                        # (B,L)
    host_row = host_cache_locs[rids]                                          # (B,H)
    lru = lru_all[rids].to(torch.int64)                                       # (B,hot)

    fast_out = torch.gather(dev_loc, 1, tok.clamp(0, buffer_len - 1)).to(torch.int32)
    fast_active = use_fast & (k_row < seq_l) & (tok >= 0)

    newest = seq_l - 1
    tok_host = tok.clamp(0, host_stride - 1)
    is_newest = tok == newest
    tok_key = torch.where(is_newest, tok.new_full((), -1), tok)
    order = torch.argsort(tok_key, dim=1)
    sorted_key = torch.gather(tok_key, 1, order)
    buf_token = torch.gather(dev_tok, 1, lru).to(torch.int64)
    pos = torch.searchsorted(sorted_key, buf_token)
    pos_c = pos.clamp(0, Kd - 1)
    matched = (pos < Kd) & (torch.gather(sorted_key, 1, pos_c) == buf_token) & (buf_token >= 0)
    match_idx = torch.gather(order, 1, pos_c)                                 # (B,hot)

    hit_active = long_active & matched
    newest_active = long_active & is_newest

    hit_topk = torch.zeros(batch_size, Kd + 1, dtype=torch.bool, device=device)
    hit_topk.scatter_(1, torch.where(matched, match_idx, sent), matched)
    hit_topk = hit_topk[:, :Kd]

    miss_mask = torch.logical_not(hit_topk) & torch.logical_not(is_newest) & long_active
    miss_rank = torch.cumsum(miss_mask.to(torch.int64), dim=1) - 1
    evict_mask = torch.logical_not(matched)
    evict_rank = torch.cumsum(evict_mask.to(torch.int64), dim=1) - 1
    slot_of_rank = torch.zeros(batch_size, hot + 1, dtype=torch.int64, device=device)
    slot_of_rank.scatter_(
        1, torch.where(evict_mask, evict_rank, scratch_idx).clamp_(0, hot), lru
    )
    rank_for_miss = torch.where(miss_mask, miss_rank, scratch_idx).clamp_(0, hot)
    miss_slot = torch.gather(slot_of_rank, 1, rank_for_miss)                  # (B,K)

    # Output slot table.  fast/newest/miss all target output column k and are
    # mutually exclusive there (fast rows are use_fast; newest vs miss disjoint by
    # miss_mask; hit's newest-token key=-1 never matches a buffer token so hit never
    # lands on a newest/miss column), so select those THREE elementwise instead of via
    # three separate scatter_ (profiling: scatter_v2_small_index was the #1 decode
    # kernel).  Only the hit case (indexed by match_idx over the hot dim) still scatters.
    newest_col = dev_loc[:, hot].reshape(batch_size, 1).to(torch.int32)
    miss_loc = torch.gather(dev_loc, 1, miss_slot).to(torch.int32)
    col_val = torch.where(
        fast_active, fast_out,
        torch.where(
            newest_active, newest_col.expand(batch_size, Kd),
            torch.where(miss_mask, miss_loc, neg_one),
        ),
    )
    out_pad = torch.full((batch_size, Kd + 1), -1, dtype=torch.int32, device=device)
    out_pad[:, :Kd] = col_val
    out_pad.scatter_(
        1, torch.where(hit_active, match_idx, sent),
        torch.gather(dev_loc, 1, lru).to(torch.int32),
    )
    top_k_device_locs.copy_(torch.where(real, out_pad[:, :Kd], neg_one))

    # device-buffer token bookkeeping (misses write their token id into the slot)
    write_slot = torch.where(miss_mask, miss_slot, scratch_idx)
    write_val = torch.where(miss_mask, tok, neg_one.to(torch.int64)).to(device_buffer_tokens.dtype)
    device_buffer_tokens[rids.reshape(batch_size, 1).expand(batch_size, Kd), write_slot] = write_val

    # miss plan for the shared-index prefetch (park non-misses at the sentinel col Kd)
    if miss_src is not None:
        plan_rank = torch.where(miss_mask, miss_rank.to(torch.int32), miss_rank.new_full((), Kd)).clamp_(0, Kd)
        src_plan = torch.full((batch_size, Kd + 1), 0, dtype=miss_src.dtype, device=device)
        src_plan[:, :Kd].copy_(miss_src)
        src_plan.scatter_(1, plan_rank.to(torch.int64), torch.gather(host_row, 1, tok_host).to(miss_src.dtype))
        miss_src.copy_(src_plan[:, :Kd])
        dst_plan = torch.full((batch_size, Kd + 1), 0, dtype=miss_dst.dtype, device=device)
        dst_plan[:, :Kd].copy_(miss_dst)
        dst_plan.scatter_(1, plan_rank.to(torch.int64), torch.gather(dev_loc, 1, miss_slot).to(miss_dst.dtype))
        miss_dst.copy_(dst_plan[:, :Kd])
        miss_count.copy_(miss_mask.to(torch.int32).sum(dim=1))

    # LRU rewrite: [evictables not used by a miss][miss-evicted evictables][hits]
    lru_row = lru_pos.reshape(1, hot)
    total_misses = miss_mask.to(torch.int64).sum(dim=1, keepdim=True)
    slot_class = torch.where(
        matched, lru_pos.new_full((), 2),
        torch.where(evict_rank >= total_misses, lru_pos.new_full((), 0), lru_pos.new_full((), 1)),
    )
    new_lru = torch.gather(lru, 1, torch.argsort(slot_class * hot + lru_row, dim=1))
    lru_all[rids] = torch.where(use_fast, lru, new_lru).to(torch.int32)

    # host -> device copy of the misses (single batched call)
    if not skip_io:
        src_locs = torch.gather(host_row, 1, tok_host).reshape(-1)
        dst_locs = torch.gather(dev_loc, 1, miss_slot).to(torch.int64).reshape(-1)
        mm_flat = miss_mask.reshape(-1)
        if is_dsv4_layout:
            dev_scratch = dst_locs.new_full((), device_buffer.shape[0] * _HISPARSE_DSV4_PAGE_SIZE - 1)
            host_scratch = src_locs.new_full((), host_cache.shape[0] * _HISPARSE_DSV4_PAGE_SIZE - 1)
            src_locs = torch.where(mm_flat, src_locs, host_scratch)
            dst_locs = torch.where(mm_flat, dst_locs, dev_scratch)
        else:
            src_locs = src_locs[mm_flat]
            dst_locs = dst_locs[mm_flat]
        _hisparse_copy_items(
            host_cache, device_buffer, src_locs, dst_locs, item_size_bytes, is_dsv4_layout
        )

    lru_slots.copy_(lru_all.to(lru_slots.dtype))


@register_jit_op(
    "sglang.kernels.ops.kvcache.hisparse", "load_cache_to_device_buffer_dsv4_mla"
)
def hisparse_load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: Optional[torch.Tensor] = None,
    miss_src: Optional[torch.Tensor] = None,
    miss_dst: Optional[torch.Tensor] = None,
    miss_count: Optional[torch.Tensor] = None,
    skip_io: bool = False,
    backed_up_len: Optional[torch.Tensor] = None,
) -> None:
    """DSv4 hisparse swap-in (page-padded device + page-padded host C4)."""
    _hisparse_load_cache_to_device_buffer(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        hot_buffer_size=hot_buffer_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
        backed_up_len=backed_up_len,
    )



@register_jit_op(
    "sglang.kernels.ops.kvcache.hisparse", "load_cache_to_device_buffer_mla"
)
def hisparse_load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: Optional[torch.Tensor] = None,
    miss_src: Optional[torch.Tensor] = None,
    miss_dst: Optional[torch.Tensor] = None,
    miss_count: Optional[torch.Tensor] = None,
    skip_io: bool = False,
    backed_up_len: Optional[torch.Tensor] = None,
) -> None:
    """Generic MLA hisparse swap-in (linear device + linear host stride)."""
    _hisparse_load_cache_to_device_buffer(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        hot_buffer_size=hot_buffer_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
        backed_up_len=backed_up_len,
    )


@register_jit_op("sglang.kernels.ops.kvcache.hisparse", "copy_cache_planned_mla")
def hisparse_copy_cache_planned_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
    skip_io: bool = False,
) -> None:
    """Replay a recorded miss plan (host -> device) for a shared-index layer."""
    del num_blocks, block_size
    if skip_io:
        return
    counts = miss_count.reshape(-1).to(torch.int64)
    n_rows, capacity = miss_src.shape
    row_ids = torch.arange(n_rows, device=miss_src.device).unsqueeze(1).expand(
        n_rows, capacity
    )
    col_ids = torch.arange(capacity, device=miss_src.device).unsqueeze(0).expand(
        n_rows, capacity
    )
    valid = col_ids < counts.unsqueeze(1)
    src_locs = miss_src[row_ids[valid], col_ids[valid]]
    dst_locs = miss_dst[row_ids[valid], col_ids[valid]]
    _hisparse_copy_items(
        host_cache, device_buffer, src_locs, dst_locs, item_size_bytes, is_dsv4_layout
    )
@register_jit_op(
    "sglang.kernels.ops.mamba.mamba_state_indices_triton",
    "fused_replay_state_indices",
)
def fused_replay_state_indices(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
    # 0.5.22 added ``v2p``: the mamba virtual->physical slot table of a unified
    # memory pool (None for the static hybrid pool, whose mapping already yields
    # physical slots). The kernel gathers through it right after the mapping
    # gather and BEFORE the padding sentinel, so it is reproduced in that order.
    v2p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Torch form of the mamba replay state-indices fusion."""
    valid = max(0, min(int(valid_bs), int(total_bs)))
    total = int(total_bs)

    # Gather first: the Triton kernel masks its load to the valid rows, so the
    # padding zero-fill below cannot influence the gathered values. Doing it in
    # this order keeps that true no matter how the buffers alias.
    if valid > 0:
        req = req_pool_indices[:valid].to(torch.long)
        idx = mamba_index_mapping[req]
        if v2p is not None:
            # 0.5.22 virtual->physical mamba slot translate (page_size 1, so a
            # plain table gather), applied before the sentinel write.
            idx = v2p[idx.to(torch.long)]
        out_state_indices[:valid] = idx.to(out_state_indices.dtype)
    if total > valid:
        # -1 is the sentinel mamba kernels test with `state_idx < 0`.
        out_state_indices[valid:total] = -1
        # Side effect the reference chain relies on: padded rows of the static
        # req_pool_indices buffer are zeroed so captured kernels that gather
        # with them stay in bounds.
        req_pool_indices[valid:total] = 0
    return out_state_indices[:total]


def _kunlun_renorm(name: str):
    """Fetch a renorm implementation from the Kunlun sgl_kernel shim, lazily.

    Imported inside the call rather than at module scope: kernel_ops is loaded
    very early in the plugin bootstrap, before the shim package is safe to pull
    in.
    """

    from sglang_kunlun.kernels import sgl_kernel_kunlun  # noqa: F401

    from sglang_kunlun.kernels.sgl_kernel_kunlun import sampling as _sampling

    return getattr(_sampling, name)


@register_jit_op(
    "sglang.srt.speculative.dflash_utils",
    "top_p_renorm_prob",
)
def dflash_top_p_renorm_prob_kunlun(
    probs: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Route DFlash/DSpark's top-p renorm to the Kunlun op instead of flashinfer.

    ``dflash_utils.py:44-47`` binds this symbol under ``if is_cuda():`` straight
    from ``flashinfer.sampling`` -- and Kunlun XPU answers ``is_cuda()`` with
    True. That bypasses ``sgl_kernel_stub.install()``, which deliberately points
    ``sgl_kernel.top_p_renorm_prob`` at the Kunlun implementation. The main
    sampler (``srt/layers/sampler.py:40-43``) imports the ``sgl_kernel``
    spelling and is therefore fine; this one file takes the flashinfer spelling
    and is not.

    flashinfer's version does not run on XPU. It does not raise either: it
    returns its uninitialised ``torch.empty_like`` output buffer. Measured on a
    real post-softmax row, ``rowsum=-1979.7``, ``max=22.8``, nothing filtered --
    and the device context is left in a state where the next op fails with
    ``invalid device function``. Feeding that to the verify sampler makes the
    residual ``p - q`` meaningless, so every emitted token is drawn from what is
    effectively a uniform distribution over the vocabulary: with
    ``temperature=0.8, top_p=0.95`` the model answers in random multilingual
    tokens.

    The failure hides well. Whether the garbage looks wrong depends on what the
    caching allocator last left in that block -- hand it the buffer a correct
    call just freed and the "result" compares bit-identical to the right answer.

    Only reachable with speculative decoding on: ``_accept_sampling_core``
    routes to ``build_dflash_verify_target_probs`` (the sole caller) as soon as
    ``need_top_p_sampling`` is set, and to ``SoftmaxTemp`` otherwise, which is
    why plain temperature and top-k were unaffected.
    """

    return _kunlun_renorm("top_p_renorm_prob")(probs, top_p)


@register_jit_op(
    "sglang.srt.speculative.dflash_utils",
    "top_k_renorm_prob",
)
def dflash_top_k_renorm_prob_kunlun(
    probs: torch.Tensor, top_k: torch.Tensor
) -> torch.Tensor:
    """Same redirect for the top-k renorm, which sits on the same import line.

    Reached on the dense branch of ``build_dflash_verify_target_probs``
    (``dflash_utils.py:1185-1191``) -- a mixed batch is enough, since one request
    at ``TOP_K_ALL`` pushes ``max_top_k`` to the vocabulary size and disables the
    sparse top-k fast path for the whole batch. Note that branch passes
    ``sampling_info.top_ks`` through unclamped, so the Kunlun op has to tolerate
    ``TOP_K_ALL == 1 << 30``; it does, treating any k past the vocabulary as no
    filtering.
    """

    return _kunlun_renorm("top_k_renorm_prob")(probs, top_k)


@register_triton_op(
    "sglang.kernels.ops.speculative.dspark.commit_swa",
    "_committed_swa_locations",
)
def committed_swa_locations_kernel(
    LOC: torch.Tensor,
    MAP: torch.Tensor,
    LENS: torch.Tensor,
    OUT: torch.Tensor,
    N: int,
    WIDTH: int,
    MAP_SIZE: int,
    BLOCK: int,
) -> None:
    """Translate committed cache locations into draft-SWA slots, in torch.

    New in 0.5.22: ``dspark_kv_inject._inject_mla`` routes every committed
    decode batch through this kernel when ``cache_loc.is_cuda`` -- true on
    Kunlun -- so the launch is unavoidable and there is no torch branch to fall
    into inside the kernel's own wrapper.

    Mirrors the kernel row for row: ``row = i // WIDTH``, ``col = i % WIDTH``, a
    lane is committed when ``col < LENS[row]``; a committed lane reads
    ``MAP[LOC[i]]`` with the negative-index wrap torch would apply
    (``LOC[i] + MAP_SIZE``), and an uncommitted lane stores ``-1``.
    """

    del BLOCK, N
    rows = LENS.numel()
    if OUT.numel() == 0:
        return
    loc = LOC.view(rows, WIDTH).to(torch.int64)
    loc = torch.where(loc < 0, loc + MAP_SIZE, loc)
    # Uncommitted lanes are overwritten with -1 below; clamp only so the gather
    # cannot index out of bounds on their garbage locations.
    loc = loc.clamp_(0, max(MAP_SIZE - 1, 0))
    swa = MAP[loc].to(torch.int32)
    committed = torch.arange(WIDTH, device=LOC.device).view(1, WIDTH) < LENS.view(
        rows, 1
    )
    OUT.view(rows, WIDTH).copy_(torch.where(committed, swa, -torch.ones_like(swa)))


@register_triton_op(
    "sglang.kernels.ops.memory.allocator",
    "get_and_clear_swa_pages_kernel",
)
def get_and_clear_swa_pages_kernel_kunlun(
    full_page_representatives: torch.Tensor,
    mapping: torch.Tensor,
    swa_pages: torch.Tensor,
    peers_mapped: torch.Tensor,
    page_mappings_valid: torch.Tensor,
    num_pages: int,
    index_stride: int,
    page_size: int,
    CHECK_PAGE_MAPPINGS: bool,
    BLOCK_PAGES: int,
    BLOCK_OFFSETS: int,
) -> None:
    """Resolve one SWA page per FULL-page representative and clear the mapping.

    New in 0.5.22: ``SWATokenToKVPoolAllocator._free_swa_pages_cuda`` does the
    resolve-and-clear in one Triton launch instead of the old gather + separate
    ``clear_full_to_swa_mapping``. There is no device gate to fall back through,
    so every SWA free on Kunlun hit "0 compatible backends for target (cuda)".

    Mirrors the kernel, including its ordering guarantee: the kernel issues
    ``tl.debug_barrier()`` so every ``mapping[full_reps]`` load retires before
    any lane zeroes ``mapping[page_start .. page_start + page_size)``. Sequential
    torch ops give that for free.

    ``page_mappings_valid`` aliases ``peers_mapped`` when the caller did not ask
    for the check (see the wrapper), so it must stay untouched in that case.
    """

    del BLOCK_PAGES, BLOCK_OFFSETS
    if num_pages == 0:
        return

    reps = full_page_representatives
    if reps.dim() != 1 or reps.stride(0) != index_stride:
        reps = reps.as_strided((num_pages,), (index_stride,))
    full_reps = reps[:num_pages].to(torch.int64)

    swa_reps = mapping[full_reps]
    # Triton's ``//`` on integers truncates toward zero; torch's ``//`` floors.
    # ``full_to_swa_index_mapping`` can hold ``-1`` sentinels, where the two
    # differ (``-1 // 4`` is 0 in the kernel and -1 in torch), so the division
    # has to match the kernel exactly rather than "look right" for non-negatives.
    swa_pages[:num_pages].copy_(_trunc_div(swa_reps, page_size))
    peers_mapped[:num_pages].copy_(swa_reps > 0)

    page_starts = _trunc_div(full_reps, page_size) * page_size
    offsets = torch.arange(page_size, device=mapping.device, dtype=torch.int64)
    mapping_offsets = page_starts[:, None] + offsets[None, :]

    if CHECK_PAGE_MAPPINGS:
        page_mapping = mapping[mapping_offsets]
        same_swa_page = _trunc_div(page_mapping, page_size) == _trunc_div(
            swa_reps[:, None], page_size
        )
        valid = (swa_reps > 0) & (
            ((page_mapping > 0) & ~same_swa_page).sum(dim=1) == 0
        )
        page_mappings_valid[:num_pages].copy_(valid)

    mapping[mapping_offsets.reshape(-1)] = 0


@register_jit_op(
    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
    "combine_topk_swa_indices",
)
def combine_topk_swa_indices_kunlun(
    topk_indices, query_start_loc, query_pos, seq_lens, gather_lens,
    compressed_base, swa_base, window_size, compress_ratio, topk,
    out_indices=None, out_lens=None, swa_indices=None, swa_lengths=None,
):
    """Combine compressed and SWA workspace indices without a CUDA/Triton launch.

    Preserve holes and scanned-prefix lengths, including explicit request-window
    indices introduced in 0.5.22. Output buffers may be reused across layers.

    Args:
        topk_indices: Per-query compressed-token selections; negative entries are holes.
        query_start_loc: Cumulative query offsets delimiting requests.
        query_pos: Absolute position of each query in its request.
        seq_lens: Total sequence length for each request.
        gather_lens: Number of SWA tokens gathered for each request.
        compressed_base: Per-request base offset in the compressed workspace.
        swa_base: Per-request base offset in the SWA workspace.
        window_size: Maximum SWA entries per query.
        compress_ratio: Number of original tokens represented by a compressed token.
        topk: Maximum number of compressed selections to include.
        out_indices: Optional reusable int32 buffer for combined indices.
        out_lens: Optional reusable int32 buffer for scanned-prefix lengths.
        swa_indices: Optional explicit per-query indices relative to swa_base.
        swa_lengths: Per-query window lengths required with explicit swa_indices.

    Returns:
        Combined workspace indices and scanned-prefix lengths, preserving holes.
    """
    assert compress_ratio >= 1 and topk_indices.shape[-1] >= topk
    n = topk_indices.shape[0]
    width = ((topk + window_size + 127) // 128) * 128
    device = topk_indices.device
    if out_indices is None:
        out_indices = torch.empty((n, width), dtype=torch.int32, device=device)
    if out_lens is None:
        out_lens = torch.empty(n, dtype=torch.int32, device=device)
    assert out_indices.shape == (n, width) and out_indices.dtype == torch.int32
    assert out_lens.shape == (n,) and out_lens.dtype == torch.int32
    if n == 0 or seq_lens.numel() == 0:
        out_indices.fill_(-1)
        out_lens.zero_()
        return out_indices, out_lens
    offsets = query_start_loc - query_start_loc[0]
    rows = torch.arange(n, device=device, dtype=offsets.dtype)
    owned = rows < offsets[-1]
    req = (torch.searchsorted(offsets, rows, right=True) - 1).clamp(
        0, seq_lens.numel() - 1
    ).long()
    pos = query_pos
    top_lens = torch.where(
        owned, ((pos + 1) // compress_ratio).clamp_max(topk), 0
    )
    if swa_indices is None:
        window_lens = torch.where(owned, (pos + 1).clamp_max(window_size), 0)
    else:
        assert swa_indices.shape == (n, window_size)
        assert swa_lengths is not None and swa_lengths.shape == (n,)
        window_lens = torch.where(owned, swa_lengths, 0)
    columns = torch.arange(width, device=device)[None, :]
    in_top = columns < top_lens[:, None]
    in_window = (columns >= top_lens[:, None]) & (
        columns < (top_lens + window_lens)[:, None]
    )
    result = torch.full((n, width), -1, dtype=torch.int32, device=device)
    if topk:
        selected = topk_indices[:, :topk].index_select(
            1, columns[0].clamp_max(topk - 1)
        )
        result = torch.where(
            in_top & (selected >= 0),
            selected + compressed_base[req, None],
            result,
        )
    relative = columns - top_lens[:, None]
    if swa_indices is not None and window_size:
        window_values = swa_indices.gather(
            1, relative.clamp(0, window_size - 1)
        )
        window_values = torch.where(
            window_values >= 0, window_values + swa_base[req, None], -1
        )
    else:
        start = (
            swa_base[req] + pos - window_lens + 1
            - (seq_lens[req] - gather_lens[req])
        )
        window_values = start[:, None] + relative
    result = torch.where(in_window, window_values, result)
    out_indices.copy_(result)
    out_lens.copy_(top_lens + window_lens)
    return out_indices, out_lens


@register_jit_op(
    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
    "build_swa_token_ids",
)
def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Build SWA geometry and physical token IDs with the Kunlun operator.

    Args:
        seq_lens: Total sequence length for each request.
        extend_seq_lens: Number of query tokens added by each request.
        req_pool_indices: Request pool rows for the batch.
        req_to_token: Mapping from request positions to full-cache token slots.
        full_to_swa: Mapping from full-cache slots to SWA slots.
        swa_window: Sliding-window width used to determine gathered tokens.
        total_swa: Total number of SWA token IDs to allocate.

    Returns:
        Physical SWA token IDs, first positions, gather lengths, and cumulative
        offsets. An empty batch has a single zero offset.
    """
    num_reqs = seq_lens.shape[0]
    device = seq_lens.device
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    swa_first_pos = torch.empty(num_reqs, dtype=torch.int32, device=device)
    swa_gather_lens = torch.empty(num_reqs, dtype=torch.int32, device=device)
    if num_reqs == 0:
        # The op returns early on an empty batch, so `swa_offsets[0] == 0` — the
        # one element upstream callers still index — has to be materialized here.
        swa_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    swa_offsets = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    torch.ops.xspeedgate_ops.build_swa_token_ids(
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        seq_lens,
        extend_seq_lens,
        req_pool_indices,
        req_to_token,
        full_to_swa,
        swa_window,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


# Width of one DSV4 KV slot: 448 nope + 64 rope, stored unpacked on Kunlun.
DSV4_KV_DIM = 512


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.dequant_k_cache",
    "dequantize_k_cache_paged",
)
def dsv4_dequantize_k_cache_paged_kunlun(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
    layout=None,
) -> torch.Tensor:
    """Gather paged DSV4 KV rows into the flat sparse-prefill workspace.

    Upstream this dequantises the V4 slot (448B fp8 nope + per-tile ue8m0
    scales + 64B bf16 rope) with a Triton kernel. The Kunlun pool stores plain
    fp16/bf16 latents -- `dsv4_quant_k_cache_kunlun` only rounds bf16 to fp16
    and `set_k_and_s_v4` writes the row as-is -- so a slot already *is* the
    512-wide vector the workspace wants and the dequant collapses to a gather
    plus a dtype cast. `_assert_v4_kv_layout` keeps that assumption honest: the
    packed V4.1 layouts have no Kunlun writer, so they cannot reach this path.

    `out` is usually a slice of a larger workspace, so the gather is copied
    into it instead of replacing it.

    Args:
        quant_k_cache: Paged cache containing plain Kunlun latent vectors.
        page_table_1_flattened: Physical token slots to gather, in workspace order.
        page_size: Number of latent-vector slots per cache page.
        out: Optional destination receiving gathered rows in its own dtype.
        layout: KV layout checked for compatibility with plain latent storage.

    Returns:
        Gathered rows shaped [tokens, 1, 512], as bf16 when out is omitted,
        or the supplied output buffer after copying.
    """
    _assert_v4_kv_layout(layout, "dequantize_k_cache_paged")
    buf = quant_k_cache.reshape(quant_k_cache.shape[0], -1)
    width = buf.shape[1]
    # A page is `page_size` contiguous 512-wide slots, possibly followed by
    # padding: the compressed pools allocate the widest ratio's page and the
    # reader slices `page_size * DSV4_KV_DIM` off the front (see the
    # `extra_cache[:, : page_width * cache_dim]` slice in the Kunlun backend).
    # So the page stride is the full row and only the slot stride is fixed.
    assert width >= page_size * DSV4_KV_DIM, (
        f"page of {width} elements cannot hold {page_size} x {DSV4_KV_DIM} slots"
    )
    token_ids = page_table_1_flattened.reshape(-1).to(torch.long)
    if width == page_size * DSV4_KV_DIM:
        gathered = buf.reshape(-1, DSV4_KV_DIM).index_select(0, token_ids)
    else:
        slot_start = (
            torch.div(token_ids, page_size, rounding_mode="floor") * width
            + (token_ids % page_size) * DSV4_KV_DIM
        )
        lane = torch.arange(DSV4_KV_DIM, device=buf.device)
        gathered = buf.reshape(-1).take(
            (slot_start.unsqueeze(1) + lane.unsqueeze(0)).reshape(-1)
        )
    gathered = gathered.reshape(-1, 1, DSV4_KV_DIM)
    if out is None:
        return gathered.to(torch.bfloat16)
    out.copy_(gathered)
    return out
