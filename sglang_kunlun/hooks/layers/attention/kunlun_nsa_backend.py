"""kunlun_nsa attention backend"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Optional

import torch

import kunlun_ops
import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops

# `xspeedgate_ops.sparse_attn_fwd` is registered by *this* extension, not by
# the `xspeedgate_ops` python package: its `__init__` only binds the proxy, the
# schema comes out of the cocopod .so. The `torch.ops.<ns>` proxy is lazy, so
# neither `dir()` nor `hasattr` can tell whether an op exists -- enumerating
# `torch._C._jit_get_all_schemas()` is the only reliable probe. That gives
#   xspeedgate_ops::sparse_attn_fwd(Tensor q, Tensor kv, Tensor indices,
#       float sm_scale, int d_v, Tensor topk_length, Tensor? attn_sink=None)
#       -> Tensor[]
# The import is for the registration side effect only, exactly like the
# `xspeedgate_ops` import above.
import cocopod  # noqa: F401  # registers torch.ops.xspeedgate_ops.sparse_attn_fwd

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAIndexerMetadata,
    DSAMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Native INT8 mixed-cache sparse MLA attention
# --------------------------------------------------------------------------- #
# The 0.5.8 baseline never dequantized the packed rows. When the pool reported
# int8 storage it branched straight to `_native_mixed_attn` -- the same helper
# from both arms, sparse prefill/verify
# (`sgl-kernel/.../attention/nsa_backend.py:1751-1759`) and decode
# (`:2235-2238`) -- and handed the 656 B rows to the vendor op as they are.
#
# Restoring that call is also what puts the three baseline kernels back into
# the torch-profiler trace. All three are launched by ONE host function,
# `xfa::dsa_prefill_with_mixed_cache<float16, int8, bfloat16, float16, float16,
# float, int>` in `torch_xmlir/libxpu_flash_attention.so` (symbol at
# 0x13ffe10; disassembled and its call targets resolved):
#   * `fa_xpu3::dsa_fwd_mixed_cache_multistream_cluster<int8, bfloat16,
#     float16, int>` -- called directly through its PLT entry,
#   * `fa_xpu3::dsa_fwd_multistream_sdnn<float16, ..., float, 0>` -- through the
#     helper at 0x2286ed0,
#   * `fa_xpu3::constant_kl` -- through the fill helper at 0x2e51050.
# The dequant + `sparse_attn_fwd` chain reaches none of them, which is why the
# fresh traces are missing exactly those three.
#
# Two python entry points land on that host function in this image, and 0.5.8
# could use either (`SGLANG_NSA_MIXED_ATTN_IMPL`, `nsa_backend.py:190-214`):
#
#   * `torch.ops.xspeedgate_ops.sparse_mixed_attn_int8` -- 0.5.8's default
#     (`nsa_backend.py:166`). Registered here too, schema
#       (q, kv, indices, out!, max_logits!, lse!, sm_scale, qlod_cpu, qlod_xpu,
#        kvlod_cpu, kvlod_xpu, is_causal, topk, side_stream,
#        max_seq_kv_bound=-1) -> int
#     and `nm -D --undefined-only` on its .so shows it links exactly the two
#     int8/bf16-rope instantiations (q=fp16 and q=bf16) plus
#     `xmlir_rt::getXdnnContext`, i.e. the runtime context that makes it
#     graph-capture safe.
#   * `kunlun_ops.dsa_prefill_with_mixed_cache` -- 0.5.8's `=kunlun` variant
#     (`nsa_backend.py:208`). Same work, different argument order, and no
#     `max_seq_kv_bound`: it derives that scalar from `kvlod_cpu` on the host,
#     so a capture has to be fed an upper-bound host lod
#     (`nsa_backend.py:2091-2110`).
#
# Row layout: both the op's own docstring and the writer agree on
# `[kv_lora_rank(512) int8 nope | 4 fp32 scales (one per 128 nope) |
# rope_head_dim(64) bf16]` = 656 B/token, which is what
# `kunlun_ops.quantize_mla_kv_cache_split` emits and what
# `kernels/mixed_int8_kv.py` documents. The 604 B
# `kv_lora_rank(int8) + rope(TROPE) + n_scale(fp32)` order in the docstring of
# `kunlun_ops.hybrid_attention_with_mixed_cache` belongs to that *other* op
# (DSV4 SWA/CSA, 448+64+7); it takes win/com caches and window/compress
# arguments this model has no use for, and nothing in 0.5.8 ever called it.
# There is no `n_scale_blocks` / `xbf16_label` selector: the rope element type
# is a template parameter of the host function (bfloat16 in every instantiation
# the wrapper links) and the scale tiling follows from the kv_lora_rank
# argument, so the field order is fixed by the instantiation, not by a flag.

_MIXED_ATTN_IMPL = None  # ("xspeed"|"kunlun"|"none", op) -- resolved once
_MIXED_ATTN_SIDE_STREAMS: dict = {}  # device index -> persistent side stream
_MIXED_ATTN_BOUND_KVLODS: dict = {}  # (rows, topk) -> capture-time host lod
_MIXED_ATTN_FALLBACK_LOGGED = False


def _resolve_mixed_attn_impl():
    """Resolve the native mixed-cache op once. ``(kind, None)`` means none."""
    global _MIXED_ATTN_IMPL
    if _MIXED_ATTN_IMPL is None:
        # A C++ extension registers its schemas when it is *imported*, not when
        # it is installed -- so the probe below answers "has anything imported
        # `xspeedgate_ops` yet", not "does the wheel contain the op". Measured
        # on this box: before `import xspeedgate_ops` the schema set has zero
        # `xspeedgate_ops::` entries, after it has 256. If the plugin happened
        # to reach this resolver before any other module imported the package,
        # it would silently pick the dequant fallback and cost 3.5x decode
        # throughput with no error. Import it here so the answer depends on the
        # wheel, not on import order.
        try:
            import xspeedgate_ops  # noqa: F401
        except Exception:  # noqa: BLE001 - absent wheel is a legitimate state
            pass
        # `torch.ops.<ns>` is a lazy proxy, so neither `dir()` nor `hasattr`
        # can answer "is this op registered" -- enumerating the schemas is the
        # only reliable probe (same note as `sparse_attn_fwd` at the top of
        # this file).
        names = {schema.name for schema in torch._C._jit_get_all_schemas()}
        has_xspeed = "xspeedgate_ops::sparse_mixed_attn_int8" in names
        has_kunlun = hasattr(kunlun_ops, "dsa_prefill_with_mixed_cache")
        # Both entry points land on the SAME vendor host function,
        # `xfa::dsa_prefill_with_mixed_cache` in
        # `torch_xmlir/libxpu_flash_attention.so` -- the xspeedgate wheel only
        # carries the torch wrapper (`nm -D --undefined-only` on it shows the
        # two int8 instantiations as `U`). So "the wheel lost the op" is a
        # wrapper problem, not a kernel problem, and `kunlun` reaches the same
        # kernels. This knob exists to measure that claim, and to serve from
        # the `kunlun` entry point if a rebuilt wheel ever drops the wrapper
        # again. 0.5.8 had the same switch (`nsa_backend.py:190-214`).
        want = os.environ.get("SGLANG_NSA_MIXED_ATTN_IMPL", "auto").strip().lower()
        if want not in ("auto", "xspeed", "kunlun", "none"):
            logger.warning(
                "sglang-kunlun: SGLANG_NSA_MIXED_ATTN_IMPL=%r is not one of "
                "auto/xspeed/kunlun/none; treating it as auto.",
                want,
            )
            want = "auto"
        if want == "none":
            _MIXED_ATTN_IMPL = ("none", None)
        elif want == "kunlun" and has_kunlun:
            _MIXED_ATTN_IMPL = ("kunlun", kunlun_ops.dsa_prefill_with_mixed_cache)
        elif want == "xspeed" and has_xspeed:
            _MIXED_ATTN_IMPL = (
                "xspeed",
                torch.ops.xspeedgate_ops.sparse_mixed_attn_int8,
            )
        elif has_xspeed:
            if want != "auto":
                logger.warning(
                    "sglang-kunlun: SGLANG_NSA_MIXED_ATTN_IMPL=%s is "
                    "unavailable; falling back to xspeed.",
                    want,
                )
            _MIXED_ATTN_IMPL = (
                "xspeed",
                torch.ops.xspeedgate_ops.sparse_mixed_attn_int8,
            )
        elif has_kunlun:
            _MIXED_ATTN_IMPL = (
                "kunlun",
                kunlun_ops.dsa_prefill_with_mixed_cache,
            )
        else:
            _MIXED_ATTN_IMPL = ("none", None)
        # Say which branch won, once per process. Without this the only signal
        # that the fast path was skipped is a warning that fires from the *call*
        # site, which reads like "neither library has it" even when the real
        # reason is that the wheel was rebuilt without the operator.
        logger.info(
            "sglang-kunlun: DSA decode attention implementation = %s "
            "(requested %s; xspeedgate_ops::sparse_mixed_attn_int8 "
            "registered: %s, kunlun_ops.dsa_prefill_with_mixed_cache: %s)",
            _MIXED_ATTN_IMPL[0],
            want,
            has_xspeed,
            has_kunlun,
        )
    return _MIXED_ATTN_IMPL


def _mixed_attn_side_stream(device: torch.device) -> torch.cuda.Stream:
    """Persistent non-default side stream for one device.

    The op is multi-stream only ("仅支持多流版本, side_stream 需外部管理" in the
    vendor docstring) and dispatches part of its work onto the handle it is
    given. 0.5.8 kept one pair alive per device rather than creating one per
    call (`nsa_backend.py:229-238`): a fresh stream per call would be recorded
    into every captured graph.
    """
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _MIXED_ATTN_SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _MIXED_ATTN_SIDE_STREAMS[index] = stream
    return stream


def _mixed_attn_bound_kvlod(rows: int, topk: int) -> torch.Tensor:
    """Capture-time upper-bound host kv lod, one ``topk``-long segment per row.

    Only `kunlun_ops.dsa_prefill_with_mixed_cache` needs this: it reads the
    batch's kv total off `kvlod_cpu` on the host, and under capture that scalar
    is baked into the recorded launch, so the warmup batch's (short) real lod
    would under-plan every replay. Each row's count is
    ``min(valid_kv_len, topk) <= topk``, so ``rows * topk`` bounds it. The
    device lod stays the real one, which is what the kernel reads per row
    (`nsa_backend.py:217-226`, `:2091-2110`).
    """
    cpu = _MIXED_ATTN_BOUND_KVLODS.get((rows, topk))
    if cpu is None:
        cpu = (torch.arange(0, rows + 1, dtype=torch.int32) * int(topk)).contiguous()
        _MIXED_ATTN_BOUND_KVLODS[(rows, topk)] = cpu
    return cpu


def _log_mixed_attn_fallback(reason: str) -> None:
    """Warn once that the packed rows are being dequantized instead."""
    global _MIXED_ATTN_FALLBACK_LOGGED
    if _MIXED_ATTN_FALLBACK_LOGGED:
        return
    _MIXED_ATTN_FALLBACK_LOGGED = True
    logger.warning(
        "sglang-kunlun: native INT8 mixed-cache attention unusable (%s); "
        "falling back to dequant + xspeedgate_ops.sparse_attn_fwd. That costs "
        "one dense copy of the whole KV table per layer per forward, and the "
        "torch-profiler trace will then lack fa_xpu3::constant_kl, "
        "fa_xpu3::dsa_fwd_mixed_cache_multistream_cluster and "
        "fa_xpu3::dsa_fwd_multistream_sdnn.",
        reason,
    )



def _paged_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pick the top-k scored positions and map them to page-1 cache slots.

    Torch port of the fused paged top-k transform. ``xspeedgate_ops.topk_transform``
    requires ``src_page_table`` to be at least as wide as the score, but on 0.5.14
    the score width is page aligned (``page_table_64.shape[1] * page_size``) while
    the page-1 table is exactly ``max_seqlen_k`` wide, so that contract no longer
    holds. Positions are returned in ascending order (the selected *set* is what
    the sparse attention consumes) with ``-1`` padding.
    """
    rows, width = score.shape
    device = score.device
    table_width = src_page_table.shape[1]

    positions = torch.arange(width, device=device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    k = min(topk, width)
    selected = score.masked_fill(~valid, float("-inf")).topk(k, dim=1).indices
    selected, _ = selected.sort(dim=1)
    selected_valid = valid.gather(1, selected)

    if src_page_table.shape[0] == rows:
        table = src_page_table
    else:
        # Prefill/verify feed one score row per query token while the page table
        # keeps one row per request; cu_seqlens_q carries that mapping.
        assert (
            cu_seqlens_q is not None
        ), "paged topk transform needs cu_seqlens_q to map tokens to requests"
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
        row_index = torch.repeat_interleave(
            torch.arange(q_lens.shape[0], device=device), q_lens
        )
        table = src_page_table[row_index]

    slots = table.gather(1, selected.clamp(max=table_width - 1))
    slots = torch.where(selected_valid, slots, -torch.ones_like(slots))
    if k < topk:
        slots = torch.nn.functional.pad(slots, (0, topk - k), value=-1)
    return slots.to(torch.int32)


@dataclass(frozen=True)
class KunlunDSAMetadata(DSAMetadata):
    """DSAMetadata + Kunlun specific ``cum_q_lod`` and host mirrors.

    The Kunlun kernels take both the device and the host copy of the lod/seqlen
    tensors, so the host copies are materialized once per forward batch here
    instead of once per layer.
    """

    # Cumulative q lengths in *token* space (upstream ``cu_seqlens_q`` is in
    # expanded/per-token space for verify/draft-extend).
    cum_q_lod: Optional[torch.Tensor] = None

    # (host mirror attribute, device field) pairs.
    _MIRRORED = (
        ("cache_seqlens_int32_cpu", "cache_seqlens_int32"),
        ("cu_seqlens_q_cpu", "cu_seqlens_q"),
        ("cu_seqlens_k_cpu", "cu_seqlens_k"),
        ("dsa_seqlens_expanded_cpu", "dsa_seqlens_expanded"),
        ("dsa_cu_seqlens_q_cpu", "dsa_cu_seqlens_q"),
        ("dsa_cu_seqlens_k_cpu", "dsa_cu_seqlens_k"),
        ("cum_q_lod_cpu", "cum_q_lod"),
    )

    def __post_init__(self):
        """Materialize the host mirrors used by the Kunlun kernels."""
        for host_name, dev_name in self._MIRRORED:
            dev = getattr(self, dev_name, None)
            object.__setattr__(self, host_name, None if dev is None else dev.cpu())

    def refresh_host_mirrors(self) -> None:
        """Update the host mirrors *in place*, keeping their storage stable.

        On the cuda-graph path this object is reused across replays and the
        Kunlun kernels were recorded against these host buffers, so a fresh
        ``.cpu()`` tensor per replay would leave the graph reading capture-time
        lengths. Copy into the existing storage instead, and only reallocate when
        the shape/dtype actually changes.

        The seven mirrors used to be seven ``host.copy_(dev)`` calls, i.e. seven
        blocking device-to-host transfers per call and three calls per decode
        step (measured 1.16 ms/step). Stage the device tensors of one dtype into
        a single contiguous device buffer first (device-to-device, no sync) and
        bring them back in one transfer per dtype; the final host-to-host copies
        into the existing mirrors keep every ``data_ptr`` stable.
        """
        plan = self._mirror_plan()
        if plan is None:
            # Shapes/dtypes changed (or a mirror had to be reallocated); the
            # mirrors are already up to date in that case.
            return
        for dtype, entries, dev_buf, host_buf in plan:
            for _, dev, offset, numel in entries:
                dev_buf[offset : offset + numel].copy_(dev.reshape(-1))
            host_buf.copy_(dev_buf)
            for host, _, offset, numel in entries:
                host.reshape(-1).copy_(host_buf[offset : offset + numel])

    def _mirror_plan(self):
        """Return the cached staging plan, rebuilding it when shapes change.

        Returns ``None`` when a rebuild happened: the rebuild path already
        refreshed every mirror with a plain ``.cpu()``, so the caller is done.
        """
        signature = tuple(
            None if (dev := getattr(self, dev_name, None)) is None
            else (tuple(dev.shape), dev.dtype)
            for _, dev_name in self._MIRRORED
        )
        cached = getattr(self, "_mirror_staging", None)
        if cached is not None and cached[0] == signature:
            return cached[1]

        by_dtype: dict[torch.dtype, list] = {}
        for host_name, dev_name in self._MIRRORED:
            dev = getattr(self, dev_name, None)
            if dev is None:
                continue
            host = getattr(self, host_name, None)
            if host is None or host.shape != dev.shape or host.dtype != dev.dtype:
                host = dev.cpu()
                object.__setattr__(self, host_name, host)
            else:
                host.copy_(dev)
            by_dtype.setdefault(dev.dtype, []).append((host, dev))

        plan = []
        for dtype, pairs in by_dtype.items():
            entries = []
            offset = 0
            for host, dev in pairs:
                numel = dev.numel()
                entries.append((host, dev, offset, numel))
                offset += numel
            dev_buf = torch.empty(offset, dtype=dtype, device=pairs[0][1].device)
            try:
                host_buf = torch.empty(
                    offset, dtype=dtype, device="cpu", pin_memory=True
                )
            except RuntimeError:
                # Pinning is unavailable on some runtimes; a pageable staging
                # buffer still collapses the seven transfers into one.
                host_buf = torch.empty(offset, dtype=dtype, device="cpu")
            plan.append((dtype, entries, dev_buf, host_buf))
        object.__setattr__(self, "_mirror_staging", (signature, plan))
        return None


@dataclass(frozen=True)
class KunlunDSAIndexerMetadata(DSAIndexerMetadata):
    """Indexer metadata exposing the host mirrors and the Kunlun topk kernel.

    ``attn_metadata`` is always a :class:`KunlunDSAMetadata` here; it is not
    re-declared because giving it a default would break the dataclass field
    ordering inherited from upstream.
    """

    def get_seqlens_expanded_cpu(self) -> torch.Tensor:
        """Host mirror of ``dsa_seqlens_expanded``."""
        return self.attn_metadata.dsa_seqlens_expanded_cpu

    def get_cu_seqlens_k_cpu(self) -> torch.Tensor:
        """Host mirror of ``cu_seqlens_k``."""
        return self.attn_metadata.cu_seqlens_k_cpu

    def get_cu_seqlens_k(self) -> torch.Tensor:
        """Device ``cu_seqlens_k`` (used by the ragged prefill top-k path)."""
        return self.attn_metadata.cu_seqlens_k

    def get_cu_seqlens_q(self) -> torch.Tensor:
        """Device ``cu_seqlens_q``."""
        return self.attn_metadata.cu_seqlens_q

    def get_cu_seqlens_q_cpu(self) -> torch.Tensor:
        """Host mirror of ``cu_seqlens_q``."""
        return self.attn_metadata.cu_seqlens_q_cpu

    def get_seqlens_int32_cpu(self) -> torch.Tensor:
        """Host mirror of ``cache_seqlens_int32``."""
        return self.attn_metadata.cache_seqlens_int32_cpu

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same control flow as upstream; the PAGED branch runs the Kunlun kernel.

        Upstream 0.5.14 delegates the whole body to ``DSATopKBackend``. That enum is
        closed, so the Kunlun fused kernel is injected here instead.
        """
        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset

        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()

        if batch_idx_list is not None:
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1

        if not envs.SGLANG_DSA_FUSE_TOPK.get() or self.force_unfused_topk:
            return self.topk_backend.topk_func(
                logits, seq_lens_topk, topk, row_starts=ks
            )
        if self.topk_transform_method == TopkTransformMethod.PAGED:
            # if fused, we return a transformed page table directly
            return _paged_topk_transform(
                score=logits,
                lengths=seq_lens_topk,
                src_page_table=page_table_size_1,
                topk=topk,
                cu_seqlens_q=cu_seqlens_q_topk,
            )
        if self.topk_transform_method == TopkTransformMethod.RAGGED:
            from sgl_kernel import fast_topk_transform_ragged_fused

            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        raise RuntimeError(f"Unsupported {self.topk_transform_method = }")


class KunlunDSAAttnBackend(DeepseekSparseAttnBackend):
    """DeepSeek sparse attention backend running on Kunlun XPU."""

    # 0.5.23 defaults this to False on the DSA backend
    # (`dsa_backend.py:505`, `:4173`) and then nulls the host mirror of
    # `seq_lens_cpu` for spec-v2 (`dsa_backend.py:1112`). The Kunlun indexer
    # reads that mirror on the MTP/nextn path
    # (`kunlun_dsa_indexer.py`, `assert forward_batch.seq_lens_cpu is not None`),
    # so decode dies at the first draft step with `AssertionError`. Declaring the
    # need is the upstream seam for exactly this: `BaseSpecWorker` ORs the flag
    # over `spec_v2_attn_backends` (`speculative/base_spec_worker.py:262-264`)
    # to decide whether the CPU mirror is materialised.
    needs_cpu_seq_lens = True


    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #

    def _compute_cum_q_lod(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Cumulative q lengths in token space, per forward mode."""
        batch_size = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            return self.get_device_int32_arange(batch_size + 1)
        if forward_batch.forward_mode.is_target_verify():
            extend_seq_lens = torch.full(
                (batch_size,),
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            return compute_cu_seqlens(extend_seq_lens)
        # draft_extend (v1/v2) and plain extend
        assert forward_batch.extend_seq_lens is not None
        # ``compute_cu_seqlens`` asserts int32; the prefill cuda-graph capture path
        # (0.5.19) hands us int64 extend_seq_lens.
        return compute_cu_seqlens(forward_batch.extend_seq_lens.to(torch.int32))

    def _cum_q_lod_for_graph(self, bs: int, forward_mode: ForwardMode) -> torch.Tensor:
        """cum_q_lod for the cuda-graph paths (fixed draft token count)."""
        if forward_mode.is_decode_or_idle():
            # NOTE: shares storage with cu_seqlens_q, so graph replay (which updates
            # cu_seqlens_q in place) keeps cum_q_lod in sync.
            return self.forward_metadata.cu_seqlens_q
        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        return compute_cu_seqlens(extend_seq_lens)

    @staticmethod
    def _to_kunlun_metadata(
        metadata: DSAMetadata, cum_q_lod: torch.Tensor
    ) -> KunlunDSAMetadata:
        """Re-wrap an upstream metadata object, sharing all tensors."""
        kwargs = {f.name: getattr(metadata, f.name) for f in fields(metadata)}
        kwargs["cum_q_lod"] = cum_q_lod
        return KunlunDSAMetadata(**kwargs)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        super().init_forward_metadata(forward_batch)
        self.forward_metadata = self._to_kunlun_metadata(
            self.forward_metadata, self._compute_cum_q_lod(forward_batch)
        )

    def _build_forward_metadata_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Capture-time metadata: upstream build + Kunlun ``cum_q_lod``."""
        super()._build_forward_metadata_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            seq_lens_cpu,
            forward_mode,
            spec_info,
            out_cache_loc=out_cache_loc,
            actual_forward_mode=actual_forward_mode,
        )
        self._rewrap_graph_metadata(bs, forward_mode)

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Shared capture+replay body: upstream apply + Kunlun ``cum_q_lod``."""
        super()._apply_cuda_graph_metadata(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_cpu,
            forward_mode,
            spec_info,
            out_cache_loc=out_cache_loc,
            actual_forward_mode=actual_forward_mode,
        )
        self._rewrap_graph_metadata(bs, forward_mode)

    def init_forward_metadata_replay_cuda_graph_from_precomputed(
        self,
        bs: int,
        precomputed,
        forward_mode: ForwardMode,
    ):
        """Replay-prep fast path (0.5.19): refresh the Kunlun host mirrors.

        The multi-step draft backend copies precomputed metadata into
        ``decode_cuda_graph_metadata[bs]`` through this method instead of
        ``_apply_cuda_graph_metadata``, so without this override the wrapper's
        host mirrors would keep the previous step's lengths.
        """
        super().init_forward_metadata_replay_cuda_graph_from_precomputed(
            bs, precomputed, forward_mode
        )
        self._rewrap_graph_metadata(bs, forward_mode)

    def _rewrap_graph_metadata(self, bs: int, forward_mode: ForwardMode) -> None:
        """Install the Kunlun metadata wrapper for this batch size, optionally updating the tensors."""
        cache = self.decode_cuda_graph_metadata
        metadata = self.forward_metadata

        if isinstance(metadata, KunlunDSAMetadata):
            # Upstream already handed back the wrapper (it lives in the cache);
            # refresh the Kunlun-owned tensors in place. cum_q_lod is per forward
            # mode, and one wrapper is shared by the decode and verify graphs of
            # the same bs, so it has to be recomputed every time.
            new_lod = self._cum_q_lod_for_graph(bs, forward_mode)
            cur_lod = metadata.cum_q_lod
            if (
                cur_lod is None
                or cur_lod.shape != new_lod.shape
                or cur_lod.dtype != new_lod.dtype
                or cur_lod.data_ptr() == new_lod.data_ptr()
            ):
                object.__setattr__(metadata, "cum_q_lod", new_lod)
            else:
                cur_lod.copy_(new_lod)
            metadata.refresh_host_mirrors()
            return

        metadata = self._to_kunlun_metadata(
            metadata, self._cum_q_lod_for_graph(bs, forward_mode)
        )
        self.forward_metadata = metadata
        if isinstance(cache, dict) and cache.get(bs) is not None:
            cache[bs] = metadata

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> KunlunDSAIndexerMetadata:
        """Indexer metadata carrying the Kunlun host mirrors."""
        force_unfused = (
            self.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        return KunlunDSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(
                forward_batch.forward_mode
            ),
            topk_backend=self.dsa_topk_backend,
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            paged_mqa_ctx_lens_2d=self.forward_metadata.paged_mqa_ctx_lens_2d,
            force_unfused_topk=force_unfused,
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def _native_mixed_attn(
        self,
        q_all: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: KunlunDSAMetadata,
    ) -> Optional[torch.Tensor]:
        """Sparse MLA attention read straight off the packed INT8 rows.

        Port of 0.5.8's `_native_mixed_attn`
        (`sgl-kernel/.../attention/nsa_backend.py:1770-1856`) and its
        `_call_mixed_attn` (`:2076-2169`), minus the decode-context-parallel
        arm: that one needs `dcp_own_topk` / `dcp_lse_merge_reduce_scatter`
        and this tree refuses DCP on the write side already
        (`kunlun_pools.py`, `_write_mla_kv_buffer`), so `dcp_size` is 1 here by
        construction.

        Returns ``None`` -- never raises -- when the op is missing or rejects
        the arguments, so the caller degrades to the dequantization chain
        instead of taking the server down on a signature change.
        """
        kind, op = _resolve_mixed_attn_impl()
        if op is None:
            _log_mixed_attn_fallback(
                "neither xspeedgate_ops::sparse_mixed_attn_int8 nor "
                "kunlun_ops.dsa_prefill_with_mixed_cache is available"
            )
            return None

        # Lod convention is the bf16 path's own semantics: every q row is its
        # own segment, and `dsa_cu_seqlens_k` -- the cumsum of
        # `dsa_cache_seqlens_int32` = min(valid_kv_len, index_topk) per row
        # (`dsa_backend.py:273-275`, `:1361`) -- says how many LEADING entries
        # of that row's top-k list are real; the tail is -1 padding. Same
        # number `sparse_attn_fwd` is handed as `topk_length` in the fallback
        # branches, only accumulated. The host mirrors are materialized once
        # per forward batch by `KunlunDSAMetadata`, which is exactly what the
        # op wants ("qlod_cpu must be a CPU int32 tensor" in its own checks)
        # and what 0.5.8 had to refresh by hand on every replay
        # (`nsa_backend.py:1348-1381`).
        ql_cpu = metadata.dsa_cu_seqlens_q_cpu
        ql_xpu = metadata.dsa_cu_seqlens_q
        kvl_cpu = metadata.dsa_cu_seqlens_k_cpu
        kvl_xpu = metadata.dsa_cu_seqlens_k
        if any(t is None for t in (ql_cpu, ql_xpu, kvl_cpu, kvl_xpu)):
            _log_mixed_attn_fallback("the DSA lods are absent from the metadata")
            return None

        kv_rows = self.token_to_kv_pool.mixed_int8_kv_rows(layer.layer_id)
        q_all = q_all.contiguous()
        n_rows = q_all.shape[0]
        n_lod = int(kvl_cpu.numel()) - 1
        if n_lod > n_rows:
            # Cuda-graph buckets size these buffers for the batch they were
            # captured at; a lod with more segments than there are q rows would
            # make the kernel read q past its end. Slicing from 0 keeps
            # `data_ptr`, so a recorded launch still reads the same storage on
            # replay.
            ql_cpu, ql_xpu = ql_cpu[: n_rows + 1], ql_xpu[: n_rows + 1]
            kvl_cpu, kvl_xpu = kvl_cpu[: n_rows + 1], kvl_xpu[: n_rows + 1]
            n_lod = n_rows
        elif n_lod < n_rows:
            # TP + partial-DP padding adds q rows the metadata never saw.
            # Extend with zero-length kv segments so those rows read nothing,
            # and build the pads on CPU first: `.cpu()` inside a capture is an
            # illegal D2H copy (0.5.8 `nsa_backend.py:1809-1827`). Starting the
            # q pad at `n_lod + 1` avoids duplicating the last value, which
            # would add a segment and make the two lods disagree on batch_num.
            pad_q = torch.arange(n_lod + 1, n_rows + 1, dtype=torch.int32)
            ql_cpu = torch.cat([ql_cpu, pad_q])
            ql_xpu = torch.cat([ql_xpu, pad_q.to(ql_xpu.device)])
            pad_kv = kvl_cpu[-1:].expand(n_rows - n_lod)
            kvl_cpu = torch.cat([kvl_cpu, pad_kv])
            kvl_xpu = torch.cat([kvl_xpu, pad_kv.to(kvl_xpu.device)])
            n_lod = n_rows

        idx = page_table_1
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)
        # -1 padding (and the EAGLE warmup's garbage ids) map to slot 0, which
        # the pool keeps zeroed; the per-row lods bound the read anyway. Same
        # clamp 0.5.8 applied (`nsa_backend.py:1798-1802`), done on the device
        # so it is capture-safe.
        idx = torch.where(
            (idx < 0) | (idx >= kv_rows.shape[0]), torch.zeros_like(idx), idx
        ).contiguous()
        topk = idx.shape[-1]

        out = torch.empty(
            (n_rows, q_all.shape[1], layer.v_head_dim),
            dtype=q_all.dtype,
            device=q_all.device,
        )
        # Both outputs are required by the op even though only `out` is used
        # here; 0.5.8 read `lse` only for its DCP merge.
        max_logits = torch.empty(
            (n_rows, q_all.shape[1]), dtype=torch.float32, device=q_all.device
        )
        lse = torch.empty_like(max_logits)

        side = _mixed_attn_side_stream(q_all.device)
        capturing = torch.cuda.is_current_stream_capturing()
        try:
            if kind == "xspeed":
                # `max_seq_kv_bound` is a by-value scalar baked into a recorded
                # launch, so a capture must not let the C++ side derive it from
                # the (short) warmup lod -- pass the per-bucket upper bound
                # `n_rows * topk`; eager passes -1 and lets the op compute the
                # exact last-first value (0.5.8 `nsa_backend.py:2125-2156`).
                # `is_causal=False` in both arms, decode and prefill alike:
                # each q row carries its own already causal-filtered top-k
                # list, so there is no triangle left to mask.
                status = op(
                    q_all,
                    kv_rows,
                    idx,
                    out,
                    max_logits,
                    lse,
                    layer.scaling,
                    ql_cpu,
                    ql_xpu,
                    kvl_cpu,
                    kvl_xpu,
                    False,
                    topk,
                    side.cuda_stream,
                    n_rows * topk if capturing else -1,
                )
            else:
                status = op(
                    q=q_all,
                    kv_cache=kv_rows,
                    indices=idx,
                    o=out,
                    max_logits=max_logits,
                    lse=lse,
                    qlod_cpu=ql_cpu,
                    qlod_xpu=ql_xpu,
                    kvlod_cpu=(
                        _mixed_attn_bound_kvlod(n_rows, topk)
                        if capturing
                        else kvl_cpu
                    ),
                    kvlod_xpu=kvl_xpu,
                    sm_scale=layer.scaling,
                    is_causal=False,
                    topk=topk,
                    side_stream=side.cuda_stream,
                    use_xfa_boost=False,
                )
            if not capturing:
                # The op farms sub-work out to `side_stream`, so this stream has
                # to join it before anyone reads `out`. Illegal during capture,
                # where the graph records the dependency itself.
                torch.cuda.current_stream(q_all.device).wait_stream(side)
        except Exception as exc:  # noqa: BLE001 - degrade, do not kill the server
            _log_mixed_attn_fallback(f"the {kind} op raised: {exc}")
            return None
        if int(status) != 0:
            _log_mixed_attn_fallback(
                f"the {kind} op returned {int(status)} (1=INVALID_PARAM "
                f"2=RUNTIME_ERROR 3=NO_ENOUGH_WORKSPACE 4=NOT_IMPLEMENT); "
                f"capturing={capturing} rows={n_rows} lod={n_lod} topk={topk} "
                f"q={tuple(q_all.shape)}/{q_all.dtype} "
                f"kv={tuple(kv_rows.shape)}/{kv_rows.dtype} "
                f"idx={tuple(idx.shape)}"
            )
            return None
        return out


    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefill / verify / draft-extend through the Kunlun sparse kernel."""
        # 0.5.22 appended `attn_sink` to DeepseekSparseAttnBackend.forward_extend
        # (learnable per-head attention sinks, produced by the DSV4 models and
        # forwarded as a kwarg by RadixAttention only when non-None). The Kunlun
        # sparse kernel has no sink input, and upstream itself rejects sinks on
        # every dsa_impl except flashmla_sparse, so refuse explicitly instead of
        # dropping them and silently computing a different softmax.
        if attn_sink is not None:
            raise NotImplementedError(
                "KunlunDSAAttnBackend.forward_extend: learnable attention sinks "
                "(attn_sink) are not supported by the Kunlun sparse kernel"
            )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        # Use MHA kernel if in MHA_ONE_SHOT mode
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert (
                layer.tp_k_head_num == layer.tp_q_head_num > 1
            ), "MHA_ONE_SHOT requires dense multi-head config"
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        assert q_rope is not None
        # `get_key_buffer` is the accessor that yields the table this op indexes:
        # one row per *token slot*, in slot order (`memory_pool.py:4893-4900`;
        # the Kunlun DSA pool overrides only the allocation, not the accessor --
        # `kunlun_pools.py`). `indices` below is built with `page_size=1`
        # from the token id, so it addresses those rows directly. A dense
        # bf16/fp16 pool row *is* that table; a mixed INT8 row is 656 bytes of
        # packed nope/scales/rope and is dequantized into it at the call site.
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)

        # Align topk_indices with q dimensions (q may be padded: TP + partial DP)
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # here we use page size = 1
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = metadata.topk_indices_offset
                assert topk_indices_offset is not None
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                assert metadata.dsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.dsa_extend_seq_lens_list,
                    page_size=1,
                )

        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # Sparse MLA prefill goes through the same vendor op the 0.5.8 baseline
        # ran (sgl-kernel `.../attention/nsa_backend.py`, sparse-prefill arm),
        # not through `kunlun_ops.sparse_prefill_fwd_opt`:
        #
        #   torch.ops.xspeedgate_ops.sparse_attn_fwd(
        #       q, kv, indices, sm_scale, d_v, topk_length, attn_sink=None)
        #       -> Tensor[]  # (out, max_logits, lse)
        #
        # `sparse_prefill_fwd_opt` is a 0.5.23-era call that has never worked on
        # this hardware: it reaches `xfa::dsa_prefill`, whose non-zero return is
        # what surfaces as `xops_block_ERROR ... prefill_dsa_attention_opt_block.cc
        # line:115` -- a vendor-block status, not a wrapper dtype/shape check.
        #
        # Shapes/semantics below are the op's own (schema + the validation
        # strings in its meta function), and they differ from the old call:
        #   q           [s_q, h_q, qk_head_dim], fp16/bf16, contiguous
        #   kv          [s_kv, qk_head_dim],     dtype == q.dtype, contiguous
        #   indices     [s_q, topk],             int32 -- 2-D, no kv-head axis
        #   sm_scale    float
        #   d_v         int, value width of `out`
        #   topk_length [s_q],                   int32
        # No q/kv lods: the batch structure is implied by the page-1 table (every
        # q row owns its own top-k list) and the only per-row bound is
        # `topk_length`. So `q_all` keeps the model's dtype instead of being
        # forced to bf16 (the op instantiates fp16 as well as bf16), and `out`
        # consequently comes back in q's dtype.
        #
        # `kv` must be the flat [tokens, qk_head_dim] pool view in q's dtype.
        #
        # * `--kv-cache-dtype int8` stores the packed mixed row
        #   `[int8 nope | fp32 per-128 scales | bf16 rope]`, 656 B/token
        #   (`kernels/mixed_int8_kv.py`, written by
        #   `KunlunNSATokenToKVPool._write_mla_kv_buffer`). This op has no scale
        #   argument and no int8 instantiation, so those rows go to the native
        #   mixed-cache op (`_native_mixed_attn`) and are only dequantized here
        #   when that op is unavailable.
        #   `mixed_int8_kv` is the pool's own answer: a *cast* int8
        #   row (what this tree produced while the pool had no mixed-row
        #   writer) does not advertise it and still fails the guard below,
        #   rather than being "dequantized" into numbers nobody quantized.
        # * fp8 DSA storage (`dsa_kv_cache_store_fp8`) packs
        #   [fp8 nope | fp32 per-128 scales | bf16 rope] into 656 u8 elements.
        #   It is not the int8 row (different element dtype, different scale
        #   tiling) and it is refused below: its row width is not
        #   `qk_head_dim` (`calculate_mla_kv_cache_dim`,
        #   `kv_cache_configurator.py:3464-3482`).
        if getattr(self.token_to_kv_pool, "mixed_int8_kv", False):
            # The op instantiates bf16/fp16 only, so a model served in either
            # runs in its own dtype. The fp32 projections the dense branch
            # documents below (`q=float` in the op's own error) fall back to
            # fp16 -- the dtype this layer chain's residual stream carries, and
            # the same value `forward_decode` hardcodes for the same reason.
            out_dtype = q_all.dtype
            if out_dtype not in (torch.float16, torch.bfloat16):
                out_dtype = torch.float16
            q_all = q_all.to(out_dtype)
            # 0.5.8 branched here instead of dequantizing
            # (`nsa_backend.py:1751-1759`, sparse prefill and target-verify
            # both). The native op reads the packed row and its embedded
            # scales, so it saves the dense copy below and is what emits the
            # three `fa_xpu3` mixed-cache kernels the operator-set gate wants.
            native = self._native_mixed_attn(q_all, page_table_1, layer, metadata)
            if native is not None:
                return native
            kv = self.token_to_kv_pool.dequant_mixed_kv_table(
                layer.layer_id, out_dtype
            )
        else:
            if kv_cache.dtype not in (torch.float16, torch.bfloat16) or (
                kv_cache.size(-1) != layer.head_dim
            ):
                raise RuntimeError(
                    "Kunlun DSA prefill: the KV pool row is "
                    f"{kv_cache.size(-1)} elements of {kv_cache.dtype}, but "
                    "sparse_attn_fwd reads kv as [tokens, "
                    f"qk_head_dim={layer.head_dim}] in q's dtype. Only the "
                    "mixed INT8 row (KunlunNSATokenToKVPool.mixed_int8_kv) and "
                    "a dense fp16/bf16 row are usable here; fp8 DSA storage "
                    "belongs to dsa_prefill_with_mixed_cache."
                )
            # Match `q` to the KV pool's dtype, not the other way round. On P800
            # the projections come out of the vendor GEMV in fp32 (`q=float` in
            # the op's own error), and `sparse_attn_fwd` instantiates only bf16
            # and fp16. The pool dtype is already the configured KV dtype, so
            # this is the one cast both sides can agree on.
            # Everything the op touches goes to the KV pool's dtype, and the
            # result comes back in the same dtype. The pool dtype is the
            # configured `--kv-cache-dtype` (model dtype for a dense pool), so
            # this is what the residual stream and the W8A8 projections
            # downstream expect -- their `kunlun_ops.matmul` rejects anything
            # else with `Check failed, matmul unimplemented`.
            out_dtype = kv_cache.dtype
            q_all = q_all.to(out_dtype)
            kv = kv_cache.contiguous().view(-1, layer.head_dim)

        # `dsa_cache_seqlens_int32` is this tree's `nsa_cache_seqlens_int32`: one
        # int32 per q row holding `min(valid_kv_len, index_topk)`
        # (`compute_dsa_seqlens` clamps to `dsa_index_topk`, `dsa/utils.py:159-161`),
        # i.e. the "how many LEADING entries of `indices` are real, the tail is
        # -1 padding" contract this op wants. Upstream reads the same field for
        # the same purpose (`_forward_flashmla_sparse`, `dsa_backend.py:2854-2868`).
        # DP/CP padding rows are already zero-length segments in it.
        topk_length = metadata.dsa_cache_seqlens_int32
        if topk_length.shape[0] != q_all.shape[0]:
            # Metadata rows are expected to match q rows, but this op has no
            # "compute the full width" fallback (its `topk_length` is required),
            # so rebuild the counts from the page-1 table: `indices >= 0` is
            # exactly the number of real slots in the row (device-side, so it is
            # capture-safe).
            topk_length = (page_table_1 >= 0).sum(dim=-1).to(torch.int32)

        o_, _, _ = torch.ops.xspeedgate_ops.sparse_attn_fwd(
            q_all,
            kv,
            page_table_1.to(torch.int32),
            layer.scaling,
            layer.v_head_dim,
            topk_length,
        )

        # Defensive: replace NaN with 0 (zero-length padding rows).
        # Back to the dtype the layer chain runs in. In the bf16 pool case the
        # op returns bf16 while the residual stream is fp16, and the next
        # `add_rmsnorm_inplace` rejects a mixed pair (`x and residual must have
        # the same dtype`).
        return torch.nan_to_num(o_, nan=0.0).to(out_dtype)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode through the Kunlun sparse kernel."""
        # Same 0.5.22 `attn_sink` tail as forward_extend; the Kunlun sparse
        # decode kernel has no sink input either.
        if attn_sink is not None:
            raise NotImplementedError(
                "KunlunDSAAttnBackend.forward_decode: learnable attention sinks "
                "(attn_sink) are not supported by the Kunlun sparse kernel"
            )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert q_rope is not None, "Kunlun DSA decode requires the absorbed MLA path"
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)

        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        # Concatenate in the attention dtype, do not let ``torch.cat`` pick it.
        # ``q_nope`` is the absorbed-MLA bmm output and ``q_rope`` the rotary
        # output, and on 0.5.23 those two arrive in *different* dtypes (bf16 and
        # fp16 respectively, because upstream's ``w_kc`` is hardcoded bf16 --
        # ``deepseek_weight_loader.py:747-749``). ``torch.cat`` then promotes the
        # pair to fp32, concatenates at double width and the ``.to(out_dtype)``
        # below casts the result back, which the 2026-11-12 decode trace shows as
        # three extra casts per layer invocation around one widened concat::
        #
        #     cluster_cast_kl3<bfloat16, float>  5.2 us   q_nope
        #     cluster_cast_kl3<float16, float>   2.4 us   q_rope
        #     concat_t_different                 6.8 us   (4.7 us in fp16 on 0.5.8)
        #     cluster_cast_kl3<float, float16>   5.2 us   q_all
        #
        # i.e. 243 casts per step for ~1.04 ms/step, where 0.5.8 ran the concat
        # alone with both operands fp16 (``sgl-kernel/.../nsa_backend.py``, same
        # ``torch.cat([q_nope, q_rope])`` with an fp16 bmm output).
        #
        # Casting the operands first costs one cast on the mismatched operand
        # instead of three on both plus the result, and once the absorbed
        # projections are restored to the served dtype
        # (``hooks/models/deepseek_v4_precision.py:180-244``) both ``.to`` calls
        # become no-ops -- ``Tensor.to`` returns ``self`` for a matching dtype --
        # and this is exactly the 0.5.8 single-kernel concat. ``out_dtype`` below
        # is the same constant and documents why fp16 is the right target.
        q_all = torch.cat(
            [q_nope.to(torch.float16), q_rope.to(torch.float16)], dim=-1
        )

        # Decode runs the SAME operator as sparse prefill, exactly as the 0.5.8
        # baseline does: `sgl-kernel/.../nsa_backend.py:2240-2244` calls
        # `torch.ops.xspeedgate_ops.sparse_attn_fwd(q_all, kv_attn, page_table_1,
        # layer.scaling, layer.v_head_dim, nsa_cache_seqlens_int32)` for both
        # arms -- on the int8 arm 0.5.8 only reaches `_native_mixed_attn` because
        # that image's op had an int8 instantiation; this one takes plain
        # values, so the mixed row is dequantized first
        # (`KunlunNSATokenToKVPool.dequant_mixed_kv_table`).
        #
        # The 0.5.23 plugin called `kunlun_ops.fwd_kvcache_mla` here instead,
        # which is a different vendor entry point and fails on this stack with
        # `decode_attention_dsa failed ret=3` once its dtype gate is satisfied.
        # Same contract as prefill: q [tokens, heads, qk_head_dim], kv
        # [tokens, qk_head_dim], indices [tokens, topk] int32, a per-row
        # `topk_length`, and the result in q's dtype.
        # bf16, not the pool dtype: the layer chain this feeds runs bf16 on
        # this platform even under `--dtype float16` (the vendor attention and
        # projection ops are bf16-only). Measured in both directions -- fp32 out
        # makes the W8A8 `o_proj` raise `matmul unimplemented`, fp16 out makes
        # the next `add_rmsnorm_inplace` raise `x and residual must have the
        # same dtype` against the bf16 residual stream.
        # fp16, the residual stream's dtype. Measured with a one-shot probe in
        # the plugin's `fused_add_rmsnorm`: x=bf16 vs residual=fp16, weight=fp16,
        # hence `x and residual must have the same dtype`. The earlier fp32
        # attempt failed the W8A8 `o_proj` (`matmul unimplemented`), so the
        # model dtype is the only value both consumers accept.
        out_dtype = torch.float16
        q_all = q_all.to(out_dtype)
        if getattr(self.token_to_kv_pool, "mixed_int8_kv", False):
            # Decode takes the same native op as sparse prefill, exactly as
            # 0.5.8 did: `nsa_backend.py:2235-2238` calls the one
            # `_native_mixed_attn` helper for both arms (with `allow_dcp=True`,
            # which is a no-op here -- see `_native_mixed_attn`).
            native = self._native_mixed_attn(q_all, page_table_1, layer, metadata)
            if native is not None:
                return native
            # Fallback: `--kv-cache-dtype int8` rows are packed
            # [int8 nope | fp32 scales | bf16 rope] and `sparse_attn_fwd` takes
            # plain values in q's dtype only, so they are dequantized per
            # layer. `.to(fp16)` here would reinterpret the packed bytes as
            # codes.
            kv_attn = self.token_to_kv_pool.dequant_mixed_kv_table(
                layer.layer_id, out_dtype
            )
        else:
            kv_attn = kv_cache.to(out_dtype).contiguous().view(-1, layer.head_dim)

        topk_length = metadata.cache_seqlens_int32
        if topk_length.shape[0] != q_all.shape[0]:
            topk_length = (page_table_1 >= 0).sum(dim=-1).to(torch.int32)

        o_, _, _ = torch.ops.xspeedgate_ops.sparse_attn_fwd(
            q_all,
            kv_attn,
            page_table_1.to(torch.int32),
            layer.scaling,
            layer.v_head_dim,
            topk_length,
        )
        return torch.nan_to_num(o_, nan=0.0)

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: KunlunDSAMetadata,
    ) -> torch.Tensor:
        """Dense MHA_ONE_SHOT path via ``kunlun_ops.attention``."""
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        cu_seqlens_q_cpu = metadata.cu_seqlens_q_cpu
        cu_seqlens_k_cpu = metadata.cu_seqlens_k_cpu

        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q)-1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k)-1} requests"
        )

        o_ = torch.empty(
            [q.shape[0], layer.tp_q_head_num, layer.v_head_dim],
            dtype=torch.bfloat16,
            device=q.device,
        )
        ds_alpha = layer.scaling * math.sqrt(layer.head_dim)
        # the kernel requires an lse buffer even though we drop it here
        softmax_lse = torch.full(
            (layer.tp_q_head_num, q.size(0)),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        kunlun_ops.attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            o_,
            is_causal=True,
            is_prefill=True,
            prefill_len=0,
            k_perchannel_scale=None,
            v_perchannel_scale=None,
            smooth=None,
            context_seq_lod_cpu=cu_seqlens_q_cpu,
            context_seq_lod_xpu=cu_seqlens_q,
            slot_mapping_cpu=None,
            slot_mapping_xpu=None,
            context_kvlen_lod_cpu=cu_seqlens_k_cpu,
            context_kvlen_lod_xpu=cu_seqlens_k,
            v_trans=False,
            v_trans_threshold=0,
            alpha=ds_alpha,
            softmax_lse=softmax_lse,
            unpadded_lse=True,
        )
        del softmax_lse
        return o_


class KunlunDSAMultiStepBackend(DeepseekSparseAttnMultiStepBackend):
    """Multi-step (MTP) wrapper around :class:`KunlunDSAAttnBackend`.

    NOTE: 0.5.14 builds ``speculative_num_steps - 1`` backends (0.5.8 built all
    ``speculative_num_steps``); this follows upstream.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                KunlunDSAAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    seed_dsa_topk_from_draft_extend=seed_dsa_topk_from_draft_extend,
                )
            )
