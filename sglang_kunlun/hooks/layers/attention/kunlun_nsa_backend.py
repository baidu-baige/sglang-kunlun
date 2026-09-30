"""kunlun_nsa attention backend for sglang 0.5.14 (DeepSeek sparse attention).

Ported from the 0.5.8 subclass
``sgl_kernel/patch/layers/attention/kunlun_nsa_backend.py``. Same design: subclass
upstream and override only what really differs on Kunlun.

0.5.14 deltas that this port had to honour (see
``/home/zx/doc/ds_v32/kunlun_dsa_0514_migration_delta.md``):

* ``NativeSparseAttnBackend`` -> ``DeepseekSparseAttnBackend``, ``NSAMetadata`` ->
  ``DSAMetadata``, ``NSAIndexerMetadata`` -> ``DSAIndexerMetadata``, all ``nsa_*``
  metadata fields -> ``dsa_*``; ``TopkTransformMethod`` moved to
  ``dsa/dsa_topk_backend.py``.
* the cuda-graph seams changed: there is no
  ``init_forward_metadata_capture_cuda_graph`` / ``..._replay_cuda_graph`` any more,
  so ``cum_q_lod`` is injected in ``_build_forward_metadata_cuda_graph`` and
  ``_apply_cuda_graph_metadata`` instead.
* ``DSAIndexerMetadata`` gained ``topk_backend`` / ``paged_mqa_ctx_lens_2d`` /
  ``force_unfused_topk`` and its ``topk_transform`` now delegates to
  ``DSATopKBackend``. ``DSATopKBackend`` is a closed Enum, so a plugin cannot add a
  member - the Kunlun fused topk stays an override of ``topk_transform``.
* ``DeepseekSparseAttnMultiStepBackend`` builds ``speculative_num_steps - 1``
  backends (0.5.8 built all of them).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Optional

import torch

import kunlun_ops
import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops

# Sentinel for the per-forward cache in _per_row_lods (None is a real result).
_UNSET = object()

# Selects between the GLM-5.3 DSA kernel path (validated on P800) and the
# upstream DeepSeek-V4 path. Default "1" = GLM path.
_USE_GLM_DSA_PATH = os.environ.get("SGLANG_KUNLUN_GLM_DSA_PATH", "1") == "1"

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

# Upstream (DSV4) uses the Triton ``transform_index`` helpers. Triton does not
# compile on P800 ("'_CudaTarget' object has no attribute 'arch'"), so the import
# is guarded and the GLM path binds the torch ``_ref`` variants instead. Upstream
# ships torch reference implementations of exactly these two helpers (a gather
# plus a -1 fill for out-of-range topk) and the kwargs at both call sites below
# are compatible with either signature.
try:
    from sglang.srt.layers.attention.dsa.transform_index import (
        transform_index_page_table_decode as _transform_index_decode_triton,
        transform_index_page_table_prefill as _transform_index_prefill_triton,
    )
except ImportError:  # pragma: no cover - Triton unavailable on P800
    _transform_index_decode_triton = None
    _transform_index_prefill_triton = None

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_ref as _transform_index_decode_ref,
    transform_index_page_table_prefill_ref as _transform_index_prefill_ref,
)

if _USE_GLM_DSA_PATH or _transform_index_decode_triton is None:
    transform_index_page_table_decode = _transform_index_decode_ref
    transform_index_page_table_prefill = _transform_index_prefill_ref
else:
    transform_index_page_table_decode = _transform_index_decode_triton
    transform_index_page_table_prefill = _transform_index_prefill_triton

from sglang.srt.layers.attention.dsa.utils import (
    is_dsa_prefill_cp_in_seq_split,
    is_dsa_prefill_cp_round_robin_split,
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


def _view_q_rope(q_rope, q_nope: torch.Tensor, layer) -> torch.Tensor:
    """Reshape the rope half of q, tolerating models without one.

    GLM-5-Next sets ``qk_rope_head_dim = 0``, so ``head_dim == v_head_dim`` and
    ``q_rope`` arrives empty (or None). An empty tensor cannot be reshaped with
    an inferred token dim, so build the zero-width view from ``q_nope`` instead;
    the later ``cat([q_nope, q_rope], -1)`` then reduces to ``q_nope``.
    """

    rope_dim = layer.head_dim - layer.v_head_dim
    if rope_dim == 0:
        return q_nope.new_empty((q_nope.shape[0], layer.tp_q_head_num, 0))
    assert q_rope is not None, "Kunlun DSA requires the absorbed MLA path"
    return q_rope.view(-1, layer.tp_q_head_num, rope_dim)


def _paged_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pick the top-k scored positions and map them to page-1 cache slots.

    Dispatches between the GLM fused ``xspeedgate_ops.topk_transform`` path
    (default, validated on P800) and upstream's pure-torch port.
    """
    impl = (
        _paged_topk_transform_xsg if _USE_GLM_DSA_PATH else _paged_topk_transform_torch
    )
    return impl(score, lengths, src_page_table, topk, cu_seqlens_q)


def _paged_topk_transform_torch(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Torch port of the fused paged top-k transform (upstream DSV4 path).

    ``xspeedgate_ops.topk_transform`` requires ``src_page_table`` to be at least
    as wide as the score, but on 0.5.14 the score width is page aligned
    (``page_table_64.shape[1] * page_size``) while the page-1 table is exactly
    ``max_seqlen_k`` wide, so that contract no longer holds. Positions are
    returned in ascending order (the selected *set* is what the sparse attention
    consumes) with ``-1`` padding.
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


def _paged_topk_transform_xsg(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ``xspeedgate_ops.topk_transform`` paged top-k (GLM P800 path).

    ``xspeedgate_ops.topk_transform`` requires
    ``src_page_table.size(1) * block_size >= score.size(1)``. The paged indexer
    pads its logits out to the graph's ``max_seq_len`` while the page-1 table is
    exactly ``max_seqlen_k`` wide (measured on a live decode: score 2048 columns
    against an 18-column table), which is why the other variant is a torch port.
    Narrowing the score to the table width restores the contract without
    changing the result: ``lengths`` never exceeds the page-1 width, so the
    dropped columns are ones ``lengths`` already masks out.

    Positions come back in the op's own order (the selected *set* is what the
    sparse attention consumes) with ``-1`` padding.
    """

    rows, width = score.shape
    if rows == 0:
        return score.new_empty((0, topk), dtype=torch.int32)

    table_width = src_page_table.shape[1]
    if width > table_width:
        score = score[:, :table_width]

    if src_page_table.shape[0] == rows:
        # One table row per score row: the op indexes it directly.
        cu_seqlens_q = None
    else:
        # Prefill/verify feed one score row per query token while the page table
        # keeps one row per request; cu_seqlens_q carries that mapping.
        assert (
            cu_seqlens_q is not None
        ), "paged topk transform needs cu_seqlens_q to map tokens to requests"
        cu_seqlens_q = cu_seqlens_q.contiguous()

    # The op only writes the entries it selects, so the buffer has to arrive
    # pre-filled: everything it does not touch is the ``-1`` padding the sparse
    # attention expects.
    dst_page_table = torch.full(
        (rows, topk), -1, dtype=torch.int32, device=score.device
    )
    torch.ops.xspeedgate_ops.topk_transform(
        score=score.contiguous(),
        lengths=lengths.to(torch.int32).contiguous(),
        src_page_table=src_page_table.contiguous(),
        dst_page_table=dst_page_table,
        topk=topk,
        cu_seqlens_q=cu_seqlens_q,
    )
    return dst_page_table


def _per_row_lods(metadata, page_table_1: torch.Tensor):
    """Per-token ``(qlod_cpu, qlod, kvlod_cpu, kvlod)``, or None if no row is padded.

    The kpool topk table is ``index_topk + pool_size - 1`` columns wide while a
    row only fills ``min(4 * (pos // 4), index_topk) + pos % 4`` of them, so 3
    rows out of 4 end in -1. The kernel takes each row's column count from
    ``min(n_cols, kvlen - qlen + r + 1)``, a per-batch quantity that cannot
    express that per-row sawtooth (it grows by 1 per row, the real count jumps
    between 2048 and 2051 with period 4). Giving every q row its own single-row
    segment collapses the formula to ``min(n_cols, kvlen)``, i.e. exactly this
    row's valid count, so the -1 tail is never used as an address.

    The counts depend only on a row's visible kv length, not on the layer, so
    they are computed once per forward batch and cached on ``metadata``.
    """
    cached = getattr(metadata, "_kunlun_per_row_lods", _UNSET)
    if cached is _UNSET:
        n_valid = (page_table_1 >= 0).sum(dim=-1).cpu()
        if int(n_valid.min()) == page_table_1.shape[-1]:
            cached = None  # nothing is padded: the per-request lods are correct
        else:
            rows = n_valid.numel()
            kv_lod_cpu = torch.zeros(rows + 1, dtype=torch.int32)
            kv_lod_cpu[1:] = (
                n_valid.clamp(min=1).to(torch.int64).cumsum(dim=0).to(torch.int32)
            )
            assert int(kv_lod_cpu[-1]) < 2**31 - 1, (
                f"kv lod overflows int32 at {rows} q tokens; keep "
                "--chunked-prefill-size below ~1M"
            )
            q_lod_cpu = torch.arange(rows + 1, dtype=torch.int32)
            device = page_table_1.device
            cached = (
                q_lod_cpu,
                q_lod_cpu.to(device),
                kv_lod_cpu,
                kv_lod_cpu.to(device),
            )
        object.__setattr__(metadata, "_kunlun_per_row_lods", cached)
    return cached


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

    def __post_init__(self):
        """Cache the host mirrors used by the Kunlun kernels."""

        def _to_cpu(name: str, tensor: Optional[torch.Tensor]):
            object.__setattr__(self, name, None if tensor is None else tensor.cpu())

        _to_cpu("cache_seqlens_int32_cpu", self.cache_seqlens_int32)
        _to_cpu("cu_seqlens_q_cpu", self.cu_seqlens_q)
        _to_cpu("cu_seqlens_k_cpu", self.cu_seqlens_k)
        _to_cpu("dsa_seqlens_expanded_cpu", self.dsa_seqlens_expanded)
        _to_cpu("dsa_cu_seqlens_q_cpu", self.dsa_cu_seqlens_q)
        _to_cpu("dsa_cu_seqlens_k_cpu", self.dsa_cu_seqlens_k)
        _to_cpu("cum_q_lod_cpu", self.cum_q_lod)


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
        return compute_cu_seqlens(forward_batch.extend_seq_lens)

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
        metadata = self.forward_metadata
        if (
            isinstance(metadata, KunlunDSAMetadata)
            and metadata.cache_seqlens_int32_cpu is not None
        ):
            # The captured Kunlun kernel keeps this host pointer.  Upstream
            # refreshes the device seqlens in place before replay, but the CPU
            # mirror created by KunlunDSAMetadata.__post_init__ is a separate
            # tensor and otherwise remains at its capture-time values.  Keep
            # the pointer stable and refresh only its contents.
            metadata.cache_seqlens_int32_cpu.copy_(
                metadata.cache_seqlens_int32.detach().cpu()
            )

        # forward_decode hands fwd_kvcache_mla_v2 a host kv LoD whose pointer is
        # captured into the graph. Replay does not re-run that python code, so
        # rewrite the buffer's contents here, before the replay, from this
        # cycle's sequence lengths. topk_width comes from the first decode
        # forward; before that there is no captured graph to keep in sync.
        topk_width = getattr(self, "_v2_topk_width", None)
        if (
            topk_width is not None
            and forward_mode.is_decode_or_idle()
            and seq_lens_cpu is not None
        ):
            self._v2_refresh_kv_lod_host(bs, topk_width, seq_lens_cpu)

    def _rewrap_graph_metadata(self, bs: int, forward_mode: ForwardMode) -> None:
        """Upgrade ``self.forward_metadata`` (and the per-bs cache) in place."""
        if isinstance(self.forward_metadata, KunlunDSAMetadata):
            # already wrapped (apply() can run right after build())
            return
        metadata = self._to_kunlun_metadata(
            self.forward_metadata, self._cum_q_lod_for_graph(bs, forward_mode)
        )
        self.forward_metadata = metadata
        cache = getattr(self, "graph_metadata", None)
        if isinstance(cache, dict) and bs in cache:
            cache[bs] = metadata

        # DeepseekSparseAttnBackend keeps one metadata object per captured
        # bucket here.  The kernel captures host pointers from this wrapped
        # object, so it must be written back to the real cache; otherwise replay
        # wraps a second object and refreshes a CPU mirror the graph never uses.
        cache = getattr(self, "decode_cuda_graph_metadata", None)
        if isinstance(cache, dict) and bs in cache:
            cache[bs] = metadata

    def _v2_refresh_kv_lod_host(self, bs, topk_width, seqlens_cpu):
        """Stable host kv LoD for the v2 decode call, refreshed in place.

        ``fwd_kvcache_mla_v2`` keeps this host pointer, and cuda-graph replay
        does not re-run the python forward, so the buffer identity must be
        stable while its contents are rewritten before every replay (see
        ``_apply_cuda_graph_metadata``). The per-sequence count is
        ``min(seq_len, topk_width)``, which is exactly how many page-table slots
        ``transform_index_page_table_decode`` keeps, so no device read is needed.
        """
        store = getattr(self, "_v2_kv_lod_host_cache", None)
        if store is None:
            store = {}
            self._v2_kv_lod_host_cache = store
        key = (bs, topk_width)
        buf = store.get(key)
        if buf is None:
            buf = torch.zeros(bs + 1, dtype=torch.int32)
            store[key] = buf
        counts = seqlens_cpu[:bs].to(torch.int32).clamp(max=topk_width)
        buf[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return buf

    def _v2_kv_lod_device(self, bs, topk_width, device):
        """Stable device kv LoD buffer mirroring the host one."""
        store = getattr(self, "_v2_kv_lod_dev_cache", None)
        if store is None:
            store = {}
            self._v2_kv_lod_dev_cache = store
        key = (bs, topk_width, str(device))
        buf = store.get(key)
        if buf is None:
            buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            store[key] = buf
        return buf

    def _dsa_index_kpool(self):
        """kpool pool size (index_kpool); the decode page table's unpooled tail
        occupies its last ``index_kpool - 1`` columns. Cached; falls back to 4
        (GLM-5-Next / GLM-5.3-Flash) if the config is unavailable."""
        v = getattr(self, "_v2_index_kpool_cached", None)
        if v is None:
            try:
                from sglang.srt.configs.model_config import get_dsa_index_kpool
                from sglang.srt.server_args import get_global_server_args

                hf = get_global_server_args().get_model_config().hf_config
                v = int(get_dsa_index_kpool(hf))
            except Exception:
                v = 4
            self._v2_index_kpool_cached = v
        return v

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
    ) -> torch.Tensor:
        """Prefill / verify / draft-extend through the Kunlun sparse kernel."""
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
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = _view_q_rope(q_rope, q_nope, layer)

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

        if _USE_GLM_DSA_PATH:
            # q/kv for sparse attention。kunlun_ops>=0.1.228 起 qk_head_dim==v_head_dim，
            # 无需 _SPARSE_QK_PAD 加宽（decode 的 fwd_kvcache_mla 仍需要）。
            q_sparse = q_all.to(kv_cache.dtype)
            kv_sparse = kv_cache.contiguous().view(-1, kv_cache.size(-1))
            # DSA 稀疏 prefill 走 XSpeedGate sparse_attn_fwd（替代 kunlun_ops.sparse_prefill_fwd_opt）。
            # 新版 XSG 已支持 2051 宽 indices（index_topk 2048 + pool_size-1）并容忍 -1（单测 replay 验证）。
            # 该算子按值返回 out/max_logits/lse，不吃 qlod/kvlod/is_causal（causal/varlen 由 topk 索引本身表达）；
            # topk_length = 每行有效(非 -1)索引数（kpool 输出为连续 valid 前缀 + -1 尾部）。
            topk_length = (page_table_1 >= 0).sum(dim=1).to(torch.int32).clamp_(min=1)
            # sparse_attn_fwd 在 fp16 下更快（bf16 vs fp16 单测见 probe_sparse_attn_fwd_timing.py）。
            # 服务整体是 bf16（prefill 全程 fp16 反而更慢），故这里**只把该算子局部 cast 成 fp16** 计算、
            # 再把输出 cast 回模型 dtype（下方 return o_.to(q_all.dtype)）。
            # ⚠️ kv_sparse 是整层 KV pool 的展平视图，`.to(fp16)` 是 O(pool) 拷贝；需用 trace 确认
            #   「fp16 注意力省的时间 > kv cast 多花的时间」，否则应改为只 cast 有效 kv 段、或让算子内部
            #   只 cast 命中的行（bf16-kv + fp16-compute）。
            q_fp16 = q_sparse.to(torch.float16)
            kv_fp16 = kv_sparse.to(torch.float16)
            o_, max_logits, lse = torch.ops.xspeedgate_ops.sparse_attn_fwd(
                q_fp16,
                kv_fp16,
                page_table_1.to(torch.int32).contiguous(),
                layer.scaling,
                layer.v_head_dim,
                topk_length,
            )
        else:
            # sparse_prefill_fwd_opt takes explicit q/kv lods and needs preallocated
            # output buffers.
            o_ = torch.zeros(
                [q_all.shape[0], layer.tp_q_head_num, layer.v_head_dim],
                dtype=torch.bfloat16,
                device=q_all.device,
            )
            max_logits = torch.zeros(
                [q_all.shape[0], layer.tp_q_head_num],
                dtype=torch.float32,
                device=q_all.device,
            )
            lse = torch.zeros(
                [q_all.shape[0], layer.tp_q_head_num],
                dtype=torch.float32,
                device=q_all.device,
            )

            if (
                forward_batch.attn_cp_metadata is not None
                and is_dsa_prefill_cp_in_seq_split()
            ):
                # Only use attn_cp_metadata attributes in "in seq split" mode;
                # round-robin split is handled by the override below.
                # prefix_kv_len = KV from prior chunks (0 for first chunk, >0 for 2nd+),
                # because kv_len_prev/next only cover the current chunk's CP range.
                cp_metadata = forward_batch.attn_cp_metadata
                prefix_kv_len = 0
                if (
                    forward_batch.extend_seq_lens_cpu is not None
                    and forward_batch.seq_lens_cpu is not None
                ):
                    prefix_kv_len = int(forward_batch.seq_lens_cpu[0].item()) - int(
                        sum(forward_batch.extend_seq_lens_cpu)
                    )

                actual_seq_q_prev = cp_metadata.actual_seq_q_prev_list[0]
                kv_len_prev_total = prefix_kv_len + cp_metadata.kv_len_prev_list[0]
                kv_len_next_total = prefix_kv_len + cp_metadata.kv_len_next_list[0]

                # lod is a cumulative offset: the kernel derives per-batch length as
                # lod[i+1] - lod[i]; batch 0 is "prev", batch 1 is "next".
                cum_q_lod_cpu = torch.tensor(
                    [0, actual_seq_q_prev, q_all.shape[0]], dtype=torch.int32
                )
                kv_lod_cpu = torch.tensor(
                    [0, kv_len_prev_total, kv_len_prev_total + kv_len_next_total],
                    dtype=torch.int32,
                )
                cum_q_lod = cum_q_lod_cpu.to(device=q_all.device)
                kv_lod = kv_lod_cpu.to(device=q_all.device)
            else:
                cum_q_lod = metadata.cu_seqlens_q
                cum_q_lod_cpu = metadata.cu_seqlens_q_cpu
                kv_lod = metadata.cu_seqlens_k
                kv_lod_cpu = metadata.cu_seqlens_k_cpu

            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                cum_q_lod = metadata.cum_q_lod
                cum_q_lod_cpu = metadata.cum_q_lod_cpu

            if is_dsa_prefill_cp_round_robin_split():
                cum_q_lod = metadata.dsa_cu_seqlens_q
                cum_q_lod_cpu = metadata.dsa_cu_seqlens_q_cpu
                kv_lod = metadata.dsa_cu_seqlens_k
                kv_lod_cpu = metadata.dsa_cu_seqlens_k_cpu

            kunlun_ops.sparse_prefill_fwd_opt(
                q=q_all,
                kv=kv_cache.contiguous().view(-1, kv_cache.size(-1)),
                indices=page_table_1.unsqueeze(1),
                out=o_,
                max_logits=max_logits,
                lse=lse,
                sm_scale=layer.scaling,
                is_causal=True,
                qlod_cpu=cum_q_lod_cpu,
                qlod_xpu=cum_q_lod,
                kvlod_cpu=kv_lod_cpu,
                kvlod_xpu=kv_lod,
            )

        # Defensive: replace NaN with 0 (e.g. when kv_lod has 0-length padding)
        o_ = torch.nan_to_num(o_, nan=0.0)

        del max_logits
        del lse
        if _USE_GLM_DSA_PATH:
            # The sparse-prefill kernel only supports a bf16/fp16 q/kv/out; under
            # --dtype float16 we ran it in fp16, so cast the result back to the
            # model dtype for the caller.
            return o_.to(q_all.dtype)
        return o_

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
    ) -> torch.Tensor:
        """Decode through the Kunlun sparse kernel."""
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
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = _view_q_rope(q_rope, q_nope, layer)

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

        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # GLM/P800 default vs upstream DSV4 decode kernel (see _USE_GLM_DSA_PATH).
        if _USE_GLM_DSA_PATH:
            # fwd_kvcache_mla_v2 accepts qk_head_dim == v_head_dim, so the
            # _SPARSE_QK_PAD copy of the whole layer KV pool is gone. It addresses KV
            # as ``kv_cache[kv_lod[b] + indices[b][j]]`` (measured: batch 0, whose
            # base is 0, matched a packed reference exactly while batches with a
            # nonzero base did not, and rebasing to negative indices is rejected),
            # so each sequence's selected rows must start exactly at kv_lod[b].
            #
            # The build now accepts kv.size(0) > kv_lod[-1], which lets that packing
            # land in a STATIC [bs*W + 1, R] buffer instead of an exactly-Tkv one:
            # every shape below is fixed for a given (bs, W) and only LoD/index
            # values change, which is what cuda-graph capture needs.
            bs = forward_batch.batch_size
            assert q_all.shape[0] == bs, (
                f"Kunlun DSA decode expects one query token per sequence, got "
                f"{q_all.shape[0]} tokens for bs={bs}"
            )
            kv_pool = kv_cache.view(-1, self.kv_cache_dim)
            slots = page_table_1.view(bs, -1)
            topk_width = slots.shape[-1]
            valid = slots >= 0

            # kunlun_ops >= 0.1.238+a959415f: fwd_kvcache_mla_v2 addresses KV with
            # ABSOLUTE pool rows -- kv_lod is a logical per-seq count only and no base
            # offset is added (verified: seq with base!=0 reads pool[index], not
            # pool[base+index]). So the whole KV pool is passed straight through and
            # the per-layer pack/scatter (index_copy of ~65k x R rows, the decode
            # hot spot) is gone. NOTE: the old tight-pack + ramp-index scheme is
            # WRONG on this build for bs>1 (ramp index 0 is absolute -> every seq
            # would read pool row 0), which is why this had to change.
            #
            # The kernel reads the first count = kv_lod[b+1]-kv_lod[b] entries of
            # indices[b], which must therefore be the valid absolute slots left-
            # compacted into [0, count). The decode page table has the kpool 2-block
            # shape -- history block [0, nA) then unpooled-tail block
            # [T, T+nB), T = topk_width-(index_kpool-1), -1 elsewhere (verified on
            # real dumps) -- so the compaction is elementwise, no scatter:
            #   col_map[r] = r               for r < nA          (history)
            #              = T + (r - nA)     for nA <= r < count (tail)
            # every shape below is static for a given (bs, topk_width), which is what
            # cuda-graph capture needs.
            self._v2_topk_width = topk_width
            counts = valid.sum(dim=1, keepdim=True).to(torch.int32)
            kv_lod_xpu = self._v2_kv_lod_device(bs, topk_width, q_all.device)
            kv_lod_xpu[0] = 0
            kv_lod_xpu[1:] = torch.cumsum(counts.view(-1), dim=0, dtype=torch.int32)
            kv_lod_cpu = self._v2_refresh_kv_lod_host(
                bs, topk_width, metadata.cache_seqlens_int32_cpu
            )

            tail_start = topk_width - (self._dsa_index_kpool() - 1)
            col = torch.arange(
                topk_width, dtype=torch.int32, device=q_all.device
            ).view(1, topk_width).expand(bs, topk_width)
            n_hist = (valid & (col < tail_start)).sum(dim=1, keepdim=True).to(torch.int32)
            col_map = torch.where(
                col < n_hist,
                col,
                torch.where(
                    col < counts, tail_start + (col - n_hist), torch.zeros_like(col)
                ),
            ).to(torch.int64)
            abs_slots = torch.gather(slots, 1, col_map)
            # left-compacted absolute pool rows for j < count, else -1 (unread).
            kernel_indices = torch.where(
                col < counts, abs_slots.to(torch.int32), torch.full_like(col, -1)
            ).view(bs, 1, topk_width)

            out = torch.zeros(
                [bs, layer.tp_q_head_num, layer.v_head_dim],
                dtype=kv_pool.dtype,
                device=q_all.device,
            )
            max_logits = torch.zeros(
                [bs, layer.tp_q_head_num], dtype=torch.float32, device=q_all.device
            )
            lse = torch.zeros(
                [bs, layer.tp_q_head_num], dtype=torch.float32, device=q_all.device
            )
            # is_causal=False: the topk selection already decides what this decode
            # token attends to, and the selected order is selection order rather than
            # true token positions, so no position-derived mask applies.
            # (Measured identical to is_causal=True for one query token per seq.)
            kunlun_ops.fwd_kvcache_mla_v2(
                q=q_all.to(kv_pool.dtype),
                kv_cache=kv_pool,
                indices=kernel_indices,
                q_lod_cpu=metadata.cum_q_lod_cpu,
                kv_lod_cpu=kv_lod_cpu,
                out=out,
                max_logits=max_logits,
                lse=lse,
                softmax_scale=layer.scaling,
                is_causal=False,
                q_lod_xpu=metadata.cum_q_lod,
                kv_lod_xpu=kv_lod_xpu,
            )
            del max_logits
            del lse

            # Restore the [bs, max_seq_q=1, heads, v_head_dim] layout the caller
            # expects; v2 returns the packed [Tq, heads, v_head_dim] form. The
            # kernel needs q/kv/out to share one dtype, so we run it at the KV
            # cache's dtype (bf16 by default; fp16 when --kv-cache-dtype float16)
            # and cast the result back to the model dtype for the caller.
            return out.view(bs, 1, layer.tp_q_head_num, layer.v_head_dim).to(q_all.dtype)
        else:
            # fwd_kvcache_mla wants a [tokens, 1, heads, head_dim] q and preallocated
            # output buffers.
            reshape_q = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
            bs = forward_batch.batch_size
            o_ = torch.zeros(
                [bs, reshape_q.shape[1], layer.tp_q_head_num, layer.v_head_dim],
                dtype=torch.bfloat16,
                device=q_all.device,
            )
            max_logits = torch.zeros(
                [bs, reshape_q.shape[1], layer.tp_q_head_num],
                dtype=torch.float32,
                device=q_all.device,
            )
            p_sums = torch.zeros(
                [bs, reshape_q.shape[1], layer.tp_q_head_num],
                dtype=torch.float32,
                device=q_all.device,
            )
            kunlun_ops.fwd_kvcache_mla(
                q_c=reshape_q,
                kv_cache=kv_cache.view(-1, self.kv_cache_dim),
                indices=page_table_1.view(bs, reshape_q.shape[1], page_table_1.shape[-1]),
                out=o_,
                max_logits=max_logits,
                p_sums=p_sums,
                softmax_scale=layer.scaling,
                kv_lod_cpu=metadata.cache_seqlens_int32_cpu,
                kv_lod_xpu=metadata.cache_seqlens_int32,
                max_seq_kv=metadata.max_seq_len_k,
            )
            del max_logits
            del p_sums
            return o_

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
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
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
                )
            )
