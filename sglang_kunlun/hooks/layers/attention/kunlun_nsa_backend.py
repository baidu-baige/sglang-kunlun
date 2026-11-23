"""kunlun_nsa attention backend"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Optional

import logging
import os

import torch

import kunlun_ops
import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
try:
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    def _cp_rowwise_split(forward_batch) -> bool:
        """Pre-0.5.23 spelling: the predicate took no per-batch argument."""
        return is_dsa_prefill_cp_round_robin_split()

except ImportError:
    # 0.5.23 renamed the round-robin split to "interleave" (same layout, token
    # i -> rank i % cp_size) and made the predicate per-batch, so the old name
    # is gone. Falling back to False is not harmless: the only call site swaps
    # in the CP-aware q/kv LoD before `sparse_prefill_fwd_opt`, so stuck at
    # False the kernel reads global LoD against rank-local rows. Measured on
    # ERNIE 5.5 (cp2 vs dp8, same 4K request), layer 7 -- the first full-MLA
    # layer this backend serves -- reaches maxrel 9.9e-1 and the served output
    # is empty. `kunlun_deepseek_v4_backend.py` already followed the rename.
    try:
        from sglang.srt.layers.attention.dsa.utils import (
            can_dsa_prefill_cp_interleave as _can_dsa_prefill_cp_interleave,
        )

        def _cp_rowwise_split(forward_batch) -> bool:
            """0.5.23 spelling of the same row-wise CP split."""
            return bool(_can_dsa_prefill_cp_interleave(forward_batch))

    except ImportError:

        def _cp_rowwise_split(forward_batch) -> bool:
            """Neither spelling available: no row-wise CP split on this build."""
            return False


try:
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
except ImportError:  # dsv4.1 删掉了 DSA prefill-CP 的 in-seq-split 模式

    def is_dsa_prefill_cp_in_seq_split() -> bool:
        """0.5.14 prefill-CP no longer supports in-seq-split.

        Unlike the rename above, this mode really is gone: it was a different
        layout (contiguous per-rank ranges), not a renamed one.
        """
        return False

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

_PAGED_TOPK_PATH_SEEN: set = set()
_PAGED_TOPK_FUSED_DISABLED = False
_PAGED_TOPK_SELFCHECK_DONE = False
_MIRROR_COALESCE_LOGGED = False


def _mark_fused_disabled(reason):
    """Disable the fused path for good and log why."""
    global _PAGED_TOPK_FUSED_DISABLED
    if not _PAGED_TOPK_FUSED_DISABLED:
        _PAGED_TOPK_FUSED_DISABLED = True
        logger.warning("ERNIE55 paged topk: fused path disabled (%s), falling back to torch", reason)


def _capturing() -> bool:
    """True while the decode cuda graph is being captured."""
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        return bool(get_is_capture_mode())
    except Exception:  # noqa: BLE001
        return False


def _selfcheck_fused(dst, score, lengths, src_page_table, topk, cu_seqlens_q):
    """Compare the fused result against the torch path once, row by row (sets, any order).

    A mismatch disables the fused path permanently: this path decides which tokens
    attention reads, so a wrong result is an accuracy bug. Runs once, cost is negligible.
    """
    global _PAGED_TOPK_SELFCHECK_DONE
    _PAGED_TOPK_SELFCHECK_DONE = True
    try:
        os.environ["ERNIE55_FUSED_PAGED_TOPK"] = "0"      # make the recursive call take the torch branch
        ref = _paged_topk_transform(score, lengths, src_page_table, topk, cu_seqlens_q)
        bad = 0
        for r in range(dst.shape[0]):
            if (set(dst[r].tolist()) - {-1}) != (set(ref[r].tolist()) - {-1}):
                bad += 1
        if bad:
            _mark_fused_disabled("selfcheck mismatch on %d/%d rows" % (bad, dst.shape[0]))
        else:
            logger.warning(
                "ERNIE55 paged topk selfcheck passed: fused and torch select the same set per row "
                "(rows=%s topk=%s width=%s table_width=%s)",
                dst.shape[0], topk, score.shape[1], src_page_table.shape[1],
            )
    except Exception as exc:
        _mark_fused_disabled("selfcheck itself raised " + repr(exc)[:140])
    finally:
        os.environ["ERNIE55_FUSED_PAGED_TOPK"] = "1"




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

    # KUNLUN_PERF: prefer the fused xspeedgate topk, which does top-K search, length
    # masking and page-table mapping in one op with no [rows, width] torch temporaries
    # (the torch port below costs 615 ms / 20 steps, 20% of device time). Guarded at
    # runtime, falls back to torch when the contract does not hold, one log line per
    # outcome. The score is narrowed to table_width first: the tail columns are page
    # padding with no legal position (all seq_len <= table_width) and the fused op only
    # scans the first seq_lens[row] columns. Cost then follows the actual sequence
    # length instead of the context_length-sized buffer width.
    # Disable with ERNIE55_FUSED_PAGED_TOPK=0.
    _score_fused = score[:, :table_width] if width > table_width else score
    _fused_ok = (
        os.environ.get("ERNIE55_FUSED_PAGED_TOPK", "1") != "0"
        and not _PAGED_TOPK_FUSED_DISABLED
        and src_page_table.shape[0] == rows
        and table_width >= _score_fused.shape[1]
        and hasattr(torch.ops.xspeedgate_ops, "topk_transform")
    )
    _key = (bool(_fused_ok), rows == src_page_table.shape[0], table_width >= width)
    if _key not in _PAGED_TOPK_PATH_SEEN:
        _PAGED_TOPK_PATH_SEEN.add(_key)
        logger.warning(
            "ERNIE55 paged topk path=%s rows=%s width=%s table_width=%s topk=%s lengths_dtype=%s",
            "xspeedgate_fused" if _fused_ok else "torch",
            rows, width, table_width, topk, lengths.dtype,
        )
    if _fused_ok:
        dst_page_table = score.new_full((rows, topk), -1, dtype=torch.int32)
        _sf = _score_fused if _score_fused.is_contiguous() else _score_fused.contiguous()
        try:
            torch.ops.xspeedgate_ops.topk_transform(
                _sf,
                lengths.clamp(max=_sf.shape[1]).to(torch.int32),
                src_page_table,
                dst_page_table,
                topk,
                None,
                1,      # page_size: this path's page table is already page-1 cache slots
                True,   # fast_path
            )
        except Exception as exc:  # shape not accepted: fall back to torch for good
            _mark_fused_disabled("topk_transform raised: " + repr(exc)[:160])
            return _paged_topk_transform(score, lengths, src_page_table, topk, cu_seqlens_q)
        # Never under capture: the reference path's kernels would be recorded into the
        # graph and replayed every step. The int64 probe caught exactly that -- a
        # [rows, context] masked_fill + topk + sort baked into one decode bucket.
        if not _PAGED_TOPK_SELFCHECK_DONE and not _capturing():
            _selfcheck_fused(dst_page_table, score, lengths, src_page_table, topk, cu_seqlens_q)
        return dst_page_table

    # KUNLUN_PERF: the torch path used to build an int64 [rows, width] mask and gather
    # through it; aten::lt alone cost 512 ms / 20 steps (17% of device time). Compare in
    # int32 and derive selected_valid on [rows, k] directly (k=2048 vs width~32k).
    # Element-wise equivalent on the selected set. Disable with ERNIE55_INT32_TOPK_MASK=0.
    if os.environ.get("ERNIE55_INT32_TOPK_MASK", "1") != "0":
        positions = torch.arange(width, device=device, dtype=torch.int32)
        lengths_i32 = lengths.to(torch.int32)
        valid = positions.unsqueeze(0) < lengths_i32.unsqueeze(1)
        k = min(topk, width)
        selected = score.masked_fill(~valid, float("-inf")).topk(k, dim=1).indices
        selected, _ = selected.sort(dim=1)
        selected_valid = selected.to(torch.int32) < lengths_i32.unsqueeze(1)
    else:
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
        """Materialize the host mirrors used by the Kunlun kernels.

        Build them through the coalesced path as well, so a mirror's storage is
        decided once here and never moves: the graph records these host pointers,
        and swapping them on the first refresh would leave it reading a freed
        buffer.
        """
        for host_name, _dev_name in self._MIRRORED:
            object.__setattr__(self, host_name, None)
        if self._refresh_coalesced():
            return
        for host_name, dev_name in self._MIRRORED:
            dev = getattr(self, dev_name, None)
            object.__setattr__(self, host_name, None if dev is None else dev.cpu())

    def _refresh_coalesced(self) -> bool:
        """One D2H for every mirror instead of one per tensor.

        Seven small transfers cost seven times the fixed latency: measured 221 us
        each, 1.55 ms/step out of graph, for a few hundred bytes of payload. Stage
        the device tensors into one buffer, copy once, and expose each mirror as a
        view of the same host buffer so the pointers the graph recorded stay put.
        Returns False (caller falls back to per-tensor copies) when the mirrors do
        not share a dtype or the layout changed.
        """
        if os.environ.get("ERNIE55_COALESCE_MIRRORS", "1") == "0":
            return False
        plan, total, dtype = [], 0, None
        for host_name, dev_name in self._MIRRORED:
            dev = getattr(self, dev_name, None)
            if dev is None:
                continue
            if dtype is None:
                dtype = dev.dtype
            elif dev.dtype != dtype:
                return False
            if not dev.is_contiguous():
                return False
            plan.append((host_name, dev, total, dev.numel(), tuple(dev.shape)))
            total += dev.numel()
        if not plan:
            return True
        key = (dtype, total, tuple((p[0], p[2], p[4]) for p in plan))
        if getattr(self, "_mirror_key", None) != key:
            dev0 = plan[0][1]
            object.__setattr__(
                self, "_mirror_stage", torch.empty(total, dtype=dtype, device=dev0.device)
            )
            try:
                host_flat = torch.empty(total, dtype=dtype, pin_memory=True)
            except Exception:  # noqa: BLE001  # pinned host memory is optional
                host_flat = torch.empty(total, dtype=dtype)
            object.__setattr__(self, "_mirror_host", host_flat)
            object.__setattr__(self, "_mirror_key", key)
            for host_name, _dev, off, numel, shape in plan:
                object.__setattr__(
                    self, host_name, self._mirror_host[off : off + numel].view(shape)
                )
        stage = self._mirror_stage
        for _host_name, dev, off, numel, _shape in plan:
            stage[off : off + numel].copy_(dev.reshape(-1))
        self._mirror_host.copy_(stage)
        global _MIRROR_COALESCE_LOGGED
        if not _MIRROR_COALESCE_LOGGED:
            _MIRROR_COALESCE_LOGGED = True
            # setproctitle wipes /proc/<pid>/environ for the schedulers, so the only
            # reliable readback of this path is a line in the server log.
            print(
                "[ERNIE55] dsa host mirrors coalesced: %d tensors -> 1 D2H (%d elems %s)"
                % (len(plan), total, dtype),
                flush=True,
            )
        return True

    def refresh_host_mirrors(self) -> None:
        """Update the host mirrors *in place*, keeping their storage stable.

        On the cuda-graph path this object is reused across replays and the
        Kunlun kernels were recorded against these host buffers, so a fresh
        ``.cpu()`` tensor per replay would leave the graph reading capture-time
        lengths. Copy into the existing storage instead, and only reallocate when
        the shape/dtype actually changes.
        """
        if self._refresh_coalesced():
            return
        for host_name, dev_name in self._MIRRORED:
            dev = getattr(self, dev_name, None)
            if dev is None:
                continue
            host = getattr(self, host_name, None)
            if host is None or host.shape != dev.shape or host.dtype != dev.dtype:
                object.__setattr__(self, host_name, dev.cpu())
            else:
                host.copy_(dev)


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
        # 0.5.22 appended `attn_sink` (per-head learnable sinks). The Kunlun
        # sparse kernel has no sink input, but the sink is a softmax-denominator
        # term we can fold in post-hoc from the kernel's natural-log `lse`
        # (see end of this method). MHA_ONE_SHOT has no such lse here, so only
        # that path still refuses a sink.
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
            if attn_sink is not None:
                raise NotImplementedError(
                    "KunlunDSAAttnBackend: attn_sink not supported on the "
                    "MHA_ONE_SHOT path"
                )
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

        # sparse_prefill_fwd_opt takes explicit q/kv lods and needs preallocated
        # output buffers.
        # out must match q's dtype (the kernel requires out.dtype == q.dtype;
        # registrations cover both bf16 and fp16). Hardcoding bf16 breaks the
        # fp16-latent path, so follow q_all.dtype.
        o_ = torch.zeros(
            [q_all.shape[0], layer.tp_q_head_num, layer.v_head_dim],
            dtype=q_all.dtype,
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

        if _cp_rowwise_split(forward_batch):
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

        if attn_sink is not None:
            # Fold the learnable per-head sink (a virtual value-0 key, natural-log
            # bias) into the output: o *= Z/(Z+exp(sink-m)) = sigmoid(lse - sink),
            # with lse = m + ln(Z) the kernel's natural-log normalizer. Matches the
            # bmu reference (torch.logaddexp(lse, sink)). o_:[rows,head,vdim],
            # lse:[rows,head], sink:[head].
            sink = attn_sink.to(lse.dtype).reshape(-1)
            o_ = o_ * torch.sigmoid(lse - sink).unsqueeze(-1)
            o_ = torch.nan_to_num(o_, nan=0.0)

        del max_logits
        del lse
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
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode through the Kunlun sparse kernel."""
        # Same 0.5.22 `attn_sink` tail as forward_extend; the Kunlun sparse
        # decode kernel has no sink input either.
        # See forward_extend: fold the per-head sink in post-hoc from the
        # decode kernel's max_logits + p_sums (lse = max_logits + ln(p_sums)).
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

        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # fwd_kvcache_mla wants a [tokens, 1, heads, head_dim] q and preallocated
        # output buffers.
        reshape_q = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        bs = forward_batch.batch_size
        o_ = torch.zeros(
            [bs, reshape_q.shape[1], layer.tp_q_head_num, layer.v_head_dim],
            dtype=q_all.dtype,
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
        if attn_sink is not None:
            # lse = max_logits + ln(p_sums); o *= sigmoid(lse - sink).
            # o_:[bs,qlen,head,vdim], max_logits/p_sums:[bs,qlen,head], sink:[head].
            lse = max_logits + torch.log(p_sums.clamp_min(1e-20))
            sink = attn_sink.to(lse.dtype).reshape(-1)
            o_ = o_ * torch.sigmoid(lse - sink).unsqueeze(-1)
            o_ = torch.nan_to_num(o_, nan=0.0)

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
            dtype=q.dtype,
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
