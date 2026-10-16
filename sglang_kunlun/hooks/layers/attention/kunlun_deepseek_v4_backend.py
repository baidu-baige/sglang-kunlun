"""Kunlun DeepSeek-V4 compressed-attention backend.

The upstream 0.5.14 backend owns metadata preparation and graph lifecycle.  This
module only replaces the FlashMLA-specific metadata/forward boundary with the
Kunlun compressed-attention operator.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import logging
import os
import cocopod  # noqa: F401  # Registers DSV4 XSpeedGate attention operators.
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
    dsv4_c4_paged_mqa_logits_torch,
    dsv4_compressed_attention_torch,
    dsv4_quant_k_cache_kunlun,
    dsv4_set_k_and_s_with_mapping_kunlun,
)

_DSV4_TENSOR_DUMP_CALLBACK = None


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


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend._create_flashmla_metadata",
    type=HookType.REPLACE,
)
@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix."
    "_create_flashmla_metadata",
    type=HookType.REPLACE,
)
def _create_flashmla_metadata_kunlun():
    """Disable CUDA FlashMLA metadata on Kunlun."""

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
    """Replace the unavailable sgl_kernel FlashMLA callable on Kunlun."""
    if args:
        raise TypeError("Kunlun FlashMLA replacement requires keyword arguments")
    return _flash_mla_with_kvcache_reference(**kwargs)


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.attn.get_paged_mqa_logits_metadata",
    type=HookType.REPLACE,
)
def _get_paged_mqa_logits_metadata_kunlun(seq_lens, page_size, num_sm):
    """Kunlun: bypass sglang JIT compile of paged_mqa_metadata.cuh.

    The upstream JIT kernel requires C++20 ``std::bit_cast`` (needs gcc-11+
    libstdc++), which the current build env (gcc-10) cannot provide. The
    downstream Kunlun path (``_compute_c4_logits_kunlun`` via XSpeedGate ops)
    does not consume ``deep_gemm_metadata``, so an empty placeholder is safe.
    """
    return torch.empty(0, dtype=torch.int32, device=seq_lens.device)


class KunlunDSV4AttnMetadata(DSV4AttnMetadata):
    """DSV4 metadata without CUDA FlashMLA scheduler objects."""

    def init_flashmla_related(self, is_prefill: bool = False) -> None:
        """Initialize Kunlun sparse metadata without FlashMLA schedulers.

        Mirrors the upstream ratio split. V4-Flash stages carry ratio 4 (and
        possibly 128); the V4.1 checkpoint carries only ratios 0/1/2, so the c4
        buffers must stay unset there and the c1/c2 buffers must be allocated --
        otherwise the low-ratio path silently degrades to window-only attention
        (and an absent c4 pool crashes this method outright).
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
            setattr(self, f"c{ratio}_sparse_raw_indices", None)
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
    """eager 的 TARGET_VERIFY 必须用 per-row LoD，和 sglang 0.5.8 一致。

    ``xspeedgate_ops.compressed_attention`` 拿不到 ``win_lengths`` /
    ``extra_lengths``，它唯一的 per-row 掩码来自 ``kv_lens`` 加 LoD 布局。把
    gamma+1 个 query 行当成一个 item、共用一个 request 级 ``kv_len``，每一行就都能
    读到整个窗口——包括后面几个 verify 位置上的草稿 token，于是 target 直接抄了它
    本该校验的草稿。

    0.5.8 的昆仑后端（该算子已验证可用）走的是 decode 型契约：
    ``qlod = arange(n+1)``、``kvseqlen = seq_lens_casual``（每个 verify 位置的精确
    因果长度 L+j+1）。``_make_cp_prefill_lod`` 正好构造这个，复用它。

    capture 期走的是 graph 专用实现（见 ``_make_verify_row_lod_graph_safe``），因为
    这里的构造需要 H2D，录不进图。``DSV4_VERIFY_ROW_LOD=0`` 可 A/B 回旧行为。
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
    # CP-V1 废弃把 round-robin 改名成 interleave
    # （`can_dsa_prefill_cp_round_robin_split` → `can_dsa_prefill_cp_interleave`），
    # 判据和 token→rank 映射都没变。
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
            cached_bs, _, _, rows_per_req = key
            if rows_per_req != 1:
                # TARGET_VERIFY per-row entry: its rows are draft rows, not
                # requests, and _compute_c4_logits_kunlun refreshes it inline.
                # Overwriting it here feeds request-shaped lengths to a
                # row-shaped buffer on the next replay.
                continue
            if cached_bs < batch_size:
                continue
            _copy_host_lengths_(aux[2], c4_context_lens, fill_value=4)
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
_DSV4_PAGE_TABLE_SPAN_LOGGED = False


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
    if torch.cuda.is_current_stream_capturing():
        # The span is a Python slice bound, so the real maximum needs a host
        # sync.  That cannot run inside capture -- and during capture the tensor
        # holds uninitialised memory, so the sync sized the page table from
        # garbage (spans around 8e8 were observed) and the attention then read
        # past it: "load/store operation exceed memory size".  Take the static
        # upper bound instead: verify rows reach committed + num_draft_tokens.
        # ``scores`` is the full compressed logits width and the page table is
        # gathered with ``raw_indices >> page_bits``, so the span has to cover
        # the logits width, not just the causal length.  One page of slack keeps
        # the last page fully described.
        return max_seq_len + _draft_token_count() + _page_slack()
    if seq_lens_casual.numel() == 0:
        return max_seq_len
    needed = int(seq_lens_casual.max().item())
    if needed <= max_seq_len:
        return max_seq_len
    global _DSV4_PAGE_TABLE_SPAN_LOGGED
    if not _DSV4_PAGE_TABLE_SPAN_LOGGED:
        _DSV4_PAGE_TABLE_SPAN_LOGGED = True
        logger.warning(
            "[DSV4_PAGE_TABLE] widening span: max_seq_len=%s needed=%s",
            max_seq_len,
            needed,
        )
    return needed


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


class KunlunDeepseekV4AttnBackend(DeepseekV4AttnBackend):
    """0.5.14 DSV4 control flow with the Kunlun attention boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Kunlun metadata builder was written against the 4.1 field names.
        # 0.5.19's upstream backend still exposes the same values as
        # ``c4_topk`` and does not publish ratio tuples on the backend object.
        # Normalize the compatibility fields once at construction time.
        self.index_topk = getattr(self, "index_topk", self.c4_topk)
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

    _WINDOW_WARNED = False

    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
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
        * `window` is static -- `DSV4_LOW_RATIO_INDEX_WINDOW` (default 32768) capped
          by the capture's max sequence length -- so no shape depends on data. For a
          compressed length that fits the window this is identical to the reference;
          beyond it the window is a documented approximation.
        * the window is chunked and a running top-k is kept across chunks, so the
          transient tensors stay bounded while the result is still the exact top-k
          over the whole window.

        Candidate publish/consume is row-indexed: within one forward the rows are
        the same for every layer, so a `[rows, window]` mask carries exactly the
        request-keyed masks the reference built.
        """
        ratio = layer.compress_ratio
        indexer = layer.indexer
        core = self.forward_metadata.core_attn_metadata
        pool = self.token_to_kv_pool
        rows = x.shape[0]
        if rows == 0:
            return
        topk = indexer.index_topk
        cap = int(os.environ.get("DSV4_LOW_RATIO_INDEX_WINDOW", "32768"))
        window = min(self.MAX_SEQ_LEN_FOR_CAPTURE // ratio + 1, cap)
        k = min(topk, window)

        page_indices = core.sparse_page_indices(ratio)
        page_indices.fill_(-1)

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [R, H, index_head_dim]
        head_w = indexer.head_weights(x)  # [R, H]
        visible = ((pos + 1) // ratio).to(torch.int64)  # compressed length per row
        req_i = req.to(torch.int64).reshape(-1)
        device = x.device
        neg = torch.finfo(torch.float32).min
        req_width = self.req_to_token.shape[1] - 1

        publish = bool(getattr(indexer, "is_candidate_source", False))
        consume = None
        if getattr(indexer, "uses_candidates", False) and isinstance(
            self.candidate_masks, dict
        ):
            consume = self.candidate_masks.get("rows")
        published = (
            torch.zeros((rows, window), dtype=torch.bool, device=device)
            if publish
            else None
        )

        if not KunlunDeepseekV4AttnBackend._WINDOW_WARNED and bool(
            (visible > window).any()
        ):
            KunlunDeepseekV4AttnBackend._WINDOW_WARNED = True
            logger.warning(
                "low-ratio indexer: visible compressed length exceeds the static "
                "window %d; candidates beyond it are not scored. Raise "
                "DSV4_LOW_RATIO_INDEX_WINDOW for exact results.",
                window,
            )

        best_val = torch.full((rows, k), neg, dtype=torch.float32, device=device)
        # Unfilled slots sort last: the attention op reads a *leading* prefix of
        # `sparse_topk_lengths` entries, so the valid positions must come first
        # (the reference sorts too). A `window` sentinel is greater than every real
        # position and is filtered by the `reach` mask below.
        best_j = torch.full((rows, k), window, dtype=torch.int64, device=device)

        from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

        jc = int(os.environ.get("DSV4_LOW_RATIO_INDEX_CHUNK", "2048"))
        rc = int(os.environ.get("DSV4_LOW_RATIO_INDEX_ROWS", "64"))
        for j0 in range(0, window, jc):
            j = torch.arange(j0, j0 + jc, device=device)  # static width
            tok_pos = (j * ratio).clamp_max(req_width)
            valid = j.unsqueeze(0) < visible.unsqueeze(1)
            for r0 in range(0, rows, rc):
                r1 = min(r0 + rc, rows)
                slots = self.req_to_token[req_i[r0:r1, None], tok_pos[None, :]] // ratio
                v = valid[r0:r1]
                index_k = pool.get_low_ratio_index_k_dequant(
                    layer.layer_id, slots.reshape(-1)
                ).reshape(r1 - r0, jc, -1)
                # `DeepseekV41Indexer.scores` with a per-row K: bf16 up to the
                # reduction, then relu, head-weight and head sum in fp32.
                dots = (q[r0:r1].unsqueeze(2) * index_k.unsqueeze(1)).sum(dim=-1)
                s = (dots.relu() * head_w[r0:r1].unsqueeze(-1)).sum(dim=1).float()
                s = s.masked_fill(~v, neg)
                if consume is not None:
                    s = s.masked_fill(~consume[r0:r1, j0 : j0 + jc], neg)
                if published is not None:
                    published[r0:r1, j0 : j0 + jc] = select_candidate_blocks(
                        s,
                        visible[r0:r1, None],
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    )
                cand_val, cand_j = s.topk(k, dim=-1, sorted=False)
                merged_val = torch.cat([best_val[r0:r1], cand_val], dim=1)
                merged_j = torch.cat([best_j[r0:r1], j0 + cand_j], dim=1)
                top = merged_val.topk(k, dim=-1, sorted=False)
                best_val[r0:r1] = top.values
                best_j[r0:r1] = merged_j.gather(1, top.indices)

        # Sort the (value, position) pairs by position so valid entries form the
        # leading prefix the attention op expects; `best_val` must follow.
        order = best_j.argsort(dim=-1)
        best_j = best_j.gather(1, order)
        best_val = best_val.gather(1, order)
        reach = (best_j < visible.unsqueeze(1)) & (best_val > neg)
        sel_pos = (best_j * ratio).clamp_max(req_width)
        sel_slots = self.req_to_token[req_i[:, None], sel_pos] // ratio
        page_indices[:, :k] = torch.where(
            reach, sel_slots, torch.full_like(sel_slots, -1)
        ).to(torch.int32)
        if published is not None:
            self.candidate_masks = {"rows": published}

    def get_swa_page_indices(
        self, seq_lens_casual, req_pool_indices_repeated, swa_replay_start=None
    ):
        """Translate request token offsets into Kunlun SWA page indices.

        ``swa_replay_start`` (4.1) is the per-row window floor used by decoder
        SWA bounded replay: offsets below it are never written at the tail
        layers, so the window is clamped at the floor instead of at 0.
        """
        # Match the contract: invalid history offsets are clamped to the
        # first physical row and remain valid indices; the length tensor masks
        # those rows. Upstream 0.5.14 writes -1 here, which changes graph replay
        # inputs and is not accepted by the Kunlun compressed-attention path.
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
        # 上游 ``_causal_swa_page_indices_kernel`` 对每次 load 都做掩码，超出该行因果
        # 长度的 lane 不会真的访存；这里是稠密改写，会读所有 lane，所以映射查表要手动
        # clamp——越界 lane 由 ``swa_topk_lengths`` 掩掉。
        mapping = self.token_to_kv_pool.full_to_swa_index_mapping
        mapped = mapping[
            raw_indices.clamp(min=0, max=mapping.shape[0] - 1)
        ].to(torch.int32)
        return mapped

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        """Initialize out-of-graph metadata and refresh replay lengths."""
        if in_capture and _is_graph_extend_mode(forward_batch.forward_mode):
            _get_graph_extend_aux(
                self._attention_graph_extend_aux,
                forward_batch.batch_size,
                forward_batch.positions.numel(),
                forward_batch.seq_lens.device,
            )
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        _refresh_graph_host_lengths(
            forward_batch,
            self._attention_decode_aux,
            self._c4_decode_aux,
            self._attention_graph_extend_aux,
        )

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

        c4_seq_lens = match_num_queries(indexer_metadata.c4_seq_lens, value=1)
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
            topk_transform = getattr(
                upstream_indexer,
                "topk_transform_pytorch_vectorized",
                upstream_indexer.topk_transform_512_pytorch_vectorized,
            )
            topk_transform(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform = getattr(upstream_indexer, "topk_transform_paged_v2", None)
            if topk_transform is None:
                upstream_indexer.topk_transform_512_pytorch_vectorized(
                    logits,
                    c4_seq_lens,
                    page_table,
                    c4_sparse_page_indices,
                    indexer_metadata.c4_page_size,
                    raw_indices,
                )
            else:
                topk_transform(
                    logits,
                    c4_seq_lens,
                    page_table,
                    c4_sparse_page_indices,
                    indexer_metadata.c4_page_size,
                    indexer_metadata.topk_metadata,
                )
        else:
            topk_transform = getattr(
                upstream_indexer,
                "topk_transform_paged",
                upstream_indexer.topk_transform_512_pytorch_vectorized,
            )
            topk_transform(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
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
                swap_seq_lens = indexer_metadata.c4_seq_lens
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
    ):
        del fn, core_metadata
        return _compute_c4_logits_kunlun(
            backend=self,
            q_fp8=q,
            kvcache_fp8=c4_indexer_kv_cache,
            weight=weights,
            seq_lens=c4_seq_lens,
            page_table=page_table,
            max_seq_len=indexer_metadata.max_c4_seq_len,
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
        )

    def on_after_cuda_graph_warmup(self):
        """Skip FlashMLA metadata refresh, matching the Kunlun backend."""
        metadata = self.forward_metadata
        if isinstance(metadata, upstream.DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, upstream.DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c0_flashmla_metadata = None
            core.c1_flashmla_metadata = None
            core.c4_flashmla_metadata = None
            core.c128_flashmla_metadata = None

        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

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
        assert self.swa_page_size == upstream.SWA_WINDOW
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
        page_table = req_to_token[
            req_pool_indices_repeated,
            : _dsv4_page_table_span(max_seq_len, seq_lens_casual) : self.page_size,
        ]
        metadata = KunlunDSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=seq_lens_casual - 1,
            page_table=(page_table // self.page_size).to(torch.int32),
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=torch.clamp(seq_lens_casual, max=upstream.SWA_WINDOW),
            c4_sparse_topk=self.index_topk,
        )
        # 0.5.19 keeps the upstream dataclass constructor contract
        # (``c4_sparse_topk``); the 4.1 compatibility fields are consumed by
        # KunlunDSV4AttnMetadata.init_flashmla_related().
        metadata.index_topk = self.index_topk
        metadata.present_ratios = self.present_ratios
        metadata.low_ratios = self.low_ratios
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
        """DSpark draft block 前向的 core metadata（只有窗口，没有压缩分支）。

        ``init_forward_metadata_dspark_draft_block`` 以 ``need_compress=False``
        走 prefill，draft 不碰 c4/c128，只用滑动窗口。上游
        ``get_dspark_swa_page_indices`` 已经为 gamma 个 draft 行构造了 per-row 因果
        窗口索引（相关 kernel 有昆仑替换实现），昆仑侧唯一要改的是无效 lane 的约定：
        填 page 0 而不是 -1，尾部由 ``swa_topk_lengths`` 掩掉。
        """
        assert self.swa_page_size == upstream.SWA_WINDOW
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
        page_table = req_to_token[
            req_pool_indices_repeated,
            : _dsv4_page_table_span(max_seq_len, seq_lens_casual) : self.page_size,
        ]
        metadata = KunlunDSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=seq_lens_casual - 1,
            page_table=(page_table // self.page_size).to(torch.int32),
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.index_topk,
        )
        metadata.index_topk = self.index_topk
        metadata.present_ratios = self.present_ratios
        metadata.low_ratios = self.low_ratios
        metadata.swa_topk_lengths = swa_topk_lengths
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
        """capture 安全的 per-row TARGET_VERIFY LoD。

        ``_make_cp_prefill_lod`` 表达的是同一个契约，但要走 D2H 再新建 CPU 张量，录不
        进图（replay 时录下的 H2D 会读到已释放的 host 缓冲）。所以 capture 期原先退回
        request-level LoD——正是让 verify 行 j 读到行 j+1.. 草稿 KV 的那个布局（见
        ``_verify_row_lod_enabled``），这就是 CUDA graph 下退化成单 token 死循环、
        accept len 钉在 gamma+1 而 eager 干净的原因。

        这里 ``q_lod`` 是常量 ``arange``（可以烘死），``kv_lens`` 由录进图的设备端 op
        从 ``seq_lens_casual`` 填充，而该缓冲每次 replay 前由 ``copy_`` 原地刷新；
        ``kv_lens_cpu`` 保持占位符，与 request-level capture 路径原本的假设一致。
        ``DSV4_VERIFY_ROW_LOD_GRAPH=0`` 可 A/B 回旧行为。
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
                # ``kv_lens_cpu`` 原先一直保持 ``ones`` 初值（只有设备端被回填），于是
                # ``compressed_topk = max(kv_lens_cpu.max() // ratio, 1)`` 恒为 1：
                # 压缩分支只读到 1 条 entry，而 257 token 时 c4 应该是 64 条。prefill
                # 的 host 长度是真值所以不受影响，这也是与 CUDA 参考实现的分歧只从
                # decode 的第一个 c4 层（layer 2）开始的原因。
                _copy_host_lengths_(
                    kv_lens_cpu, _seq_lens_cpu_i32(forward_batch), fill_value=1
                )
            return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

        if _is_graph_extend_mode(forward_batch.forward_mode) and not getattr(
            forward_batch, "_kunlun_ragged_draft_extend", False
        ):
            if _verify_row_lod_enabled(forward_batch):
                return _make_cp_prefill_lod(
                    self.forward_metadata.core_attn_metadata, num_queries, device
                )
            if (
                forward_batch.forward_mode.is_target_verify()
                and os.environ.get("DSV4_VERIFY_ROW_LOD_GRAPH", "1") == "1"
            ):
                return self._make_verify_row_lod_graph_safe(
                    self.forward_metadata.core_attn_metadata, num_queries, device
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
                self.forward_metadata.core_attn_metadata, num_queries, device
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
        q_lod = q_lod_cpu.to(device, non_blocking=False)
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
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)
        assert k is v, "DeepseekV4 shares k and v"
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)

        core = self.forward_metadata.core_attn_metadata
        pool = self.token_to_kv_pool
        cache_dim = pool.swa_kv_pool.kv_cache_total_dim
        # `swa_window_size` 在新版 `DeepSeekV4TokenToKVPool` 里被删了，它一直只是
        # `swa_page_size`（SWA ring 的每请求宽度）的别名。
        swa_size = pool.swa_page_size
        win_cache = pool.get_swa_key_buffer_radix(layer.layer_id)
        win_cache = win_cache.reshape(-1, cache_dim)

        win_indices = self._match_queries(core.swa_page_indices, q.shape[0], 0)
        win_indices = win_indices[:, :swa_size].contiguous()
        win_lengths = self._match_queries(core.swa_topk_lengths, q.shape[0], 0)
        if self.is_dspark_draft:
            # draft 窗口构造器把 ``swa_topk_lengths`` 之后的 lane 标成 -1，而
            # ``compressed_attention`` 没有 window-length 参数。所以 -1 不能简单
            # clamp 成 page 0——算子会把它当成真 KV 参与 softmax（实测 192 个 lane 里
            # 有 59 个无效，约 31% 权重落在无关 token 上）。它的窗口循环上界是
            # ``min(kv_len, max_window_size)``，因此掩掉尾部的办法是把 kv_len 夹到
            # 窗口长度，见下面的 clamp。
            if (
                win_lengths.numel()
                and os.environ.get("DSV4_KUNLUN_DRAFT_WINDOW_COMPACT", "0") == "1"
            ):
                # DSV4_KUNLUN_DRAFT_WINDOW_KVLEN=0 时的旧退路：压缩 lane 维度让
                # ``max_window_size`` 自己成为掩码。但 ``valid`` 是设备端取值、
                # 结果 shape 依赖数据，录不进 CUDA graph。
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
            # ``compressed_attention`` 的窗口循环上界是 ``min(kv_len,
            # max_window_size)``，而 draft 的 kv_len 是整个序列长度，远超
            # ``swa_topk_lengths``，无效 lane 会全被 attend。把 kv_len 夹到窗口长度
            # 等价于 lane 压缩，但 shape 不依赖数据，能被 capture。draft 各行共用同一
            # 个窗口长度，所以这里取 ``max()`` 是精确的。
            # 只夹设备端张量：``kv_lens_cpu`` 在 capture 期是占位符。实测同时夹它
            # 产生逐字节相同的 836 token 输出。
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
                    # ``compressed_topk`` 是作用于所有 query 行的单个标量，原先取自
                    # 最长的那一行。TARGET_VERIFY 里最长的是最后一个草稿位置，于是每
                    # 一行（包括只有它是已提交的第 0 行）都被允许读到覆盖草稿块的那条
                    # 压缩项，而这条项就是本次前向里发布的。改成按最短行（即已提交
                    # 前缀）来定上界：0..K-1 项只覆盖 < K*ratio <= 前缀的位置，任何行
                    # 都到不了未提交位置。信息没丢——被排除的 token 本来就在精确 SWA
                    # 窗口（>=128）里。
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
        q_op = q_3d.contiguous()
        win_cache_op = win_cache.contiguous()
        win_indices_op = win_indices.contiguous()
        extra_cache_op = extra_cache.contiguous()
        extra_indices_op = extra_indices.contiguous()
        out_op = out.contiguous()
        max_logits_op = max_logits.contiguous()
        lse_op = lse.contiguous()
        q_lod_cpu_op = q_lod_cpu.contiguous()
        q_lod_op = q_lod.contiguous()
        kv_lens_cpu_op = kv_lens_cpu.contiguous()
        kv_lens_op = kv_lens.contiguous()
        attn_sink_op = local_sink.contiguous() if local_sink is not None else None
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
    forward_batch, c4_seq_lens, page_table, device, num_queries
):
    """Build the request-level LoD contract used by the extend path."""
    global_extend_lens = [int(value) for value in forward_batch.extend_seq_lens_cpu]
    cp_ranks = _dsa_cp_prefill_ranks(forward_batch)

    if cp_ranks is None:
        # c4_flat is only used here; keeping it off the CP path avoids a dead per-layer D2H sync.
        c4_flat = c4_seq_lens.reshape(-1).to("cpu", dtype=torch.int32)
        extend_lens = global_extend_lens
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


_C4_VERIFY_CHUNK_LOGGED = False


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
    global _C4_VERIFY_CHUNK_LOGGED
    if not _C4_VERIFY_CHUNK_LOGGED:
        _C4_VERIFY_CHUNK_LOGGED = True
        logger.warning(
            "[DSV4_C4_VERIFY] per-row path active: num_queries=%s num_requests=%s "
            "page_rows=%s",
            num_queries,
            num_requests,
            page_table.shape[0],
        )
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
):
    """Compute C4 logits with explicit Kunlun request ownership."""

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
            forward_batch, seq_lens, page_table, q.device, q.shape[0]
        )
        if (
            contract["cp_ranks"] is None
            and q.shape[1:] == (64, 128)
            and weights.shape[1] == 64
            and kvcache_fp8.shape[1:] == (64, 1, 132)
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
    weights = weight.unsqueeze(1)

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
    block_table = page_table[:, :num_pages].to(torch.int32)
    scale_pages = (
        cache_flat[:, scale_offset:]
        .contiguous()
        .view(torch.float32)
        .view(-1, block_size)
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
    raw_seq_lens_cpu = _seq_lens_cpu_i32(forward_batch)
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


# ── CP 预填索引器：dense fp4 logits 仅 deep_gemm 有，改走上游的纯 torch 实现 ──
#
# 现象：P 开 CP 后，**真实请求**进来时 8 个 rank 全挂
#     ImportError: cannot import name 'fp8_fp4_mqa_logits' from 'deep_gemm'
# 链路：
#     forward_low_ratio_sources(2832) → _forward_low_ratio_sources_cp(2902)
#       → _low_ratio_index_topk_dense(3313) → _dense_fp4_mqa_logits(339)
#         → from deep_gemm import fp8_fp4_mqa_logits
#
# 上游本来有分派（`_low_ratio_index_topk`，3200-3233）：
#     elif self._use_dense_fp4_prefill_indexer(fb) and _is_sm100_or_newer():
#         self._low_ratio_index_topk_extend(...)   # dense fp4（deep_gemm）
#     else:
#         self._low_ratio_index_topk_torch(...)    # 纯 torch 回退
# 其中 `_use_dense_fp4_prefill_indexer` 读的就是 `SGLANG_DSV41_TORCH_PREFILL_INDEXER`
# （脚本里设成 1）。**但 CP 分支绕过了这个分派**，直接调 `_low_ratio_index_topk_dense`
# ⇒ 开关对 CP 不生效，请求期必然撞 deep_gemm。
#
# 为什么 1-token 的 warmup 没炸：插件里 `_cp_short_prompt_needs_reference` 把退化批次
# 送去了 Torch 参考路径；只有真实 prompt 才走到这里。
#
# 做法：AROUND 挂 `_low_ratio_index_topk_dense`，当「开关要求 torch」或「deep_gemm
# 取不到 fp8_fp4_mqa_logits」时，重定向到上游自带的 `_low_ratio_index_topk_torch`。
# **不复制上游任何方法**，直接复用那份纯 torch 实现 —— 它自带分块
# （`_TORCH_INDEXER_SCORE_BUDGET_BYTES` 控制 `[rows, heads, lc]` 中间张量，就是为
# 16k prompt 控显存加的），语义上也已按请求分组处理。
#
# `req` 怎么构造：dense 版把 `q_lens`（tensor）与 `q_lens_cpu`（list）作为参数收进来了，
# 它们是本 rank **每个请求的本地行数**（CP 下每 rank 只算自己的行），
# 据此 `repeat_interleave` 可精确重建每行的请求下标；torch 版正是按
# `torch.unique_consecutive(req)` 分请求遍历的，而本 rank 的行天然按请求连续排列。
#
# 开关沿用 `SGLANG_DSV41_TORCH_PREFILL_INDEXER`；此外 deep_gemm 不可用时也会自动
# 走这条路，免得忘记设开关就炸。


def _dense_fp4_indexer_usable() -> bool:
    """dense fp4 索引器是否可用（依赖 deep_gemm 的 fp8_fp4_mqa_logits）。"""
    if envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get():
        return False
    try:
        from deep_gemm import fp8_fp4_mqa_logits  # noqa: F401
    except Exception:  # noqa: BLE001 - 任何导入失败都说明不可用
        return False
    return True
