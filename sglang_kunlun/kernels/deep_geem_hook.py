"""Kunlun overrides for deep_gemm helper symbols."""

from __future__ import annotations

import importlib
import logging
import sys
import types

import torch

from sglang_kunlun.kernels.kernel_ops import register_jit_op

logger = logging.getLogger(__name__)
_WARNED_PAGED_MQA_METADATA = False


def _get_or_create_deep_gemm_module():
    try:
        return importlib.import_module("deep_gemm")
    except ImportError:
        deep_gemm = types.ModuleType("deep_gemm")
        sys.modules["deep_gemm"] = deep_gemm
        return deep_gemm


def _register_deep_gemm_stub(symbol_name: str):
    def decorator(fn):
        deep_gemm = _get_or_create_deep_gemm_module()
        if not hasattr(deep_gemm, symbol_name):
            setattr(deep_gemm, symbol_name, fn)
        return register_jit_op("deep_gemm", symbol_name)(fn)

    return decorator


class _NoopPagedMqaSchedule:
    """Placeholder for deep_gemm's paged MQA schedule metadata.

    The Kunlun indexer does not consume the schedule metadata, but the upstream
    DSA backend still computes it and, during cuda graph replay, calls ``.copy_()``
    on the captured value. Returning this no-op object keeps the upstream code
    paths working unchanged: it must not be ``None``, otherwise replay would try to
    assign to a frozen dataclass field.
    """

    def copy_(self, other):
        """No-op: nothing to refresh on Kunlun."""
        return self


@_register_deep_gemm_stub("get_num_sms")
def get_num_sms() -> int:
    """Return the Kunlun SM count used by NSA indexer scheduling."""

    return 32


@_register_deep_gemm_stub("get_paged_mqa_logits_metadata")
def get_paged_mqa_logits_metadata(seqlens=None, *args, **kwargs):
    """Return the no-op paged MQA schedule the Kunlun indexer ignores."""

    global _WARNED_PAGED_MQA_METADATA
    if not _WARNED_PAGED_MQA_METADATA:
        logger.warning(
            "deep_gemm.get_paged_mqa_logits_metadata is not supported on Kunlun; "
            "returning a no-op schedule placeholder"
        )
        _WARNED_PAGED_MQA_METADATA = True
    # Must be the no-op object, not an empty tensor: cuda-graph replay calls
    # .copy_() on the captured value (see _NoopPagedMqaSchedule docstring).
    return _NoopPagedMqaSchedule()


@_register_deep_gemm_stub("fp8_paged_mqa_logits")
def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata=None,
    max_seq_len=None,
    clean_logits: bool = False,
):
    """Kunlun paged MQA logits via ``kunlun_ops.I8_paged_mqa_logits``.

    P800 has no fp8, so the whole indexer path is int8 + fp32 scale (see
    ``hooks/layers/attention/nsa/kunlun_dsa_indexer.py`` and the kpool writers in
    ``kernels/kpool_hook.py``). The index cache page layout is
    ``page_size * head_dim`` quantized bytes followed by the per-slot fp32 scales,
    which is what ``gather_paged_scale`` expects.

    This lives on the ``deep_gemm`` stub rather than on an indexer method so both
    ``dsa_indexer.py`` and ``dsa_indexer_kpool.py`` (a separate class, whose
    ``_get_topk_paged`` calls ``deep_gemm`` directly) are covered.
    """
    import torch

    import kunlun_ops

    # The authoritative batch is the query axis: ``_get_topk_paged`` slices q and
    # weights down to the real request count but leaves block_tables at its
    # allocated size (bs * draft_token_num on the verify path). The NVIDIA
    # deep_gemm op derives batch from q, gather_paged_scale from
    # block_tables.size(0) -- so trim the trailing padding rows here. Rows are
    # filled leading-prefix, so [:batch] are the right ones.
    batch = q.shape[0]
    assert block_tables.shape[0] >= batch, (block_tables.shape, q.shape)
    block_tables = block_tables[:batch].contiguous()

    num_pages = kv_cache.shape[0]
    head_dim_with_sf = kv_cache.shape[-1]
    head_dim = head_dim_with_sf - 4
    flat = kv_cache.reshape(num_pages, -1)
    page_size = flat.shape[1] // head_dim_with_sf
    if max_seq_len is None:
        max_seq_len = block_tables.shape[1] * page_size

    # 零拷贝：I8_paged_mqa_logits 支持 dim-0 带 stride 的 k_cache（内三维连续，见
    # kunlun_ops/_deepgemm.py 的 stride 断言），直接对交织 buffer 做 as_strided view，
    # 避免每个 DSA 层每步把整池做一次 uint8->int8 compaction 拷贝
    # （旧写法 ~0.5ms/call、x11 层 ~5.5ms/step；view 版 ~0.005ms）。
    # uint8.view(int8) 与旧的 .to(int8) bit 一致（test_kcast_strided.py 已验证）。
    k_i8 = flat.view(torch.int8).as_strided(
        (num_pages, page_size, 1, head_dim),
        (page_size * head_dim_with_sf, head_dim, head_dim, 1),
    )
    lens_xpu = context_lens.reshape(-1).to(torch.int32).contiguous()
    k_scale = torch.ops.xspeedgate_ops.gather_paged_scale(
        kv_cache=flat.view(torch.int8),  # the op only accepts Char, not Byte
        block_tables=block_tables,
        page_size=page_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        scale_offset=page_size * head_dim,
        context_lens=lens_xpu,
    )
    if q.dtype is torch.int8:
        # q carries no scale of its own; fold its 1/127 normalizer into k's.
        k_scale = k_scale / 127

    # kunlun_ops wants weights as [num_q, next_n, heads]; callers hand over
    # either [num_q, heads] or [num_q, heads, 1].
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.dim() == 2:
        weights = weights.unsqueeze(1)
    num_q, next_n = q.shape[0], q.shape[1]
    assert weights.shape[1] == next_n, (weights.shape, q.shape)
    logits = torch.empty(
        (num_q, next_n, max_seq_len), dtype=torch.float32, device=q.device
    )
    kunlun_ops.I8_paged_mqa_logits(
        q=q.contiguous(),
        fused_kv_cache=[k_i8, k_scale.contiguous()],
        weights=weights.contiguous(),
        context_lens=[lens_xpu.cpu(), lens_xpu],
        block_table=block_tables,
        max_context_len=max_seq_len,
        clean_logits=clean_logits,
        out=logits,
        use_xfa_boost=True,
    )
    return logits.reshape(-1, max_seq_len)


@_register_deep_gemm_stub("fp8_mqa_logits")
def fp8_mqa_logits(
    q,
    kv,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    clean_logits: bool = False,
):
    """Kunlun ragged MQA logits via ``kunlun_ops.I8_mqa_logits``.

    Upstream contract (NVIDIA deep_gemm): return ``[total_q, total_k]`` where row
    ``i`` is only valid on columns ``[cu_seqlen_ks[i], cu_seqlen_ke[i])`` - i.e.
    per-query-row offsets into one concatenated ragged K buffer, not per-sequence
    lengths. The DSA kpool indexer relies on that layout: it hands the result
    straight to ``_topk_from_kpool_logits(..., row_starts=ks)``.

    ``I8_mqa_logits`` addresses K by cu_seqlens instead, so the whole batch is
    driven as a single sequence (``[0, total_q]`` / ``[0, total_k]``) to get the
    full matrix, and the per-row window is then applied by
    ``xspeedgate_ops.mask_for_I8_mqa_logits`` - the same two-step the non-kpool
    indexer already uses (``hooks/layers/attention/nsa/kunlun_dsa_indexer.py``
    lines 623-633 and 496-501).
    Scale bookkeeping: P800 has no fp8, so q is int8 from the hooked
    ``act_quant`` and K is int8 from the kpool writers. The two use *different*
    scale conventions, measured rather than assumed:

    - ``kunlun_ops.quant2d`` (q) returns ``scale = absmax``, so the real value is
      ``q_int8 * scale / 127``. Upstream already folded that ``scale`` into
      ``weights`` via ``_get_logits_head_gate``, so only the ``/ 127`` is left to
      apply here.
    - the kpool writer ops store ``scale = absmax/127``
      already, so ``k_scale`` needs no correction.

    This lives on the ``deep_gemm`` stub because the kpool indexer class itself is
    not hooked - it calls ``deep_gemm`` directly from three ragged paths
    (``dsa_indexer_kpool.py`` lines 1008, 1122 and 1354).
    """
    import torch

    import kunlun_ops

    k, k_scale = kv
    if k.dtype is not torch.int8:
        # The caller views the uint8 index cache as float8_e4m3fn; on Kunlun the
        # bytes are int8, so reinterpret rather than convert.
        k = k.view(torch.int8)
    k_scale = k_scale.reshape(k.shape[0]).float()

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if q.dtype is torch.int8:
        weights = weights / 127

    total_q = q.shape[0]
    total_k = k.shape[0]
    device = q.device

    logits = torch.zeros((total_q, total_k), dtype=torch.float32, device=device)
    if total_q == 0 or total_k == 0:
        return logits

    q_lens_cpu = torch.tensor([0, total_q], dtype=torch.int32, device="cpu")
    k_lens_cpu = torch.tensor([0, total_k], dtype=torch.int32, device="cpu")
    def _ctg(t):
        # torch_xmlir 的 .contiguous() 对已连续张量也会 alloc+copy（q 单次 32MB），按需即可。
        return t if t.is_contiguous() else t.contiguous()

    kunlun_ops.I8_mqa_logits(
        q=_ctg(q),
        fused_kv_cache=(_ctg(k), _ctg(k_scale)),
        weights=_ctg(weights),
        context_q_lens=(q_lens_cpu, q_lens_cpu.to(device)),
        context_k_lens=(k_lens_cpu, k_lens_cpu.to(device)),
        logits=logits,
        clean_logits=clean_logits,
        use_xfa_boost=False,
    )
    # mask_for_I8_mqa_logits does no bounds/emptiness checking and traps the
    # device ("core trap exception" in dmesg, surfacing as status=719 at the next
    # device op) on two input patterns, both measured on P800:
    #   * ke > seq_len_kv  (out-of-range window end)
    #   * ke <= ks         (empty window; ks==ke and ks>ke both trap)
    # Non-empty in-range windows are fine at any position. The ragged plan does
    # produce empty windows: a query row whose causally visible pooled length is
    # 0 (the first tokens of a sequence, before pool_size tokens have been
    # pooled) gets ke == ks. So clamp the range, widen empty windows to one
    # column so the kernel has something legal to do, then blank those rows.
    # (clamp, not clamp_: .to() is a no-op when already int32, so an in-place
    # clamp would corrupt the caller's plan tensors.)
    ks_i32 = cu_seqlen_ks.to(torch.int32).clamp(0, total_k)
    ke_i32 = cu_seqlen_ke.to(torch.int32).clamp(0, total_k)
    ke_safe = torch.maximum(ke_i32, ks_i32 + 1).clamp(max=total_k)
    ks_safe = torch.minimum(ks_i32, ke_safe - 1).clamp(min=0)
    torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
        seq_len_kv=total_k,
        cu_seqlen_ks=ks_safe.contiguous(),
        cu_seqlen_ke=ke_safe.contiguous(),
        logits=logits,
    )
    # An empty window means nothing is visible → 只对这几行(序列开头不足 pool 的行)写 -inf，
    # 不再对整块 [total_q, total_k]（最大 ~623MB）做整趟 broadcast masked_fill_。
    empty_rows = ke_i32 <= ks_i32
    logits[empty_rows] = float("-inf")
    return logits
