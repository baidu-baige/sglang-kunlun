# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun subclass of ``sglang.srt.layers.attention.deepseek_v4_backend``.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional, Tuple, List

import cocopod  # noqa: F401  (registers torch.ops.xspeedgate_ops.*)
import xspeedgate_ops  # noqa: F401  (the installed wheel: newer ops, e.g. cast_*)
import kunlun_ops
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
    DSV4AttnMetadata,
    _as_int_list,
    _pad_tensor_to_size,
)
from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks

# Bind the module, not the function: ``kernel_ops.install()`` swaps
# ``dsv41_sparse.token_req_indices`` for the Kunlun replacement (host-mirror
# permutation, one H2D, no ``repeat_interleave`` launch), and its
# imported-binding rewrite deliberately skips ``sglang_kunlun.*`` modules -- a
# ``from ... import token_req_indices`` here keeps pointing at the upstream
# ``_PADFIX`` version for the whole process.
from sglang.srt.layers.attention.dsv4 import dsv41_sparse as _dsv41_sparse
from sglang.srt.layers.cp.interleave import interleave_rows_per_request
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.runtime_context import get_parallel
from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.layers.attention.dsv4 import indexer as _upstream_indexer
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    NUM_SCALE_TILES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
    dequantize_k_cache_paged,
    fp8_dtype,
)
from sglang.srt.layers.attention.dsv4.metadata import (
    _LARGE_INDEXER_QUERY_THRESHOLD,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

from sglang.srt.layers.attention.dsv4.compressor import (
    CompressorBackendMixin as _V1CompressorBackendMixin,
)

logger = logging.getLogger(__name__)

# ---- hot-path config probes, resolved once (r89) -----------------------------
# These all sit on the per-layer attention path (40 layers/forward) and each one
# is pure python work that re-does the same lookup: ``envs.*.get()`` re-parses
# ``os.environ`` (measured 1.3-1.5 us per call in this build: getenv +
# encode/decode/lower/parse), a function-level ``from X import Y`` re-runs the
# import machinery, and ``is_dsa_enable_prefill_cp`` re-derives the ModelConfig
# (``process_model_config`` -> ``model_config_of`` -> ``is_deepseek_dsa``/
# ``is_deepseek_v4``) on every call.  All of them are pure functions of the
# frozen server args / model config, so probe once.
# Effect (single-variable A/B, TP-0 trace, same pod): the
# ``set_k_and_s_v4_kernel -> infer_cast_dtype`` launch-bound bubble shrinks
# 333 -> 311 us median (n=120 both arms), and the ModelConfig rebuild
# (``__getattribute__`` 3056 -> 2420) disappears from inside it.
# KL_CFG_CACHE=0 restores the per-call behaviour (A/B arm).
_CFG_CACHE = os.environ.get("KL_CFG_CACHE", "1") != "0"
_CP_ROUND_ROBIN_SPLIT: Optional[bool] = None
_PUBLISHED_MASKS = None
_SPARSE_PREFILL_OPT: Optional[bool] = None


def _cp_round_robin_split() -> bool:
    """``is_dsa_prefill_cp_round_robin_split()`` cached (see block comment)."""
    global _CP_ROUND_ROBIN_SPLIT
    if _CP_ROUND_ROBIN_SPLIT is not None and _CFG_CACHE:
        return _CP_ROUND_ROBIN_SPLIT
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    value = bool(is_dsa_prefill_cp_round_robin_split())
    if _CFG_CACHE:
        _CP_ROUND_ROBIN_SPLIT = value
    return value


def _published_masks():
    """``candidate_indexer.published_masks`` handle, resolved once."""
    global _PUBLISHED_MASKS
    if _PUBLISHED_MASKS is not None and _CFG_CACHE:
        return _PUBLISHED_MASKS
    from sglang.srt.layers.attention.dsv4.candidate_indexer import published_masks

    if _CFG_CACHE:
        _PUBLISHED_MASKS = published_masks
    return published_masks


def _sparse_prefill_opt() -> bool:
    """``SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`` cached (it is read every layer)."""
    global _SPARSE_PREFILL_OPT
    if _SPARSE_PREFILL_OPT is not None and _CFG_CACHE:
        return _SPARSE_PREFILL_OPT
    value = bool(envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())
    if _CFG_CACHE:
        _SPARSE_PREFILL_OPT = value
    return value


def _capturing_cuda_graph() -> bool:
    """Whether the current forward is being recorded into a CUDA graph.

    ``torch.cuda.is_current_stream_capturing()`` is unreliable on the
    cuda-mapped XPU: it returns False *inside* a capture (verified by tracing
    this build: ``get_is_capture_mode()`` is True while the torch query is
    False). Callers that branch on it then issue pageable H2D/D2H copies or
    skip their capture-only fallbacks, which is fatal in a graph: the copy
    waits on the capture stream (driver reports ``error code= 702`` after
    ~20 min) and the Triton kernels that should have been replaced run inside
    the capture (C2C fabric retry storm, node wedge).

    sglang's runner publishes a process-global capture flag via
    ``model_capture_mode()`` for exactly this purpose; prefer it, and keep the
    torch check only as a fallback for builds where the helper is missing.
    """
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        return bool(get_is_capture_mode())
    except Exception:
        pass
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


# Per-device copy of the fp4 dequant LUT, materialised before capture.
_FP4_LUT_CACHE: dict = {}


def _fp4_lut(device) -> torch.Tensor:
    # `torch.cuda.current_device()` returns an int; normalise so the pre-warm
    # and the in-graph lookup share the same cache key. The conversion is a
    # device copy + dtype cast, which must not happen inside the capture.
    if not isinstance(device, torch.device):
        device = torch.device("cuda", int(device))
    key = (device.type, device.index)
    lut = _FP4_LUT_CACHE.get(key)
    if lut is None:
        lut = DSV4_DEQUANT_FP4_TABLE.to(device).float().contiguous()
        _FP4_LUT_CACHE[key] = lut
    return lut


@plugin_hook(
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
    "DecodeCudaGraphRunner.capture",
    type=HookType.BEFORE,
)
def _prewarm_fp4_lut(*args, **kwargs):
    """Materialise the fp4 dequant LUT before the decode graph capture."""
    try:
        _fp4_lut(torch.cuda.current_device())
    except Exception as exc:  # pragma: no cover
        logger.warning("sglang-kunlun: fp4 LUT pre-warm failed: %r", exc)


_HEAD_DIM = DIM_NOPE + DIM_ROPE  # 512
_E4M3_LUT = None


def _e4m3_lut(device):
    global _E4M3_LUT
    if _E4M3_LUT is None or _E4M3_LUT.device != device:
        lut_cpu = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).to(torch.float32)
        _E4M3_LUT = lut_cpu.to(device)
    return _E4M3_LUT


def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.reshape(-1, cache.shape[-1]).contiguous()


# CP prefill: the ratio 1/2 compressor projects hidden rows independently, yet
# the upstream CP path ships the whole prompt's hidden state to every rank and
# projects all of it there (cp_size-fold redundant GEMM + a full-width gather).
# Project this rank's own rows first and gather only the narrow (kv, score)
# rows instead. Same global rows, same math, different fp32 accumulation order.
# KL_CP_PROJECT_FIRST=0 restores the full-hidden gather; the /tmp marker flips
# arms inside one engine process.
def _cp_project_first() -> bool:
    if os.path.exists("/tmp/kl_cp_off"):
        return False
    return os.environ.get("KL_CP_PROJECT_FIRST", "1") != "0"


# fp16<->bf16 casts carrying the attention round trip (q bf16->fp16 in, out
# fp16->bf16 out).  aten's converter runs the fp16->bf16 direction at ~800 GB/s
# on P800 while bf16->fp16 reaches ~1350 GB/s and a plain copy ~1800 GB/s; the
# xspeedgate SIMD kernel does widening-load / narrowing-store in one pass and
# is bit-identical to ``.to`` (RNE, inf/nan pass through, saturation to +-inf).
_XSG_CAST_OPS = {
    (torch.float16, torch.bfloat16): "cast_fp16_bf16",
    (torch.bfloat16, torch.float16): "cast_bf16_fp16",
}

# Keep the prefill compressed-attention output in the dtype its kernel already
# produces (fp16) instead of casting it to the model dtype (bf16), which the
# very next consumer -- the inverse RoPE plus wo_a's per-group fp16 GEMM --
# casts straight back.  Decode / target-verify keep the old cast: their graph
# contracts were captured with the bf16 output.
_ATTN_OUT_KEEP_FP16 = os.environ.get("KL_ATTN_OUT_FP16", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)


def _xsg_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``tensor.to(dtype)``, routed through the xspeedgate converter when able."""
    name = _XSG_CAST_OPS.get((tensor.dtype, dtype))
    if name is not None and tensor.device.type == "cuda" and tensor.is_contiguous():
        op = getattr(torch.ops.xspeedgate_ops, name, None)
        if op is not None:
            return op(tensor)
    return tensor.to(dtype)


# [KL_ATTN_PREP] Kill switch / verification for the per-forward launch-metadata
# memo below. ``KL_ATTN_PREP_MEMO=0`` rebuilds on every layer (= the old
# behaviour); ``KL_ATTN_PREP_SELFCHECK=1`` rebuilds too and logs any field that
# disagrees with the memoised value.
_KL_ATTN_PREP_MEMO = os.environ.get("KL_ATTN_PREP_MEMO", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)
_KL_ATTN_PREP_SELFCHECK = os.environ.get(
    "KL_ATTN_PREP_SELFCHECK", "0"
).lower() not in ("0", "", "false", "off", "none")
_KL_ATTN_PREP_CHECKS = [0]

_KL_ATTN_PREP_FIELDS = (
    "qlod_cpu",
    "qlod_xpu",
    "kvseqlen_cpu",
    "kvseqlen_xpu",
    "kvseqlen_true_max",
    "win_indices",
    "win_size",
    "com_indices",
    "com_ratio",
    "com_topk",
)

# [host-diet r85] ``max_logits`` / ``lse`` returned by ``compressed_attention``
# are write-only outputs: nothing reads them after the op returns, and the next
# layer overwrites the whole surface.  In the extend path they were allocated
# twice per layer (2 x 40 ``aten::empty*`` dispatches per forward); reuse a
# scratch instead.  The tensors are handed to a native op and never retained, so
# aliasing across layers is safe (same stream, sequential).  Decode / cuda-graph
# capture keeps the plain ``torch.empty`` because those allocations get baked
# into the graph.
#
# [rel-7th qps-oom] r85 那版是**按 shape** 缓存（key 里带 ``tuple(shape)``），在
# 多 shape 的 QPS 流量下会退化成无界缓存：business 数据集每个请求的 extend token
# 数 T 都不同，9 分钟 QPS 实测约 370 个不同 T，每个 T 各留一份 ``[T,64]`` fp32
# （512·T 字节）且永不释放 —— 几百 MiB 就这样被钉住。随后一次"完整 QPS 发压"的
# 大请求里，``dsv41_cp_indexer._expand()`` 要的那块全宽 bool 平面（约 758 MiB
# 连续块）就分配不出来，直接 ``torch.OutOfMemoryError``（free 2.79 GiB 但没有
# 够大的连续段）。因此这里改成**每个 (tag, device) 只留一块 grow-only 缓冲**，
# 每次调用从头部切出自己那一份视图；容量只增不减，稳定态就是
# ``2 × maxT × 64 × 4 B``（T=4096…65536 时 ≤ 32 MiB），相对 pre-5625 的每层
# 分配一块不额外多占内存。
_ATTN_AUX_CACHE: dict = {}
_ATTN_AUX_SCRATCH = os.environ.get("KL_ATTN_AUX_SCRATCH", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)

# [c4 dual-stream r90] ``compressed_attention`` ships a second, ~1.5x faster
# kernel family (``compressed_attention_c4_kernel_dual_stream``) that the
# vendor gates on ``compress_ratio == 4`` plus
# ``COCOPOD_COMPRESSED_ATTN_C4_DUAL_STREAM``.  This model only ever uses
# ratios 1/2 (``config.compress_ratios``), so that gate never fires.  It is
# safe to hand the op ratio 4 anyway because, measured on production geometry
# (q [4096,64,512] fp16, topk 512, win 128, uneven per-request kvlen), ratio
# is *only* a visibility cap -- ``ceil(kvlen/ratio)`` slots -- and is
# independent of the index values, of the window size and of causal masking:
# ratios 1/2/4 produce bit-identical ``o`` / ``max_logits`` / ``lse``
# (md5-compared, also with a non-zero ``attn_sink``) as soon as the larger
# ratio's cap still leaves ``com_topk + 128`` slots.  ``_c4_dual_ok`` builds
# the dispatch half of that precondition from the op's own gate (fp16 KV
# cache, h_q 64, d_qk == d_v == 512); the ``com_topk >= 256`` and per-request
# kvlen tests need the prepared metadata, so they live in
# ``_build_kl_attn_prep`` where the lengths are already on the host.  Decode /
# cuda-graph keeps the layer ratio: the substitution is only wired into the
# extend prep, and the cap depends on the live kvlen, which a captured graph
# cannot re-evaluate.  ``KL_C4_DUAL_RATIO=0`` restores the old behaviour.
#
# ``kvseqlen`` is per *query row* (``seq_lens_casual`` is expanded to one entry
# per token before it reaches the backend), and a chunk always starts at
# position 0, so the first ``4*(com_topk+128)/cp_degree`` rows of every request
# are below the boundary.  Those rows keep the layer ratio and everything after
# them gets ratio 4: the batch is cut into maximal runs of equal safety and each
# run is one op call on a contiguous row slice (row slices of every operand stay
# contiguous, so the split costs no copy -- just one extra launch per request).
_C4_DUAL_RATIO = os.environ.get("KL_C4_DUAL_RATIO", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)
_C4_QLOD: dict = {}


def _c4_qlod(n: int, device):
    """``(qlod_cpu, qlod_xpu)`` for an ``n``-row single-row-per-request batch."""
    key = (n, device)
    t = _C4_QLOD.get(key)
    if t is None:
        t = (
            torch.arange(n + 1, dtype=torch.int32),
            torch.arange(n + 1, dtype=torch.int32, device=device),
        )
        if len(_C4_QLOD) < 64:
            _C4_QLOD[key] = t
    return t


def _c4_dual_ok(q: torch.Tensor, win_cache: torch.Tensor, head_dim_v: int) -> bool:
    """Vendor dual-stream c4 dispatch gate, minus the kvlen / com_topk tests."""
    ok = bool(
        _C4_DUAL_RATIO
        and q.dim() == 4
        and q.shape[2] == 64
        and q.shape[3] == 512
        and win_cache.dtype == torch.float16
        and head_dim_v == 512
    )
    return ok


def _grow_cap(old: int, need: int) -> int:
    """grow-only 容量：够用即可，否则按 1.25 倍留余量（别让每个新 T 都重分配）。"""
    return need if old <= 0 else max(need, (old * 5 + 3) // 4)


def _attn_aux(tag: str, shape, device) -> torch.Tensor:
    shape = tuple(shape)
    numel = 1
    for d in shape:
        numel *= int(d)
    key = (tag, device)
    buf = _ATTN_AUX_CACHE.get(key)
    if buf is None or buf.numel() < numel:
        old = 0 if buf is None else buf.numel()
        # 先把旧缓冲的引用放掉再分配：峰值是「一块」而不是「两块」（同
        # zcp_gather_chunked._grow_reserve 的教训）。
        buf = None
        _ATTN_AUX_CACHE.pop(key, None)
        buf = torch.empty(_grow_cap(old, numel), dtype=torch.float32, device=device)
        _ATTN_AUX_CACHE[key] = buf
    # ``[:numel]`` 是连续前缀视图（stride 与旧 ``torch.empty(shape)`` 完全一致），
    # 再 ``view(shape)`` 还原调用方要的维度。
    return buf[:numel].view(shape)


def _build_kl_attn_prep(
    core_attn_metadata,
    q: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_window_size: int,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    compress_ratio: int,
    is_causal: bool,
    win_cache: torch.Tensor,
    com_cache: torch.Tensor,
    dual_ok: bool = False,
):
    """Launch metadata for the non-cuda-graph (extend) compressed-attention path.

    Every value is a pure function of ``core_attn_metadata`` -- which one
    forward shares across all of its attention layers -- and of the pool
    shapes, so each layer used to rebuild identical tensors.  Measured on the
    r14 trace, that redundancy was the single largest host stall of the step:
    40 pageable H2D copies of ``qlod`` (7.7 ms of implied stream synchronise
    each, ~308 ms/step, 25% of the host budget) plus one blocking D2H and a
    re-slice/re-clamp of the window indices per layer.
    """
    seq_lens = core_attn_metadata.seq_lens_casual[: q.shape[0]]
    bs = seq_lens.shape[0]
    qlod_cpu = torch.arange(bs + 1, dtype=torch.int32)
    # [KL_ATTN_PREP] Build qlod on the device. ``qlod_cpu.to(q.device)`` is a
    # *pageable* H2D, so torch turns it into "copy + cudaStreamSynchronize" and
    # the host waits for the whole queued device backlog once per layer.
    qlod_xpu = torch.arange(bs + 1, dtype=torch.int32, device=q.device)
    kvseqlen_xpu = torch.clamp(seq_lens.to(dtype=torch.int32, device=q.device), min=0)
    kvseqlen_cpu = torch.clamp(kvseqlen_xpu.to("cpu", non_blocking=False), min=0)
    kvseqlen_true_max = int(kvseqlen_cpu.max().item()) if kvseqlen_cpu.numel() else 0
    # [KUNLUN CP int16 guard] see the caller: the op reads the per-token KV
    # length as int16, and CP interleave hands out global cumulative positions,
    # so clamp the op's copy to 32767. The true max is kept for com_topk.
    if not is_causal and kvseqlen_true_max > 32767:
        kvseqlen_cpu = kvseqlen_cpu.clamp(max=32767)
        kvseqlen_xpu = kvseqlen_xpu.clamp(max=32767)
    win_indices = swa_page_indices.squeeze(1).contiguous()
    win_size = min(swa_window_size, win_indices.shape[1])
    win_indices = win_indices[:, :win_size].contiguous()
    if extra_k_cache is None or extra_indices is None:
        com_indices = win_indices.new_empty((win_indices.shape[0], 0))
        com_ratio = 1
        com_topk = 0
    else:
        com_indices = extra_indices.squeeze(1).contiguous()
        com_ratio = compress_ratio
        com_topk = max(kvseqlen_true_max // com_ratio, 1)
        com_topk = min(com_topk, com_indices.shape[1])
        com_indices = com_indices[:, :com_topk].contiguous()
    # [c4 dual-stream r90] Hand the op ratio 4 so its dual-stream kernel family
    # becomes reachable; ``com_topk`` above stays sized by the layer's real
    # ratio, so the only thing that changes is the op's visibility cap.  The
    # guard is the measured equivalence boundary: the output stays
    # bit-identical while ``ceil(kvlen/4) >= com_topk + 128``, with kvlen taken
    # per row exactly as the op sees it (i.e. after the int16 clamp).  Rows
    # below the boundary keep the layer ratio; see ``c4_runs`` below.
    c4_runs = None
    if dual_ok and com_ratio in (1, 2) and com_topk >= 256 and kvseqlen_cpu.numel():
        need = 4 * (com_topk + 128)
        safe = kvseqlen_cpu >= need
        n_rows = int(kvseqlen_cpu.numel())
        n_safe = int(safe.sum())
        if n_safe:
            # Maximal runs of equal safety -> one op call each, at that run's
            # ratio.  Rows are ordered by position within a request and a chunk
            # restarts at 0, so this is normally "short prefix + long suffix".
            vals = safe.tolist()
            c4_runs = []
            start = 0
            for i in range(1, n_rows + 1):
                if i == n_rows or vals[i] != vals[start]:
                    c4_runs.append((start, i, 4 if vals[start] else com_ratio))
                    start = i
    # [KUNLUN CG-SAFE] Clamp out-of-range index slots to a VALID cache row
    # instead of -1: a clamped index addresses a real row and the per-request
    # device kvseqlen mask zeroes those slots out, so the output is unchanged.
    win_indices = win_indices.clamp(min=0, max=win_cache.shape[0] - 1).contiguous()
    if com_indices.numel() and com_cache.shape[0] > 0:
        com_indices = com_indices.clamp(min=0, max=com_cache.shape[0] - 1).contiguous()
    return (
        qlod_cpu,
        qlod_xpu,
        kvseqlen_cpu,
        kvseqlen_xpu,
        kvseqlen_true_max,
        win_indices,
        win_size,
        com_indices,
        com_ratio,
        com_topk,
        c4_runs,
    )


def _kl_attn_prep(
    core_attn_metadata,
    key,
    build,
):
    """Memoise ``build()`` on the metadata object (dies with the forward)."""
    if not _KL_ATTN_PREP_MEMO:
        return build()
    cache = getattr(core_attn_metadata, "_kl_attn_prep", None)
    if cache is not None and cache[0] == key:
        if _KL_ATTN_PREP_SELFCHECK:
            # Bounded to a handful of hits so the check itself stays free.
            budget = _KL_ATTN_PREP_CHECKS
            budget[0] += 1
            if budget[0] <= 8:
                fresh = build()
                for name, cached, new in zip(_KL_ATTN_PREP_FIELDS, cache[1], fresh):
                    same = (
                        bool(torch.equal(cached, new))
                        if torch.is_tensor(cached)
                        else cached == new
                    )
                    if same:
                        continue
                    logger.error(
                        "[KL_ATTN_PREP] selfcheck MISMATCH on %s (key=%s)",
                        name,
                        key,
                    )
                logger.info("[KL_ATTN_PREP] selfcheck #%d ok (key=%s)", budget[0], key)
        return cache[1]
    prep = build()
    try:
        core_attn_metadata._kl_attn_prep = (key, prep)
    except Exception:  # pragma: no cover - metadata is a plain dataclass
        pass
    return prep


def _copy_host_lengths_(destination, source, fill_value: int) -> None:
    """Copy per-request lengths into a pointer-stable host buffer in place."""
    count = min(source.numel(), destination.numel())
    destination[:count].copy_(source[:count])
    if count < destination.numel():
        destination[count:].fill_(fill_value)


def _seq_lens_cpu_i32(forward_batch) -> torch.Tensor:
    """Real per-request KV lengths on the host (no D2H when seq_lens_cpu exists)."""
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.to("cpu", non_blocking=False)
    return seq_lens_cpu[: forward_batch.batch_size].to(torch.int32)


class KunlunDeepseekV4AttnBackend(_V1CompressorBackendMixin, DeepseekV4AttnBackend):
    """KunlunDeepseekV4AttnBackend
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # [KUNLUN CG-SAFE] Persistent per-batch-size LoD / KV-length buffers for
        # cuda-graph decode. Allocated once per bs and refreshed in place via a
        # *recorded* copy_ kernel, so the captured compressed_attention reads a
        # stable address whose contents are refreshed on every replay (fresh
        # per-call allocations land in the churning graph pool and go stale at
        # replay -> illegal memory access, esp. at bs>1). Mirrors hunheyun's
        # `_attention_decode_aux`.
        self._attention_decode_aux: dict = {}
        # [KUNLUN CG-SAFE] Persistent per-B buffers for the C4 indexer decode path
        # (qlod + context_lens). Same rationale as _attention_decode_aux: the decode
        # branch of _kunlun_c4a_paged_mqa_logits otherwise does an H2D (qlod.to(dev))
        # and a D2H (context_lens.to("cpu")) that are ILLEGAL inside cuda-graph
        # capture -> illegal memory access at the first ratio-4 layer. Refreshed
        # out-of-graph in init_forward_metadata_out_graph. Mirrors hunheyun _c4_decode_aux.
        self._c4_decode_aux: dict = {}

    def make_core_attn_metadata(self, *args, **kwargs):
        """[kunlun-adapt] Keep padded target-verify metadata self-consistent.

        ``ForwardBatch.prepare_mlp_sync_batch`` pads the verify token count to
        a tensor-parallel tier. The padding is **global**: bs * cfg real rows
        followed by ``phys - bs * cfg`` trailing rows (block=5 with 5 requests:
        30 -> 32). Upstream builds ``seq_lens_casual`` from the configured
        draft width, so every real query would attend ``padded - configured``
        too few keys and the first verify logits are already wrong
        (repetitive output); the trailing rows also break
        ``init_compression_metadata``'s length assertion.

        Rebuild the causal chain for the physical width here. Trailing rows
        reuse the last real row's causal length and request index: their
        outputs are discarded, and keeping the length unchanged cannot walk
        past the request's allocated KV.
        """
        seq_lens_casual = kwargs.get("seq_lens_casual")
        out_loc = kwargs.get("out_loc")
        num_groups = kwargs.get("num_groups")
        if (
            seq_lens_casual is not None
            and out_loc is not None
            and num_groups
            and not kwargs.get("is_prefill", False)
        ):
            real = seq_lens_casual.numel()
            phys = out_loc.numel()
            if phys > real and real % num_groups == 0:
                cfg = real // num_groups
                base = seq_lens_casual.view(num_groups, cfg)[:, -1:]
                index = torch.arange(
                    cfg,
                    device=seq_lens_casual.device,
                    dtype=seq_lens_casual.dtype,
                )
                head = (base - cfg + 1 + index).reshape(-1)
                pad = phys - real
                tail = base[-1:].repeat(pad, 1).reshape(-1)
                kwargs["seq_lens_casual"] = torch.cat([head, tail]).to(
                    seq_lens_casual.dtype
                )
                repeat = kwargs.get("req_pool_indices_repeated")
                if repeat is not None and repeat.numel() == real:
                    kwargs["req_pool_indices_repeated"] = torch.cat(
                        [repeat, repeat[-1:].repeat(pad)]
                    )
        return super().make_core_attn_metadata(*args, **kwargs)

    def get_dspark_swa_page_indices(self, *args, **kwargs):
        """[kunlun-race-fix] Serialize the DSpark draft window-metadata build.

        The draft metadata kernels read ``req_to_token`` / the full->SWA
        mapping and the freshly accepted verify slots. At larger draft batches
        (bs>=5, 25 tokens) a producer on another stream could still be in
        flight, and the metadata kernel then read a partially updated table and
        walked an invalid page range (async error 700 surfacing at the next
        sync). Serializing this build is cheap relative to a decode step and
        makes the read-after-write ordering explicit. Set
        SGLANG_KUNLUN_DSPARK_META_SYNC=0 to disable for benchmarking.
        """
        import os

        from sglang_kunlun.hooks.graph_state import is_graph_capturing

        capturing = is_graph_capturing()
        sync = (
            os.environ.get("SGLANG_KUNLUN_DSPARK_META_SYNC", "1") == "1"
            and not capturing
        )
        if sync:
            torch.cuda.synchronize()
        result = super().get_dspark_swa_page_indices(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        return result

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        """Refresh the pointer-stable decode KV-length buffers before every graph
        replay (this runs OUT of the captured graph). The captured
        compressed_attention sizes its per-row KV walk from the HOST kvseqlen; if
        it stayed at the capture placeholder the op would walk past the window/topk
        index columns at replay -> OOB (illegal access at bs>1). Mirrors hunheyun
        init_forward_metadata_out_graph + _refresh_graph_host_lengths.
        """
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        try:
            # [KUNLUN CG-SAFE] The metadata glue graph (MetadataGlueGraph) may
            # RECORD this whole method into its own captured graph; an H2D
            # ``aux[3].copy_(aux[2])`` recorded there replays a STAGED host
            # buffer and leaves stale per-row KV lengths -> decode drift. Skip
            # the refresh while capturing. Device kvseqlen_xpu no longer needs
            # this refresh at all: the attention call records its own in-graph
            # D2D copy from the runner's static forward_batch.seq_lens (the
            # hunheyun _make_lod contract), so only the HOST kvseqlen_cpu
            # (read by the op via its captured CPU pointer) is refreshed here
            # — host writes are plain memory stores, replay-safe.
            _capturing = _capturing_cuda_graph()
            if _capturing:
                return
            if not forward_batch.forward_mode.is_decode_or_idle():
                return
            seq_lens_cpu = _seq_lens_cpu_i32(forward_batch)
            batch_size = forward_batch.batch_size
            for cached_bs, aux in self._attention_decode_aux.items():
                if cached_bs < batch_size:
                    continue
                # aux = (qlod_cpu, qlod_xpu, kvseqlen_cpu, kvseqlen_xpu)
                _copy_host_lengths_(aux[2], seq_lens_cpu, fill_value=1)
                # [KUNLUN CG-SAFE FIX] Also refresh the DEVICE kvseqlen_xpu here,
                # out of graph, from the same real per-request lengths. It was
                # previously left frozen at the capture placeholder (all-ones) on
                # the assumption a metadata-glue graph would rebuild it — but this
                # build ships no glue, so the captured compressed_attention read a
                # stale device kvseqlen. Harmless for locally-prefilled decode
                # (host length also passed) but wrong under PD, where the first
                # decode step must attend the full transferred-prompt context.
                _copy_host_lengths_(
                    aux[3], seq_lens_cpu.to(aux[3].device), fill_value=1
                )
            # C4 indexer host context lengths (original token units, multiple of 4).
            c4_ctx = (seq_lens_cpu // 4) * 4
            for cached_b, aux in self._c4_decode_aux.items():
                if cached_b < batch_size:
                    continue
                # aux = (qlod_cpu, qlod_xpu, context_lens_cpu)
                _copy_host_lengths_(aux[2], c4_ctx, fill_value=4)
        except Exception as exc:  # pragma: no cover - never break the forward
            logger.warning("kunlun decode host-length refresh skipped: %s", exc)


    # [kunlun-adapt] NOTE: forward_c4_indexer 仅被 C4Indexer（compress_ratio
    # 4/128 层）调用；DSV4.1（ratio 0/1/2）不触达。其内部对旧 indexer API
    # （topk_transform_512*/_forward_prepare_*）的引用在 V4.1 上为死路径，
    # 若需支持 V4 全 KV 模型需按上游 dsv4/indexer.py 新 API 重写。
    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
        skip_compressor: bool = False,
    ) -> None:
        """Upstream ``C4IndexerBackendMixin.forward_c4_indexer`` with the Kunlun
        indexer-logits kernel called inline.

        The env-selected ``fn`` ladder (deep_gemm / tilelang / aiter / torch) is
        replaced by ``self._kunlun_c4a_paged_mqa_logits``. The topk / hisparse /
        indexer-capturer tail is unchanged, resolved lazily off the upstream module
        so ``register_jit_op`` replacements still apply.
        """
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        # [kunlun-adapt] upstream _forward_prepare_{normal,multi_stream} dropped
        # the token_to_kv_pool arg and now return only (q, weights); the indexer
        # KV cache is fetched by the caller via get_index_k_with_scale_buffer.
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
            c4_indexer.layer_id
        )

        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        if c4_indexer.use_fp4_indexer:
            # The fp4 path needs deep_gemm's fp8_fp4_paged_mqa_logits; there is no
            # Kunlun equivalent.
            raise NotImplementedError(
                "DeepSeek V4 FP4 indexer is not supported on Kunlun XPU."
            )
        head_dim_with_sf = 132

        assert len(q_indexer.shape) == 3
        q = q_indexer.unsqueeze(1)

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        query_rows = q_indexer.shape[0]

        def match_num_queries(tensor: torch.Tensor, value: int) -> torch.Tensor:
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (0, query_rows - tensor.shape[0])
            return F.pad(tensor, pad, value=value)

        c4_seq_lens = match_num_queries(indexer_metadata.compressed_seq_lens, value=1)
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )

        logits = self._kunlun_c4a_paged_mqa_logits(
            q_fp8=q,
            kvcache_fp8=c4_indexer_kv_cache,
            weight=weights,
            seq_lens=c4_seq_lens,
            page_table=page_table,
            max_seq_len=indexer_metadata.max_compressed_seq_len,
            forward_batch=forward_batch,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = _upstream_indexer.get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = self.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        # [kunlun-adapt] Upstream removed SGLANG_TOPK_TRANSFORM_512_TORCH and
        # the topk_transform_512* API for a DSATopKBackend-dispatched
        # topk_transform_paged* family; the sgl_kernel / flashinfer paged-topk
        # kernels have no XPU backend, so force the graph-safe pure torch path.
        _upstream_indexer.topk_transform_pytorch_vectorized(
            logits,
            c4_seq_lens,
            page_table,
            c4_sparse_page_indices,
            indexer_metadata.compressed_page_size,
            raw_indices,
        )

        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.compressed_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                # flash_mla C4 attention requires int32 page indices.
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

    def _c4a_cp_extend_logits_torch(
        self,
        q_fp8: torch.Tensor,
        kvcache_fp8: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Indexer logits for the NSA prefill context-parallel extend path.

        The kunlun ragged kernel is driven by a per-request q-lod, but under prefill
        context parallelism a rank owns a strided subset of the sequence's queries, so
        the request's extend length does not describe the local query rows; the paged
        kernel in turn returns all-zero scores for them. Zero scores make the
        downstream 512-topk keep the *oldest* 512 compressed states, which silently
        truncates the context of any prompt longer than ``topk * compress_ratio``
        tokens and degenerates the output. Compute the scores directly instead:

            score[q, k] = sum_h weight[q, h] * relu(q[q, h] . k[k]) * k_scale[k]

        which is the same formula as the reference paged fallback, evaluated for the
        local rows only. Chunked over queries to bound the temporary [chunk, H, K].
        """
        B, _sq, H, D = q_fp8.shape
        device = q_fp8.device
        scale_offset = block_size * D
        total_dim = block_size * (D + 4)

        lens = seq_lens.to(torch.int32)
        max_len = int(lens.max().item())
        if max_len <= 0:
            return torch.zeros((B, max_seq_len), dtype=torch.float32, device=device)

        num_pages = (max_len + block_size - 1) // block_size
        num_pages = min(num_pages, page_table.shape[1])
        row = int(lens.argmax().item())
        pages = page_table[row, :num_pages].to(torch.int64).clamp(min=0)

        kv_flat = kvcache_fp8.reshape(-1, total_dim)
        gathered = kv_flat[pages]
        k_int8 = (
            gathered[:, :scale_offset].reshape(-1, D)
            if gathered.dtype == torch.int8
            else gathered[:, :scale_offset].contiguous().view(torch.int8).reshape(-1, D)
        )
        k_scale = (
            gathered[:, scale_offset:].contiguous().view(torch.float32).reshape(-1)
        )
        # Only the first ``max_len`` compressed states of the page range are written
        # by this request; the rest of the last page is whatever the pool happens to
        # hold (uninitialised garbage right after start-up). Never read past the
        # valid length, and accumulate the dot products in fp32: int8 keys and
        # queries reach |q.k| ~ 127*127*D, far above the fp16 range, so an fp16
        # matmul saturates to +inf and the weighted sum below turns it into NaN,
        # which then poisons the whole attention output.
        K = min(k_int8.shape[0], num_pages * block_size, max_len)
        k_f = k_int8[:K].to(torch.float32)
        k_scale = k_scale[:K].to(torch.float32)

        q_int8 = q_fp8[:, 0]
        q_f = (q_int8 if q_int8.dtype == torch.int8 else q_int8.view(torch.int8)).to(
            torch.float32
        )

        w = weight.to(torch.float32)
        logits = torch.zeros((B, K), dtype=torch.float32, device=device)
        chunk = int(os.environ.get("KL_INDEXER_CHUNK", "128"))
        for beg in range(0, B, chunk):
            end = min(beg + chunk, B)
            dots = torch.relu(torch.matmul(q_f[beg:end], k_f.t()))  # [c, H, K]
            logits[beg:end] = torch.einsum("chk,ch->ck", dots, w[beg:end])
        logits *= k_scale.unsqueeze(0)

        positions = torch.arange(K, device=device).unsqueeze(0)
        logits = logits.masked_fill(positions >= lens.unsqueeze(1), 0.0)

        if K < max_seq_len:
            logits = F.pad(logits, (0, max_seq_len - K), value=0.0)
        else:
            logits = logits[:, :max_seq_len]
        return logits

    def _kunlun_c4a_paged_mqa_logits(
        self,
        *,
        q_fp8: torch.Tensor,
        kvcache_fp8: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        forward_batch,
    ) -> torch.Tensor:
        """Indexer paged-mqa-logits routed to Kunlun native ops.

        Inlined ``fp8_paged_mqa_logits_torch`` replacement. Dispatch by ForwardMode:
        * decode / target_verify → ``kunlun_ops.c4a_paged_mqa_logits_with_mixed_cache``
          (paged KV, no host-side gather, CUDA-Graph safe)
        * extend / prefill       → gather KV contiguous + ``kunlun_ops.c4a_mqa_logits``
        """
        _COMPRESS_RATIO = 4
        _BLOCK_SIZE = 64

        logical_mode = getattr(
            forward_batch, "actual_forward_mode", forward_batch.forward_mode
        )
        is_extend = logical_mode.is_extend_without_speculative()
        # decode / target_verify / draft_extend / idle all use the paged op
        is_decode = not is_extend
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)

        # Normalise seq_lens to 1-D.
        _seq_lens = seq_lens.squeeze(-1) if seq_lens.dim() == 2 else seq_lens

        B, _sq, H, D = q_fp8.shape
        assert D == 128 and kvcache_fp8.shape[1] == _BLOCK_SIZE and _sq == 1

        total_dim = _BLOCK_SIZE * (D + 4)

        # NSA prefill context parallelism hands this rank a strided subset of the
        # sequence's queries, so the request's extend length is not the number of
        # local query rows and the ragged (q-lod based) logits kernel cannot describe
        # it: it would address rows outside the logits buffer, leaving most queries
        # with all-zero scores so topk keeps only the oldest ``topk`` compressed
        # states. Compute the scores for the local rows directly instead
        # (see _c4a_cp_extend_logits_torch).
        _cp_local = False
        if not is_decode and extend_seq_lens_cpu is not None:
            if len(extend_seq_lens_cpu) > 0:
                _cp_local = int(sum(int(v) for v in extend_seq_lens_cpu)) != B

        if _cp_local:
            return self._c4a_cp_extend_logits_torch(
                q_fp8,
                kvcache_fp8,
                weight,
                _seq_lens,
                page_table,
                max_seq_len,
                block_size=_BLOCK_SIZE,
            )

        if is_decode:
            # ── Decode / Target-Verify: paged op with mixed cache layout ──
            # kunlun_ops 0.1.203+ provides ``c4a_paged_mqa_logits_with_mixed_cache``
            # which accepts the mixed KV page buffer directly (K + scale
            # interleaved per page); no host-side gather needed.
            kv_cache_flat = kvcache_fp8.reshape(-1, total_dim)

            c4_seq_lens = _seq_lens.to(torch.int32).clamp(0, max_seq_len)
            context_lens_xpu = c4_seq_lens * _COMPRESS_RATIO
            # [KUNLUN CG-SAFE] Under cuda-graph capture the fresh `qlod.to(device)`
            # (H2D) and `context_lens_xpu.to("cpu")` (D2H) are illegal and cause an
            # illegal memory access at the first ratio-4 layer. Use persistent per-B
            # buffers: qlod built on-device (no H2D); context_lens_cpu a cached host
            # buffer refreshed OUT of graph in init_forward_metadata_out_graph
            # (no D2H). Device context_lens flows through the recorded context_lens_xpu.
            _capturing = _capturing_cuda_graph()
            if _capturing:
                aux = self._c4_decode_aux.get(B)
                if aux is None:
                    _qc = torch.arange(B + 1, dtype=torch.int32)
                    _qx = torch.arange(B + 1, dtype=torch.int32, device=q_fp8.device)
                    _clc = torch.ones(B, dtype=torch.int32)
                    aux = (_qc, _qx, _clc)
                    self._c4_decode_aux[B] = aux
                qlod_cpu, qlod_xpu, context_lens_cpu = aux
            else:
                qlod_cpu = torch.arange(B + 1, dtype=torch.int32)
                qlod_xpu = qlod_cpu.to(q_fp8.device, non_blocking=False)
                context_lens_cpu = context_lens_xpu.to("cpu", non_blocking=False)
            block_table_i32 = (
                page_table.to(torch.int32)
                if page_table.dtype != torch.int32
                else page_table
            )
            q_call = q_fp8.view(dtype=torch.int8) if q_fp8.dtype != torch.int8 else q_fp8

            logits = torch.empty(
                (B, 1, max_seq_len), dtype=torch.float32, device=q_fp8.device
            )

            kunlun_ops.c4a_paged_mqa_logits_with_mixed_cache(
                q=q_call,  # [B, 1, num_heads, head_dim] int8
                weight=weight.unsqueeze(1),  # [B, 1, num_heads] float32
                k_cache=kv_cache_flat,  # mixed page buffer
                logits=logits,  # [B, 1, max_seq_len]
                max_context_len=max_seq_len * _COMPRESS_RATIO,
                qlod_cpu=qlod_cpu,
                qlod_xpu=qlod_xpu,
                context_lens_cpu=context_lens_cpu,
                context_lens_xpu=context_lens_xpu,
                block_table=block_table_i32,
                compress_ratio=_COMPRESS_RATIO,
                clean_logits=True,
            )
            return logits.squeeze(1)

        # ── Extend / Prefill: gather KV contiguous, then c4a_mqa_logits ─
        if extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0:
            extend_lens = torch.as_tensor(extend_seq_lens_cpu, dtype=torch.int32)
            cum = torch.zeros(len(extend_seq_lens_cpu) + 1, dtype=torch.int64)
            cum[1:] = torch.cumsum(extend_lens.long(), dim=0)
            last_idx = (cum[1:] - 1).clamp(0, _seq_lens.shape[0] - 1)
            per_req_k_lens = _seq_lens[last_idx].to(torch.int32).cpu()
            num_requests = len(extend_seq_lens_cpu)
            qlod_cpu = torch.zeros(num_requests + 1, dtype=torch.int32)
            qlod_cpu[1:] = torch.cumsum(extend_lens, dim=0)
            max_seq_q = int(extend_lens.max().item())
            if int(qlod_cpu[-1].item()) != B:
                # NSA prefill context parallelism gives this rank only a strided
                # subset of the sequence's queries, so the request's extend length is
                # not the number of local query rows. Feeding the full length as the
                # q-lod makes the logits kernel address rows that do not exist: it
                # writes inf into part of the buffer and leaves the tail untouched, so
                # every query past ~topk compressed states keeps all-zero scores and
                # topk degenerates to "the oldest 512 states". Describe the local rows
                # instead and let the per-query mask in the topk transform apply
                # causality (which it does anyway), so drop the kernel-side mask.
                per_req_k_lens = _seq_lens.to(torch.int32).cpu().max().reshape(1)
                num_requests = 1
                qlod_cpu = torch.tensor([0, B], dtype=torch.int32)
                max_seq_q = B
                is_causal_logits = False
            else:
                is_causal_logits = True
        else:
            per_req_k_lens = _seq_lens.to(torch.int32).cpu()
            num_requests = B
            qlod_cpu = torch.arange(num_requests + 1, dtype=torch.int32)
            max_seq_q = 1
            is_causal_logits = True

        kv_cache_flat = kvcache_fp8.reshape(-1, total_dim)
        SCALE_OFFSET = _BLOCK_SIZE * D
        k_data_flat = kv_cache_flat[:, :SCALE_OFFSET].view(-1, _BLOCK_SIZE, D)
        scale_flat = (
            kv_cache_flat[:, SCALE_OFFSET:]
            .contiguous()
            .view(dtype=torch.float32)
            .reshape(-1, _BLOCK_SIZE)
        )

        total_k = int(per_req_k_lens.sum().item())
        if total_k == 0:
            return torch.zeros(
                (B, max_seq_len), dtype=torch.float32, device=q_fp8.device
            )

        k_contiguous = torch.empty((total_k, D), dtype=torch.int8, device=q_fp8.device)
        k_scale_contiguous = torch.empty(
            total_k, dtype=torch.float32, device=q_fp8.device
        )
        max_seq_k_compressed = 0
        offset = 0
        pt_rows = (qlod_cpu[1:] - 1).clamp(0, page_table.shape[0] - 1).tolist()
        for i in range(num_requests):
            seq_len_i = int(per_req_k_lens[i].item())
            if seq_len_i == 0:
                continue
            max_seq_k_compressed = max(max_seq_k_compressed, seq_len_i)
            num_blocks = (seq_len_i + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            pt_row = min(int(pt_rows[i]), page_table.shape[0] - 1)
            pages = page_table[pt_row, :num_blocks].long()
            k_pages = k_data_flat[pages].reshape(-1, D)[:seq_len_i]
            k_contiguous[offset : offset + seq_len_i] = k_pages.to(torch.int8)
            k_scale_contiguous[offset : offset + seq_len_i] = scale_flat[pages].reshape(
                -1
            )[:seq_len_i]
            offset += seq_len_i

        pre_k_lens = per_req_k_lens * _COMPRESS_RATIO
        klod_cpu = torch.zeros(num_requests + 1, dtype=torch.int32)
        klod_cpu[1:] = torch.cumsum(pre_k_lens, dim=0)
        klod_xpu = klod_cpu.to(q_fp8.device, non_blocking=False)
        qlod_xpu = qlod_cpu.to(q_fp8.device, non_blocking=False)
        com_k_start_cpu = torch.zeros(num_requests, dtype=torch.int32)
        com_k_start_xpu = com_k_start_cpu.to(q_fp8.device, non_blocking=False)

        logits = torch.zeros(
            (B, max_seq_k_compressed), dtype=torch.float32, device=q_fp8.device
        )
        # c4a_mqa_logits requires q dtype int8; upstream quant is already
        # int8 on Kunlun, defensive view for safety.
        q_2d = q_fp8[:, 0]  # [B, H, D]
        if q_2d.dtype != torch.int8:
            q_2d = q_2d.view(dtype=torch.int8)

        kunlun_ops.c4a_mqa_logits(
            q=q_2d,
            weight=weight,
            k=k_contiguous,
            k_scale=k_scale_contiguous,
            logits=logits,
            max_seq_q=max_seq_q,
            max_seq_k=max_seq_k_compressed * _COMPRESS_RATIO,
            qlod_cpu=qlod_cpu,
            qlod_xpu=qlod_xpu,
            klod_cpu=klod_cpu,
            klod_xpu=klod_xpu,
            com_k_start_cpu=com_k_start_cpu,
            com_k_start_xpu=com_k_start_xpu,
            is_causal=is_causal_logits,
            compress_ratio=_COMPRESS_RATIO,
            clean_logits=True,
        )
        if max_seq_k_compressed < max_seq_len:
            logits = F.pad(logits, (0, max_seq_len - max_seq_k_compressed), value=0.0)
        else:
            logits = logits[:, :max_seq_len]
        return logits

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs=None):
        """No-op: DSV4 has no draft-dependent verify buffers to refresh."""
        return None

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
    ):
        """Upstream ``DeepseekV4AttnBackend.forward`` with the Kunlun attention
        """
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        core_attn_metadata = self.forward_metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if not isinstance(core_attn_metadata, DSV4AttnMetadata):
            raise NotImplementedError("ragged attention")

        if save_kv_cache:
            self.store_cache(layer_id, swa_k, forward_batch)
        swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

        # ``actual_forward_mode`` carries the real runtime mode when this
        # backend is driven through cuda-graph capture (see ``_kunlun_
        # compressed_attention``); hoisted here because the extra-KV wiring
        # below is gated on it.
        logical_mode = getattr(
            forward_batch, "actual_forward_mode", forward_batch.forward_mode
        )
        # [REPAIR dsa-ratio-1-2] This plugin override only wired the extra
        # (long-range compressed/indexed) KV path for ratio 4/128. DeepSeek
        # V4.1-Flash uses only compress_ratios {0,1,2}, so every long-range
        # (ratio 1/2) layer fell through to window-only (extra_indices=None ->
        # com_topk=0) and decode could not attend prompt context older than
        # the 128-token sliding window -> correct <128-token prompts but
        # garbled long (GSM8K ~546-token) prompts. Mirror the upstream core
        # backend and use the generic per-ratio accessors for ALL non-zero
        # ratios (ratio 4/128 keep identical values via these accessors).
        #
        # [MODE GATE] This repair must hold in EVERY mode. Upstream
        # (sglang/srt/layers/attention/deepseek_v4_backend.py, the
        # ``if compress_ratio != 0:`` block) wires the extra compressed KV
        # unconditionally, and EXTEND needs it exactly like decode: with it
        # dropped, every prefill query on a ratio-1/2 layer attends only its
        # 128-token SWA window, i.e. silently loses all long-range context.
        # A previous revision defaulted to a decode-only gate (``auto``) to
        # dodge the ~13.9% prefill cost measured 2026-10-19; that premise
        # ("the repair buys no correctness in prefill") was WRONG -- it only
        # ever looked fast because prefill numerics were degraded on 38 of
        # the 40 layers. There is no operating point without it, so the
        # ``SGLANG_KUNLUN_DSA_RATIO_REPAIR`` switch (off / decode / all) was
        # removed 2026-10-23: the other two values only made it possible to
        # run a numerically wrong prefill by accident -- the image default
        # (decode-only) is exactly what produced the bogus 29.9k reading --
        # and an env whose every non-default value is wrong is a foot-gun,
        # not a knob. The dead {4,128} accessor branch went with it (it was
        # unreachable once the repair covers every mode). Restore either
        # from git history if the cost of the correct path has to be
        # re-measured (roadmap: operator-side compressed attention, see
        # r59-operator-req/REQ-compressed_attention-20261023.md).
        extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
        if compress_ratio != 0:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)
            extra_topk_lengths = core_attn_metadata.sparse_topk_lengths(compress_ratio)

        # [kunlun-adapt] pool attr renamed swa_window_size -> swa_page_size
        # (slots per SWA page); same value/meaning.
        swa_window_size = token_to_kv_pool.swa_page_size
        assert swa_k_cache.ndim == 2
        k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
        swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
            swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
        )
        if extra_k_cache is not None:
            # [REPAIR dsa-ratio-1-2] Size the extra-cache view from the pool's
            # per-layer compressed page size / bytes-per-token (mirrors upstream
            # core backend) instead of a hardcoded {4,128} table, so ratio 1/2
            # layers get a correctly shaped compressed KV view.
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_total_dim = token_to_kv_pool.get_extra_key_bytes_per_token(layer_id)
            extra_k_cache = extra_k_cache[
                :, : extra_page_size * extra_total_dim
            ].view(
                extra_k_cache.shape[0],
                extra_page_size,
                1,
                extra_total_dim,
            )
        swa_page_indices = core_attn_metadata.swa_page_indices
        swa_topk_lengths = core_attn_metadata.swa_topk_lengths

        def match_num_queries(x, value):
            if x is None or x.shape[0] == q.shape[0]:
                return x
            if x.shape[0] > q.shape[0]:
                return x[: q.shape[0]]
            return _pad_tensor_to_size(x, q.shape[0], value=value)

        swa_page_indices = match_num_queries(swa_page_indices, value=0)
        swa_topk_lengths = match_num_queries(swa_topk_lengths, value=1)
        extra_indices = match_num_queries(extra_indices, value=-1)
        extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)

        if q.ndim == 3:
            q = q.unsqueeze(1)
        if swa_page_indices.ndim == 2:
            swa_page_indices = swa_page_indices.unsqueeze(1)
        if extra_indices is not None and extra_indices.ndim == 2:
            extra_indices = extra_indices.unsqueeze(1)

        assert attn_sink is not None
        # NOTE: upstream also fetches ``get_flashmla_metadata(compress_ratio)``
        # here to feed the kernel's tile scheduler. The Kunlun ops take no such
        # argument and the accessor is side-effect free, so it is dropped.
        assert (
            swa_page_indices.shape[-1] % 64 == 0
        ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
        if extra_indices is not None:
            assert (
                extra_indices.shape[-1] % 64 == 0
            ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

        if forward_batch.forward_mode.is_extend_without_speculative() and (
            q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD or _sparse_prefill_opt()
        ):
            return self._forward_prefill_sparse(
                q=q,
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
            )
        # ---- inlined ``flash_mla_with_kvcache`` ---------------------------
        is_extend = logical_mode.is_extend_without_speculative()
        # Only DECODE / TARGET_VERIFY / EXTEND can drive the Kunlun native op
        # Everything else (e.g. draft-extend) uses the pure-torch
        # reference path below.
        if logical_mode.is_decode() or logical_mode.is_target_verify() or is_extend:
            return self._kunlun_compressed_attention(
                q=q,
                logical_mode=logical_mode,
                swa_k_cache=swa_k_cache,
                swa_page_indices=swa_page_indices,
                swa_window_size=swa_window_size,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
                forward_batch=forward_batch,
            )
        else:
            # Pure-torch reference for the modes the native op cannot serve.
            n_q, _s_q, h_q, d_qk = q.shape
            q_flat = q.reshape(n_q, h_q, d_qk)
            keys, valid = self._dequant_sparse_keys(
                swa_k_cache, swa_page_indices.reshape(n_q, -1), swa_topk_lengths
            )
            if extra_k_cache is not None and extra_indices is not None:
                extra_keys, extra_valid = self._dequant_sparse_keys(
                    extra_k_cache, extra_indices.reshape(n_q, -1), extra_topk_lengths
                )
                keys = torch.cat([keys, extra_keys], dim=1)
                valid = torch.cat([valid, extra_valid], dim=1)
            return self._sparse_mla_attention(
                q_flat, keys, valid, attn_sink, self.softmax_scale, self.head_dim_v
            )

    def _kunlun_compressed_attention(
        self,
        q: torch.Tensor,
        logical_mode,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_window_size: int,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        compress_ratio: Literal[0, 1, 2, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        forward_batch=None,
    ) -> torch.Tensor:
        """Inlined ``flash_mla_with_kvcache``: build the qlod / kvseqlen context
        the retired ContextVar wrapper used to publish, then call the Kunlun op.

        ``q`` is ``(n_q, 1, h_q, d_qk)`` and the result is
        ``(n_q, h_q, head_dim_v)``.
        """
        is_extend = logical_mode.is_extend_without_speculative()
        is_causal = (not _cp_round_robin_split()) if is_extend else True
        # CUDA graph replays only the recorded kernels, so any Python scalar
        # derived from buffer contents here would freeze at capture time.
        is_cuda_graph_mode = logical_mode.is_decode() or logical_mode.is_target_verify()

        seq_lens = core_attn_metadata.seq_lens_casual[: q.shape[0]]
        bs = seq_lens.shape[0]
        q_dtype = q.dtype
        win_cache = _flatten_cache(swa_k_cache)
        com_cache = (
            win_cache.new_empty((0, win_cache.shape[1]))
            if extra_k_cache is None or extra_indices is None
            else _flatten_cache(extra_k_cache)
        )
        if is_cuda_graph_mode:
            c4_runs = None  # [c4 r90] the ratio-4 split is extend-only
            # [KUNLUN CG-SAFE] Use persistent per-bs buffers with a *recorded*
            # in-place refresh. qlod is static (arange). kvseqlen_xpu is refreshed
            # by a copy_ kernel that is captured into the graph and re-runs on
            # every replay, reading the framework-refreshed seq_lens_casual, so
            # the captured compressed_attention always sees fresh per-req KV
            # lengths at a stable address. Fresh per-call allocations here land in
            # the churning graph pool and read stale/OOB memory at replay.
            # [KUNLUN CG-SAFE bs>1] kvseqlen_cpu must carry the REAL per-request
            # length, NOT MAX_SEQ_LEN_FOR_CAPTURE: the op sizes its per-row KV walk
            # from the HOST length, so a huge host value makes it walk far past the
            # window/topk index columns -> reads win_indices/com_indices out of
            # bounds (benign slack at bs=1, illegal access at bs>1). We init it to
            # 1 (the capture fill value) and refresh it (host + device) OUT of graph
            # in init_forward_metadata_out_graph before every replay, mirroring
            # hunheyun _make_lod + _refresh_graph_host_lengths.
            aux = self._attention_decode_aux.get(bs)
            if aux is None:
                _qlod_cpu = torch.arange(bs + 1, dtype=torch.int32)
                # Build qlod on-device directly: `_qlod_cpu.to(device)` would be an
                # H2D copy, illegal when this first-alloc path runs inside capture.
                _qlod_xpu = torch.arange(bs + 1, dtype=torch.int32, device=q.device)
                _kvseqlen_cpu = torch.ones(bs, dtype=torch.int32)
                _kvseqlen_xpu = torch.ones(bs, dtype=torch.int32, device=q.device)
                aux = (_qlod_cpu, _qlod_xpu, _kvseqlen_cpu, _kvseqlen_xpu)
                self._attention_decode_aux[bs] = aux
            qlod_cpu, qlod_xpu, kvseqlen_cpu, kvseqlen_xpu = aux
            # [REPAIR kvseqlen-stale] The persistent aux KV-length buffers are
            # only refreshed to the real per-request seq_lens by
            # init_forward_metadata_out_graph, which runs for cuda-graph REPLAY.
            # In eager decode (--disable-cuda-graph) that hook never runs, so the
            # buffers stay at the all-ones capture placeholder and the native op
            # walks only kv=1 row -> every decode step attends ONLY the current
            # token and drops all prior context. Refresh host+device from the real
            # seq_lens whenever NOT inside graph capture (safe outside capture).
            # [CG-DETECT] ``torch.cuda.is_current_stream_capturing()`` returns
            # False inside an XPU capture (see ``_capturing_cuda_graph``), so
            # ask the framework capture flag first: the refreshes below issue
            # pageable H2D/D2H copies, which are illegal in a graph.
            _capturing_now = _capturing_cuda_graph()
            if not _capturing_now:
                _real = torch.clamp(seq_lens.to(dtype=torch.int32), min=0)
                _real_cpu = _real.to("cpu", non_blocking=False)
                _n = min(_real_cpu.shape[0], kvseqlen_cpu.shape[0])
                kvseqlen_cpu[:_n] = _real_cpu[:_n]
                kvseqlen_xpu[:_n] = _real.to(kvseqlen_xpu.device)[:_n]
            # [KUNLUN CG-SAFE] The device kvseqlen is refreshed ONLY out of
            # graph (init_forward_metadata_out_graph), never recorded in-graph.
            # Two in-graph copy variants were tried and BOTH regressed steady
            # replay: (a) copying from the in-graph-rebuilt seq_lens_casual
            # (frozen capture-time source) and (b) copying from a
            # forward_batch.seq_lens view à la hunheyun _make_lod — on this
            # build the recorded D2D copy still baked a stale source and every
            # steady single request drifted. The op tolerates the aux[3]
            # placeholder (glue-rebuilt metadata drives correctness).

            # [KUNLUN CP int16 guard] The native compressed_attention op consumes
            # the per-token KV length as a signed 16-bit quantity internally: a
            # token whose cumulative position reaches 32768 (= 2**15) overflows
            # and the op reads out of bounds -> illegal memory access. Measured
            # precisely: clean at kvmax 32767, crash at kvmax 32768. Prefill CP
            # hands each rank the GLOBAL cumulative positions, so any prompt whose
            # length crosses 32768 always trips this. The op only uses
            # min(kvseqlen, max_window_size) for its window walk, so any value >=
            # max_window_size is equivalent. The TRUE max is kept for com_topk so
            # the compressed path is untouched. Never clamp the cuda-graph decode
            # buffers (they are refreshed in place; an H2D inside capture is
            # illegal). The non-graph path does the same inside
            # ``_build_kl_attn_prep``.
            _kvseqlen_true_max = (
                int(kvseqlen_cpu.max().item()) if kvseqlen_cpu.numel() else 0
            )
            win_indices = swa_page_indices.squeeze(1).contiguous()
            win_size = min(swa_window_size, win_indices.shape[1])
            win_indices = win_indices[:, :win_size].contiguous()
            if extra_k_cache is None or extra_indices is None:
                com_indices = win_indices.new_empty((win_indices.shape[0], 0))
                com_ratio = 1
                com_topk = 0
            else:
                com_indices = extra_indices.squeeze(1).contiguous()
                com_ratio = compress_ratio
                max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
                com_topk = max(max_seq_len // com_ratio, 1)
                com_topk = min(com_topk, com_indices.shape[1])
                com_indices = com_indices[:, :com_topk].contiguous()

            # [KUNLUN CG-SAFE] Clamp out-of-range index slots to a VALID cache
            # row instead of -1. Under cuda-graph capture the op walks the full
            # window/topk index width for every row (its per-row loop bound comes
            # from the host kvseqlen placeholder), so any -1 padding slot it
            # visits would dereference cache[-1] -> out-of-bounds (benign slack at
            # bs=1, an illegal access at bs>1 that surfaces at the next TP
            # all-reduce). A clamped index always addresses a real row; the
            # per-request device kvseqlen_xpu mask then zeroes those padding slots
            # out, so the output is unchanged. Mirrors hunheyun
            # kunlun_deepseek_v4_backend.py:1582 which clamps extra_indices to
            # [0, extra_cache.shape[0]-1].
            win_indices = win_indices.clamp(
                min=0, max=win_cache.shape[0] - 1
            ).contiguous()
            if com_indices.numel() and com_cache.shape[0] > 0:
                com_indices = com_indices.clamp(
                    min=0, max=com_cache.shape[0] - 1
                ).contiguous()
        else:
            # [KL_ATTN_PREP] qlod / kvseqlen / window+compressed index prep is a
            # pure function of ``core_attn_metadata`` (shared by every layer of
            # this forward) and of the pool shapes, so memoise it on the
            # metadata object instead of rebuilding it 40x per step.
            dual_ok = _c4_dual_ok(q, win_cache, self.head_dim_v)
            prep_key = (
                q.shape[0],
                compress_ratio,
                bool(is_causal),
                int(swa_window_size),
                int(win_cache.shape[0]),
                int(com_cache.shape[0]),
                q.device,
            )
            (
                qlod_cpu,
                qlod_xpu,
                kvseqlen_cpu,
                kvseqlen_xpu,
                _kvseqlen_true_max,
                win_indices,
                win_size,
                com_indices,
                com_ratio,
                com_topk,
                c4_runs,
            ) = _kl_attn_prep(
                core_attn_metadata,
                prep_key,
                lambda: _build_kl_attn_prep(
                    core_attn_metadata,
                    q,
                    swa_page_indices,
                    swa_window_size,
                    extra_k_cache,
                    extra_indices,
                    compress_ratio,
                    is_causal,
                    win_cache,
                    com_cache,
                    dual_ok,
                ),
            )

        q_3d = q.squeeze(1)
        if q_3d.dtype != win_cache.dtype:
            q_3d = _xsg_cast(q_3d, win_cache.dtype)
        if not q_3d.is_contiguous():
            q_3d = q_3d.contiguous()

        out = torch.empty(
            (q_3d.shape[0], q_3d.shape[1], self.head_dim_v),
            dtype=q_3d.dtype,
            device=q.device,
        )
        # [host-diet r85] ``max_logits`` / ``lse`` are pure outputs of the op and
        # are dead once it returns, but the extend path used to allocate both on
        # every layer (80 ``aten::empty*`` dispatches per forward).  Reuse a
        # per-shape scratch outside cuda-graph mode; decode keeps the old
        # allocation because those tensors are captured into the graph.
        if is_cuda_graph_mode or not _ATTN_AUX_SCRATCH:
            max_logits = torch.empty(q_3d.shape[:2], dtype=torch.float32, device=q.device)
            lse = torch.empty_like(max_logits)
        else:
            max_logits = _attn_aux("max_logits", q_3d.shape[:2], q.device)
            lse = _attn_aux("lse", q_3d.shape[:2], q.device)
        # [host-diet r85] The launch metadata is built once per forward by
        # ``_build_kl_attn_prep`` and is already contiguous (arange / clamp / .to
        # outputs), so the five ``.contiguous()`` calls here were five no-op
        # ``aten::contiguous`` dispatches per layer (~200 per forward) for
        # nothing.  Guard instead: a non-contiguous input still gets copied.
        qlod_cpu_c = qlod_cpu if qlod_cpu.is_contiguous() else qlod_cpu.contiguous()
        qlod_xpu_c = qlod_xpu if qlod_xpu.is_contiguous() else qlod_xpu.contiguous()
        kvseqlen_cpu_c = (
            kvseqlen_cpu if kvseqlen_cpu.is_contiguous() else kvseqlen_cpu.contiguous()
        )
        kvseqlen_xpu_c = (
            kvseqlen_xpu if kvseqlen_xpu.is_contiguous() else kvseqlen_xpu.contiguous()
        )
        attn_sink_c = attn_sink if attn_sink.is_contiguous() else attn_sink.contiguous()
        if q_3d.shape[1] == 64:
            if c4_runs is None:
                torch.ops.xspeedgate_ops.compressed_attention(
                    q_3d,
                    win_cache,
                    win_indices,
                    com_cache,
                    com_indices,
                    out,
                    max_logits,
                    lse,
                    qlod_cpu_c,
                    qlod_xpu_c,
                    kvseqlen_cpu_c,
                    kvseqlen_xpu_c,
                    self.softmax_scale,
                    is_causal,
                    win_size,
                    com_ratio,
                    com_topk,
                    attn_sink_c,
                    side_stream=-1,
                )
            else:
                # [c4 dual-stream r90] One call per run of equal ratio.  A row
                # slice of every operand is still contiguous, so the split is
                # pure views plus one extra launch per run.
                for start, end, ratio in c4_runs:
                    ql_cpu, ql_xpu = _c4_qlod(end - start, q.device)
                    torch.ops.xspeedgate_ops.compressed_attention(
                        q_3d[start:end],
                        win_cache,
                        win_indices[start:end],
                        com_cache,
                        com_indices[start:end],
                        out[start:end],
                        max_logits[start:end],
                        lse[start:end],
                        ql_cpu,
                        ql_xpu,
                        kvseqlen_cpu[start:end],
                        kvseqlen_xpu[start:end],
                        self.softmax_scale,
                        is_causal,
                        win_size,
                        ratio,
                        com_topk,
                        attn_sink_c,
                        side_stream=-1,
                    )
        else:
            kunlun_ops.hybrid_attention(
                q=q_3d,
                win_kv_cache=win_cache,
                win_indices=win_indices,
                com_kv_cache=com_cache,
                com_indices=com_indices,
                o=out,
                max_logits=max_logits,
                lse=lse,
                qlod_cpu=qlod_cpu,
                qlod_xpu=qlod_xpu,
                kvseqlen_cpu=kvseqlen_cpu,
                kvseqlen_xpu=kvseqlen_xpu,
                sm_scale=self.softmax_scale,
                is_causal=is_causal,
                max_window_size=win_size,
                compress_ratio=com_ratio,
                com_topk=com_topk,
                attn_sink=attn_sink_c,
                side_stream=-1,
            )
        if out.dtype == q_dtype:
            return out
        if (
            _ATTN_OUT_KEEP_FP16
            and is_extend
            and out.dtype == torch.float16
            and q_dtype == torch.bfloat16
        ):
            # fp16 straight through to the inverse RoPE + wo_a's fp16 GEMM.
            return out
        return _xsg_cast(out, q_dtype)

    def _forward_prefill_sparse(
        self,
        q: torch.Tensor,
        layer_id: int,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        forward_batch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream ``_forward_prefill_sparse`` with ``flash_mla_sparse_fwd``
        replaced by pure-torch sparse attention over the gathered workspace.
        """
        # q is (b, 1, h_q, d_qk); the sparse attention takes (s_q, h_q, d_qk).
        q_flat = q.squeeze(1)

        cache = self.forward_metadata.sparse_prefill_cache
        if cache is None:
            # [kunlun-adapt] 0.5.14+ 把 build 参数（query_lens/query_pos/
            # max_seq_len/total_swa）抽进基类 _build_sparse_prefill_chunk_cache。
            # [kunlun-adapt] 0.5.20 dropped the core_attn_metadata parameter again:
            # upstream's signature is now
            #   _build_sparse_prefill_chunk_cache(self, forward_batch, *, num_qo_tokens)
            # (sglang/srt/layers/attention/deepseek_v4_backend.py:1473), so passing
            # core_attn_metadata positionally raises
            #   TypeError: ... takes 2 positional arguments but 3 positional arguments
            #   (and 1 keyword-only argument) were given
            # and kills every scheduler. Only reachable on the large-query sparse
            # prefill path, which is why it stays hidden until a configuration that
            # selects it (e.g. running without --enable-nsa-prefill-context-parallel).
            cache = self._build_sparse_prefill_chunk_cache(
                forward_batch, core_attn_metadata, num_qo_tokens=q_flat.shape[0]
            )
            self.forward_metadata.sparse_prefill_cache = cache
        # Resolve the workspace + indices for this ratio, then dequant
        # SWA + compressed regions directly into the workspace (no torch.cat).
        compressed_slice = None
        extra_k_cache = None
        extra_page_size = None
        flat_token_ids = None
        if compress_ratio == 0:
            # [kunlun-adapt] 0.5.14+ workspace 改由基类 sparse_prefill_workspace
            # 分配器提供（cache 不再持有 c0_workspace/c4_workspace 字段）。
            workspace = self.sparse_prefill_workspace.get(cache.swa_token_ids.shape[0])
            combined_indices = cache.c0_combined_indices
            combined_lens = cache.c0_combined_lens
            swa_slice = workspace
        else:
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            if compress_ratio == 128:
                assert core_attn_metadata.c128_page_indices is not None
                cache.ensure_c128(core_attn_metadata.c128_page_indices)
                flat_token_ids = cache.c128_flat_token_ids
                combined_indices = cache.c128_combined_indices
                combined_lens = cache.c128_combined_lens
            else:
                raw_indices = core_attn_metadata.sparse_raw_indices(compress_ratio)
                assert raw_indices is not None, (
                    f"sparse-prefill c{compress_ratio} path requires the raw "
                    "top-k indices (allocated in init_flashmla_related when "
                    "is_prefill=True)"
                )
                gather = cache.ensure_compressed(
                    compress_ratio, core_attn_metadata.page_table, extra_page_size
                )
                flat_token_ids = gather.flat_token_ids
                combined_indices, combined_lens = cache.combine_compressed(
                    compress_ratio, raw_indices[: cache.num_qo_tokens]
                )
            n_compressed = flat_token_ids.shape[0]
            workspace = self.sparse_prefill_workspace.get(
                n_compressed + cache.swa_token_ids.shape[0]
            )
            compressed_slice = workspace[:n_compressed]
            swa_slice = workspace[n_compressed:]

        if compressed_slice is not None:
            dequantize_k_cache_paged(
                extra_k_cache,
                flat_token_ids,
                page_size=extra_page_size,
                out=compressed_slice,
            )
        dequantize_k_cache_paged(
            token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            cache.swa_token_ids,
            page_size=cache.swa_page_size,
            out=swa_slice,
        )
        # ---- inlined ``flash_mla_sparse_fwd`` (pure torch) ------------------
        # q_flat: (s_q, h_q, d_qk); workspace: (s_kv, 1, d_qk) bf16;
        # combined_indices: (s_q, topk) rebased locs into the workspace.
        s_q = q_flat.shape[0]
        kv_flat = workspace.reshape(workspace.shape[0], -1)
        idx = combined_indices.reshape(s_q, -1).to(torch.int64)
        ar = torch.arange(idx.shape[1], device=q.device).unsqueeze(0)
        # NSA prefill context parallelism pads the per-query index rows, and the
        # padded rows can carry indices past the gathered workspace. An
        # out-of-range gather reads garbage (nondeterministic NaN rows) or faults
        # with an illegal memory access, so treat out-of-range slots as invalid
        # and clamp the gather index.
        in_range = (idx >= 0) & (idx < kv_flat.shape[0])
        if combined_lens is not None:
            valid = (ar < combined_lens.to(torch.int64).unsqueeze(1)) & in_range
        else:
            valid = in_range
        safe_idx = torch.where(in_range, idx, torch.zeros_like(idx))
        keys = kv_flat[safe_idx.reshape(-1)].reshape(s_q, idx.shape[1], -1)
        return self._sparse_mla_attention(
            q_flat, keys, valid, attn_sink, self.softmax_scale, self.head_dim_v
        )

    def _dequant_paged_torch(
        self,
        buf_u8_flat: torch.Tensor,
        flat_idx: torch.Tensor,
        page_size: int,
        bytes_per_page: int,
    ) -> torch.Tensor:
        """Decode N tokens from the v4 paged layout to (N, 512) bf16, pure torch."""
        device = buf_u8_flat.device
        loc = flat_idx.to(torch.int64)
        N = loc.shape[0]
        page_idx = loc // page_size
        in_page = loc % page_size
        page_byte_base = page_idx * bytes_per_page
        token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
        s_offset_bytes = page_size * NOPE_ROPE_BYTES
        token_scale_base = (
            page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
        )

        # nope fp8 bytes -> float via LUT
        nope_byte = (
            token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
        )
        nope_u8 = buf_u8_flat[nope_byte.reshape(-1)].to(torch.int64)
        nope_f = _e4m3_lut(device)[nope_u8].reshape(N, DIM_NOPE)

        # ue8m0 scales -> 2^(s-127), zeroing subnormals
        scale_byte = (
            token_scale_base[:, None]
            + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
        )
        scale_u8 = buf_u8_flat[scale_byte.reshape(-1)].reshape(N, NUM_SCALE_TILES)
        scale_pow2 = torch.exp2(scale_u8.to(torch.float32) - 127.0)
        scale_pow2 = torch.where(
            scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
        )
        scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
        nope = (nope_f * scale_full).to(torch.bfloat16)

        # rope bf16: reinterpret the same bytes as bf16 and gather
        buf_bf16 = buf_u8_flat.view(torch.bfloat16)
        rope_bf16_base = (token_data_base + DIM_NOPE) // 2
        rope_idx = (
            rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
        )
        rope = buf_bf16[rope_idx.reshape(-1)].reshape(N, DIM_ROPE)

        out = torch.empty((N, _HEAD_DIM), dtype=torch.bfloat16, device=device)
        out[:, :DIM_NOPE] = nope
        out[:, DIM_NOPE:] = rope
        return out

    def _dequant_sparse_keys(
        self,
        k_cache_4d: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize v4 FP8-packed keys at per-query token locations.

        Returns ``(keys (n_q, K, 512) bf16, valid_mask (n_q, K) bool)``; invalid
        slots are zeroed.
        """
        num_pages, page_size, _, _total_dim = k_cache_4d.shape
        # k_cache_4d is a strided slice-view of the pool's contiguous
        # (num_pages, bytes_per_page_padded) uint8 buffer. Materializing it with
        # .contiguous() triggers an unsupported strided-uint8 copy on Kunlun XPU,
        # so read from the contiguous base buffer using its real page stride.
        base = k_cache_4d
        while base._base is not None:
            base = base._base
        # If the pool buffer was realigned (sglang-kunlun PD), `base` is the padded
        # flat storage; recover the real KV region [off, numel).
        _lg = getattr(base, "_kunlun_logical", None)
        if _lg is not None and len(_lg) == 2:
            base = base.narrow(0, _lg[0], _lg[1])
        assert base.is_contiguous(), "expected contiguous base kv buffer"

        ar = torch.arange(indices.shape[1], device=indices.device).unsqueeze(0)
        idx_i64 = indices.to(torch.int64)
        valid_mask = (ar < lengths.to(torch.int64).unsqueeze(1)) & (idx_i64 >= 0)

        # bf16 KV (align-to-reference): direct 512-bf16 gather, no fp8 dequant.
        if base.dtype in (torch.bfloat16, torch.float16):
            flat = base.reshape(-1, _HEAD_DIM)
            safe = torch.where(idx_i64 >= 0, idx_i64, torch.zeros_like(idx_i64))
            safe = safe.clamp(0, flat.shape[0] - 1)  # KUNLUN_IDX_CLAMP idle/dummy safety
            keys = (
                flat[safe.reshape(-1)]
                .view(indices.shape[0], indices.shape[1], _HEAD_DIM)
                .to(torch.bfloat16)
            )
            # Slots this rank never wrote (other CP ranks' pages, or the page-aligned
            # tail) hold uninitialised memory that reads back as NaN. The mask only
            # zeroes known-invalid slots, so scrub the rest instead of letting a NaN
            # key wipe out the whole softmax row.
            keys = torch.nan_to_num(keys) * valid_mask.unsqueeze(-1)
            return keys, valid_mask

        # The pool stores bytes as uint8, but get_key_buffer().view(dtype) may hand
        # us an fp8/bf16-typed view. Reinterpret the flat storage as raw bytes.
        base_u8 = base.reshape(-1).view(torch.uint8)
        assert base_u8.numel() % num_pages == 0, "base buffer not page-aligned"
        bytes_per_page = base_u8.numel() // num_pages

        safe_idx = torch.where(idx_i64 >= 0, idx_i64, torch.zeros_like(idx_i64))
        safe_idx = safe_idx.clamp(0, max(num_pages * page_size - 1, 0))  # KUNLUN_IDX_CLAMP
        deq = self._dequant_paged_torch(
            base_u8, safe_idx.reshape(-1).to(torch.int32), page_size, bytes_per_page
        )
        keys = deq.view(indices.shape[0], indices.shape[1], _HEAD_DIM)
        return keys, valid_mask

    def _sparse_mla_attention(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        valid_mask: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        softmax_scale: float,
        head_dim_v: int,
    ) -> torch.Tensor:
        """MLA softmax attention with per-head attention-sink virtual key.

        ``q`` is (n_q, h_q, 512), ``keys`` (n_q, K, 512) bf16 (== values in absorbed
        MLA), ``valid_mask`` (n_q, K); returns (n_q, h_q, head_dim_v) in q.dtype.
        """
        n_q, h_q, _d = q.shape
        # KUNLUN_ATTN_IDLE_GUARD: DP-attention idle/dummy forward can hand us n_q==0
        # or K==0. The degenerate softmax/reduce over a size-0 dim illegal-accesses
        # on Kunlun XPU. The idle output is unused, so return zeros of the right shape.
        if n_q == 0 or keys.shape[1] == 0:
            return q.new_zeros((n_q, h_q, head_dim_v))
        qf = q.float()
        kf = keys.float()  # (n_q, K, d)

        # scores: (n_q, h_q, K)
        scores = torch.einsum("qhd,qkd->qhk", qf, kf) * softmax_scale
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), neg_inf)

        if attn_sink is not None:
            sink = attn_sink.float().view(1, h_q, 1).expand(n_q, h_q, 1)
            scores_ws = torch.cat([scores, sink], dim=-1)  # (n_q, h_q, K+1)
            probs = torch.softmax(scores_ws, dim=-1)[..., :-1]  # drop virtual key
        else:
            probs = torch.softmax(scores, dim=-1)

        # For a query with zero valid keys softmax over all -inf yields nan; zero it.
        any_valid = valid_mask.any(dim=-1)  # (n_q,)
        probs = torch.where(any_valid.view(n_q, 1, 1), probs, torch.zeros_like(probs))

        out = torch.einsum("qhk,qkd->qhd", probs, kf)  # (n_q, h_q, d)
        return out[..., :head_dim_v].to(q.dtype)

    # ------------------------------------------------------------------
    # DSV4.1 low-ratio indexer: xspeedgate fused decode path
    # Replaces the upstream _low_ratio_index_topk_sm90_decode which falls
    # through to the pure-torch fp4_index_logits_decode shim (~1s per call).
    # xspeedgate_ops.dsv41_low_ratio_indexer fuses queries→scoring→topk
    # into a single C++ kernel.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # CP prefill: project the compressor on this rank's own rows first
    # ------------------------------------------------------------------
    # Upstream ships the whole prompt's hidden state to every CP rank and runs
    # the full-width ratio-1/2 projection on all of it. ``Compressor.project``
    # is row-wise, so projecting this rank's rows and gathering only the narrow
    # (kv, score) rows produces the same global rows for 1/cp_size of the GEMM
    # and far less gather traffic. ``KL_CP_PROJECT_FIRST=0`` (or the
    # /tmp/kl_cp_off marker for one process) falls back to the base method.
    def _forward_low_ratio_sources_cp(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch,
        run_compressor,
        run_indexer,
    ) -> None:
        if not _cp_project_first():
            return super()._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                run_compressor=run_compressor,
                run_indexer=run_indexer,
            )
        cp_meta = forward_batch.attn_cp_metadata
        total = int(cp_meta.total_seq_lens)
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens_cpu = tail.local_lens_cpu
            req_global, pos_global = tail.req_global, tail.pos_global
        else:
            q_lens_cpu = interleave_rows_per_request(
                _as_int_list(forward_batch.extend_seq_lens_cpu),
                get_parallel().attn_cp_rank,
                get_parallel().attn_cp_size,
            )
            req_global = _dsv41_sparse.token_req_indices(
                forward_batch, num_tokens=total
            )
            pos_global = forward_batch.positions[:total].to(torch.int64)
        num_local = sum(q_lens_cpu)
        if run_compressor and layer.compressor is not None:
            kv_local, score_local = layer.compressor.project(x[:num_local])
            stream = torch.cuda.current_stream()
            kv_global = cp_materialize_global_token_order(
                kv_local, forward_batch, stream
            )[:total]
            score_global = (
                None
                if score_local is None
                else cp_materialize_global_token_order(
                    score_local, forward_batch, stream
                )[:total]
            )
            self._low_ratio_compress_torch(
                layer,
                None,
                req_global,
                pos_global,
                projected=(kv_global, score_global),
            )
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk_dense(
                layer,
                x[:num_local],
                q_lora[:num_local],
                positions[:num_local].to(torch.int64),
                forward_batch,
                torch.tensor(q_lens_cpu, dtype=torch.int32, device=x.device),
                q_lens_cpu,
            )

    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        bs = req.shape[0]
        assert pos.shape[0] == bs, (
            f"decode expects one token per request, {pos.shape=} {bs=}"
        )
        if bs == 0:
            return
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None
        lmax = min(
            metadata.max_compressed_seq_len,
            self.req_to_token.shape[1] // ratio,
        )
        if lmax == 0:
            return

        # xspeedgate requires window to be divisible by candidate_block_size.
        cbs = indexer.candidate_block_size
        if cbs > 0 and lmax % cbs != 0:
            lmax = ((lmax + cbs - 1) // cbs) * cbs
            # Clamp back to req_to_token capacity so slot mapping stays in-bounds.
            max_capacity = self.req_to_token.shape[1] // ratio
            lmax = min(lmax, (max_capacity // cbs) * cbs)
        if lmax == 0:
            return

        # --- Prepare inputs for xspeedgate fused kernel ---
        q = indexer.queries(q_lora, layer.freqs_cis[pos])     # [R, 32, 128] bf16
        weights = indexer.head_weights(x)                       # [R, 32] bf16
        k_pool = pool.get_index_k_with_scale_buffer(layer.layer_id)
        fp4_lut = _fp4_lut(pos.device)
        req_to_token = self.req_to_token.to(torch.int32).contiguous()
        req_i = req.to(torch.int32).contiguous()
        pos_i32 = pos.to(torch.int32).contiguous()
        topk = min(indexer.index_topk, lmax)

        is_publish = indexer.is_candidate_source
        is_consume = indexer.uses_candidates and not indexer.is_candidate_source

        # Build candidate mask for consume layers
        if is_consume:
            consume_mask = _published_masks()(
                self.forward_metadata.candidate_metadata
            ).mask
            assert torch.is_tensor(consume_mask) and consume_mask.shape[0] == bs
            # Pad/trim to window width
            if consume_mask.shape[1] < lmax:
                cand_mask = F.pad(
                    consume_mask, (0, lmax - consume_mask.shape[1]), value=False
                )
            else:
                cand_mask = consume_mask[:, :lmax]
            cand_mask = cand_mask.contiguous()
        else:
            cand_mask = torch.zeros(
                bs, lmax, dtype=torch.bool, device=pos.device
            )

        # --- Call fused xspeedgate kernel ---
        fused_page_indices, published = (
            torch.ops.xspeedgate_ops.dsv41_low_ratio_indexer(
                q.contiguous(),
                weights.to(torch.bfloat16).contiguous(),
                k_pool.contiguous(),
                fp4_lut,
                req_to_token,
                req_i,
                pos_i32,
                cand_mask,
                ratio,
                topk,
                lmax,
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
                is_publish,
                is_consume,
            )
        )
        # fused_page_indices: [R, topk] int32, ascending, -1 padded
        # published:          [R, lmax]  uint8, candidate block bitmap

        # --- Write outputs ---
        page_indices[:bs, :topk] = fused_page_indices

        if raw_indices is not None:
            # xspeedgate returns page (physical slot) indices, not raw
            # (logical compressed position) indices. Recover raw positions
            # from the slots: raw = page_indices * ratio, then look up in
            # req_to_token to reverse-map. For the decode path the upstream
            # code does idx.sort → slots.gather → page_indices, where idx
            # is the logical position. Since page_indices == slots[idx],
            # and slots[j] = req_to_token[r, j*ratio] // ratio, the raw
            # position is not directly recoverable without the slot table.
            # However, on the sm90 decode path, raw_indices is only
            # allocated for prefill metadata (sparse_prefill_utils) and is
            # None for decode. This branch is defensive only.
            pass

        if is_publish:
            # Convert uint8 bitmap → bool mask for CandidateMasks
            mask = published.to(torch.bool)
            self.forward_metadata.candidate_metadata = CandidateMasks(mask=mask)


class KunlunDeepseekV4MultiStepBackend(_V1CompressorBackendMixin, DeepseekV4MultiStepBackend):
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