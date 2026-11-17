"""W8A8-INT8 engram embedding table for DeepSeek-V4.1 on Kunlun/XPU.

Upstream `EngramEmbedding` stores the table as an fp8_e4m3 payload with e8m0
scales, one per 32-element block (the FP8 release). The W8A8-INT8 release stores

    layers.N.engram.embed.weight        I8   [num_embeddings, dim]
    layers.N.engram.embed.weight_scale  BF16 [num_embeddings, 1]

i.e. compressed-tensors channel-wise int8: one scale per row. The row sharding,
host-table machinery and the all-reduce contract are identical; only the payload
dtype, the scale layout and the dequant arithmetic differ, so this class keeps the
upstream base for the fp8 format and adds the int8 layout alongside it.

Dequant: ``values = weight[ids].float() * weight_scale[ids].float()`` -> bf16,
which is exactly what the compressed-tensors channel-wise scale means.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from torch import nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import EngramEmbedding


_INT8_CACHE: Optional[bool] = None
# Persistent pinned staging for the pipelined host gather, grown on demand.
_PIN: dict = {"rows": None, "scales": None}
_GATHER_POOL = None


def _env_int(name: str, default: int) -> int:
    """Read an integer tuning option, using the default for malformed values."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _gather_pool(threads: int):
    """Process-wide thread pool for the chunked host gather."""
    global _GATHER_POOL
    if _GATHER_POOL is None or _GATHER_POOL._max_workers != threads:
        from concurrent.futures import ThreadPoolExecutor

        _GATHER_POOL = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="engram-gather"
        )
    return _GATHER_POOL


def _pinned_staging(n: int, dim: int):
    """`[n, dim]` int8 + `[n, 1]` bf16 pinned buffers, reallocated only to grow.

    Pinned memory is what lets the per-chunk H2D actually overlap with the next
    chunk's gather; a copy out of ordinary pageable memory is synchronous.
    """
    rows, scales = _PIN["rows"], _PIN["scales"]
    if rows is None or rows.shape[0] < n or rows.shape[1] != dim:
        rows = torch.empty(n, dim, dtype=torch.int8).pin_memory()
        scales = torch.empty(n, 1, dtype=torch.bfloat16).pin_memory()
        _PIN["rows"], _PIN["scales"] = rows, scales
    return rows[:n], scales[:n]


def _stage_lookup(embed, indices: torch.Tensor):
    """Serve the lookup from the pre-staged HBM buffer, or None to gather as before.

    See `dsv41_engram_stage.py`: with the table in host memory the CPU gather
    cannot be captured into a decode CUDA graph, so the rows a step needs are
    prefetched into a fixed buffer before replay and the in-graph lookup becomes
    a reshape of that buffer. Imported lazily to keep the hook import order free.
    """
    try:
        from sglang_kunlun.hooks.models.dsv41_engram_stage import get_stager
    except Exception:  # noqa: BLE001 - the stage is an optimization, never a dep
        return None
    stager = get_stager()
    if stager is None:
        return None
    return stager.lookup(embed, indices)


def _host_gather_batched_sharded(embed, indices: torch.Tensor):
    """`_host_gather_batched` for the `per_rank` host layout (one shard per rank).

    The shared layout gets the batched gather through `EngramEmbedding.forward`,
    but `per_rank` falls through to upstream's `_owned_rows`, whose pointer-based
    `engram_gather` lands on `engram_gather_kunlun` -- a Python loop that reads
    one row at a time with `ctypes.from_address`. A P-side prefill chunk is
    `chunked_prefill_size * 24` rows per engram layer (16384 -> 393216, twice for
    the two layers), so that loop is what shows up as the TTFT regression.

    Same math as the shared path, plus the shard contract upstream relies on:
    ids outside `[row_start, row_start + rows)` must come back as zero rows, and
    the caller's TP all-reduce then reassembles the full lookup.

    Only this rank's own rows are moved. Under prefill CP the caller looks up the
    CP group's whole token range, so with `tp=8` about 1/8 of the ids fall in this
    shard; gathering all of them and zeroing the rest afterwards (what this used
    to do) cost a `n * 8`-byte id D2H, an `n * dim`-byte pinned staging + H2D and
    a device `masked_fill` for eight times the rows the shard owns. Compacting
    first makes every one of those proportional to the owned count instead.
    Disable with `SGLANG_KUNLUN_ENGRAM_COMPACT_OWNED=0`.
    """
    if os.environ.get(
        "SGLANG_KUNLUN_ENGRAM_BATCH_GATHER_SHARDED", "1"
    ).strip().lower() in ("0", "false", "no"):
        return None
    weight = embed.weight.data
    scale = getattr(embed, "weight_scale", None)
    if scale is None:
        return None
    scale = scale.data
    if weight.device.type != "cpu" or scale.device.type != "cpu":
        return None
    if weight.dtype is not torch.int8 or scale.dtype is not torch.bfloat16:
        return None
    if weight.shape[0] != int(embed.rows):
        return None
    device = indices.device
    if os.environ.get("SGLANG_KUNLUN_ENGRAM_COMPACT_OWNED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        return _owned_only_gather(embed, indices, weight, scale, device)
    local = (indices.reshape(-1) - int(embed.row_start)).to("cpu", torch.int64)
    owned = (local >= 0) & (local < int(embed.rows))
    local = local.masked_fill(~owned, 0).contiguous()
    min_rows = _env_int("SGLANG_KUNLUN_ENGRAM_GATHER_MIN_ROWS", 8192)
    if local.numel() >= min_rows:
        rows, scales = _pipelined_host_gather(weight, scale, local, embed.dim, device)
    else:
        rows = weight[local].to(device, non_blocking=True)
        scales = scale[local].to(device, non_blocking=True)
    values = _dequant_on_device(rows, scales, embed.dim)
    values = values.masked_fill(~owned.to(device).unsqueeze(-1), 0)
    return values.view(*indices.shape, embed.dim)


def _owned_only_gather(embed, indices: torch.Tensor, weight, scale, device):
    """Gather just the ids this shard owns and scatter them into a zero buffer.

    `positions` stays on the device -- it is only needed to place the gathered
    rows back -- so the sole host crossing is the owned row ids, 1/8 of the ids
    the CP lookup passes in. `nonzero` already reports its own output length, so
    the compaction costs no extra synchronization beyond the one this path has
    always paid for the id D2H.

    Compacting on the host instead (`local.to("cpu")` first, then a CPU
    `masked_select`) removes the `nonzero` stream drain, but the drain was only
    ever waiting for work the device still owed: measured on the 64K P-side step
    it bought nothing and the CPU pass over one chunk's 393216 ids added ~2.5 ms
    of serial host time per engram layer, which came straight out of TTFT
    (11.0 s -> 11.1 s). Keep the compaction on the device.
    """
    n = indices.numel()
    out = torch.zeros(n, embed.dim, dtype=torch.bfloat16, device=device)
    if n == 0:
        return out.view(*indices.shape, embed.dim)
    local = indices.reshape(-1) - int(embed.row_start)
    owned = (local >= 0) & (local < int(embed.rows))
    positions = torch.nonzero(owned, as_tuple=True)[0]
    owned_rows = local[positions].to("cpu", torch.int64)
    k = owned_rows.numel()
    if k == 0:
        return out.view(*indices.shape, embed.dim)
    min_rows = _env_int("SGLANG_KUNLUN_ENGRAM_GATHER_MIN_ROWS", 8192)
    if k >= min_rows:
        rows, scales = _pipelined_host_gather(
            weight, scale, owned_rows, embed.dim, device
        )
    else:
        rows = weight[owned_rows].to(device, non_blocking=True)
        scales = scale[owned_rows].to(device, non_blocking=True)
    out.index_copy_(0, positions, _dequant_on_device(rows, scales, embed.dim))
    return out.view(*indices.shape, embed.dim)


def _identity_ids(n: int, device) -> torch.Tensor:
    """Create positional dequant IDs without retaining every observed size.

    During capture, arange and its storage belong to the graph's memory pool.
    No eager cache can later grow, replace or overwrite that captured storage.

    Args:
        n: Number of gathered rows.
        device: Device on which the dequant kernel consumes the IDs.

    Returns:
        A fresh int32 tensor containing [0, n).
    """
    return torch.arange(n, dtype=torch.int32, device=device)


def _pipelined_host_gather(weight, scale, idx, dim, device):
    """Chunked gather whose H2D overlaps the next chunk's CPU work.

    One prefill chunk needs `chunked_prefill_size * 24 = 196608` rows, i.e. a
    48 MiB random-access gather over the 92 GiB mmap followed by a 48 MiB H2D.
    Done in one shot those two phases serialize and the gather is single
    threaded. Splitting the row range lets a pool of workers gather different
    slices concurrently while the already-finished slices are in flight to the
    device, and staging through pinned memory is what makes those copies async.

    Chunk and thread counts are tunable; measured on one chunk-sized call
    (196608 rows, dim 256, 8 GiB stand-in table, machine otherwise idle):

        one shot, pageable staging          19.0 ms
        8 chunks / 2 threads                 8.4 ms
        8 chunks / 3 threads                 7.6 ms   <- best
        8 chunks / 4 threads                 8.1 ms   (default, more headroom)
        8 chunks / 6 threads                 9.5 ms
        16 chunks / 4 threads                11.6 ms

    Results are bit-identical to the one-shot gather.
    """
    n = idx.numel()
    chunks = max(1, _env_int("SGLANG_KUNLUN_ENGRAM_GATHER_CHUNKS", 8))
    threads = max(1, _env_int("SGLANG_KUNLUN_ENGRAM_GATHER_THREADS", 4))
    pin_rows, pin_scales = _pinned_staging(n, dim)
    dev_rows = torch.empty(n, dim, dtype=torch.int8, device=device)
    dev_scales = torch.empty(n, 1, dtype=torch.bfloat16, device=device)
    bounds = [(i * n // chunks, (i + 1) * n // chunks) for i in range(chunks)]

    def gather(bound):
        """Fill one disjoint slice of the pinned row and scale buffers."""
        lo, hi = bound
        part = idx[lo:hi]
        torch.index_select(weight, 0, part, out=pin_rows[lo:hi])
        torch.index_select(scale, 0, part, out=pin_scales[lo:hi])
        return bound

    # `index_select` releases the GIL, so the workers really do run in parallel.
    done = _gather_pool(threads).map(gather, bounds) if threads > 1 else map(
        gather, bounds
    )
    for lo, hi in done:
        dev_rows[lo:hi].copy_(pin_rows[lo:hi], non_blocking=True)
        dev_scales[lo:hi].copy_(pin_scales[lo:hi], non_blocking=True)
    return dev_rows, dev_scales


def _dequant_on_device(rows: torch.Tensor, scale: torch.Tensor, dim: int):
    """`int8 rows * per-row bf16 scale -> bf16`, run on the XPU.

    `xspeedgate_ops::engram_gather` is exactly this kernel; feeding it the already
    gathered sub-table with identity ids keeps the documented
    `fp32(w) * fp32(scale) -> bf16` rounding. The torch expression is the fallback
    for packages that do not export the op.
    """
    n = rows.shape[0]
    try:
        return torch.ops.xspeedgate_ops.engram_gather(
            rows, scale, _identity_ids(n, rows.device), dim, dim, 0, n
        )
    except Exception:  # noqa: BLE001
        return rows.to(torch.bfloat16) * scale


def _host_gather_batched(embed, indices: torch.Tensor):
    """One batched gather for the whole call, with the dequant moved to the device.

    The pointer-based `engram_gather` falls through to `engram_gather_kunlun`
    (`sglang_kunlun/kernels/dsv41_low_ratio.py`), which walks `ids.tolist()` and
    reads each row with `ctypes.from_address`. Prefill makes that the dominant
    cost: a chunk of `chunked_prefill_size=8192` tokens needs
    `8192 * 24 = 196608` rows per engram layer, so a 64 K prompt drives ~3.1 M
    Python-level row reads and TTFT goes from ~70 s to ~170 s.

    In the shared layout the table is already a CPU tensor view over the mmap, so
    the CPU only has to do the row gather. Dequantizing there would cost far more
    than the gather itself -- `weight[idx].float()` materializes an fp32
    [196608, 256] temporary (201 MiB) and the bf16 result is another 100 MiB to
    push over PCIe -- so the raw int8 rows and their scales go to the device and
    the multiply happens there. Measured on one chunk-sized call (196608 rows):

        CPU gather + CPU dequant + H2D bf16    124.9 ms
        CPU gather + H2D int8 + device dequant  19.6 ms   (6.4x, bit-identical)

    Returns None when the shapes/dtypes are not the expected int8 + per-row-bf16
    host views, leaving the original path in place.

    Disable with `SGLANG_KUNLUN_ENGRAM_BATCH_GATHER=0`.
    """
    if os.environ.get("SGLANG_KUNLUN_ENGRAM_BATCH_GATHER", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return None
    weight = embed.weight.data
    scale = getattr(embed, "weight_scale", None)
    if scale is None:
        return None
    scale = scale.data
    if weight.device.type != "cpu" or scale.device.type != "cpu":
        return None
    if weight.dtype is not torch.int8 or scale.dtype is not torch.bfloat16:
        return None
    # Ids are global rows into the whole shared table; out-of-range can only be
    # graph padding, whose output is discarded, so clamping is enough.
    idx = indices.reshape(-1).to("cpu", torch.int64).clamp_(0, weight.shape[0] - 1)
    device = indices.device
    # Small calls (eager decode fallback, the tail chunk of a prompt) are not
    # worth splitting: the chunking and hand-off cost more than they save.
    min_rows = _env_int("SGLANG_KUNLUN_ENGRAM_GATHER_MIN_ROWS", 8192)
    if idx.numel() >= min_rows:
        rows, scales = _pipelined_host_gather(weight, scale, idx, embed.dim, device)
    else:
        rows = weight[idx].to(device, non_blocking=True)
        scales = scale[idx].to(device, non_blocking=True)
    values = _dequant_on_device(rows, scales, embed.dim)
    return values.view(*indices.shape, embed.dim)


@plugin_hook(
    "sglang.srt.layers.engram._HostTable._open_shared_fd",
    type=HookType.AROUND,
)
def engram_open_shared_fd_compat(original_fn, self, nbytes, name):
    """Preserve shared memfd/proc-fd lifetime without patching os globally.

    Args:
        original_fn: Upstream shared-file creation method.
        self: Host table owning the TP group.
        nbytes: Required shared-file size in bytes.
        name: Diagnostic name of the anonymous shared file.

    Returns:
        A rank-local file descriptor for the shared table.
    """
    if hasattr(os, "memfd_create"):
        return original_fn(self, nbytes, name)
    from sglang_kunlun.compat import memfd_create

    fd = None
    try:
        owner = None
        if self.group.rank_in_group == 0:
            fd = memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = self.group.broadcast_object(owner, src=0)
        if self.group.rank_in_group == 0:
            return fd
        try:
            return os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                "engram host table: cannot open rank 0's memfd through /proc; "
                "the TP ranks must share a PID namespace"
            ) from e
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise


# Retain the existing dispatch override; diagnostic tensor probes are removed.
#   SGLANG_KUNLUN_ENGRAM_GATE=torch  falls back to the upstream torch fallback (= pre-change behavior)
_ENGRAM_GATE_MODE = os.environ.get("SGLANG_KUNLUN_ENGRAM_GATE", "fused")


@plugin_hook("sglang.srt.layers.engram.engram_gate", type=HookType.AROUND)
def engram_gate_fp16(original_fn, x, kv, q_weight, k_weight, eps, clamp_value):
    """Route the fp16 serving path to the fused op.

    Upstream's ``engram_gate`` guard admits bf16/fp32 io only, so an fp16 run
    always drops to its torch expression (fp32 temporaries, and an OOM on long
    prefills). The XSpeedGate op has same-dtype fp16 kernels, so hand fp16 io to
    the Kunlun replacement, which re-checks the op contract and keeps its own
    torch fallback for the shapes/dtypes the op declines. Everything else stays
    on upstream.

    Args:
        original_fn: Upstream gate implementation.
        x: Residual activations.
        kv: Retrieved engram values.
        q_weight: Query normalization weights.
        k_weight: Key normalization weights.
        eps: Normalization epsilon.
        clamp_value: Gate clamp bound.

    Returns:
        The selected implementation's gated engram output.
    """
    if x.dtype is not torch.float16 or _ENGRAM_GATE_MODE != "fused":
        return original_fn(x, kv, q_weight, k_weight, eps, clamp_value)
    if kv.dtype is not x.dtype:
        # ``wkv``'s quantized output carries the checkpoint's bf16, not the
        # serving dtype; bf16->fp16 is exact for in-range values.
        kv = kv.to(x.dtype)
    from sglang_kunlun.kernels.dsv41_low_ratio import fused_engram_gate_kunlun

    return fused_engram_gate_kunlun(x, kv, q_weight, k_weight, eps, clamp_value)


def _index_has_int8_engram(model_path: str) -> bool:
    """Whether the checkpoint index names `engram.embed.weight_scale`."""
    try:
        for fn in os.listdir(model_path):
            if not fn.endswith(".safetensors.index.json"):
                continue
            with open(os.path.join(model_path, fn)) as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(k.endswith("engram.embed.weight_scale") for k in weight_map)
    except Exception:  # noqa: BLE001
        return False
    return False


def engram_embed_weight_is_int8() -> bool:
    """Whether the engram table is channel-wise int8 (vs the fp8 block layout).

    Decided from, in order: the ``DSV4_ENGRAM_INT8`` override, the checkpoint's
    safetensors index (``MODEL_PATH``), and the HF quantization config. The
    index path matters because ``EngramEmbedding`` is constructed while the model
    is still being built -- ``get_model()`` is not published yet, so a
    config-only probe would silently fall back to the fp8 layout.
    """
    global _INT8_CACHE
    if _INT8_CACHE is not None:
        return _INT8_CACHE

    override = os.environ.get("DSV4_ENGRAM_INT8")
    if override is not None and override.strip() != "":
        _INT8_CACHE = override.strip().lower() not in ("0", "false", "no")
        return _INT8_CACHE

    model_path = os.environ.get("MODEL_PATH") or os.environ.get("DSV4_MODEL_PATH") or ""
    if model_path and _index_has_int8_engram(model_path):
        _INT8_CACHE = True
        return True

    try:
        hf = get_model().model_config.hf_config
        qc = getattr(hf, "quantization_config", None)
        if qc is None:
            _INT8_CACHE = False
            return False
        if not isinstance(qc, dict):
            qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
        for group in (qc.get("config_groups") or {}).values():
            targets = group.get("targets") or []
            if not any("engram" in t and "embed" in t for t in targets):
                continue
            weights = group.get("weights") or {}
            _INT8_CACHE = weights.get("type") == "int" and weights.get("num_bits") == 8
            return _INT8_CACHE
    except Exception:  # noqa: BLE001 - detection must never break model build
        pass
    _INT8_CACHE = False
    return False


@plugin_hook(
    "sglang.srt.layers.engram.drop_checkpoint_page_cache",
    type=HookType.AROUND,
)
def drop_checkpoint_page_cache_kunlun(original_fn):
    """Honor the existing launcher's explicit opt-out from checkpoint eviction.

    Clearing a large shared checkpoint from every TP rank can serialize in the
    filesystem for minutes. Keep upstream behavior unless the caller explicitly
    disables this optional host-memory optimization.
    """
    if os.environ.get("SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE", "").lower() in (
        "0",
        "false",
    ):
        return 0, 0
    return original_fn()


@plugin_hook("sglang.srt.layers.engram.EngramEmbedding", type=HookType.REPLACE)
class EngramEmbeddingKunlun(EngramEmbedding):
    """`EngramEmbedding` that also serves the W8A8-INT8 table layout."""

    def __init__(self, num_embeddings: int, dim: int, layer_id: int):
        """Build the checkpoint's FP8 or per-row INT8 embedding storage.

        Args:
            num_embeddings: Global number of embedding rows.
            dim: Number of values per row.
            layer_id: Layer identifier used for shared host storage.

        Returns:
            None.
        """
        self._int8 = engram_embed_weight_is_int8()
        if not self._int8:
            # FP8 checkpoint: keep the upstream fp8_e4m3 + e8m0 layout verbatim.
            super().__init__(num_embeddings, dim, layer_id)
            return
        nn.Module.__init__(self)
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.row_start = num_embeddings * tp_rank // self.tp_size
        row_end = num_embeddings * (tp_rank + 1) // self.tp_size
        self.rows = row_end - self.row_start
        self.host_table: Optional[object] = None
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table_int8(num_embeddings, dim, layer_id)
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.int8), requires_grad=False
            )
            self.weight_scale = nn.Parameter(
                torch.empty(self.rows, 1, dtype=torch.bfloat16), requires_grad=False
            )
        # `_load_rows` is dtype/name agnostic: it copies this rank's row slice.
        self.weight.weight_loader = self._load_rows
        self.weight_scale.weight_loader = self._load_rows

    def _init_host_table_int8(self, num_embeddings: int, dim: int, layer_id: int):
        """Map shared INT8 payload and BF16 scales into non-trainable views."""
        # 0.5.22 validates the explicit layout in _HostTable.__init__.
        layout = envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        n = num_embeddings if layout == "shared" else self.rows
        w_bytes = n * dim  # int8 payload
        s_bytes = n * 2  # one bf16 scale per row
        self.host_table = engram_mod._HostTable(
            layout,
            max(1, w_bytes + s_bytes),  # mmap needs storage even for an empty shard
            f"sglang_engram_{layer_id}",
            get_tp_group(),
        )
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.int8).view(n, dim)
        scale = raw[w_bytes : w_bytes + s_bytes].view(torch.bfloat16).view(n, 1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch=None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Gather and dequantize rows, using the fixed stage during capture.

        Args:
            indices: Global embedding row IDs.
            forward_batch: Batch metadata used by upstream CP/shard handling.
            cp_all_tokens: Whether CP lookup covers all tokens before sharding.

        Returns:
            Embeddings with shape ``(*indices.shape, dim)``.
        """
        if not self._int8:
            return super().forward(
                indices, forward_batch, cp_all_tokens=cp_all_tokens
            )
        if os.environ.get("DSV4_KUNLUN_ZERO_ENGRAM") == "1":
            # Accuracy A/B: make the engram lookup a well-defined no-op (the
            # residual then carries `x + gate * 0`). Every rank takes this branch,
            # so the TP all-reduce contract stays consistent.
            return self._empty(indices).zero_()
        staged = _stage_lookup(self, indices)
        if staged is not None:
            return staged
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            batched = _host_gather_batched(self, indices)
            if batched is not None:
                return batched
            out = self._empty(indices)
            from sglang.kernels.ops.embeddings.engram_gather import engram_gather

            # block_size == dim selects the one-scale-per-row (int8) decode.
            # `weight_ptr` is the whole shared table's base, so ids are absolute
            # rows and `row_lo` stays 0. `row_hi` must be the table's row count:
            # the op turns out-of-range ids into zero rows, whereas the default
            # `row_hi=2**62` lets `weight_ptr + id * dim` overflow the 64-bit
            # address in `engram_gather_kunlun` (hit during decode CUDA graph
            # capture, where the dummy batch's hash ids are not in range).
            engram_gather(
                self.weight.data_ptr(),
                self.weight_scale.data_ptr(),
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                self.dim,
                0,
                self.weight.shape[0],
            )
            return out
        return super().forward(indices, forward_batch, cp_all_tokens=cp_all_tokens)

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather this rank's rows and zero unowned IDs before TP reduction."""
        if not self._int8:
            return super()._owned_rows(indices)
        if self.rows == 0:
            return self._empty(indices).zero_()
        if self.host_table is None:
            # Device-resident shard: plain tensor gather, no raw pointers.
            local = (indices - self.row_start).to(torch.int64)
            owned = (local >= 0) & (local < self.rows)
            local = local.masked_fill(~owned, 0)
            values = self.weight[local].float() * self.weight_scale[local].float()
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        batched = _host_gather_batched_sharded(self, indices)
        if batched is not None:
            return batched
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        out = self._empty(indices)
        engram_gather(
            self.weight.data_ptr(),
            self.weight_scale.data_ptr(),
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            self.dim,
            row_lo=self.row_start,
            row_hi=self.row_start + self.rows,
        )
        return out
