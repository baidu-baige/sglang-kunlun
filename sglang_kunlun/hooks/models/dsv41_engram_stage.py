"""Host engram table + decode CUDA graph: per-step prefetch into a fixed HBM stage.

PROBLEM
-------
With `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` the two ~94 GiB engram tables stay
in host memory, and the lookup runs through `engram_gather_kunlun`
(`sglang_kunlun/kernels/dsv41_low_ratio.py`): `ids.tolist()` -> ctypes row reads
-> `out.copy_(values.to(device))`.

P800 cannot DMA from a host address (verified: `GM2LM` on a host VA raises
`KL_XID_KERNEL_EXCEPTION` / NoC timeout), so that CPU detour is the only way to
read the table. But it is not capturable: `tolist()` is a host sync and the
ctypes reads are plain Python, so a decode CUDA graph records *only* the trailing
`copy_` and bakes the capture-time source buffer in as a constant. Replay then
feeds every step the same stale embedding -- the server runs, no error, and the
output is fluent garbage.

APPROACH (two-stage: eager IO, in-graph compute)
------------------------------------------------
A decode step touches very few rows: `tokens * n_hash_cols` per engram layer
(bs=64 -> 1536 rows -> ~390 KiB, and `tokens = bs * draft_token_num` under
DSPARK), so the whole working set of one step fits in a small HBM buffer whose
address never moves:

  before replay (eager, outside the graph)
      1. hash the batch once more, WITHOUT committing history, to learn the rows
         this step needs
      2. gather + dequantize those rows out of the host table on the CPU
      3. async H2D into a fixed per-layer stage buffer

  inside the graph
      `EngramEmbedding.forward` stops gathering and just *reshapes* its slice of
      the stage buffer. The stage is laid out positionally -- row k of the stage
      is flat index k of `hash_ids[:, layer]` -- so no indexing is needed and
      nothing data-dependent is recorded.

The in-graph hasher still runs and still commits history; its `hash_ids` are
simply unused by the lookup. Correctness therefore requires the pre-replay hash
to equal the in-graph hash, which holds because both read the same pre-commit
`history` rows with the same kernel.

Both captured decode modes are staged: plain DECODE (one token per request) and
DSPARK's TARGET_VERIFY (`draft_token_num` tokens per request, hashed with
MODE_VERIFY). Verify never commits history, so the pre-replay hash is a pure
read there.

SCOPE
-----
`shared` host-table layout only (the default, and what `run_server.sh` uses):
there every rank holds the whole table and the lookup needs no TP all-reduce, so
each rank can stage its own copy. `per_rank` keeps the old path, since its
zero-unowned-rows + all-reduce contract would have to be reproduced in the stage.

Disable with `SGLANG_KUNLUN_ENGRAM_STAGE=0` to get the previous behaviour back.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_DECODE_RUNNER = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner"
)
MODE_DECODE = 0
MODE_VERIFY = 1
# `EngramHasher.forward` passes this to the hash kernel; the pre-replay hash has
# to pass the same value or multimodal-padded ids hash differently.
MM_PAD_SHIFT_VALUE = 1_000_000


def _stage_enabled() -> bool:
    """Host table in `shared` layout, and the feature not switched off."""
    if os.environ.get("SGLANG_KUNLUN_ENGRAM_STAGE", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    if not envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
        return False
    return envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() == "shared"


class EngramHostStager:
    """Per-step host->HBM prefetch of the rows one decode step will look up."""

    def __init__(self) -> None:
        """Start without stage storage; allocation must precede graph capture."""
        # (hash column index, EngramEmbedding) for every engram layer, in model order.
        self._layers: List[Tuple[int, object]] = []
        self._hasher = None
        self._cols = 0
        self._dim = 0
        self._max_rows = 0
        # Fixed-address HBM stage, one per engram layer. Captured graphs read these.
        self._stage: List[torch.Tensor] = []
        # Pinned host staging so the H2D can be async on the current stream.
        self._pinned: List[torch.Tensor] = []
        # int8-row staging for the device-side dequant: the host then only has
        # to gather `dim` bytes per row instead of running the fp32 dequant.
        self._pin_rows: List[torch.Tensor] = []
        self._pin_scales: List[torch.Tensor] = []
        self._dev_rows: List[torch.Tensor] = []
        self._dev_scales: List[torch.Tensor] = []
        self._device_dequant = os.environ.get(
            "SGLANG_KUNLUN_ENGRAM_STAGE_DEVICE_DEQUANT", "1"
        ).strip().lower() not in ("0", "false", "no")
        # The row gather itself is random access over the 92 GiB mmap and stays
        # on the host; >1 splits it across the shared pool (index_select drops
        # the GIL). 4 matches the prefill path's default and measured
        # 4482 -> 4522 tok/s at 220 concurrency on 206.
        try:
            self._gather_threads = max(
                1, int(os.environ.get("SGLANG_KUNLUN_ENGRAM_STAGE_GATHER_THREADS", 4))
            )
        except ValueError:
            self._gather_threads = 4
        self._in_capture = False
        self._prepared_tokens = 0
        self._ready = False

    # ---------------------------------------------------------------- discovery

    def _discover(self, model_runner) -> bool:
        """Find the hasher and the engram layers, and remember each one's slot."""
        if self._layers:
            return True
        model = getattr(model_runner, "model", None)
        inner = getattr(model, "model", None)
        hasher = getattr(inner, "engram_hasher", None)
        layers = getattr(inner, "layers", None)
        if hasher is None or layers is None:
            return False
        found = []
        for layer in layers:
            engram = getattr(layer, "engram", None)
            if engram is None:
                continue
            found.append((int(engram.layer_hash_index), engram.embed))
        if not found:
            return False
        # `_shared` means one full copy of the table per rank; see module docstring.
        if not all(getattr(embed, "_shared", False) for _, embed in found):
            logger.info(
                "engram stage: host table is not in the shared layout, keeping the "
                "CPU gather path"
            )
            return False
        # Only the W8A8-INT8 layout (int8 payload + one bf16 scale per row) is
        # staged; the fp8 e4m3 + e8m0 release keeps the original path.
        if not all(
            getattr(embed, "weight_scale", None) is not None for _, embed in found
        ):
            logger.info(
                "engram stage: table is not channel-wise int8, keeping the CPU "
                "gather path"
            )
            return False
        self._hasher = hasher
        self._layers = found
        self._cols = int(hasher.offsets.shape[1])
        self._dim = int(found[0][1].dim)
        for slot, (_, embed) in enumerate(found):
            embed._stage_slot = slot
        return True

    # --------------------------------------------------------------- allocation

    def ensure_alloc(self, model_runner, max_tokens: int) -> bool:
        """Allocate the stage before capture, sized for the widest captured graph.

        `max_tokens` is the token count of the largest captured batch, not its
        batch size: under DSPARK a captured verify graph carries
        `bs * draft_token_num` tokens, and one stage row is needed per hash
        column of every token.

        The address has to outlive every graph, so this runs once, before the
        first capture, and never reallocates.

        Args:
            model_runner: Owner of the hasher and host embedding tables.
            max_tokens: Largest captured token count.

        Returns:
            Whether the fixed stage can cover the requested capture.
        """
        if self._ready:
            return int(max_tokens) * self._cols <= self._max_rows
        if not self._discover(model_runner):
            return False
        rows = max(1, int(max_tokens)) * self._cols
        device = next(iter(self._layers))[1].weight.device
        if device.type != "cuda":  # host table: take the device from the ids later
            device = torch.device("cuda", torch.cuda.current_device())
        for _ in self._layers:
            self._stage.append(
                torch.zeros(rows, self._dim, dtype=torch.bfloat16, device=device)
            )
            self._pinned.append(
                torch.zeros(rows, self._dim, dtype=torch.bfloat16).pin_memory()
            )
            self._pin_rows.append(
                torch.zeros(rows, self._dim, dtype=torch.int8).pin_memory()
            )
            self._pin_scales.append(
                torch.zeros(rows, 1, dtype=torch.bfloat16).pin_memory()
            )
            self._dev_rows.append(
                torch.zeros(rows, self._dim, dtype=torch.int8, device=device)
            )
            self._dev_scales.append(
                torch.zeros(rows, 1, dtype=torch.bfloat16, device=device)
            )
        self._max_rows = rows
        self._ready = True
        logger.info(
            "engram stage: %d layer(s), %d rows x %d dim per layer (%.2f MiB HBM), "
            "max_tokens=%d cols=%d",
            len(self._layers),
            rows,
            self._dim,
            len(self._layers) * rows * self._dim * 2 / 1024 / 1024,
            max_tokens,
            self._cols,
        )
        return True

    # ------------------------------------------------------------------ capture

    def begin_capture(self) -> None:
        """Enable fixed-stage lookups for capture and its eager warmup."""
        self._in_capture = True

    def end_capture(self) -> None:
        """Restore ordinary gathers after capture, including failed captures."""
        self._in_capture = False

    # -------------------------------------------------------------- per-step IO

    def prepare(self, model_runner, forward_batch) -> bool:
        """Refill every layer before authorizing replay of this batch.

        Args:
            model_runner: Runner owning the model (kept for the hook interface).
            forward_batch: Live decode or equal-block target-verify batch.

        Returns:
            True only after all copies are issued, or for a 0-token idle batch
            that has nothing to look up; False on an unsupported batch or a
            missing hash. Exceptions propagate and must prevent replay.
        """
        self._prepared_tokens = 0
        if not self._ready:
            return False
        ids = getattr(forward_batch, "input_ids", None)
        if ids is None or ids.shape[0] == 0:
            # DP idle participation builds a 0-token `ForwardMode.IDLE` batch
            # (dspark_draft.py:454-478) whose only job is to join the MoE
            # all-to-all; its output is discarded. No row can be looked up, so
            # zero the stage: replay then reads defined rows instead of the
            # previous step's, while any batch that does carry tokens still has
            # to stage successfully before replay is authorized.
            for staged in self._stage:
                staged.zero_()
            return True
        mode = forward_batch.forward_mode
        if mode.is_decode():
            kmode, block = MODE_DECODE, 1
        elif mode.is_target_verify():
            # DSPARK: `bs * draft_token_num` tokens, one equal block per request.
            spec_info = getattr(forward_batch, "spec_info", None)
            block = int(getattr(spec_info, "draft_token_num", 0) or 0)
            if block <= 0:
                return False
            kmode = MODE_VERIFY
        else:
            return False
        num_tokens = forward_batch.input_ids.shape[0]
        if (
            num_tokens == 0
            or num_tokens * self._cols > self._max_rows
            or num_tokens != forward_batch.req_pool_indices.shape[0] * block
        ):
            return False
        hash_ids = self._hash_without_commit(forward_batch, kmode, block)
        if hash_ids is None:
            return False
        if (
            hash_ids.ndim != 3
            or hash_ids.shape[0] != num_tokens
            or hash_ids.shape[2] != self._cols
            or any(hidx >= hash_ids.shape[1] for hidx, _ in self._layers)
        ):
            return False
        # [T, L, cols] -> host, one sync for ~12 KiB.
        ids_host = hash_ids.to("cpu", torch.int64, non_blocking=False)
        num_tokens = ids_host.shape[0]
        rows = num_tokens * self._cols
        if rows == 0 or rows > self._max_rows:
            return False
        for slot, (hidx, embed) in enumerate(self._layers):
            flat = ids_host[:, hidx, :].reshape(-1)
            if self._device_dequant and self._stage_on_device(slot, embed, flat):
                continue
            self._gather_into_pinned(embed, flat, self._pinned[slot][:rows])
            self._stage[slot][:rows].copy_(self._pinned[slot][:rows], non_blocking=True)
        self._prepared_tokens = num_tokens
        return True

    def _hash_without_commit(
        self, forward_batch, kmode: int, block: int
    ) -> Optional[torch.Tensor]:
        """Re-hash the batch to learn its rows, leaving `history` untouched.

        The in-graph hasher is the one that advances history (decode runs the
        fused `engram_hash_ids_and_commit`), so this call must use the plain
        `engram_hash_ids` with the same mode/block and the same pre-commit
        `history` rows. `MODE_VERIFY` commits nothing in either path.
        """
        hasher = self._hasher
        if hasher is None or hasher.history is None:
            return None
        try:
            from sglang.kernels.ops.embeddings.engram_hash import engram_hash_ids

            hash_ids, _tokens = engram_hash_ids(
                forward_batch.input_ids,
                forward_batch.positions,
                mode=kmode,
                history=hasher.history,
                token_map=hasher.token_map,
                multipliers=hasher.multipliers,
                primes=hasher.primes,
                offsets=hasher.offsets,
                pad_id=hasher.pad_id,
                num_real=int(forward_batch.input_ids.shape[0]),
                req_slots=forward_batch.req_pool_indices,
                block=block,
                row=None,
                starts=None,
                image_token_id=hasher.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
            return hash_ids
        except Exception as e:
            raise RuntimeError("engram stage: pre-replay hash failed; replay denied") from e

    def _stage_on_device(self, slot: int, embed, flat_rows: torch.Tensor) -> bool:
        """Gather the raw int8 rows and dequantize them on the XPU.

        `_gather_into_pinned` runs the whole dequant on the CPU:
        `weight[idx].float() * scale[idx].float() -> bf16` makes four full passes
        over `[rows, dim]` and two fp32 temporaries. Measured on 206 at bs=32
        that is 5.7 ms of `aten::copy_`/`aten::_to_copy` per decode round, and
        it is 2.3x slower on Hygon than on the Intel reference host, so it is
        the single largest host-side item in the round.

        Here the host only gathers `dim` int8 bytes plus one bf16 scale per row
        (half the H2D bytes of the bf16 stage) and the multiply happens on the
        device. The arithmetic is unchanged: int8 -> bf16 is exact, the scale is
        already bf16, and a bf16 multiply rounds once, so the result is bitwise
        equal to the fp32 host expression (verified over 890k random elements in
        `debug/engram_dequant_bitcmp.py`).

        Returns:
            True when the stage slot was filled; False for a table layout this
            path does not handle, leaving the CPU dequant to run as before.
        """
        weight = embed.weight.data
        scale = getattr(embed, "weight_scale", None)
        if scale is None:
            return False
        scale = scale.data
        if weight.dtype is not torch.int8 or scale.dtype is not torch.bfloat16:
            return False
        if weight.device.type != "cpu" or scale.device.type != "cpu":
            return False
        n = flat_rows.numel()
        if n > self._pin_rows[slot].shape[0]:
            return False
        idx = flat_rows.clamp_(0, weight.shape[0] - 1)
        pin_rows = self._pin_rows[slot][:n]
        pin_scales = self._pin_scales[slot][:n]
        threads = self._gather_threads if n >= 256 * self._gather_threads else 1
        if threads > 1:
            from sglang_kunlun.hooks.models.dsv41_engram_int8 import _gather_pool

            def gather(bound):
                """Fill one disjoint row slice of both pinned buffers."""
                lo, hi = bound
                part = idx[lo:hi]
                torch.index_select(weight, 0, part, out=pin_rows[lo:hi])
                torch.index_select(scale, 0, part, out=pin_scales[lo:hi])

            bounds = [
                (i * n // threads, (i + 1) * n // threads) for i in range(threads)
            ]
            list(_gather_pool(threads).map(gather, bounds))
        else:
            torch.index_select(weight, 0, idx, out=pin_rows)
            torch.index_select(scale, 0, idx, out=pin_scales)
        dev_rows = self._dev_rows[slot][:n]
        dev_scales = self._dev_scales[slot][:n]
        dev_rows.copy_(pin_rows, non_blocking=True)
        dev_scales.copy_(pin_scales, non_blocking=True)
        torch.mul(
            dev_rows.to(torch.bfloat16), dev_scales, out=self._stage[slot][:n]
        )
        return True

    @staticmethod
    def _gather_into_pinned(embed, flat_rows: torch.Tensor, dst: torch.Tensor) -> None:
        """`dst[k] = weight[flat_rows[k]] * weight_scale[flat_rows[k]]` on the CPU.

        `weight`/`weight_scale` are the CPU views over the mmap'd host table, so
        this is an ordinary tensor gather -- still the only way to read the table
        on P800, but now it happens outside the graph where a host sync is legal.
        Out-of-range ids are clamped rather than zeroed: in the shared layout the
        table covers every id the hasher can produce, so a clamp only ever fires
        for graph padding, whose output is discarded.
        """
        weight = embed.weight.data
        scale = embed.weight_scale.data
        idx = flat_rows.clamp_(0, weight.shape[0] - 1)
        rows = weight[idx]
        dst.copy_((rows.float() * scale[idx].float()).to(torch.bfloat16))

    # ----------------------------------------------------------------- in-graph

    def lookup(self, embed, indices: torch.Tensor) -> Optional[torch.Tensor]:
        """The in-graph replacement for the gather: a reshape of the stage.

        Only taken while a graph is being captured. Replay never runs Python, so
        what matters is that capture records a read of this fixed address; every
        later step just has to refill it in `prepare`. Eager forwards (warmup,
        prefill, oversized decode batches) keep using the original CPU gather,
        which is slower but needs no prefetch.

        Args:
            embed: Embedding module assigned a stage slot during discovery.
            indices: Hash IDs whose shape selects the positional stage view.

        Returns:
            A view of the fixed stage during capture, otherwise None.

        Raises:
            RuntimeError: Capture cannot safely read the allocated stage.
        """
        slot = getattr(embed, "_stage_slot", None)
        if not self._in_capture:
            return None
        if not self._ready or slot is None:
            raise RuntimeError("engram stage: missing capture stage")
        num_tokens = indices.shape[0]
        rows = num_tokens * self._cols
        if rows == 0 or rows > self._max_rows or indices.shape[-1] != self._cols:
            raise RuntimeError("engram stage: unsupported capture lookup shape")
        return self._stage[slot][:rows].view(*indices.shape, self._dim)


_STAGER: Optional[EngramHostStager] = None


def get_stager() -> Optional[EngramHostStager]:
    """The process-wide stager, or None when the feature does not apply."""
    global _STAGER
    if not _stage_enabled():
        return None
    if _STAGER is None:
        _STAGER = EngramHostStager()
    return _STAGER


@plugin_hook(f"{_DECODE_RUNNER}.capture", type=HookType.AROUND)
def engram_stage_capture(original_fn, self, *args, **kwargs):
    """Allocate the stage before the first graph, and record reads of it.

    Allocation has to happen here rather than lazily in `forward`: the address
    is baked into every captured graph, so it must be final before capture and
    must be sized for the largest captured batch.

    Args:
        original_fn: Wrapped capture method.
        self: DecodeCudaGraphRunner instance.
        *args: Positional arguments forwarded to capture.
        **kwargs: Keyword arguments forwarded to capture.

    Returns:
        The wrapped capture result, with the stager retained by this runner.
    """
    stager = get_stager()
    if stager is None:
        return original_fn(self, *args, **kwargs)
    # `max_num_token` is the widest captured graph's token count (it already
    # accounts for DSPARK's `captured_req_width` tokens per request); the
    # product is the fallback for runners that do not expose it.
    max_tokens = getattr(self, "max_num_token", None)
    if not max_tokens:
        capture_bs = getattr(self, "capture_bs", None) or [1]
        width = int(getattr(self, "captured_req_width", 1) or 1)
        max_tokens = max(capture_bs) * width
    if not stager.ensure_alloc(self.model_runner, max_tokens):
        raise RuntimeError("engram stage: cannot allocate capture stage; disable CUDA graph")
    self._kunlun_engram_stager = stager
    stager.begin_capture()
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        stager.end_capture()


@plugin_hook(f"{_DECODE_RUNNER}.execute", type=HookType.AROUND)
def engram_stage_execute(original_fn, self, forward_batch, *args, **kwargs):
    """Refill the stage for this step, then replay.

    The copy is issued on the current stream before `original_fn` launches the
    graph, so the graph's reads are ordered after it without an extra sync.

    Args:
        original_fn: Wrapped execute method, which loads the batch and replays.
        self: DecodeCudaGraphRunner owning the captured stage.
        forward_batch: Live batch to stage.
        *args: Positional execution arguments.
        **kwargs: Keyword execution arguments.

    Returns:
        The wrapped execution result, only after successful staging.

    Raises:
        RuntimeError: A staged graph cannot be supplied fresh rows.
    """
    # Capture owns this dependency, even if feature flags later change.
    stager = getattr(self, "_kunlun_engram_stager", None)
    if stager is not None:
        if getattr(self, "ragged_verify_mode", False):
            raise RuntimeError("engram stage: ragged verify replay is unsupported")
        if not stager.prepare(self.model_runner, forward_batch):
            raise RuntimeError("engram stage: batch preparation failed; replay denied")
    return original_fn(self, forward_batch, *args, **kwargs)
