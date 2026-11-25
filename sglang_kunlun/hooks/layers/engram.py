# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Kunlun override for the DeepSeek-V4.1 engram host-table lookup.

Upstream ``sglang.srt.layers.engram.EngramEmbedding`` has two host-table paths
that hand a *host* pointer to the triton ``engram_gather`` kernel:

* the ``shared`` layout fast path in ``forward``;
* the ``per_rank`` shard path in ``_owned_rows``.

``engram_gather`` takes raw integer addresses and dereferences them inside the
kernel, which only works where the device can reach host memory (Grace-Blackwell
ATS over NVLink-C2C). On Kunlun P800 there is no UVA/ATS: dereferencing a pinned
host pointer never returns, the driver reports a task timeout and raises
``KL_XID_KERNEL_EXCEPTION`` (device wedge). Both paths are therefore unusable
exactly where they were meant to save memory.

The saving itself is still worth having — the engram table is ~198 GB across the
two layers (25.9% of the checkpoint), ~25 GB per rank once TP-sharded, and the
host-table layout moves it off the device entirely. This module keeps that
saving but moves the lookup to the CPU: gather the payload rows and their scales
with torch, dequantise to bf16 (exact for both layouts — see below), and ship
bf16 to the device. Measured on P800 the CPU path is 76 ms/layer against the
shipped device kernel's 99 ms/layer (T=16384), i.e. free — even a little faster
— in exchange for ~22 GB/rank.

Two table layouts exist and both are handled here:

* **fp8** (the FP8 release): payload ``float8_e4m3fn`` [n, dim], one ``e8m0``
  scale per 32-element block. ``fp8 -> bf16`` keeps every mantissa bit and the
  block scale only shifts the exponent, so the product is exact.
* **int8** (the W8A8-INT8 release, ``hooks/models/dsv41_engram_int8.py``):
  payload ``int8`` [n, dim], one ``bf16`` scale per *row*. ``values =
  weight[ids].float() * weight_scale[ids].float()`` -> bf16, which is exactly
  what the compressed-tensors channel-wise scale means.

Three hard XPU constraints shape the fp8 implementation and are why it is not
simpler:

* the CPU fp8 tensor cannot be indexed (``UnknownScalarType``), so the gather
  goes through a ``.view(torch.uint8)`` round-trip;
* XPU cannot cast fp8/e8m0 to bf16, so the dequantisation cannot be pushed to
  the device and the bf16 result is what crosses the bus.

Neither applies to the int8 payload (int8/bf16 index and cast natively on CPU),
so that path stays plain torch.

The device path (``host_table is None``) is untouched: the hooks only intercept
when a host table is present, which requires ``SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1``.
Set ``SGLANG_KUNLUN_ENGRAM_CPU_GATHER=0`` to restore the upstream (host-pointer)
path for an A/B.

Also installs an ``os.memfd_create`` shim: the ``shared`` host-table layout needs
it and this Python build does not expose it, and without the shim the layout
falls back to ``per_rank`` (8 copies, plus an all-reduce per lookup).
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_parallel

from sglang_kunlun.hooks.utils import engram_precompute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# os.memfd_create shim. Installed at import time, before the engine builds the
# host table, so ``_HostTable._open_shared_fd`` finds it.
# ---------------------------------------------------------------------------
if not hasattr(os, "memfd_create"):
    import ctypes as _ctypes

    _libc_memfd = _ctypes.CDLL(None, use_errno=True)
    _libc_memfd.memfd_create.restype = _ctypes.c_int
    _libc_memfd.memfd_create.argtypes = [_ctypes.c_char_p, _ctypes.c_uint]

    def _os_memfd_create(name, flags=0):
        fd = _libc_memfd.memfd_create(
            name.encode() if isinstance(name, str) else name, flags
        )
        if fd < 0:
            err = _ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return fd

    os.memfd_create = _os_memfd_create
    logger.info("sglang-kunlun: installed os.memfd_create shim (shared engram host table)")


def _enabled() -> bool:
    """CPU gather is the default; ``=0`` restores the upstream host-pointer path."""
    return os.environ.get("SGLANG_KUNLUN_ENGRAM_CPU_GATHER", "1") != "0"


def _rs_enabled() -> bool:
    """CP prefill: reduce-scatter the per-rank row slices instead of all-reducing.

    Each hash id has exactly one owning TP shard, so the slice this rank keeps
    after the CP all-gather is the only nonzero contribution of the ranks that
    own those rows: reduce-scatter hands every rank the same bytes the old
    all-reduce slice did. ``KL_CP_ENGRAM_RS=0`` restores the all-reduce; the
    /tmp marker flips arms inside one engine process.
    """
    if os.path.exists("/tmp/kl_engram_off"):
        return False
    return os.environ.get("KL_CP_ENGRAM_RS", "1") != "0"


def _is_int8(table) -> bool:
    """Whether this engram table is the channel-wise int8 layout.

    The kunlun int8 subclass (``hooks/models/dsv41_engram_int8.py``) sets
    ``_int8``; the fp8 base class has no such attribute and its payload is
    ``float8_e4m3fn``. Prefer the explicit flag, fall back to the dtype.
    """
    if getattr(table, "_int8", False):
        return True
    weight = getattr(table, "weight", None)
    return weight is not None and weight.dtype == torch.int8


def _fp8_block_size() -> int:
    from sglang.srt.layers.engram import FP8_BLOCK_SIZE

    return FP8_BLOCK_SIZE


def _gather_rows_fp8(table: torch.Tensor, local: torch.Tensor):
    """Gather fp8 rows and block scales from a host table by flat row index.

    CPU fp8 cannot be indexed, hence the uint8 view round-trip on both the
    weight and the e8m0 scale.
    """
    sel = table.weight.view(torch.uint8)[local].view(torch.float8_e4m3fn)
    scl = table.scale.view(torch.uint8)[local].view(torch.float8_e8m0fnu)
    return sel, scl


def _dequant_fp8(sel: torch.Tensor, scl: torch.Tensor, dim: int) -> torch.Tensor:
    """bf16-native fp8 * block-scale -> [n, dim] bf16 (exact, see module docstring)."""
    block = _fp8_block_size()
    rows = sel.to(torch.bfloat16).unflatten(-1, (-1, block))
    return (rows * scl.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)


def _device_dequant() -> bool:
    """Dequantise int8 on the device (default); ``=0`` keeps it on the CPU."""
    return os.environ.get("SGLANG_KUNLUN_ENGRAM_DEVICE_DEQUANT", "1") != "0"


def _dequant_int8(table, local: torch.Tensor) -> torch.Tensor:
    """Channel-wise int8 * per-row bf16 scale -> [n, dim] bf16 (exact)."""
    weight = table.weight[local].to(torch.bfloat16)  # [n, dim]
    scale = table.scale[local].to(torch.bfloat16)  # [n, 1]
    return weight * scale


def _dequant_int8_to(table, local: torch.Tensor, device) -> torch.Tensor:
    """Gather int8 rows + per-row scale on the host, dequantise on ``device``.

    Only the *gather* has to stay on the CPU (the table lives in host memory,
    so the XPU cannot dereference it). The cast and the scale multiply are both
    far cheaper on the device — measured on P800, the [n, 6144] cast+multiply
    is 0.14 ms on the XPU against 9.4 ms on the CPU — and shipping the raw int8
    payload is 4.4x cheaper than shipping the dequantised bf16 (0.94 ms vs
    4.17 ms for a 1659x6144 pageable H2D). The copy is also what made the slow
    ranks lag by ~170 ms at the engram all-reduce, so shrinking it shrinks the
    barrier wait that it causes.
    """
    weight = table.weight[local].to(device)  # int8 [n, dim]
    scale = table.scale[local].to(device)  # bf16 [n, 1]
    return weight.to(torch.bfloat16) * scale


def _clamp_rows(table, flat: torch.Tensor) -> torch.Tensor:
    """Fold out-of-range hash ids onto row 0 so the CPU gather cannot raise.

    The device ``engram_gather`` kernel bounds-checks every row (row_lo/row_hi)
    and writes zeros for a row it does not hold, so junk ids are harmless there,
    and the ``per_rank`` path below masks them for the same reason. ``shared``
    indexes the table directly, and ``table.weight[ids]`` raises IndexError --
    which cuda-graph capture triggers, because the hasher capture-time index
    buffers are uninitialised (observed: index 9205357640488583168 is out of
    bounds for dimension 0 with size 384006168). Clamping rather than masking
    the output is enough: capture results are discarded, and on real traffic
    every id the hasher emits is already in range, so this never fires.
    """
    if flat.numel() == 0:
        return flat
    bad = (flat < 0) | (flat >= table.weight.shape[0])
    if bool(bad.any()):
        flat = flat.masked_fill(bad, 0)
    return flat


def _host_cpu_lookup(table, indices: torch.Tensor) -> torch.Tensor:
    """Full-table gather for the ``shared`` layout: one rank can answer any row."""
    flat = _clamp_rows(table, indices.reshape(-1).to("cpu", torch.int64))
    if _is_int8(table):
        if _device_dequant():
            values = _dequant_int8_to(table, flat, indices.device)
            return values.view(*indices.shape, table.dim)
        values = _dequant_int8(table, flat)
    else:
        sel, scl = _gather_rows_fp8(table, flat)
        values = _dequant_fp8(sel, scl, table.dim)
    return values.to(indices.device).view(*indices.shape, table.dim)


def _host_cpu_lookup_sharded(table, indices: torch.Tensor) -> torch.Tensor:
    """Shard gather for the ``per_rank`` layout: zero rows this rank does not own."""
    if table.rows == 0:
        return table._empty(indices).zero_()
    local = (indices - table.row_start).to("cpu", torch.int64).reshape(-1)
    owned = (local >= 0) & (local < table.rows)
    if _is_int8(table) and _device_dequant():
        # Only the rows this rank OWNS need gathering: unowned slots contribute a
        # zero to the TP all-reduce that ``_lookup`` runs afterwards, which
        # reassembles the full lookup from the other ranks' shards. The shipped
        # path gathers all ``len(indices)`` rows (reading row 0 for every unowned
        # slot) -- on the CP engine each rank looks up the *whole* prompt, so
        # that is a tp_size-fold redundant gather of a host table that is itself
        # the latency bottleneck (random access into ~12 GB/rank). Selecting the
        # owned rows first cuts the gather AND the H2D by ~tp_size.
        out = torch.zeros(
            local.numel(), table.dim, dtype=torch.bfloat16, device=indices.device
        )
        sel = local[owned]
        if sel.numel():
            vals = _dequant_int8_to(table, sel, indices.device)  # [n_owned, dim]
            out.index_copy_(0, owned.nonzero(as_tuple=False).squeeze(1).to(indices.device), vals)
        return out.view(*indices.shape, table.dim)
    local = local.masked_fill(~owned, 0)
    if _is_int8(table):
        values = _dequant_int8(table, local)
    else:
        sel, scl = _gather_rows_fp8(table, local)
        values = _dequant_fp8(sel, scl, table.dim)
    values = values.view(*indices.shape, table.dim).to(indices.device)
    # ``owned`` is a CPU mask; XPU ``masked_fill`` rejects a host-side mask
    # ([INVALID PARAMETER]), so move it to the device with the values.
    owned = ~owned.view(*indices.shape, 1).to(indices.device)
    return values.masked_fill(owned, 0)


# ---------------------------------------------------------------------------
# EngramHasher.forward — DP-attention idle batches, and the token->request
# index without the vendor's scalar ``repeat_interleave``.
#
# The extend branch builds ``row = torch.repeat_interleave(torch.arange(bs),
# lens)``. ``repeat_interleave_fwd<long>`` is a single-threaded scalar kernel:
# 61,436 elements take 354 us (5.8 ns/element, 1.4 GB/s). ``row`` is just
# request r repeated ``lens[r]`` times, and ``lens`` already has a host mirror
# (``forward_batch.extend_seq_lens_cpu``), so build it from that and cache it by
# ``(device, lens)``. The swap is scoped to one hasher call: anything that is
# not that exact one-dimensional int64 arange-by-lens call goes to the original
# kernel.
# ---------------------------------------------------------------------------
_ROW_REPEAT_ORIG = torch.repeat_interleave
_ROW_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
_ROW_CACHE_MAX = 8
# bs values whose call site was verified to pass ``torch.arange(bs)``.  The
# check costs one D2H (a blocking read can wait for the whole device queue), so
# it runs once per bs for the process; a *negative* verdict is never cached, we
# simply fall back to the vendor kernel.
_ROW_SITE_ARANGE: set[int] = set()


def _row_repeat_enabled() -> bool:
    """``KL_ENGRAM_ROW=0`` restores the upstream repeat_interleave."""
    return os.environ.get("KL_ENGRAM_ROW", "1") != "0"


def _site_is_arange(bs: int, input: torch.Tensor) -> bool:
    if bs in _ROW_SITE_ARANGE:
        return True
    same = torch.equal(
        input, torch.arange(bs, dtype=torch.int64, device=input.device)
    )
    if same:
        _ROW_SITE_ARANGE.add(bs)
    return same


def _cached_row(device, lens: tuple[int, ...]) -> torch.Tensor:
    key = (str(device), lens)
    row = _ROW_CACHE.get(key)
    if row is None:
        row = torch.tensor(
            [r for r, n in enumerate(lens) for _ in range(n)],
            dtype=torch.int64,
            device=device,
        )
        if len(_ROW_CACHE) >= _ROW_CACHE_MAX:
            _ROW_CACHE.clear()
        _ROW_CACHE[key] = row
    return row


def _extend_lens(forward_batch, input_ids) -> tuple[int, ...] | None:
    """Per-request extend lengths from host state, or ``None``.

    Batches built from host lists carry ``extend_seq_lens_cpu``; the gpu_only
    path hands device tensors straight in and leaves the mirror unset.  A
    single request needs only its total, and the scheduler keeps that on the
    host as ``extend_num_tokens`` (= ``sum(extend_seq_lens)``), so the
    delivery pose still gets the cached row.
    """
    slots = getattr(forward_batch, "req_pool_indices", None)
    if slots is None:
        return None
    bs = int(slots.shape[0])
    lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if lens:
        lens = tuple(int(v) for v in lens)
        if len(lens) == bs and min(lens) > 0 and 0 < sum(lens) <= input_ids.shape[0]:
            return lens
        return None
    if bs == 1:
        total = getattr(forward_batch, "extend_num_tokens", None)
        if total is not None and 0 < int(total) <= input_ids.shape[0]:
            return (int(total),)
    return None


class _CachedRowRepeat:
    """Serve ``repeat_interleave(arange(bs), lens)`` from ``self.row``.

    ``EngramHasher.forward`` issues exactly one ``torch.repeat_interleave``; the
    first call in the window that matches the shape guard is that one, and the
    input is checked to really be ``arange(bs)`` (once per bs, see
    ``_ROW_SITE_ARANGE``).  Every other call goes to the vendor kernel.
    """

    def __init__(self, row: torch.Tensor, bs: int):
        self.row = row
        self.bs = bs
        self.used = False

    def __enter__(self):
        torch.repeat_interleave = self._call
        return self

    def __exit__(self, *exc):
        torch.repeat_interleave = _ROW_REPEAT_ORIG
        return False

    def _call(self, input, repeats, dim=None, *, output_size=None):
        if (
            not self.used
            and dim is None
            and output_size is None
            and isinstance(input, torch.Tensor)
            and isinstance(repeats, torch.Tensor)
            and input.dim() == 1
            and repeats.dim() == 1
            and input.numel() == self.bs
            and repeats.numel() == self.bs
            and input.dtype == torch.int64
            and _site_is_arange(self.bs, input)
        ):
            self.used = True
            return self.row
        return _ROW_REPEAT_ORIG(input, repeats, dim, output_size=output_size)


# ---------------------------------------------------------------------------
# EngramHasher.forward — allow target-verify batches padded to attn_tp_size.
#
# ``ForwardBatch.prepare_mlp_sync_batch`` rounds the verify token count up to
# the attention tensor-parallel width (block=5: 6 real rows -> 8). Upstream
# asserts ``num_tokens == bs * draft_token_num`` and leaves ``num_real`` at
# ``num_tokens``, so the tier-padding rows are hashed as if they were real
# history. Relax the assert and mask the trailing rows via ``num_real``.
#
# The change is inside the function body, which no AROUND hook can reach, so
# the plugin re-compiles this one method from its own source at import time
# instead of editing sglang on disk. If upstream changes the layout the patch
# is skipped (with a warning) and the original assertion keeps failing loudly.
# It must run BEFORE the AROUND hook below wraps the method, otherwise the
# recompiled original would replace the wrapper and drop the idle/row hooks.
# ---------------------------------------------------------------------------
def _install_padded_verify_patch() -> None:
    import inspect
    import textwrap

    from sglang.srt.layers.engram import EngramHasher

    fn = EngramHasher.forward
    if getattr(fn, "_kunlun_padded_verify", False):
        return
    source = textwrap.dedent(inspect.getsource(fn))
    marker = "        block = int(forward_batch.spec_info.draft_token_num)\n"
    old_assert = "        assert num_tokens == bs * block, ("
    if marker not in source or old_assert not in source:
        logger.warning(
            "sglang-kunlun: EngramHasher.forward layout changed; "
            "padded target-verify patch not applied"
        )
        return
    source = source.replace(
        marker, marker + "        num_real = bs * block\n", 1
    )
    source = source.replace(
        old_assert, "        assert num_tokens >= bs * block, (", 1
    )
    namespace: dict = {}
    exec(compile(source, fn.__code__.co_filename, "exec"), fn.__globals__, namespace)
    patched = namespace["forward"]
    patched._kunlun_padded_verify = True
    EngramHasher.forward = patched
    logger.info("sglang-kunlun: padded target-verify engram patch installed")


_install_padded_verify_patch()


@plugin_hook(
    "sglang.srt.layers.engram.EngramHasher.forward",
    type=HookType.AROUND,
)
def _engram_hasher_forward_kunlun(original_fn, self, input_ids, forward_batch):
    """Dummy hash ids for a DP-attention idle batch; cached row otherwise.

    With DP attention, a rank with no local work still runs an IDLE batch so
    it participates in the collectives. ``EngramHasher.forward`` asserts the
    mode is extend/target-verify/decode, so the idle batch aborts with
    ``engram serves extend, target-verify and decode, not 4``. The hash ids of
    an idle batch are discarded; only their row count and shape matter, and
    row 0 is a valid table row, so zeros are safe and avoid touching history.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None and mode.is_idle():
        return torch.zeros(
            (input_ids.shape[0], self.primes.shape[0], self.offsets.shape[1]),
            dtype=torch.int64,
            device=input_ids.device,
        )
    if mode is not None and mode.is_extend() and _row_repeat_enabled():
        lens = _extend_lens(forward_batch, input_ids)
        if lens is not None:
            with _CachedRowRepeat(_cached_row(input_ids.device, lens), len(lens)):
                return original_fn(self, input_ids, forward_batch)
        _log_row_miss(forward_batch)
    return original_fn(self, input_ids, forward_batch)


_ROW_MISS_LOGGED = False


def _log_row_miss(forward_batch):
    """One line per process: a silent fallback would look like "no win"."""
    global _ROW_MISS_LOGGED
    if _ROW_MISS_LOGGED:
        return
    _ROW_MISS_LOGGED = True
    logger.info(
        "engram row cache off: mode=%s slots=%s extend_seq_lens_cpu=%s "
        "extend_num_tokens=%s",
        getattr(forward_batch, "forward_mode", None),
        getattr(getattr(forward_batch, "req_pool_indices", None), "shape", None),
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        getattr(forward_batch, "extend_num_tokens", None),
    )


# ---------------------------------------------------------------------------
# EngramEmbedding.forward — intercept the host-table fast path (shared layout).
#
# Target the *upstream* path on purpose: ``apply_hooks`` applies class REPLACE
# hooks first, so ``sglang.srt.layers.engram.EngramEmbedding`` resolves to the
# kunlun replacement class (int8 or fp8) and these AROUND hooks wrap its
# ``forward``/``_owned_rows`` rather than the base implementations.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.layers.engram.EngramEmbedding.forward",
    type=HookType.AROUND,
)
def _engram_forward_kunlun(
    original_fn, self, indices, forward_batch=None, *, cp_all_tokens=False
):
    # Capture/replay of the full decode graph: the host gather already ran in
    # the pre-replay pass, so the captured graph must only read that buffer.
    # See hooks/utils/engram_precompute.py.
    if engram_precompute.active():
        return engram_precompute.take_precomputed(self, indices)
    if _enabled() and self._shared:
        if indices.shape[0] == 0:
            return self._empty(indices)
        return _host_cpu_lookup(self, indices)
    if cp_all_tokens and not self._shared and self.tp_size > 1 and _rs_enabled():
        parallel = get_parallel()
        # attn_cp aliases the TP group when the sizes match, so the row shards are
        # disjoint across it and each hash id has exactly one nonzero owner: the
        # slice this rank keeps after the all-gather is the only contribution that
        # survives the old all-reduce, so reduce-scatter hands it the same bytes.
        if parallel.attn_cp_size == self.tp_size:
            local_rows = indices.shape[0]
            all_indices = indices.new_empty(
                (parallel.attn_cp_size * local_rows, *indices.shape[1:])
            )
            attn_cp_all_gather_into_tensor(all_indices, indices.contiguous())
            owned = self._owned_rows(all_indices)
            out = owned.new_empty((local_rows, *owned.shape[1:]))
            attn_cp_reduce_scatter_tensor(out, owned.contiguous())
            return out
    return original_fn(self, indices, forward_batch, cp_all_tokens=cp_all_tokens)


# ---------------------------------------------------------------------------
# EngramEmbedding._owned_rows — intercept the per_rank host-table shard.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.layers.engram.EngramEmbedding._owned_rows",
    type=HookType.AROUND,
)
def _engram_owned_rows_kunlun(original_fn, self, indices):
    if _enabled() and self.host_table is not None:
        return _host_cpu_lookup_sharded(self, indices)
    return original_fn(self, indices)
