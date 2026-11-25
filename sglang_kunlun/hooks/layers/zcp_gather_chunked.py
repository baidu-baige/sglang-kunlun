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
"""Kunlun XPU: chunked interleave-CP gather to bound the reorder copy.

At ``--chunked-prefill-size 61440`` upstream
``InterleaveCPStrategy._gather_interleaved_tensor`` allocates the full gathered
tensor (``cp * prl`` rows) and then a *second* full-size contiguous copy for the
``(cp, prl) -> (prl, cp)`` reorder::

    gathered = x.new_empty((cp * prl, *rest))          # buffer 1
    attn_cp_all_gather_into_tensor(gathered, padded_x)
    gathered.view(cp, prl, *rest).transpose(0, 1).reshape(total, *rest)  # buffer 2

For the hyper-connection-expanded hidden state (``tokens x 4 * hidden_size =
7680 x 20480`` bf16 per rank) that is 2.34 GiB twice.  The second allocation
dies inside a fragmented allocator (4.2 GiB free, 6.6 GiB reserved but
unallocated) and kills every scheduler on the first 61440-token chunk.

This AROUND hook keeps upstream semantics but gathers the ``prl`` rows in
blocks and writes each block straight into its final rows of one output
tensor::

    out[j * cp + r] = gathered[r * prl + j]

so the peak is ``out + one block`` instead of ``out + full gathered``.  The
block size targets 512 MiB of all-gather output, and only gathers whose output
would exceed 1.5 GiB are touched -- smaller tensors (the per-layer Kv-cache
materialisation, the plain hidden gather at chunk 32768) and the
``gather_index`` tail path go through the untouched upstream implementation.

The token mapping is identical to upstream:
``out[t] = gathered[(t % cp) * prl + t // cp]`` with ``t < total``; when ranks
are equal-length this is exactly the upstream transpose-reshape fast path.

A third hook reserves the indexer logits plane itself.  The ``[7680, 61436]``
fp32 plane is 1.76 GiB and cannot be allocated at forward time even with 3-4
GiB of reported free memory: the XPU allocator has no expandable segments, and
the forward peak leaves only small holes, so the request fails on whichever
rank draws the short straw.  The pool-configuration hook takes the plane once,
right after the weights are loaded and before the KV pool is charged
(``available_gpu_mem`` is ~24 GB there), and subtracts it from the KV budget so
the pool does not grow into the reservation.  ``_index_logits_init`` then hands
out contiguous prefix views of that block, so no forward-time allocation of
that size happens at all.  ``KL_IDX_PLANE_PREALLOC=0`` reverts to allocating per
forward.

A second, smaller hook on ``cp_gather_after_forward`` drops the second item of
the model-boundary tuple.  The DSV4 body returns
``(hidden_states[5120], pre_hc_head[4 * hidden = 20480])`` and
``LogitsProcessor`` only ever reads the pre-norm states when it is asked to
*capture hidden states*: ``_get_pruned_states`` slices both,
``_get_hidden_states_to_store`` returns ``pruned_states_before_norm`` and
``_get_logits`` never touches it.  On a plain generation request the 2.34 GiB
all-gather of ``pre_hc_head`` is therefore pure waste -- and it is the single
largest allocation of the forward, i.e. the one that cannot find a hole.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_TARGET = (
    "sglang.srt.layers.cp.interleave.InterleaveCPStrategy"
    "._gather_interleaved_tensor"
)
_MODEL_BOUNDARY = "sglang.srt.layers.cp.utils.cp_gather_after_forward"
_LOGITS_INIT = "sglang_kunlun.hooks.layers.dsv41_cp_indexer._index_logits_init"
_POOL_CALC = (
    "sglang.srt.model_executor.pool_configurator"
    ".DSV4PoolConfigurator.calculate_pool_sizes"
)


def _env_mb(name: str, default_mb: float) -> int:
    try:
        return int(float(os.environ.get(name, default_mb)) * (1 << 20))
    except (TypeError, ValueError):
        return int(default_mb * (1 << 20))


# Only gathers whose reordered output would exceed this go through the chunked
# path; everything else must stay bit-for-bit upstream (and keeps the 32768
# canonical pose on the original code path).
_MIN_TOTAL_OUT_BYTES = _env_mb("KL_CP_GATHER_MIN_MB", 1536)
# Target size of one all-gather output block; <= 0 means one single block.
_BLOCK_OUT_BYTES = _env_mb("KL_CP_GATHER_BLOCK_MB", 512)
# Debug-only: also run upstream and compare (needs free memory for both).
_SELFCHECK = os.environ.get("KL_CP_GATHER_SELFCHECK", "0") == "1"
_SELFCHECK_LOG = os.environ.get("KL_CP_GATHER_SELFCHECK_LOG", "")

# Drop the unused ``pre_hc_head`` half of the model-boundary gather.
_DROP_BEFORE_NORM = os.environ.get("KL_CP_DROP_BEFORE_NORM", "1") == "1"
# Memory trace of the first few big gathers (headroom for the next chunk size).
# Off by default: a diagnostic writer must be opted into, and its destination is
# the caller's, never a hard-coded developer directory.
_DIAG = os.environ.get("KL_CP_GATHER_DIAG", "0") == "1"
_DIAG_LOG = os.environ.get("KL_CP_GATHER_DIAG_LOG", "")
_DIAG_CALLS = [0]
_DIAG_LIMIT = 6


def _mem(tag: str, extra: str = "") -> str:
    try:
        free, total = torch.cuda.mem_get_info()
        return (
            f"{tag} free={free / 2**30:.2f} tot={total / 2**30:.2f} "
            f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} "
            f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} "
            f"rsvd={torch.cuda.memory_reserved() / 2**30:.2f} {extra}"
        )
    except (RuntimeError, AttributeError):
        return f"{tag} <no meminfo> {extra}"


def _log(line: str) -> None:
    if not _DIAG_LOG:
        return
    try:
        with open(_DIAG_LOG, "a") as fh:
            fh.write(f"[pid{os.getpid()}] {line}\n")
    except OSError:
        pass


def _diag_should_log() -> bool:
    if not _DIAG or not _DIAG_LOG:
        return False
    if _DIAG_CALLS[0] >= _DIAG_LIMIT:
        return False
    _DIAG_CALLS[0] += 1
    return True


# Upstream fallbacks that survived only by luck; see _upstream_gather below.
_OOM_RETRIES = [0]
_OOM_LOG_LIMIT = 20


def _upstream_gather(original_fn, self, x, forward_batch):
    """Run upstream's monolithic gather, handing the fragment cache back once.

    The guards below deliberately leave *mid-size* gathers on upstream's path,
    and upstream keeps two tensors of that size live at once (the all-gather
    buffer and the reorder copy).  At the accuracy pose -- ``--max-total-tokens
    250000`` with four concurrent business requests -- 61440 tokens x 10240 B is
    600 MiB, so the transient is 1.2 GiB, and the rank that draws the short
    straw reports 2.20 GiB free alongside 4.97 GiB reserved-but-unallocated and
    dies inside the allocator (2026-10-31 01:12:15, perf-024 attempt 000002);
    SIGQUIT then tears every scheduler down and the leg reports a health
    timeout instead of a measurement.  Nothing there has to be re-shaped or
    re-measured: the cache only has to be handed back before the retry.
    """
    try:
        return original_fn(self, x, forward_batch)
    except torch.OutOfMemoryError:
        _OOM_RETRIES[0] += 1
        if _OOM_RETRIES[0] <= _OOM_LOG_LIMIT:
            try:
                _log(
                    f"upstream-gather OOM #{_OOM_RETRIES[0]} "
                    f"shape={tuple(x.shape)} | " + _mem("oom")
                )
            except OSError:
                pass
        torch.cuda.empty_cache()
        return original_fn(self, x, forward_batch)


@plugin_hook(_TARGET, type=HookType.AROUND)
def _gather_interleaved_tensor_chunked(original_fn, self, x, forward_batch):
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return _upstream_gather(original_fn, self, x, forward_batch)

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if metadata is None or metadata.gather_index is not None:
        return _upstream_gather(original_fn, self, x, forward_batch)

    total_tokens = int(metadata.total_seq_lens)
    if total_tokens <= 0:
        return _upstream_gather(original_fn, self, x, forward_batch)

    per_tok_bytes = (x.numel() // x.shape[0]) * x.element_size()
    total_out_bytes = total_tokens * per_tok_bytes
    if total_out_bytes < _MIN_TOTAL_OUT_BYTES:
        return _upstream_gather(original_fn, self, x, forward_batch)

    logical_lens = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    local_logical_len = int(logical_lens[self.cp_rank])
    if x.shape[0] < local_logical_len:
        # Let upstream raise its own diagnostic for a malformed shard.
        return _upstream_gather(original_fn, self, x, forward_batch)

    cp = int(self.cp_size)
    prl = max(int(n) for n in metadata.per_rank_actual_token)
    if prl == 0 or cp == 0:
        return _upstream_gather(original_fn, self, x, forward_batch)

    rest = tuple(x.shape[1:])
    if _BLOCK_OUT_BYTES <= 0:
        block_rows = prl
    else:
        block_rows = max(
            1, min(prl, _BLOCK_OUT_BYTES // max(1, cp * per_tok_bytes))
        )

    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import (
        attn_cp_all_gather_into_tensor,
        is_allocation_symmetric,
    )
    from sglang.srt.runtime_context import get_parallel

    if local_logical_len < prl:
        src_all = x.new_zeros((prl, *rest))
        src_all[:local_logical_len] = x[:local_logical_len]
    else:
        src_all = x

    group = get_parallel().attn_cp_group
    sym_disabled = not is_allocation_symmetric()

    def _assemble():
        # One output tensor in the final (prl, cp) row grid; row j of rank r
        # lands at out[j * cp + r], so a contiguous j-range is a contiguous
        # out-range.
        out = x.new_empty((prl * cp, *rest))
        dst = out.view(prl, cp, *rest)
        for j0 in range(0, prl, block_rows):
            j1 = min(j0 + block_rows, prl)
            n = j1 - j0
            src = src_all[j0:j1]
            if not src.is_contiguous():
                src = src.contiguous()
            with use_symmetric_memory(group, disabled=sym_disabled):
                tmp = x.new_empty((cp * n, *rest))
            attn_cp_all_gather_into_tensor(tmp, src)
            # tmp.view(cp, n)[r, j - j0] = rank r's row j  ->  out[j * cp + r]
            dst[j0:j1].copy_(tmp.view(cp, n, *rest).transpose(0, 1))
        return out

    diag = _diag_should_log()
    if diag:
        _log(
            f"gather-in shape={tuple(x.shape)} total={total_tokens} cp={cp} "
            f"prl={prl} block_rows={block_rows} out={total_out_bytes / 2**30:.2f}GiB | "
            + _mem("before")
        )

    try:
        out = _assemble()
    except torch.OutOfMemoryError:
        # At the largest chunk the allocator can be sitting on several GiB of
        # reserved-but-unallocated segments and still fail a 2.5 GiB request;
        # hand them back once and retry before letting the scheduler die.
        try:
            _log(
                f"OOM shape={tuple(x.shape)} total={total_tokens} prl={prl} | "
                + _mem("oom")
            )
        except OSError:
            pass
        torch.cuda.empty_cache()
        out = _assemble()

    if diag:
        _log(f"gather-out total={total_tokens} | " + _mem("after"))

    result = out[:total_tokens]
    if _SELFCHECK:
        ref = original_fn(self, x, forward_batch)
        ok = torch.equal(ref, result)
        try:
            with open(_SELFCHECK_LOG, "a") as fh:
                fh.write(
                    f"shape={tuple(x.shape)} total={total_tokens} cp={cp} "
                    f"prl={prl} block_rows={block_rows} equal={ok}\n"
                )
        except OSError:
            pass
        return ref
    return result


@plugin_hook(_MODEL_BOUNDARY, type=HookType.AROUND)
def _cp_gather_after_forward_drop_pre_hc(original_fn, x, forward_batch, stream=None):
    """Skip the CP all-gather of ``pre_hc_head`` when nothing will read it.

    ``DeepseekV4Model.forward`` returns ``(hidden_states[hidden], pre_hc_head
    [4 * hidden])``; ``cp_gather_after_forward`` gathers both, i.e. a 0.59 GiB
    and a 2.34 GiB all-gather per forward at chunk 61440.  The pre-norm half is
    consumed only by ``LogitsProcessor``'s hidden-state *capture* path, so on a
    plain generation request it is dead weight.  Returning a bare tensor makes
    ``_execute_extend_cp`` pass ``hidden_states_before_norm=None``.
    """
    # Last layer of the forward is done: hand the indexer logits plane back so
    # the rest of the teardown (this gather included) sees upstream's peak.
    _drop_logits_planes()
    if not _DROP_BEFORE_NORM or not (isinstance(x, tuple) and len(x) == 2):
        return original_fn(x, forward_batch, stream)
    final, pre_hc = x
    if not (isinstance(final, torch.Tensor) and isinstance(pre_hc, torch.Tensor)):
        return original_fn(x, forward_batch, stream)
    if final.shape[1:] == pre_hc.shape[1:]:
        # Cannot tell which one is the pre-norm states: leave it to upstream.
        return original_fn(x, forward_batch, stream)
    mode = getattr(forward_batch, "capture_hidden_mode", None)
    try:
        if mode.need_capture():
            return original_fn(x, forward_batch, stream)
    except AttributeError:
        return original_fn(x, forward_batch, stream)
    if getattr(forward_batch, "extend_return_logprob", False):
        return original_fn(x, forward_batch, stream)

    diag = _diag_should_log()
    if diag:
        _log(
            f"model-boundary drop pre_hc={tuple(pre_hc.shape)} "
            f"keep={tuple(final.shape)} | " + _mem("before")
        )
    out = original_fn(final, forward_batch, stream)
    if diag:
        _log("model-boundary gather done | " + _mem("after"))
    return out


# ---------------------------------------------------------------------------
# Indexer logits plane: one block per forward instead of eight.
#
# ``dsv41_cp_indexer._index_logits_init`` allocates ``[num_tokens, max_seqlen_k]``
# fp32 -- 7680 x 61436 = 1.76 GiB in this pose -- and it is called once from
# *each* of the 8 indexer layers, so a single forward burns 14 GiB of
# allocator churn through that one size class.  The block is the largest any
# 61440-token forward asks for, so the churn both fragments the pool and
# leaves the next request's request for the same 1.76 GiB depending on whether
# anything carved up the freed block in between; two of the observed scheduler
# deaths were exactly this (``Tried to allocate 1.76 GiB ... 4.61 GiB free``).
#
# Holding one plane for the whole forward turns 8 allocations into 1 while
# keeping the *peak* identical to upstream: the block is dropped at the model
# boundary hook below, i.e. as soon as the last layer is done, before the
# boundary gather and the rest of the forward's teardown.  ``KL_IDX_PLANE_CACHE``
# selects the behaviour: 2 (default) = keep the plane resident across forwards,
# 1 = one plane per forward (drop at the model boundary), 0 = upstream, a fresh
# plane per layer.  Resident is the default because the XPU allocator cannot
# hand out a 1.76 GiB *contiguous* block out of its fragmented tail even with
# 4.6 GiB reported free, so the only reliable way to get that block is to take
# it once, when the device is still empty, and never give it back.
_PLANE_CACHE = {}
_PLANE_MODE = os.environ.get("KL_IDX_PLANE_CACHE", "2")
# Reserve the plane once, at pool-configuration time, instead of at forward time.
_PLANE_PREALLOC = os.environ.get("KL_IDX_PLANE_PREALLOC", "1") == "1"
_PLANE_RESERVE = {}  # str(device index) -> flat fp32 tensor holding the plane
_PLANE_RESERVE_LIMIT = 4 << 30  # refuse to reserve more than 4 GiB
# Widest plane the run will ask for, in columns.  ``chunked_prefill_size`` bounds
# the *chunk* a rank takes, not the *sequence* it attends over: a 110K request is
# served in the same 61440-token chunk but its score plane is
# ``[ceil(chunk/cp), max_seqlen_k]``, i.e. 112636 columns.  The reservation is
# taken once, before the pool is carved, so it has to be sized for the widest
# cell of the grid rather than for the first one; 0 (the default) keeps the
# chunk width, which is enough for any grid whose contexts fit in one chunk.
try:
    _PLANE_COLS = int(os.environ.get("KL_IDX_PLANE_COLS", "0") or 0)
except ValueError:  # a typo must not take the engine down at import time
    _PLANE_COLS = 0
# Row width handed out by this module must match ``dsv41_cp_indexer._PLANE_ALIGN``:
# that module rounds the plane up to the candidate block size (8) so the
# candidate publisher can pool a whole row as ``unflatten(-1, (n, 8))``.  Core
# asks for ``ceil_align(max(lc_per_req), 4)`` = 61436, and 61436 % 8 == 4, so a
# 61436-wide view here would hand the publisher a strided row and ATen would
# materialise a contiguous body copy first (``strided_slice_copy2d``,
# 0.26 ms x 32 = 2.10 ms/forward).  The reservation is a flat buffer whose
# ``cols`` is the chunk size (a multiple of 8), so the four extra columns of a
# row already exist in it -- this only changes how the view is shaped.  Same env
# var and same cold-start semantics as the indexer (``KL_IDX_PLANE_ALIGN8=0``
# turns both off).
_PLANE_ALIGN = 8 if os.environ.get("KL_IDX_PLANE_ALIGN8", "1") not in (
    "",
    "0",
    "false",
    "False",
) else 1


def _plane_cols(width):
    """Row width the publisher can pool in place: ``width`` rounded up to 8."""
    width = int(width)
    if _PLANE_ALIGN > 1:
        return (width + _PLANE_ALIGN - 1) // _PLANE_ALIGN * _PLANE_ALIGN
    return width


def _dev_key(device):
    if isinstance(device, torch.device):
        return str(device.index if device.index is not None else torch.cuda.current_device())
    return str(device)


def _plane_capacity():
    """``(rows, cols)`` the logits plane must cover in this pose, or ``None``.

    ``_index_logits_init`` is called with this rank's token count and the widest
    sequence in the batch: ``ceil(tokens / cp)`` rows by ``chunked_prefill_size``
    columns covers every prefill chunk that fits in one scheduler budget, and
    ``KL_IDX_PLANE_COLS`` widens the reservation when the grid serves contexts
    longer than one chunk (see the constant).
    """
    try:
        from sglang.srt.runtime_context import get_parallel, get_server_args

        chunk = int(getattr(get_server_args(), "chunked_prefill_size", 0) or 0)
        if chunk <= 0:
            return None
        par = get_parallel()
        cp = 0
        for name in ("attn_cp_size", "cp_size", "tp_size"):
            try:
                cp = int(getattr(par, name) or 0)
            except (AttributeError, TypeError, ValueError):
                cp = 0
            if cp > 0:
                break
        if cp <= 1:
            # No CP: the dense CP indexer path (and hence this plane) is unused.
            return None
        rows = (chunk + cp - 1) // cp + 2
        # One chunk is not the largest forward: the scheduler packs a whole chunk
        # *plus* the remainder of the next request (observed on the business
        # dataset: 61,440 + 3,712 tokens => 8,144 rows against the 7,682 a bare
        # chunk needs).  An eighth of a chunk of margin covers that remainder
        # twice over; it costs 0.44 GiB of the budget the ``--max-total-tokens``
        # cap leaves unused, so the pool does not shrink and the business leg
        # does not need a mid-forward reallocation.  ``_grow_reserve`` below is
        # still the safety net for anything bigger.
        rows += max(2, chunk // cp // 8)
        return rows, _plane_cols(max(chunk, _PLANE_COLS))
    except Exception:
        return None


def _reserve_plane() -> int:
    """Allocate the logits plane before the KV pool takes its memory."""
    if not _PLANE_PREALLOC or _PLANE_MODE == "0":
        return 0
    if _PLANE_RESERVE:
        return sum(t.numel() * t.element_size() for t in _PLANE_RESERVE.values())
    dims = _plane_capacity()
    if dims is None:
        return 0
    rows, cols = dims
    need = rows * cols * 4
    if need > _PLANE_RESERVE_LIMIT:
        _log(f"plane prealloc refused rows={rows} cols={cols} need={need / 2**30:.2f}GiB")
        return 0
    try:
        dev = torch.cuda.current_device()
        free, _total = torch.cuda.mem_get_info()
        if free < need * 2:
            _log(
                f"plane prealloc skipped dev={dev} free={free / 2**30:.2f}GiB "
                f"need={need / 2**30:.2f}GiB"
            )
            return 0
        buf = torch.empty(rows * cols, dtype=torch.float32, device=dev)
        _PLANE_RESERVE[str(dev)] = buf
        _log(
            f"plane prealloc rows={rows} cols={cols} {need / 2**30:.2f}GiB "
            f"dev={dev} | " + _mem("prealloc")
        )
        return need
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        _log(f"plane prealloc failed: {type(exc).__name__}: {exc}")
        return 0


def _plane_from_reserve(num_tokens, max_seqlen_k, device):
    """A contiguous view of the reserved plane, or ``None`` if it does not fit.

    The row is widened to the candidate block size when the reservation allows
    it (``_plane_cols``); the four extra columns of a 61436-column request are
    already inside the reserved chunk-width rows, so this costs no memory and
    only makes the row poolable without a copy.  Falls back to the requested
    width when the reservation is too small for the aligned shape.
    """
    buf = _PLANE_RESERVE.get(_dev_key(device))
    if buf is None:
        return None
    n, w = int(num_tokens), int(max_seqlen_k)
    if n <= 0 or w <= 0:
        return None
    wide = _plane_cols(w)
    if n * wide > buf.numel():
        wide = w
        if n * wide > buf.numel():
            return None
    # Scratch: every layer writes ``[0, w)`` of its own rows and the tail is
    # either sentinel-filled or never read, so reusing the block is safe.
    return buf[: n * wide].view(n, wide)


def _resolve_device(device):
    """A real ``torch.device`` for the string key the plane helpers carry.

    ``_plane_key``/``_dev_key`` hand the cache key -- ``"0"``.."7" -- to the
    reserve helpers, and ``torch.empty(device="0")`` raises ``RuntimeError:
    Invalid device string: '0'`` instead of allocating.  That made ``_grow_reserve``
    unable to allocate *anything*: every oversized batch fell through to the
    unguarded retry in ``_alloc_logits_plane`` and took the scheduler down with
    it (business leg, 2026-10-29 20:23, ``Tried to allocate 4.52 GiB``).
    """
    if isinstance(device, torch.device):
        return device
    text = str(device)
    if text.isdigit():
        return torch.device("cuda", int(text))
    return torch.device("cuda", torch.cuda.current_device())


_FACT_FUNCS = (
    "_low_ratio_index_topk_dense_kunlun",
    "_dense_fp4_mqa_logits_kunlun",
    "_dense_fp4_mqa_logits_torch",
    "_dense_bf16_mqa_logits_torch",
)


def _fact_value(value):
    if isinstance(value, (int, float, bool, str)):
        return value
    try:
        if hasattr(value, "numel") and hasattr(value, "tolist"):
            if value.numel() <= 16:
                return value.tolist()
            return f"t{list(value.shape)}"
        if hasattr(value, "shape"):
            return f"t{list(value.shape)}"
        return list(value)[:8]
    except Exception:
        return type(value).__name__


def _caller_facts(limit=3):
    """Locals of the indexer frames that decided this plane's geometry.

    The hook only sees ``(num_tokens, max_seqlen_k, device)``.  When a batch asks
    for a plane the reservation cannot cover, the numbers that produced it
    (``seq_lens_cpu``, ``ratio``, ``q_lens``) are what the next reservation has
    to be sized from -- and they only exist in the caller's frame.
    """
    facts, depth = [], 0
    try:
        import sys

        frame = sys._getframe(1)
    except Exception:
        return facts
    while frame is not None and depth < 40 and len(facts) < limit:
        name = frame.f_code.co_name
        if name in _FACT_FUNCS:
            loc = frame.f_locals
            item = {"fn": name}
            for field in ("ratio", "num_tokens", "max_seqlen_k", "epi_width",
                          "q_lens", "seq_lens_cpu", "lc_per_req", "ks", "ke"):
                if field in loc:
                    item[field] = _fact_value(loc[field])
            facts.append(item)
        frame = frame.f_back
        depth += 1
    return facts


def _grow_reserve(n, w, device):
    """Replace the reservation with a plane of ``n`` x ``w`` floats, or ``None``.

    The reservation is taken before the KV pool is carved, so its geometry is a
    guess at the largest forward (see ``_plane_capacity``).  When a batch beats
    that guess the old code fell through to a *second* allocation while the
    reservation was still held, i.e. it asked for 6.76 GiB of tail to service a
    3.48 GiB request -- which is how the business leg died at ``4.21 GiB free``.
    Handing the reservation back first makes the new plane cost exactly what it
    asks for; keeping it as the reservation means the high-water mark is paid
    once instead of once per forward.
    """
    key = _dev_key(device)
    real_dev = _resolve_device(device)
    # Same block-aligned row as ``_plane_from_reserve`` so a grown plane keeps
    # the publisher's zero-copy view instead of regressing to the strided one.
    w = _plane_cols(w)
    n = int(n)
    need = n * w
    old = _PLANE_RESERVE.pop(key, None)
    del old  # release the reservation before asking for its replacement
    _PLANE_CACHE.clear()
    free = 0
    try:
        torch.cuda.empty_cache()
        free, _total = torch.cuda.mem_get_info(real_dev)
        buf = torch.empty(need, dtype=torch.float32, device=real_dev)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        _log(
            f"plane grow failed rows={n} cols={w} need={need * 4 / 2**30:.2f}GiB "
            f"free={free / 2**30:.2f}GiB dev={real_dev}: {type(exc).__name__}: {exc}"
            f" | facts={_caller_facts()}"
        )
        return None
    _PLANE_RESERVE[key] = buf
    _log(
        f"plane reserve grown rows={n} cols={w} {need * 4 / 2**30:.2f}GiB "
        f"free_before={free / 2**30:.2f}GiB dev={key} | " + _mem("grown")
    )
    return buf[:need].view(n, w)


@plugin_hook(_POOL_CALC, type=HookType.AROUND)
def _dsv4_pool_calc_reserve_plane(original_fn, self, available_bytes, page_size):
    """Take the logits plane before the KV pool is charged for the memory.

    ``available_bytes`` is the byte budget the caller measured for the pool; the
    reservation is subtracted so the pool does not grow into it.  With
    ``--max-total-tokens`` set the final pool is token-capped anyway, so this
    only keeps the accounting honest.
    """
    reserved = _reserve_plane()
    if reserved > 0:
        available_bytes = max(int(available_bytes) - reserved, 0)
    return original_fn(self, available_bytes, page_size)


def _drop_logits_planes(force: bool = False) -> None:
    if not force and _PLANE_MODE == "2":
        return
    if _PLANE_CACHE:
        _PLANE_CACHE.clear()


def _plane_key(args, kwargs):
    """Cache key for an ``_index_logits_init(num_tokens, max_seqlen_k, device)``."""
    if kwargs or len(args) != 3:
        return None
    num_tokens, max_seqlen_k, device = args
    if not isinstance(num_tokens, int) or not isinstance(max_seqlen_k, int):
        return None
    return (num_tokens, max_seqlen_k, _dev_key(device))


def _alloc_logits_plane(original_fn, args, kwargs):
    try:
        return original_fn(*args, **kwargs)
    except torch.OutOfMemoryError:
        try:
            shapes = [
                tuple(a.shape) if isinstance(a, torch.Tensor) else a
                for a in args
            ]
            _log(f"index-logits OOM args={shapes} | " + _mem("oom"))
        except OSError:
            pass
        _drop_logits_planes(force=True)
        torch.cuda.empty_cache()
        out = original_fn(*args, **kwargs)
        _log("index-logits retry ok | " + _mem("after"))
        return out


@plugin_hook(_LOGITS_INIT, type=HookType.AROUND)
def _index_logits_init_plane_cache(original_fn, *args, **kwargs):
    """Reuse the logits plane across the 8 indexer layers of one forward.

    The plane is scratch: every layer writes its own ``[0, width)`` per row and
    the tail is either sentinel-filled or never read (see ``_logit_tail_needed``),
    so a reused block is indistinguishable from the freshly allocated one that
    upstream hands out.  A real OOM still propagates -- the retry after
    ``empty_cache`` is not guarded.
    """
    key = _plane_key(args, kwargs)
    if key is None or _PLANE_MODE not in ("1", "2"):
        return _alloc_logits_plane(original_fn, args, kwargs)
    if _PLANE_RESERVE:
        num_tokens, max_seqlen_k, dev = key
        view = _plane_from_reserve(num_tokens, max_seqlen_k, dev)
        if view is not None:
            return view
        _log(
            f"plane reserve miss key={key} reserved="
            f"{', '.join(f'{k}:{v.numel()}' for k, v in _PLANE_RESERVE.items())}"
            f" | facts={_caller_facts()}"
        )
        grown = _grow_reserve(num_tokens, max_seqlen_k, dev)
        if grown is not None:
            return grown
    buf = _PLANE_CACHE.get(key)
    if buf is not None:
        return buf
    buf = _alloc_logits_plane(original_fn, args, kwargs)
    _PLANE_CACHE.clear()
    _PLANE_CACHE[key] = buf
    return buf
