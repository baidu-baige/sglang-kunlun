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
"""Kunlun XPU fixes for the DSA prefill-CP path.

Two independent concerns share this module:

1. **dtype mismatch** (always on).  ``dsa_cp_gather_hidden_states`` gathers every
   CP rank's local hidden shard into one buffer::

       out = get_local_dp_buffer(attn_cp_group)            # dtype = dp.buffer_dtype
       attn_cp_all_gather_into_tensor(out, hidden_states)  # XCCL

   ``dp.buffer_dtype`` is stamped once from ``model_config.dtype``
   (``_DpGatheredBufferWrapper.set_metadata``). On this deployment the model's
   activations are bfloat16 (the engram path ships bf16, and the rest of the
   network follows) while ``model_config.dtype`` is float16 (from ``--dtype
   float16``), so the pre-allocated output and the activation disagree and XCCL
   refuses the op before any gather runs::

       TypeError: output tensor must have the same type as input tensor

   That kills every scheduler on the first CP prefill, so prefill context
   parallelism never comes up.

   The fix is to hand the upstream function a buffer of the *activation's* dtype:
   temporarily point ``dp.buffer_dtype`` at ``hidden_states.dtype`` for the call,
   so ``get_local_dp_buffer`` allocates exactly the buffer the upstream path
   expects -- same shape, same device, and still through ``use_symmetric_memory``
   (the Kunlun box runs with ``XSHMEM_MODE`` / ``BKCL_USE_PEERMEM_XDR``, so the
   collective buffer must stay a symmetric allocation) -- then restore it. Nothing
   is reallocated by hand and the upstream gather is otherwise untouched.  It is
   inert unless the two dtypes actually disagree, and only touches the prefill-CP
   gather: the inverse ``dsa_cp_reduce_scatter_hidden_states`` already splits one
   tensor into both roles, so it can never mismatch.

2. **the C2C-fabric wedge fix** (default ON, ``SGLANG_KUNLUN_CP_DRAIN=off`` to
   disable).  A full-speed CP prefill faults the node's C2C fabric: the engine
   dies with ``kl3ChannelCheckErrors ... status= 299`` (or ``700``) and dmesg
   shows ``KL_XID_KERNEL_EXCEPTION`` on every device, followed by a
   ``KL_XID_C2C_EVENT`` retry storm (type-1, port 3, code ``0x01030000``) that
   only a reboot clears.  Serializing *all* device work with
   ``CUDA_LAUNCH_BLOCKING=1`` removes the fault but costs ~30x throughput, so it
   is a diagnosis, not a deployment.

   The fault is *non-collective* device work upstream of the low-ratio indexer.
   The fix (run-030, single-variable bisect) is a single device drain right after
   ``_low_ratio_compress_torch`` -- the compress that runs on the gathered global
   tokens, immediately before the indexer.  Serialising that one boundary removes
   the wedge: CP-on then runs a full-speed 127728-token prefill at ~4.9k tok/s
   with ``noc idle=0``, ``kl3=0`` and a clean dmesg.

   The drain is unconditional -- run-046 showed the fault is NOT CP-specific (the
   CP-off arm wedges too, at a ~8k-token batch), so both A/B arms carry it.
   ``SGLANG_KUNLUN_CP_DRAIN=off`` makes it inert for bisection.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_reported = False


# In-path drain -- THE wedge fix (default ON, for BOTH arms).  run-026/027 localised
# the fault to *non-collective* device work upstream of the low-ratio indexer, but
# could not say which op: the wedge always surfaces at the first host<->device sync,
# which is the indexer's.  run-029/030 bisected the two candidate boundaries
# single-variably:
#   - drain only after `cp_materialize_global_token_order`  -> WEDGE (identical site)
#   - drain only after `_low_ratio_compress_torch`          -> PASS (full-speed
#     127728-token prefill, ~4.9k tok/s, noc idle=0 / kl3=0 / dmesg clean)
# so the compress boundary carries the fault and is the shipped fix.
#
# run-046 correction: the fault is NOT CP-specific.  With the all-features arm
# (deepep + engram + W8A8/W4A8) the CP-OFF arm wedges at a ~8k-token batch
# (error 299 "noc idle timeout" at `_low_ratio_index_topk_torch`'s first sync,
# `get_index_k_dequant`), and the wedge POISONS the node's C2C fabric (KL_XID_C2C_EVENT
# 0 -> 625 on a clean node, after which a fresh engine can no longer init).  An
# earlier CP-gated variant left CP-off undrained, so it wedged; draining the
# compress boundary on CP-off too (A/B on one clean node) clears the wedge:
# 4x1988-token concurrent batch passes, noc idle=0, C2C=0.  The drain is therefore
# unconditional now -- both A/B arms must carry it for the comparison to be meaningful.
#   SGLANG_KUNLUN_CP_DRAIN = "1"/unset -> drain compress + SWA-write (default)
#                          = "off"/"0" -> inert (bisection)
#
# run-047 fix (the 60k deadlock): the drain used a full-device
# ``torch.cuda.synchronize()`` -- a host barrier on *every* stream of the device.
# That is (a) the ~3% latency regression (it also waits on the XCCL collective
# stream, which has nothing to do with the compress->indexer host read) and (b)
# the cause of the 60k multi-chunk HANG: with interleave CP the per-rank token
# split diverges across chunks, so the ranks hit the compress/SWA drains a
# different number of times and in a different interleaving with their collectives;
# a per-rank *full*-device sync dropped into that collective-synchronised region
# deadlocks (rank A's synchronize() waits on a collective rank B has not posted yet
# because B is itself parked at an earlier synchronize()).  All 8 ranks then sit at
# 0% util -- the exact signature observed on the 60k single-request prefill (chunk-1
# at 16384 passes, the chunk crossing seq 32768 wedges).
#
# The producer->consumer race the drain exists to close is purely on the COMPUTE
# stream (the indexer reads host-side offsets/counts derived from the compress /
# SWA-write kernels), so draining only the *current* stream
# (``torch.cuda.current_stream().synchronize()``) is sufficient to order it, while
# leaving the collective stream free -- which removes the cross-rank deadlock and
# recovers the latency.  ``SGLANG_KUNLUN_CP_DRAIN_SYNC=full`` restores the old
# whole-device barrier for A/B.
#
_DRAIN = os.environ.get("SGLANG_KUNLUN_CP_DRAIN", "1")
_DRAIN_COMPRESS = _DRAIN not in ("off", "0", "none", "")
# The 60k multi-chunk path faults at the SWA-write boundary too (seq 32768), so the
# SWA drain is part of the default set now (was opt-in in run-046).  Explicit
# SGLANG_KUNLUN_CP_DRAIN_SWA=0 can still turn it off for bisection.
_DRAIN_SWA = os.environ.get(
    "SGLANG_KUNLUN_CP_DRAIN_SWA", "1" if _DRAIN_COMPRESS else "0"
) not in ("off", "0", "none", "")
# Host-barrier scope for the drain.  The default is ``safe``.
#
#   full   -- whole-device ``torch.cuda.synchronize()`` (all streams incl. the XCCL
#             collective PG stream).  Closes the OOB but DEADLOCKS the 60k multi-chunk
#             path: with interleave CP the per-rank token split diverges across chunks,
#             so a host barrier that waits on the collective stream parks rank A on a
#             collective rank B has not posted yet (B is itself parked) -> all 8 ranks
#             sit at 0% util at the chunk crossing seq 32768.  Diagnosis only.
#   stream -- current-stream-only ``current_stream().synchronize()``.  The compress and
#             SWA drains fire INSIDE ``with torch.cuda.stream(stream_sources/stream_kv)``
#             (deepseek_v4.py:1498 / 1527), so this drains exactly those two producer
#             side-streams.  Deadlock-free, but TOO WEAK at 60k: the indexer's host
#             read-back (deepseek_v4_backend.py:3275 ``torch.tensor(starts,...)`` ->
#             ``ks`` windows; 3615 ``int(lens.max().item())``) also consumes ``q_lora``/
#             ``q`` produced on the DEFAULT (compute) stream (1509/1514), which this
#             mode never drains -> OOB (error-700, addr 0xffffffff).
#   safe   -- (default) host-sync the current stream AND the default (compute) stream.
#             Covers the compute-stream producer the ``stream`` mode missed while still
#             NEVER touching the collective PG stream, so it closes the OOB without the
#             cross-rank deadlock.  A CUDA event between compress and the indexer is a
#             no-op here: they already share ``stream_sources`` (1498 & 1517) and the
#             barrier's real job is to gate a HOST read, which an event cannot do.
_DRAIN_SYNC = os.environ.get("SGLANG_KUNLUN_CP_DRAIN_SYNC", "safe")
_drain_reported = False
# [host-diet r85] ``torch.cuda.default_stream()`` / ``current_stream()`` without an
# explicit device go through the torch.device fast path and cost 6.3 us per call
# (measured on XPU, ``probe_hostdiet.py``); passing the device index costs 2.4 us,
# and the default stream handle itself is immutable for the life of the process,
# so it is built once per device.  ``current_stream()`` was verified identical to
# ``current_stream(current_device())`` including inside ``torch.cuda.stream(s)``.
_DEFAULT_STREAM_CACHE: dict = {}


def _default_stream_of(dev: int):
    s = _DEFAULT_STREAM_CACHE.get(dev)
    if s is None:
        s = torch.cuda.default_stream(dev)
        _DEFAULT_STREAM_CACHE[dev] = s
    return s


def _drain_device() -> None:
    """Order the producer kernels against the following host-side read.

    See ``_DRAIN_SYNC`` above for why ``safe`` (current + default/compute stream) is
    the shipped scope: it drains every producer the indexer reads back on the host
    (compress/indexer on the current stream, ``q_lora``/``q`` on the default stream)
    without the whole-device barrier that deadlocks the 60k multi-chunk path.
    """
    if _DRAIN_SYNC == "full":
        torch.cuda.synchronize()
    elif _DRAIN_SYNC == "stream":
        torch.cuda.current_stream().synchronize()
    else:  # "safe" (default)
        dev = torch.cuda.current_device()
        torch.cuda.current_stream(dev).synchronize()
        _default_stream_of(dev).synchronize()


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_compress_torch",
    type=HookType.AROUND,
)
def _low_ratio_compress_torch_drain(original_fn, *args, **kwargs):
    global _drain_reported
    out = original_fn(*args, **kwargs)
    if _DRAIN_COMPRESS:
        if not _drain_reported:
            logger.warning(
                "sglang-kunlun: SGLANG_KUNLUN_CP_DRAIN active (sync=%s) -- draining "
                "the device after _low_ratio_compress_torch (the CP prefill wedge fix)",
                _DRAIN_SYNC,
            )
            _drain_reported = True
        _drain_device()
    return out


# SWA-write drain.  The 60k CP prefill faults at seq position 32768 with only the
# compress drain; the fault surfaces at the next host<->device sync, which
# (per the entry/pre/post-op bisection) is in the window [previous layer's
# post-attention, this layer's attention entry].  The only device op in that window
# not yet covered is the SWA cache write (`set_swa_key_buffer_radix_fused_norm_rope`,
# the gather+RoPE scatter of the whole chunk's KV).  It is part of the default drain
# set now (gated with the compress drain); SGLANG_KUNLUN_CP_DRAIN_SWA=0 disables it.
_swa_reported = False


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope",
    type=HookType.AROUND,
)
def _set_swa_key_buffer_radix_fused_norm_rope_drain(original_fn, *args, **kwargs):
    global _swa_reported
    out = original_fn(*args, **kwargs)
    if _DRAIN_SWA:
        if not _swa_reported:
            logger.warning(
                "sglang-kunlun: SGLANG_KUNLUN_CP_DRAIN_SWA active (sync=%s) -- draining "
                "the device after set_swa_key_buffer_radix_fused_norm_rope",
                _DRAIN_SYNC,
            )
            _swa_reported = True
        _drain_device()
    return out


@plugin_hook(
    "sglang.srt.layers.communicator_dsa_cp.dsa_cp_gather_hidden_states",
    type=HookType.AROUND,
)
def _dsa_cp_gather_hidden_states_kunlun(original_fn, hidden_states):
    global _reported
    from sglang.srt.runtime_context import get_flags

    dp = get_flags().dp
    if dp.buffer_dtype == hidden_states.dtype:
        return original_fn(hidden_states)
    if not _reported:
        logger.warning(
            "sglang-kunlun: DSA prefill-CP gather dtype mismatch "
            "(buffer=%s activation=%s); allocating the gather buffer in the "
            "activation dtype",
            dp.buffer_dtype,
            hidden_states.dtype,
        )
        _reported = True
    saved = dp.buffer_dtype
    dp.buffer_dtype = hidden_states.dtype
    try:
        return original_fn(hidden_states)
    finally:
        dp.buffer_dtype = saved
