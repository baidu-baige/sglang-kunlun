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
"""Kunlun fixes for the DSV4.1 attn-TP scatter/gather around the MoE a2a.

With a MoE a2a backend (DeepEP) and ``attn_tp_size > 1``, DSV4.1 splits the
hidden states over the attention-TP ranks before the dispatch and gathers them
back after the combine (``deepseek_v4.py``: ``_run_moe_ffn_dp_sync``). That
gather is an ``attn_tp_group`` collective *inside* the decode cuda graph, and on
Kunlun P800 both the collective itself and its first use break the capture. The
module keeps the split (it is what keeps the decode all-to-all from being
replicated ``attn_tp_size`` times) and makes it capturable; an env-gated
fallback that drops the split entirely is provided for bisection.

**1. The gather form.** Upstream ``dp_attention.attn_tp_all_gather`` uses the
*list* form of the collective (``parallel_state.py:1347``, under its own
"TODO(ch-wan): support other backends"), which lands in the XMLIR eager copy
path and aborts capture::

    ValueError: [RUNTIME ERROR]: error code= 0 ... cudaGetErrorString: no error
    (aten_capture/eager_customized/copy_kernel.cpp:428)

``attn_tp_all_gather_kunlun`` replaces it with either an
``all_gather_into_tensor`` (pad-to-max staging, because ``tensor_split`` chunks
differ by a row) or a zero-fill + SUM ``all_reduce`` emulation, selected by
``SGLANG_KUNLUN_ATTN_TP_GATHER`` (``allreduce`` default / ``into_tensor``).

**2. Cold attn-TP communicator.** Either form then faults with ``CUDA error: an
illegal memory access`` *inside* capture, because without DeepEP nothing ever
drives a collective on ``attn_tp_group`` -- its communicator resources get
created on first use, and first use is inside the capture, which is illegal.
Upstream's ``--pre-warm-nccl`` only warms ``tp_group``
(``bootstrap.py:_prewarm_nccl``), so ``prewarm_attn_tp_group`` drives the exact
ops on that group before ``DecodeCudaGraphRunner.capture`` runs.

**3. The same capture also has to pre-create DeepEP's low-latency RDMA buffer.**
The ~2 GB registration (``[XCCL] deepep.cpp:1944
bkcl_init_low_latency_buffer``) is triggered by the *first* low-latency
dispatch, which is the capture's first forward; ``prewarm_before_capture`` runs
one dummy dispatch eagerly so the registration happens outside the capture.
Note the buffer is sized by ``num_max_dispatch_tokens_per_rank``, whose default
is far larger than a decode step needs -- leaving
``SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`` unset makes the registration
~2 GiB and the first captured combine then faults; size it to the real per-rank
token count (e.g. ``32`` for the tp8/ep8 decode above) so it fits.

**Fallback: ``SGLANG_KUNLUN_DISABLE_A2A_SCATTER=1``** drops the split instead of
fixing it (``run_moe_ffn_dp_sync_kunlun`` below). Correct -- every rank then
dispatches all its tokens, as older builds did -- but it multiplies the decode
all-to-all traffic by ``attn_tp_size``, so it is the bisect / escape hatch, not
the default.
"""

from __future__ import annotations

import logging
import os
from typing import List

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_MODE = os.environ.get("SGLANG_KUNLUN_ATTN_TP_GATHER", "allreduce")
_PREWARM = os.environ.get("SGLANG_KUNLUN_PREWARM_ATTN_TP", "1") not in (
    "0",
    "off",
    "none",
    "",
)
_DISABLE_SCATTER = os.environ.get(
    "SGLANG_KUNLUN_DISABLE_A2A_SCATTER", "0"
) not in ("0", "", "off", "none")
_reported = False


def _gather_via_all_reduce(group, output_list: List[torch.Tensor], input: torch.Tensor):
    rows = [int(t.shape[0]) for t in output_list]
    tail = tuple(input.shape[1:])
    total = sum(rows)
    if total == 0:
        return None
    offsets = [0]
    for n in rows:
        offsets.append(offsets[-1] + n)
    rank = group.rank_in_group
    buf = input.new_zeros((total, *tail))
    if rows[rank] > 0:
        buf[offsets[rank] : offsets[rank + 1]].copy_(input)
    out = group.all_reduce(buf)
    if out is None:
        out = buf
    for i, dst in enumerate(output_list):
        if rows[i] > 0:
            dst.copy_(out[offsets[i] : offsets[i + 1]])
    return None


def _gather_via_into_tensor(group, output_list: List[torch.Tensor], input: torch.Tensor):
    rows = [int(t.shape[0]) for t in output_list]
    tail = tuple(input.shape[1:])
    max_rows = max(rows)
    if max_rows == 0:
        return None
    send = input
    if int(input.shape[0]) != max_rows:
        send = input.new_zeros((max_rows, *tail))
        if int(input.shape[0]) > 0:
            send[: input.shape[0]].copy_(input)
    elif not input.is_contiguous():
        send = input.contiguous()
    recv = input.new_empty((group.world_size * max_rows, *tail))
    group.all_gather_into_tensor(recv, send)
    recv = recv.view(group.world_size, max_rows, *tail)
    for i, (dst, n) in enumerate(zip(output_list, rows)):
        if n > 0:
            dst.copy_(recv[i, :n])
    return None


@plugin_hook(
    target="sglang.srt.layers.dp_attention.attn_tp_all_gather",
    type=HookType.REPLACE,
)
def attn_tp_all_gather_kunlun(output_list: List[torch.Tensor], input: torch.Tensor):
    """all_gather into a caller-provided list, without the uncapturable paths."""
    global _reported
    group = get_parallel().attn_tp_group
    if group.world_size == 1:
        output_list[0].copy_(input)
        return None
    assert len(output_list) == group.world_size, (
        f"attn_tp_all_gather expects one output per rank: "
        f"{len(output_list)} vs {group.world_size}"
    )
    if not _reported:
        logger.info("sglang-kunlun: attn_tp_all_gather via %s", _MODE)
        _reported = True
    if _MODE == "into_tensor":
        return _gather_via_into_tensor(group, output_list, input)
    return _gather_via_all_reduce(group, output_list, input)


@plugin_hook(
    target=(
        "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync"
    ),
    type=HookType.AROUND,
)
def run_moe_ffn_dp_sync_kunlun(
    original_fn, self, hidden_states, forward_batch, *, input_ids, input_ids_global
):
    """Drop the attn-TP split instead of making its gather capturable.

    ``_run_moe_ffn_dp_sync`` gates the split/gather on an inline
    ``get_parallel().attn_tp_size > 1``. With ``SGLANG_KUNLUN_DISABLE_A2A_SCATTER``
    set, this hook runs the body with that one width scoped to 1, so the gate is
    False and the whole ``attn_tp_group`` collective disappears from the captured
    graph -- every rank then dispatches all of its tokens, which is what older
    builds did. Correct, but it multiplies the decode all-to-all by
    ``attn_tp_size``; the capturable gather above is the default fix and this is
    the bisect / escape hatch.

    ``ParallelContext.override`` is the supported scoped form (the mechanism
    ``patch_tensor_parallel_group`` uses), so the width is restored on exit.
    """
    if not _DISABLE_SCATTER:
        return original_fn(
            self,
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
    with get_parallel().override(attn_tp_size=1):
        return original_fn(
            self,
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )


@plugin_hook(
    target=(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.capture"
    ),
    type=HookType.BEFORE,
)
def prewarm_before_capture(*args, **kwargs):
    """Warm everything that would otherwise initialise *inside* the capture.

    Two things, both fatal with ``illegal memory access`` when they land inside
    the capture:

    * ``attn_tp_group`` collectives -- nothing drives that group without DeepEP,
      and upstream ``--pre-warm-nccl`` only warms ``tp_group``
      (``bootstrap.py:_prewarm_nccl``);
    * DeepEP's low-latency RDMA buffer -- the registration
      (``[XCCL] deepep.cpp:1944 bkcl_init_low_latency_buffer``) is triggered by
      the first low-latency dispatch, which is the capture's first forward.
      ``DeepEPBuffer.clean_buffer()`` performs the same sizing call eagerly.
    """
    if not _PREWARM:
        return
    group = get_parallel().attn_tp_group
    device = torch.cuda.current_device()
    if group.world_size > 1:
        try:
            for numel in (1, 10240, 163840):
                buf = torch.zeros(numel, dtype=torch.float16, device=device)
                group.all_reduce(buf)
                out = torch.zeros(
                    numel * group.world_size, dtype=torch.float16, device=device
                )
                group.all_gather_into_tensor(out, buf)
            torch.cuda.synchronize()
            logger.info(
                "sglang-kunlun: pre-warmed attn_tp_group collectives (world=%d)",
                group.world_size,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.warning("sglang-kunlun: attn_tp_group pre-warm failed: %r", exc)

    try:
        _prewarm_deepep_low_latency(args[0] if args else None, device)
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.warning("sglang-kunlun: DeepEP low-latency pre-warm failed: %r", exc)


def _model_low_latency_dispatchers(runner):
    """Select communication geometry from this runner's model, including TBO."""
    model = getattr(getattr(runner, "model_runner", None), "model", None)
    if model is None:
        return
    seen = set()
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        for outer in getattr(dispatcher, "_inners", (dispatcher,)):
            impl = getattr(outer, "_low_latency_dispatcher", None)
            if impl is None or not impl.deepep_mode.enable_low_latency():
                continue
            key = (
                impl.group, impl.hidden_size, impl.params_bytes, impl.deepep_mode,
                impl.num_max_dispatch_tokens_per_rank, impl.num_experts,
                impl.router_topk,
            )
            if key not in seen:
                seen.add(key)
                yield impl


def _prewarm_deepep_low_latency(runner, device) -> None:
    """Run one dummy low-latency dispatch so the RDMA buffer is registered here.

    ``DeepEPBuffer.clean_buffer()`` is not enough: the registration
    (``[XCCL] deepep.cpp:1944 bkcl_init_low_latency_buffer``) is emitted by
    ``low_latency_dispatch`` itself, and its size depends only on
    ``num_max_dispatch_tokens_per_rank`` / hidden / experts -- so a single dummy
    token allocates exactly the same buffer the captured graph will use.
    """
    # The global last-active buffer may belong to the other DSpark model.
    # Warming it here can mutate native state used by an already captured graph.
    for impl in _model_low_latency_dispatchers(runner):
        buffer = impl._get_buffer()
        if not getattr(buffer, "low_latency_mode", False):
            continue
        hidden = impl.hidden_size
        num_experts = impl.num_experts
        tokens_per_rank = impl.num_max_dispatch_tokens_per_rank
        topk = impl.router_topk
        x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
        topk_ids = torch.zeros((1, topk), dtype=torch.int64, device=device)
        result = buffer.low_latency_dispatch(
            x,
            topk_ids,
            tokens_per_rank,
            num_experts,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=False,
        )
        torch.cuda.synchronize()
        del result
        logger.info(
            "sglang-kunlun: pre-warmed model-owned DeepEP low-latency dispatch "
            "(tokens/rank=%s, hidden=%s, experts=%s, topk=%s) before capture",
            tokens_per_rank,
            hidden,
            num_experts,
            topk,
        )
