"""Collapse the DP-attention sync readback into one ``tolist()``.

``MLPSyncBatchInfo.all_gather`` ends with eleven separate ATen calls over the
small host tensor it just copied down (``dp_attn.py:286-301``): four
slice+reduce pairs, three slice+``tolist`` pairs, and the scalar conversions.
Every one of them is a column view, i.e. a *strided* 8-element tensor, and a
strided reduction on this platform's torch build costs 4x what the contiguous
one does. Measured on this box (``debug/dp_sync_readback_bench.py``, dp_size=8,
attn_tp*attn_cp=4, int64, 20k iters x 7 reps, min):

    (a) upstream's 11 ops           70595 ns/call
    (b) one tolist() + python        7113 ns/call
    (c) one numpy() + column ops    19894 ns/call

    t[:, 2]                          3037 ns   (the slice alone)
    t[:, 2].min()                    8134 ns   strided 8-element reduce
    col.min() on a contiguous copy   2017 ns
    t.tolist()                       2673 ns   all 88 elements

So the whole readback is cheaper as one host copy plus python than as any of
its individual column reductions, and (b) wins over (c) because numpy pays the
same per-column overhead for columns it still has to materialise. Thread count
is irrelevant (identical at 1 and 128 threads); the scheduler runs at 1.

This is a ~63 us saving per gather on a step measured in tens of ms, so it is
worth exactly what it costs and no more: a fork of upstream's method body,
which is why the fast path refuses to run unless that body is the one it was
forked from (``_UPSTREAM_FINGERPRINT``) and why ``KUNLUN_DP_SYNC_FAST=0``
restores upstream without a code change.

Semantics are the upstream ones, statement for statement:
``bool(t.min())`` is kept as ``bool(min(...))`` and not ``all(...)`` -- those
differ when a column holds negative values -- ``.tolist()`` columns stay python
``list[int]``, ``int(t.max())`` stays a python ``int``, and ``self.tp0_info_cpu``
stays the tensor, because ``_update_gather_batch`` (``:335``),
``prepare_mlp_sync_batch_raw`` (``:659``, ``:691``) and
``TboDPAttentionPreparer.compute_output`` index it.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import time

import torch

from sglang.srt.managers.scheduler_components import dp_attn as _upstream
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_TARGET = (
    "sglang.srt.managers.scheduler_components.dp_attn.MLPSyncBatchInfo.all_gather"
)

# Captured at import, before ``HookRegistry.apply_hooks`` rebinds the attribute,
# so the fast path always has upstream to fall back to (same trick as
# ``hooks/models/absorbed_mla_bmm.py``'s ``_ATEN_BMM``).
_UPSTREAM_ALL_GATHER = _upstream.MLPSyncBatchInfo.all_gather

# sha1 of the upstream body this fork was taken from, whitespace-normalised.
# The collective half of the method is copied verbatim below; if upstream edits
# it, the copy would keep running the old collective, which is the one failure
# mode of a REPLACE hook that has no symptom. A mismatch is not fatal -- it
# costs 63 us -- so it logs once and delegates.
_UPSTREAM_FINGERPRINT = "8f951a93f691ce71"


def _fingerprint(fn) -> str | None:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover - source-less deployment
        return None
    body = "\n".join(line.rstrip() for line in src.splitlines()).strip()
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def _fast_path_enabled() -> bool:
    if not get_bool_env_var("KUNLUN_DP_SYNC_FAST", "true"):
        return False
    seen = _fingerprint(_UPSTREAM_ALL_GATHER)
    if seen == _UPSTREAM_FINGERPRINT:
        return True
    logger.warning(
        "sglang-kunlun: MLPSyncBatchInfo.all_gather body changed (fingerprint "
        "%s, forked from %s); using upstream's readback. Re-port "
        "hooks/managers/scheduler_components/dp_attn.py against the new body "
        "and update _UPSTREAM_FINGERPRINT.",
        seen,
        _UPSTREAM_FINGERPRINT,
    )
    return False


_FAST = _fast_path_enabled()

# One ``MLPSyncBatchInfo`` is built per forward, so per-instance caching buys
# nothing; these live at module scope. ``all_gather`` runs on the scheduler
# thread only and ends in a blocking ``.cpu()``, so a buffer handed to the
# collective is always drained before the next call rewrites it.
_LOCAL_STAGING: dict = {}
_FALLBACK_CACHE: dict = {}


def _fallback_tensor_cached(self, device, dtype=torch.int64) -> torch.Tensor:
    """``_get_fallback_tensor`` without the per-call host-to-device build.

    Its contents are constants and every use is a read (``repeat``, masked
    assignment), so one tensor per (device, dtype) is enough. Upstream rebuilt
    it on every call: three allocations and three transfers per decode step.
    """
    key = (str(device), dtype)
    cached = _FALLBACK_CACHE.get(key)
    if cached is None:
        cached = self._get_fallback_tensor(device=device, dtype=dtype)
        _FALLBACK_CACHE[key] = cached
    return cached


def _local_tensor_cached(self, device, dtype=torch.int64) -> torch.Tensor:
    """``_get_local_tensor`` through persistent pinned host + device buffers.

    ``torch.tensor([...], device=xpu)`` allocates a pinned staging buffer per
    call; measured 34.85 such ``to()`` calls and 2.72 ms per decode step. Fill
    one persistent pinned buffer and copy it into one persistent device buffer.
    The value order is upstream's ``_get_local_tensor`` order.
    """
    values = (
        self.num_tokens,
        self.num_tokens_for_logprob,
        int(self.can_run_decode_cuda_graph),
        int(self.is_extend_in_batch),
        int(self.local_can_run_tbo),
        self.local_forward_mode,
        int(self.can_run_prefill_cuda_graph),
        self.prefill_cuda_graph_max_prefix_len,
        int(self.can_run_draft_cuda_graph),
        int(self.is_health_check),
        self.attention_pairs,
    )
    key = (str(device), dtype, len(values))
    staging = _LOCAL_STAGING.get(key)
    if staging is None:
        # The value list above is a copy of upstream's ``_get_local_tensor``
        # body, and nothing in ``_UPSTREAM_FINGERPRINT`` covers that method, so
        # check it against upstream once per (device, dtype) before trusting it.
        expected = self._get_local_tensor(device="cpu", dtype=dtype).tolist()
        if list(values) != expected:
            logger.warning(
                "sglang-kunlun: MLPSyncBatchInfo._get_local_tensor layout "
                "changed (upstream %s vs fork %s); using upstream's builder.",
                expected,
                list(values),
            )
            _LOCAL_STAGING[key] = False
            return self._get_local_tensor(device=device, dtype=dtype)
        try:
            host = torch.empty(
                len(values), dtype=dtype, device="cpu", pin_memory=True
            )
        except RuntimeError:
            host = torch.empty(len(values), dtype=dtype, device="cpu")
        staging = (host, torch.empty(len(values), dtype=dtype, device=device))
        _LOCAL_STAGING[key] = staging
    elif staging is False:
        return self._get_local_tensor(device=device, dtype=dtype)
    host, dev = staging
    for i, value in enumerate(values):
        host[i] = value
    dev.copy_(host, non_blocking=host.is_pinned())
    return dev


def read_sync_fields(tp0_info_cpu: torch.Tensor):
    """The ten host-side values of ``dp_attn.py:288-300``, from one ``tolist()``.

    Returned in column order so the caller's assignments read like upstream's.
    Pure: no sglang import reaches this function, so the CPU test can load it
    on its own and diff it against upstream's statements.
    """
    rows = tp0_info_cpu.tolist()
    return (
        [r[0] for r in rows],  # global_num_tokens
        [r[1] for r in rows],  # global_num_tokens_for_logprob
        bool(min(r[2] for r in rows)),  # can_run_decode_cuda_graph
        bool(max(r[3] for r in rows)),  # is_extend_in_batch
        [r[5] for r in rows],  # local_forward_mode, for DPCooperationInfo
        bool(min(r[6] for r in rows)),  # can_run_prefill_cuda_graph
        int(max(r[7] for r in rows)),  # prefill_cuda_graph_max_prefix_len
        bool(min(r[8] for r in rows)),  # can_run_draft_cuda_graph
        bool(max(r[9] for r in rows)),  # any_health_check
        [r[10] for r in rows],  # global_attention_pairs
    )


@plugin_hook(_TARGET, type=HookType.REPLACE)
def all_gather_kunlun(self, device, group, use_all_reduce: bool = False):
    """``MLPSyncBatchInfo.all_gather`` with the readback done once.

    Lines up to ``tp0_info_cpu`` are upstream's, unchanged; the tail is
    :func:`read_sync_fields`. ``num_dp_ranks < 1`` (and every disabled case)
    goes to upstream so the degenerate shapes keep upstream's exceptions.
    """
    if not _FAST or self.num_dp_ranks < 1:
        return _UPSTREAM_ALL_GATHER(
            self, device=device, group=group, use_all_reduce=use_all_reduce
        )

    sync_start = time.perf_counter()
    local_info_tensor = _local_tensor_cached(self, device=device)
    fallback_tensor = _fallback_tensor_cached(self, device=device)
    info_width = local_info_tensor.numel()
    global_info_tensor = fallback_tensor.repeat(
        self.num_dp_ranks, self.tp_size * self.cp_size, 1
    )

    if use_all_reduce:
        global_info_tensor.zero_()
        flat_info = global_info_tensor.view(-1, info_width)
        rank = torch.distributed.get_rank(group)
        if 0 <= rank < flat_info.shape[0]:
            flat_info[rank] = local_info_tensor
        torch.distributed.all_reduce(
            global_info_tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        missing = flat_info.abs().sum(dim=1) == 0
        flat_info[missing] = fallback_tensor
    else:
        _upstream.exchange_sync_rows(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )

    tp_info = global_info_tensor.view(
        self.num_dp_ranks * self.tp_size * self.cp_size, info_width
    )
    num_ranks_in_tp_info = tp_info.shape[0]
    tp_group = _upstream.get_parallel().tp_group
    if device == "cpu":
        tp_active_ranks = tp_group.active_ranks_cpu
    else:
        tp_active_ranks = tp_group.active_ranks
    if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
        tp_active_ranks = torch.ones(
            num_ranks_in_tp_info,
            dtype=tp_active_ranks.dtype,
            device=tp_active_ranks.device,
        )
    tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

    # One D2H for every field: each `.item()` / `.tolist()` on a device
    # tensor is its own stream sync. Copy the whole tensor, not the
    # `[:, 0, :]` slice -- that slice is non-contiguous once
    # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
    tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
    self.tp0_info_cpu = tp0_info_cpu
    (
        self.global_num_tokens,
        self.global_num_tokens_for_logprob,
        self.can_run_decode_cuda_graph,
        self.is_extend_in_batch,
        forward_modes,
        self.can_run_prefill_cuda_graph,
        self.prefill_cuda_graph_max_prefix_len,
        self.can_run_draft_cuda_graph,
        self.any_health_check,
        self.global_attention_pairs,
    ) = read_sync_fields(tp0_info_cpu)
    self.sync_wait_seconds = time.perf_counter() - sync_start
    if _upstream._ENABLE_METRICS_DP_ATTENTION:
        self.dp_cooperation_info = _upstream.DPCooperationInfo.create(forward_modes)
