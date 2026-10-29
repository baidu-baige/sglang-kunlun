"""Align the TP ranks around graph capture.

`DecodeCudaGraphRunner.capture_prepare` builds the warmup batch with
``seq_lens_sum=seq_lens.sum().item()`` -- a blocking D2H that has to drain the
device queue -- while the full graph backend's warmup does
``device.synchronize(); tp_group.barrier()`` on the *device* group. On XPU a rank
that reaches the D2H while a peer is already parked in the device barrier
deadlocks: py-spy showed two ranks in ``capture_prepare`` and one in
``capture_one -> tp_group.barrier``.

Rendezvous on the CPU (gloo) group and drain the queue *before* capture_prepare
builds the batch, so no rank can enter the D2H while another holds a device
collective. It runs once per capture bucket, i.e. only during boot.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
    "DecodeCudaGraphRunner.capture_prepare",
    type=HookType.AROUND,
)
def capture_prepare_aligned_kunlun(original, self, *args, **kwargs):
    # On by default. Without it one rank enters capture_prepare's DP collective
    # (the DP-global-token path) while the other ranks are parked in
    # capture_one's TP barrier; the mismatch spins on CPU forever (py-spy: 7x
    # capture_one->barrier, 1x capture_prepare). Rendezvousing on the gloo group
    # first keeps the collectives in the same order on every rank.
    import os

    if os.environ.get("DSV4_CAPTURE_ALIGN", "1") != "1":  # default on
        return original(self, *args, **kwargs)
    try:
        import torch.distributed as dist

        from sglang.srt.distributed.parallel_state import get_tp_group

        torch.cuda.synchronize()
        group = get_tp_group().cpu_group
        if group is not None and dist.is_initialized():
            dist.barrier(group=group)
    except Exception as exc:  # noqa: BLE001 - alignment is best effort
        logger.debug("capture_prepare alignment skipped: %s", exc)
    return original(self, *args, **kwargs)
