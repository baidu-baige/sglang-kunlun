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
"""Give the DP-attention MLP-sync gather a CPU-capable collective.

With decode DP attention and Mooncake as the process-group backend, the
coordinator's ``cpu_group`` is a ``mooncake-cpu`` group. The disagg-decode
scheduler gathers its per-rank MLP-sync metadata on that group with
``device="cpu"``, and Mooncake has no CPU collective, so the first decode batch
dies with ``No backend type associated with device type cpu``.

A Gloo group is the only CPU-capable option in this torch build. It is created
lazily at the first CPU gather, where every DP rank enters the same collective,
so ``new_group`` is symmetric and cannot deadlock.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_WORLD_GLOO = None
_ANNOUNCED = False
_SWAP_ANNOUNCED = False


def _ensure_world_gloo():
    """Create the world Gloo group on first use."""
    global _WORLD_GLOO
    if _WORLD_GLOO is not None:
        return _WORLD_GLOO
    if not dist.is_available() or not dist.is_initialized() or not dist.is_gloo_available():
        return None
    _WORLD_GLOO = dist.new_group(backend="gloo", group_desc="kunlun:world-gloo")
    logger.warning("sglang-kunlun: created a world Gloo group for CPU collectives")
    return _WORLD_GLOO


def _backend_of(group):
    try:
        return dist.get_backend(group)
    except Exception:  # noqa: BLE001
        return ""


def _is_cpu_device(device) -> bool:
    """Accept both the string and the torch.device spelling of CPU."""
    if device == "cpu":
        return True
    return isinstance(device, torch.device) and device.type == "cpu"


def _all_gather_via_gloo(real, self, device, group, use_all_reduce=False):
    """Call the real gather, swapping a non-CPU group for the world Gloo group."""
    global _ANNOUNCED, _SWAP_ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        logger.warning(
            "sglang-kunlun: mlp-sync gather probe device=%r type=%s backend=%r "
            "initialized=%s gloo_available=%s",
            device, type(device).__name__, _backend_of(group),
            dist.is_initialized(), dist.is_gloo_available(),
        )
    if _is_cpu_device(device) and group is not None and _backend_of(group) != "gloo":
        gloo = _ensure_world_gloo()
        if gloo is not None:
            try:
                same_size = dist.get_world_size(group) == dist.get_world_size(gloo)
            except Exception:  # noqa: BLE001
                same_size = False
            if same_size:
                if not _SWAP_ANNOUNCED:
                    _SWAP_ANNOUNCED = True
                    logger.warning(
                        "sglang-kunlun: MLP-sync CPU gather on %r; using the world "
                        "Gloo group instead (logged once per process)",
                        _backend_of(group),
                    )
                group = gloo
            else:
                logger.error(
                    "sglang-kunlun: cannot replace the CPU gather group; sizes differ"
                )
    return real(self, device, group, use_all_reduce)


@plugin_hook(
    "sglang.srt.managers.scheduler_components.dp_attn.prepare_mlp_sync_batch_raw",
    type=HookType.AROUND,
)
def prepare_mlp_sync_batch_raw_kunlun(original_fn, *args, **kwargs):
    """Install the Gloo swap for the duration of one MLP-sync preparation.

    ``MLPSyncBatchInfo`` is a msgspec type; replacing its method permanently was
    not picked up by the scheduler's call site. Replacing the attribute around
    the caller's own execution guarantees the call dispatches through the
    wrapper, and restores the original immediately afterwards.
    """
    global _ANNOUNCED
    from sglang.srt.managers.scheduler_components import dp_attn as mod

    cls = mod.MLPSyncBatchInfo
    real = cls.all_gather
    cls.all_gather = lambda self, device, group, use_all_reduce=False: (
        _all_gather_via_gloo(real, self, device, group, use_all_reduce)
    )
    try:
        return original_fn(*args, **kwargs)
    finally:
        cls.all_gather = real
