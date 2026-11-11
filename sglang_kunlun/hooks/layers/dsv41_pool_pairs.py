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
"""Kunlun XPU fast path for the ratio-compressor's 2-way score pooling.

Upstream ``DeepseekV41Compressor.pool_pairs`` is

    return (kv2 * score2.softmax(dim=1)).sum(dim=1)

with ``kv2`` / ``score2`` shaped ``[n, 2, D]`` (the two pooled KV candidates and
their fp32 scores).  The softmax runs over a *2-long* dim whose stride is ``D``,
so on this XPU build it lowers to a transpose plus a generic ``softmax_fwd``
kernel: 2.79 ms per call at the launch pose, ~0.05 s per 5-step window.

For a 2-element dim the softmax has a closed form::

    softmax([a, b]) = [sigmoid(a - b), sigmoid(b - a)]

and ``sigmoid(b - a) = 1 - sigmoid(a - b)``, so the pooled value is

    kv0 * s + kv1 * (1 - s),   s = sigmoid(a - b)

which is elementwise over ``[n, D]``: no transpose, one sigmoid, two muls and an
add.  In fp32 the two forms agree to the sigmoid's own rounding (the ``1 - s``
form is exact to 1 ulp of the complement), and the downstream consumer only sees
the pooled ``[n, D]`` tensor, so no other kernel changes.

``KL_POOL_PAIRS_SIGMOID=0`` restores the softmax form (A/B and rollback).
``KL_POOL_PAIRS_SELFCHECK=1`` recomputes the softmax form on the first call and
logs ``max|diff|``; it stays off in production because it costs one extra
softmax on that call.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_TARGET = (
    "sglang.srt.layers.attention.dsv4.dsv41_sparse."
    "DeepseekV41Compressor.pool_pairs"
)

_enabled = os.environ.get("KL_POOL_PAIRS_SIGMOID", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)
_selfcheck = os.environ.get("KL_POOL_PAIRS_SELFCHECK", "0") == "1"
_state = {"calls": 0, "reported": False}


@plugin_hook(_TARGET, type=HookType.REPLACE)
def _pool_pairs_sigmoid(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
    """Kunlun replacement for ``DeepseekV41Compressor.pool_pairs``."""
    if not _enabled or score2.shape[1] != 2:
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)

    s = torch.sigmoid(score2[:, 0] - score2[:, 1])
    out = kv2[:, 0] * s + kv2[:, 1] * (1.0 - s)

    _state["calls"] += 1
    if _selfcheck and not _state["reported"]:
        _state["reported"] = True
        ref = (kv2 * score2.softmax(dim=1)).sum(dim=1)
        scale = ref.abs().max().item()
        diff = (out - ref).abs().max().item()
        rel = diff / scale if scale else diff
        logger.warning(
            "sglang-kunlun: pool_pairs sigmoid selfcheck "
            "(max|diff|/max|ref| = %.3e, shape=%s)", rel, tuple(score2.shape)
        )
    elif not _state["reported"] and _state["calls"] == 1:
        _state["reported"] = True
        logger.info(
            "sglang-kunlun: pool_pairs 2-way softmax -> sigmoid "
            "(shape=%s)", tuple(score2.shape)
        )
    return out
