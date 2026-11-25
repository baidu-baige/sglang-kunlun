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
"""Native fp16<->bf16 cast, shared by the hooks that move 16-bit tensors.

``aten`` lowers ``t.to(dtype)`` to ``cluster_cast_kl3<T, O>``.  On P800 that
kernel runs the fp16 -> bf16 direction at ~800 GB/s and bf16 -> fp16 at ~1350
GB/s, while a plain copy of the same bytes reaches ~1800 GB/s.  The xspeedgate
SIMD converter (``cast_fp16_bf16`` / ``cast_bf16_fp16``, wheel
``xSGops20261021r`` and later) does a widening load and a narrowing store in one
pass, is bit-identical to ``.to`` (RNE, inf/nan pass through, out-of-range
finite values saturate to +-inf) and reaches 1.3-1.8 TB/s.

r68 trace, per capture (= 8 forwards, TP-0), every site routed here:

    wo_a exit    [T, 8, 1024]     fp16 -> bf16   50.05 ms   807 GB/s
    wo_a weight  [8, 1024, 4096]  bf16 -> fp16   31.77 ms  1351 GB/s
    MoE exit     [T, 5120]        bf16 -> fp16   19.15 ms  1317 GB/s
    mqa logits   [T, 32, 128]     fp16 -> bf16    3.12 ms  1300 GB/s
    hc_post      [T, 4, 5120]     fp16 -> bf16    3.08 ms   818 GB/s

``xsg_cast`` falls back to ``.to`` when the op is missing, the dtype pair is not
one of the two 16-bit pairs, the source is not contiguous, or the tensor is
empty (so the native kernel never meets a shape it was not gated for).
"""

from __future__ import annotations

import torch

_NAMES = {
    (torch.float16, torch.bfloat16): "cast_fp16_bf16",
    (torch.bfloat16, torch.float16): "cast_bf16_fp16",
}

# Resolved once; an older wheel leaves the entry as None and every call site
# keeps using ``.to`` (no exception, no per-call probing of the extension).
_RESOLVED = False
_OPS: dict = {}


def _op(name: str):
    global _RESOLVED
    if not _RESOLVED:
        _RESOLVED = True
        try:
            import xspeedgate_ops  # noqa: F401 -- registers torch.ops.xspeedgate_ops
        except Exception:  # noqa: BLE001 -- old/absent wheel: stay on .to()
            pass
        for n in _NAMES.values():
            try:
                op = getattr(torch.ops.xspeedgate_ops, n, None)
            except Exception:  # noqa: BLE001 -- namespace not registered
                op = None
            if op is not None:
                _OPS[n] = op
    return _OPS.get(name)


def xsg_cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t.to(dtype)``, through the xspeedgate SIMD converter when it can take it."""
    name = _NAMES.get((t.dtype, dtype))
    if (
        name is not None
        and t.device.type == "cuda"
        and t.is_contiguous()
        and t.numel() > 0
    ):
        op = _op(name)
        if op is not None:
            return op(t)
    return t.to(dtype)
