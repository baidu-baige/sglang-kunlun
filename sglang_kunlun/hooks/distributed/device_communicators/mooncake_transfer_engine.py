# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""HookRegistry registration for coalesced mooncake KV/state buffer registration.
"""

from __future__ import annotations

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_GAP = int(os.environ.get("SGLANG_KUNLUN_KV_REG_GAP", str(2 * 1024 * 1024)))
_ALIGN = int(os.environ.get("SGLANG_KUNLUN_KV_REG_ALIGN", str(2 * 1024 * 1024)))


def _coalesce(ptrs, lengths):
    """Merge overlapping / near (ptr, len) pairs into fewer regions.

    Args:
        ptrs: list of buffer start addresses.
        lengths: list of buffer byte lengths (paired with ``ptrs``).

    Returns:
        Sorted list of ``[start, end)`` merged regions.
    """
    gap = _GAP
    items = [(int(p), int(l)) for p, l in zip(ptrs, lengths) if int(l) > 0]
    items.sort(key=lambda x: x[0])
    merged = []
    for p, l in items:
        end = p + l
        # gap < 0 => NO merging at all (register each buffer individually,
        # so no region can straddle two distinct XPU allocation segments).
        if gap >= 0 and merged and p <= merged[-1][1] + gap:
            if end > merged[-1][1]:
                merged[-1][1] = end
        elif gap < 0 and merged and p < merged[-1][1] and end <= merged[-1][1]:
            # KUNLUN_GAPNEG_NOMERGE: gap<0 => register each buffer individually;
            # only DROP pure containment (fully inside prev) to avoid duplicate MRs;
            # never merge distinct/partial buffers (no MR straddles two allocations).
            pass
        elif gap >= 0 and merged and p < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([p, end])
    return merged


def _subtract_registered(regions, already):
    """Return the sub-ranges of ``regions`` not already covered by ``already``.

    Args:
        regions: candidate ``[start, end)`` regions to register.
        already: previously registered ``[start, end)`` regions.

    Returns:
        List of ``[start, end)`` sub-ranges that still need registration.
    """
    out = []
    for s, e in regions:
        cur = s
        for rs, re in already:
            if re <= cur or rs >= e:
                continue
            if rs > cur:
                out.append([cur, min(rs, e)])
            cur = max(cur, re)
            if cur >= e:
                break
        if cur < e:
            out.append([cur, e])
    return out


def _merge_into(already, regions):
    """Union two region lists into a minimal sorted non-overlapping set.

    Args:
        already: existing ``[start, end)`` regions.
        regions: new ``[start, end)`` regions to fold in.

    Returns:
        Merged sorted list of ``[start, end)`` regions.
    """
    m = []
    for s, e in sorted(already + regions):
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


def _align_regions(regions):
    """Round each region out to the ``_ALIGN`` (2MiB) boundary and re-merge.

    Args:
        regions: list of ``[start, end)`` regions.

    Returns:
        List of alignment-expanded, overlap-merged ``[start, end)`` regions.
    """
    a = _ALIGN
    if a <= 1:
        return regions
    if _GAP < 0:
        # KUNLUN_GAPNEG_NOMERGE: do not expand/merge; register raw per-buffer regions
        return regions
    aligned = []
    for s, e in regions:
        ns = s - (s % a)
        ne = e if e % a == 0 else e + (a - e % a)
        aligned.append([ns, ne])
    # merge overlaps produced by alignment
    m = []
    for s, e in sorted(aligned):
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


@plugin_hook(
    "sglang.srt.distributed.device_communicators.mooncake_transfer_engine."
    "MooncakeTransferEngine.batch_register",
    type=HookType.AROUND,
)
def batch_register_kunlun(original_fn, self, ptrs, lengths):
    """Register KV buffers with mooncake after coalescing/dedup/2MiB align.

    Args:
        original_fn: upstream ``MooncakeTransferEngine.batch_register``.
        ptrs: buffer start addresses to register.
        lengths: buffer byte lengths (paired with ``ptrs``).

    Returns:
        Return code from the underlying engine (0 on success / skip).
    """
    fresh = []
    regions = []
    try:
        regions = _coalesce(list(ptrs), list(lengths))
        regions = _align_regions(regions)
        already = getattr(self, "_kunlun_reg_ranges", [])
        fresh = _subtract_registered(regions, already)
        self._kunlun_reg_ranges = _merge_into(already, regions)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sglang-kunlun[coalesce]: coalesce failed (%s); registering raw", exc)
        return original_fn(self, ptrs, lengths)
    if not fresh:
        logger.info(
            "sglang-kunlun[coalesce]: %d region(s) already registered; skip", len(ptrs))
        return 0
    cp = [s for s, _ in fresh]
    cl = [e - s for s, e in fresh]
    logger.info(
        "sglang-kunlun[coalesce]: KV reg %d -> %d fresh region(s) (gap=%d)",
        len(ptrs), len(cp), _GAP)
    if os.environ.get("SGLANG_KUNLUN_KV_REG_DIAG") == "1":
        for _s, _l in zip(cp, cl):
            try:
                _ret = self.engine.register_memory(_s, _l)
            except Exception as _e:  # noqa: BLE001
                _ret = "EXC:%s" % _e
            logger.warning(
                "sglang-kunlun[regdiag]: register ptr=0x%x len=%d ret=%s", _s, _l, _ret)
        return 0
    return original_fn(self, cp, cl)
