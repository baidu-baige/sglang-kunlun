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
"""Tiered ``swa_step`` policy for AttentionStore SWA checkpoint placement.

The SWA KV backup writes one *checkpoint* whenever a request crosses a
``swa_step`` boundary.  With a single fixed ``swa_step`` (== page_size), long
requests produce a checkpoint every page, causing heavy write pressure on the
``chunk_tail_swa_pool`` and the AS SDK.

This module places checkpoints at *position-dependent* intervals: fine-grained
near the start of a request and coarser further out, e.g.

    positions [0, 1K)   -> step 256
    positions [1K, 4K)  -> step 1024
    positions [4K, inf) -> step 4096

Both the write path and the read path resolve checkpoint positions through the
*same* ``SwaStepPolicy`` instance, guaranteeing the invariant:

    floor_to_checkpoint(p) in checkpoint_positions(p)

so a read only ever restores at a position that the write path actually backed
up (and with the same block_step -> same block->swa_slot mapping in the SDK).

The policy is pure (no torch, no side effects) to keep it unit-testable.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# A tier is (upper_bound, step). ``upper_bound`` is exclusive on the high side
# and归属下档 (a position exactly on a tier's upper bound belongs to the LOWER
# tier). The last tier uses upper_bound == None to mean +inf.
Tier = Tuple[object, int]  # (Optional[int] upper_bound, int step)


class SwaStepPolicy:
    """Maps token positions <-> SWA checkpoints under tiered step sizes.

    Args:
        tiers: list of ``(upper_bound, step)`` in ascending ``upper_bound``
               order.  The final tier MUST have ``upper_bound == None`` (=inf).
               Every ``step`` must be a positive multiple of ``page_size``,
               ``step`` must be strictly increasing, and each finite
               ``upper_bound`` must be a multiple of that tier's ``step``.
        page_size: KV page size in tokens (checkpoint positions are always a
               multiple of this).
    """

    def __init__(self, tiers: List[Tier], page_size: int):
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if not tiers:
            raise ValueError("tiers must be non-empty")
        if tiers[-1][0] is not None:
            raise ValueError("last tier must have upper_bound=None (inf)")

        prev_bound = 0
        prev_step = 0
        for i, (bound, step) in enumerate(tiers):
            is_last = i == len(tiers) - 1
            if step <= 0 or step % page_size != 0:
                raise ValueError(
                    f"tier[{i}] step={step} must be a positive multiple of "
                    f"page_size={page_size}"
                )
            if step <= prev_step:
                raise ValueError(
                    f"tier[{i}] step={step} must be strictly greater than "
                    f"previous step={prev_step}"
                )
            if not is_last:
                if bound is None:
                    raise ValueError(f"non-last tier[{i}] must have finite bound")
                if bound <= prev_bound:
                    raise ValueError(
                        f"tier[{i}] bound={bound} must exceed previous "
                        f"bound={prev_bound}"
                    )
                if bound % step != 0:
                    raise ValueError(
                        f"tier[{i}] bound={bound} must be a multiple of "
                        f"step={step}"
                    )
                prev_bound = bound
            prev_step = step

        self.page_size = page_size
        # Normalize to (lower_bound_inclusive_open, upper_bound, step) segments.
        # segment i covers positions in (lower[i], upper[i]] mapped via step[i].
        self._lowers: List[int] = []
        self._uppers: List[object] = []  # int or None(inf)
        self._steps: List[int] = []
        lower = 0
        for bound, step in tiers:
            self._lowers.append(lower)
            self._uppers.append(bound)
            self._steps.append(step)
            lower = bound if bound is not None else lower

    # ── construction helpers ─────────────────────────────────────────────

    @classmethod
    def single(cls, swa_step: int, page_size: int) -> "SwaStepPolicy":
        """Single-tier policy replicating the legacy fixed-``swa_step`` behavior.

        Used when dynamic tiering is disabled so the whole pipeline runs the
        same code path with zero behavioral change.
        """
        return cls([(None, swa_step)], page_size)

    @classmethod
    def from_spec(cls, spec: str, page_size: int) -> "SwaStepPolicy":
        """Parse a config string like ``"1024:256,4096:1024,0:4096"``.

        Each comma-separated entry is ``bound:step``; ``bound == 0`` denotes the
        final +inf tier (must be the last entry).
        """
        tiers: List[Tier] = []
        entries = [e.strip() for e in spec.split(",") if e.strip()]
        if not entries:
            raise ValueError(f"empty swa_step_tiers spec: {spec!r}")
        for idx, entry in enumerate(entries):
            try:
                bound_s, step_s = entry.split(":")
                bound_i, step_i = int(bound_s), int(step_s)
            except ValueError as e:
                raise ValueError(
                    f"invalid swa_step_tiers entry {entry!r}: expected 'bound:step'"
                ) from e
            bound: object = None if bound_i == 0 else bound_i
            if bound is None and idx != len(entries) - 1:
                raise ValueError("bound=0 (inf) is only allowed as the last tier")
            tiers.append((bound, step_i))
        if tiers[-1][0] is not None:
            raise ValueError("swa_step_tiers must end with a '0:<step>' (inf) tier")
        return cls(tiers, page_size)

    # ── internal ─────────────────────────────────────────────────────────

    def _step_for_position(self, pos: int) -> int:
        """Return the step of the tier owning ``pos`` (boundary归下档).

        ``pos`` here is the *upper* end of a candidate checkpoint: a position
        exactly on a tier bound belongs to the lower tier (``pos <= bound``).
        """
        for lower, upper, step in zip(self._lowers, self._uppers, self._steps):
            if upper is None or pos <= upper:
                return step
        return self._steps[-1]  # unreachable (last upper is None)

    # ── public API ───────────────────────────────────────────────────────

    def checkpoint_positions(self, limit: int) -> List[int]:
        """All checkpoint token positions ``<= limit`` (ascending)."""
        out: List[int] = []
        for lower, upper, step in zip(self._lowers, self._uppers, self._steps):
            # first checkpoint strictly greater than ``lower``, aligned to step
            start = (lower // step + 1) * step
            hi = limit if upper is None else min(upper, limit)
            pos = start
            while pos <= hi:
                out.append(pos)
                pos += step
            if upper is not None and limit <= upper:
                break
        return out

    def num_checkpoints(self, length: int) -> int:
        """Number of checkpoints with position ``<= length``.

        Replaces the legacy ``length // swa_step``.
        """
        if length <= 0:
            return 0
        total = 0
        for lower, upper, step in zip(self._lowers, self._uppers, self._steps):
            hi = length if upper is None else min(upper, length)
            if hi <= lower:
                continue
            # count multiples of ``step`` in (lower, hi]
            total += hi // step - lower // step
            if upper is not None and length <= upper:
                break
        return total

    def checkpoint_pos(self, idx: int) -> int:
        """Token position of the ``idx``-th (0-based) checkpoint.

        Replaces the legacy ``(idx + 1) * swa_step``.
        """
        if idx < 0:
            raise IndexError(f"checkpoint index must be >= 0, got {idx}")
        remaining = idx
        for lower, upper, step in zip(self._lowers, self._uppers, self._steps):
            if upper is None:
                return (lower // step + 1 + remaining) * step
            count = upper // step - lower // step
            if remaining < count:
                return (lower // step + 1 + remaining) * step
            remaining -= count
        # unreachable: last tier is inf
        raise IndexError(idx)

    def block_step_at(self, idx: int) -> int:
        """``step // page_size`` for the tier owning the ``idx``-th checkpoint.

        This is the per-call ``swa_block_step`` scalar handed to the AS SDK for
        the segment containing this checkpoint.
        """
        pos = self.checkpoint_pos(idx)
        return self._step_for_position(pos) // self.page_size

    def block_step_for_position(self, pos: int) -> int:
        """``step // page_size`` of the tier owning token position ``pos``.

        Used on the read path: ``pos`` is a checkpoint-aligned hit boundary, so
        this returns the same ``swa_block_step`` the write path used for the
        segment ending at that checkpoint.
        """
        return self._step_for_position(pos) // self.page_size

    def floor_to_checkpoint(self, pos: int) -> int:
        """Largest checkpoint position ``<= pos`` (0 if none).

        Used on the read path to align an AS hit length to a real checkpoint,
        so the restored SWA state exists in AS.  Guaranteed to be an element of
        ``checkpoint_positions(pos)``.
        """
        if pos <= 0:
            return 0
        step = self._step_for_position(pos)
        return pos // step * step

    def segments(self, start: int, end: int) -> List[Tuple[int, int, int]]:
        """Split checkpoint index range ``[start, end)`` into same-``block_step``
        runs.

        Returns a list of ``(seg_start, seg_end, block_step)`` where each run
        covers a contiguous set of checkpoints sharing one ``block_step``, so
        each run maps to a single AS SDK ``write``/``read`` call carrying that
        scalar ``swa_block_step``.
        """
        segments: List[Tuple[int, int, int]] = []
        i = start
        while i < end:
            bs = self.block_step_at(i)
            j = i + 1
            while j < end and self.block_step_at(j) == bs:
                j += 1
            segments.append((i, j, bs))
            i = j
        return segments


def resolve_swa_step(server_args, page_size: int) -> int:
    """Resolve the ``swa_step`` field's ``0 == "use page_size"`` sentinel.

    Every consumer must go through this: ``SwaStepPolicy.single(0, page_size)``
    would raise, and the AS SDK is handed the same resolved value so both sides
    compute checkpoint positions identically.
    """
    return getattr(server_args, "swa_step", 0) or page_size


def build_swa_step_policy(server_args, page_size: int) -> SwaStepPolicy:
    """Construct the SWA step policy from server args.

    When ``enable_dynamic_swa_step`` is set, parses ``swa_step_tiers``;
    otherwise returns a single-tier policy replicating the fixed ``swa_step``
    behavior (zero behavioral change).
    """
    if getattr(server_args, "enable_dynamic_swa_step", False):
        policy = SwaStepPolicy.from_spec(server_args.swa_step_tiers, page_size)
        logger.info(
            "SWA step policy: dynamic tiers spec=%r page_size=%d -> segments=%s",
            server_args.swa_step_tiers,
            page_size,
            list(zip(policy._lowers, policy._uppers, policy._steps)),
        )
        return policy
    swa_step = resolve_swa_step(server_args, page_size)
    policy = SwaStepPolicy.single(swa_step, page_size)
    logger.info(
        "SWA step policy: fixed swa_step=%d page_size=%d (dynamic disabled)",
        swa_step,
        page_size,
    )
    return policy
