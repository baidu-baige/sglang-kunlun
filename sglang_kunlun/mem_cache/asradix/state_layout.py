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
"""Generic packed state layouts for checkpoint endpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Mapping, Optional, Sequence

import torch


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class StateTensorSpec:
    """A tensor slice addressed by a logical slot group."""

    name: str
    tensor: torch.Tensor
    group: str = "state"
    slot_dim: int = 0
    items_per_slot: int = 1
    base_index: int = 0
    slot_count_limit: Optional[int] = None

    def __post_init__(self):
        """Normalize and validate the logical tensor slice."""
        if not self.name:
            raise ValueError("state tensor name must not be empty")
        if not self.group:
            raise ValueError(f"state tensor {self.name}: group must not be empty")
        if self.tensor.ndim == 0:
            raise ValueError(
                f"state tensor {self.name}: scalar tensors are unsupported"
            )
        slot_dim = self.slot_dim % self.tensor.ndim
        object.__setattr__(self, "slot_dim", slot_dim)
        if self.items_per_slot <= 0:
            raise ValueError(
                f"state tensor {self.name}: items_per_slot must be positive"
            )
        if self.base_index < 0 or self.base_index >= self.tensor.shape[slot_dim]:
            raise ValueError(
                f"state tensor {self.name}: invalid base_index={self.base_index}"
            )
        remaining = self.tensor.shape[slot_dim] - self.base_index
        if self.slot_count_limit is not None:
            if (
                self.slot_count_limit <= 0
                or self.slot_count_limit * self.items_per_slot > remaining
            ):
                raise ValueError(
                    f"state tensor {self.name}: invalid "
                    f"slot_count_limit={self.slot_count_limit}"
                )
        elif remaining % self.items_per_slot != 0:
            raise ValueError(
                f"state tensor {self.name}: remaining slot dimension {remaining} "
                f"is not divisible by items_per_slot={self.items_per_slot}"
            )

    @property
    def slot_count(self) -> int:
        """Return the number of addressable logical slots."""
        if self.slot_count_limit is not None:
            return self.slot_count_limit
        return (
            self.tensor.shape[self.slot_dim] - self.base_index
        ) // self.items_per_slot

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Return the tensor shape occupied by one logical slot."""
        shape = list(self.tensor.shape)
        shape[self.slot_dim] = self.items_per_slot
        return tuple(shape)

    @property
    def bytes_per_slot(self) -> int:
        """Return the byte size of one logical slot."""
        count = 1
        for dim in self.item_shape:
            count *= int(dim)
        return count * self.tensor.element_size()

    def select(self, slot: int) -> torch.Tensor:
        """Return the tensor view for one logical slot."""
        if slot < 0 or slot >= self.slot_count:
            raise IndexError(
                f"state tensor {self.name}: slot {slot} outside [0, {self.slot_count})"
            )
        start = self.base_index + slot * self.items_per_slot
        return self.tensor.narrow(self.slot_dim, start, self.items_per_slot)


@dataclass(frozen=True)
class _LayoutEntry:
    spec: StateTensorSpec
    offset: int


class PackedStateLayout:
    """Deterministic byte layout for one self-contained cache block."""

    ALIGNMENT = 16

    def __init__(
        self,
        specs: Sequence[StateTensorSpec],
        *,
        layout_name: str,
        pack_validator: Optional[Callable[[Sequence[Mapping[str, int]]], None]] = None,
        unpack_hook: Optional[Callable[[Sequence[Mapping[str, int]]], None]] = None,
    ):
        """Build a deterministic packed layout from tensor specifications."""
        if not specs:
            raise ValueError("packed state layout requires at least one tensor")

        entries = []
        offset = 0
        group_slot_counts: dict[str, int] = {}
        device = specs[0].tensor.device
        for spec in specs:
            if spec.tensor.device != device:
                raise ValueError(
                    "all state tensors must reside on one device: "
                    f"expected {device}, got {spec.tensor.device} for {spec.name}"
                )
            previous = group_slot_counts.setdefault(spec.group, spec.slot_count)
            if previous != spec.slot_count:
                raise ValueError(
                    f"state group {spec.group!r} has inconsistent slot counts: "
                    f"{previous} and {spec.slot_count} ({spec.name})"
                )
            offset = _align_up(offset, max(self.ALIGNMENT, spec.tensor.element_size()))
            entries.append(_LayoutEntry(spec=spec, offset=offset))
            offset += spec.bytes_per_slot

        if not layout_name:
            raise ValueError("layout name must not be empty")
        self.layout_name = layout_name.upper()
        self.entries = tuple(entries)
        self.device = device
        self.group_slot_counts = group_slot_counts
        self.bytes_per_block = _align_up(offset, self.ALIGNMENT)
        self.layout_id = self._compute_layout_id()
        self._pack_validator = pack_validator
        self._unpack_hook = unpack_hook

    @property
    def groups(self) -> tuple[str, ...]:
        """Return the logical slot groups used by this layout."""
        return tuple(self.group_slot_counts)

    def _compute_layout_id(self) -> str:
        payload = [
            {
                "name": entry.spec.name,
                "group": entry.spec.group,
                "dtype": str(entry.spec.tensor.dtype),
                "shape": entry.spec.item_shape,
                "slot_dim": entry.spec.slot_dim,
                "items_per_slot": entry.spec.items_per_slot,
                "offset": entry.offset,
                "bytes": entry.spec.bytes_per_slot,
            }
            for entry in self.entries
        ]
        encoded = json.dumps(
            {
                "schema_version": 2,
                "layout_name": self.layout_name,
                "segments": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def allocate_staging(self, capacity: int = 1) -> torch.Tensor:
        """Allocate contiguous byte staging for packed blocks."""
        if capacity <= 0:
            raise ValueError(f"staging capacity must be positive, got {capacity}")
        return torch.empty(
            (capacity, self.bytes_per_block),
            dtype=torch.uint8,
            device=self.device,
        )

    def _normalize_slots(self, slots: Mapping[str, int] | int) -> Mapping[str, int]:
        if isinstance(slots, torch.Tensor):
            if slots.numel() != 1:
                raise ValueError(
                    f"state slot tensor must contain one index, got {tuple(slots.shape)}"
                )
            slots = int(slots.item())
        if isinstance(slots, Integral):
            if len(self.groups) != 1:
                raise ValueError(
                    f"layout groups {self.groups} require an explicit slot mapping"
                )
            slots = {self.groups[0]: int(slots)}
        missing = set(self.groups) - set(slots)
        extra = set(slots) - set(self.groups)
        if missing or extra:
            raise ValueError(
                f"state slot groups mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return slots

    def _validate_staging(self, staging: torch.Tensor, count: int) -> None:
        if (
            staging.dtype != torch.uint8
            or staging.ndim != 2
            or staging.shape[0] < count
            or staging.shape[1] != self.bytes_per_block
            or staging.device != self.device
            or not staging.is_contiguous()
        ):
            raise ValueError(
                "invalid packed-state staging tensor: expected contiguous "
                f"uint8 [>= {count}, {self.bytes_per_block}] on {self.device}, "
                f"got dtype={staging.dtype}, shape={tuple(staging.shape)}, "
                f"device={staging.device}, contiguous={staging.is_contiguous()}"
            )

    def pack(
        self,
        slots: Sequence[Mapping[str, int] | int],
        staging: torch.Tensor,
    ) -> None:
        """Pack logical state slots into staging rows."""
        self._validate_staging(staging, len(slots))
        normalized_slots = [self._normalize_slots(raw_slots) for raw_slots in slots]
        if self._pack_validator is not None:
            self._pack_validator(normalized_slots)
        for row, slot_map in enumerate(normalized_slots):
            staging[row].zero_()
            for entry in self.entries:
                source = entry.spec.select(int(slot_map[entry.spec.group]))
                source_bytes = source.contiguous().view(torch.uint8).reshape(-1)
                end = entry.offset + entry.spec.bytes_per_slot
                staging[row, entry.offset : end].copy_(source_bytes)

    def unpack(
        self,
        slots: Sequence[Mapping[str, int] | int],
        staging: torch.Tensor,
    ) -> None:
        """Restore logical state slots from staging rows."""
        self._validate_staging(staging, len(slots))
        normalized_slots = [self._normalize_slots(raw_slots) for raw_slots in slots]
        for row, slot_map in enumerate(normalized_slots):
            for entry in self.entries:
                target = entry.spec.select(int(slot_map[entry.spec.group]))
                end = entry.offset + entry.spec.bytes_per_slot
                source = staging[row, entry.offset : end]
                typed_source = source.view(entry.spec.tensor.dtype).reshape(
                    entry.spec.item_shape
                )
                target.copy_(typed_source)
        if self._unpack_hook is not None:
            self._unpack_hook(normalized_slots)
