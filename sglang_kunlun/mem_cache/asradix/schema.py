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
"""Uniform AttentionStore endpoint schema helpers."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class PaddedPointerBuffers:
    """A uniform pointer array and any backing buffers created for padding."""

    pointers: tuple[int, ...]
    owners: tuple[Any, ...] = ()


def pad_layer_buffers(buffers: Sequence[Any], layer_num: int, name: str) -> list[Any]:
    """Pad one PP shard to the namespace's uniform per-shard layer count."""
    buffers = list(buffers)
    if not buffers:
        raise ValueError(f"{name} requires at least one local layer")
    if len(buffers) > layer_num:
        raise ValueError(
            f"{name} local layer count {len(buffers)} exceeds schema {layer_num}"
        )
    return buffers + [buffers[-1]] * (layer_num - len(buffers))


def pad_component_major_pointers(
    pointers: Sequence[int], local_layer_num: int, layer_num: int
) -> tuple[int, ...]:
    """Pad component-major pointers by aliasing each component's last layer."""
    pointers = tuple(map(int, pointers))
    if local_layer_num <= 0 or len(pointers) % local_layer_num != 0:
        raise ValueError("invalid component-major pointer layout")
    padded = []
    for start in range(0, len(pointers), local_layer_num):
        component = pointers[start : start + local_layer_num]
        padded.extend(pad_layer_buffers(component, layer_num, "component state"))
    return tuple(padded)


def pad_pointer_buffers(
    pointers: Sequence[int],
    layer_num: int,
    name: str,
    *,
    placeholder_factory: Callable[[], Any] | None = None,
) -> PaddedPointerBuffers:
    """Normalize a raw pointer family to a namespace-wide layer count."""
    pointers = tuple(map(int, pointers))
    if layer_num < 0:
        raise ValueError(f"{name} layer count must not be negative")
    if layer_num == 0:
        if pointers:
            raise ValueError(f"{name} has buffers but its schema has no layers")
        return PaddedPointerBuffers(())

    owners = ()
    if not pointers:
        if placeholder_factory is None:
            raise ValueError(f"{name} requires at least one local layer")
        placeholder = placeholder_factory()
        pointers = (int(placeholder.data_ptr()),)
        owners = (placeholder,)

    padded = pad_layer_buffers(pointers, layer_num, name)
    return PaddedPointerBuffers(tuple(padded), owners)


def merge_endpoint_schemas(
    schemas: Sequence[Mapping[str, int]],
) -> dict[str, int]:
    """Merge node-local endpoint schemas into one uniform shard contract."""
    keys = {key for schema in schemas for key in schema}
    merged = {}
    for key in keys:
        values = [int(schema.get(key, 0)) for schema in schemas]
        if any(value < 0 for value in values):
            raise ValueError(f"endpoint schema {key!r} must not be negative")
        nonzero = {value for value in values if value}
        if key.endswith("_stride") and len(nonzero) > 1:
            raise ValueError(
                f"endpoint schema {key!r} has incompatible strides: {sorted(nonzero)}"
            )
        merged[key] = max(values, default=0)
    return merged
