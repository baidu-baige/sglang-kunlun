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
"""Capacity planning for AttentionStore cache endpoint groups."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from .types import (
    AttentionStoreEndpointRegistration,
    AttentionStoreEndpointRole,
    AttentionStorePipelineMember,
)


@dataclass(frozen=True)
class AttentionStoreNodeEndpoint:
    """Physical capacity inputs for one node-local Agent endpoint."""

    member_id: int
    pp_stage_start: int
    pp_stage_end: int
    agent_address: str
    bytes_per_block: int
    chunk_count: int


def plan_pipeline_endpoint_registration(
    *,
    group_id: str,
    endpoint_name: str,
    endpoint_role: AttentionStoreEndpointRole,
    local_member_id: int,
    shared_memory_size: int,
    total_chunk_count: int,
    nodes: Sequence[AttentionStoreNodeEndpoint],
    coordinator_member_id: int = 0,
) -> AttentionStoreEndpointRegistration:
    """Negotiate one logical block count from heterogeneous node layouts."""
    # Registration descriptors are exchanged between ranks. Canonicalize the
    # member order before both serialization and generation hashing so callers
    # cannot create different generations from the same physical topology.
    nodes = tuple(sorted(nodes, key=lambda node: node.member_id))
    if not group_id or not endpoint_name:
        raise ValueError("AttentionStore PP group and endpoint names must be set")
    if not isinstance(endpoint_role, AttentionStoreEndpointRole):
        raise ValueError("AttentionStore PP endpoint role must be explicit")
    if shared_memory_size <= 0 or total_chunk_count <= 0:
        raise ValueError("AttentionStore shared memory geometry must be positive")
    if not nodes:
        raise ValueError("AttentionStore endpoint requires at least one Agent")
    if len({node.member_id for node in nodes}) != len(nodes):
        raise ValueError("AttentionStore PP node member IDs must be unique")
    member_ids = {node.member_id for node in nodes}
    if local_member_id not in member_ids:
        raise ValueError("AttentionStore PP local member is absent")
    if coordinator_member_id not in member_ids:
        raise ValueError("AttentionStore PP coordinator member is absent")
    if len({node.agent_address for node in nodes}) != len(nodes):
        raise ValueError("AttentionStore PP Agent addresses must be unique")

    ordered_nodes = tuple(sorted(nodes, key=lambda node: node.pp_stage_start))
    if ordered_nodes[0].pp_stage_start != 0:
        raise ValueError("AttentionStore PP stage ranges must start at zero")
    for previous, current in zip(ordered_nodes, ordered_nodes[1:]):
        if current.pp_stage_start != previous.pp_stage_end:
            raise ValueError("AttentionStore PP stage ranges must be contiguous")

    chunk_size = shared_memory_size // total_chunk_count
    if chunk_size <= 0:
        raise ValueError("AttentionStore Agent chunk size is zero")

    capacities = []
    for node in nodes:
        if node.pp_stage_start < 0 or node.pp_stage_start >= node.pp_stage_end:
            raise ValueError(f"invalid PP stage range for member {node.member_id}")
        if not node.agent_address:
            raise ValueError(f"AS_AGENT_ADDRESS is missing for member {node.member_id}")
        if node.bytes_per_block <= 0 or node.chunk_count <= 0:
            raise ValueError(
                f"invalid endpoint geometry for member {node.member_id}: "
                f"bytes_per_block={node.bytes_per_block}, chunks={node.chunk_count}"
            )
        blocks_per_chunk = chunk_size // node.bytes_per_block
        if blocks_per_chunk <= 0:
            raise ValueError(
                f"member {node.member_id} block size {node.bytes_per_block} "
                f"exceeds Agent chunk size {chunk_size}"
            )
        capacities.append(node.chunk_count * blocks_per_chunk)

    desired_block_count = min(capacities)
    members = tuple(
        AttentionStorePipelineMember(
            member_id=node.member_id,
            pp_stage_start=node.pp_stage_start,
            pp_stage_end=node.pp_stage_end,
            agent_address=node.agent_address,
            bytes_per_block=node.bytes_per_block,
            desired_block_count=desired_block_count,
        )
        for node in nodes
    )
    generation_payload = {
        "schema_version": 1,
        "group_id": group_id,
        "endpoint_name": endpoint_name,
        "endpoint_role": endpoint_role.value,
        "coordinator_member_id": coordinator_member_id,
        "shared_memory_size": shared_memory_size,
        "total_chunk_count": total_chunk_count,
        # ``chunk_count`` is an input to capacity planning but is intentionally
        # not part of the SDK member descriptor. Keep it in the generation
        # identity so a changed Agent partition cannot reuse an old group.
        "members": [
            {**asdict(member), "chunk_count": node.chunk_count}
            for member, node in zip(members, nodes)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(generation_payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    # Generation participates in torch.int64 topology consensus during startup.
    # Keep the deterministic digest in the positive signed-int64 range.
    generation = (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1
    return AttentionStoreEndpointRegistration(
        group_id=group_id,
        generation=generation,
        endpoint_name=endpoint_name,
        endpoint_role=endpoint_role,
        local_member_id=local_member_id,
        desired_block_count=desired_block_count,
        members=members,
        coordinator_member_id=coordinator_member_id,
    )
