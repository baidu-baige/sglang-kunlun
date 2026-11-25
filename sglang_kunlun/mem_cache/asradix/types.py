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
"""Public AttentionStore binding contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import torch


AS_BLOCK_SIZE = 256


class AttentionStoreEndpointRole(str, Enum):
    """Identify whether an endpoint stores pages or checkpoints."""

    PAGE = "PAGE"
    CHECKPOINT = "CHECKPOINT"


@dataclass(frozen=True)
class AttentionStorePipelineMember:
    """Describe one Agent participating in a PP endpoint group."""

    member_id: int
    pp_stage_start: int
    pp_stage_end: int
    agent_address: str
    bytes_per_block: int
    desired_block_count: int


@dataclass(frozen=True)
class AttentionStoreEndpointRegistration:
    """Carry a pre-negotiated PP endpoint capacity into SDK registration."""

    group_id: str
    generation: int
    endpoint_name: str
    endpoint_role: AttentionStoreEndpointRole
    local_member_id: int
    desired_block_count: int
    members: Sequence[AttentionStorePipelineMember]
    coordinator_member_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.endpoint_role, AttentionStoreEndpointRole):
            raise ValueError("AttentionStore PP endpoint role must be explicit")
        if not self.group_id or self.generation <= 0 or not self.endpoint_name:
            raise ValueError("AttentionStore PP endpoint identity must be complete")
        if self.desired_block_count <= 0 or not self.members:
            raise ValueError("AttentionStore endpoint requires shared capacity")
        member_ids = {member.member_id for member in self.members}
        if len(member_ids) != len(self.members):
            raise ValueError("AttentionStore PP endpoint member IDs must be unique")
        if self.local_member_id not in member_ids:
            raise ValueError("local AttentionStore PP member is absent")
        if self.coordinator_member_id not in member_ids:
            raise ValueError("AttentionStore PP coordinator member is absent")
        ordered = sorted(self.members, key=lambda member: member.pp_stage_start)
        if ordered[0].pp_stage_start != 0 or any(
            current.pp_stage_start != previous.pp_stage_end
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                "AttentionStore PP stage ranges must be contiguous from zero"
            )


@dataclass(frozen=True)
class AttentionStoreInitContext:
    """Provide runtime resources used to construct an AttentionStore binding."""

    kv_cache: Any
    req_to_token_pool: Any
    namespace_base: str
    pod_name: str
    model_version: str
    page_size: int
    block_size: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    node_pp_start: int
    node_pp_end: int
    dp_rank: int
    device_id: int
    gpu_count: int
    target_layer_ids: Sequence[int] = ()
    draft_kv_cache: Any = None
    draft_layer_ids: Sequence[int] = ()
    page_layer_ids: Sequence[int] = ()
    page_shard_id: int = -1
    page_shard_num: int = 0
    page_layer_num: int = 0
    state_layer_ids: Sequence[int] = ()
    state_shard_id: int = -1
    state_shard_num: int = 0
    state_layer_num: int = 0
    endpoint_schema: Mapping[str, int] = field(default_factory=dict)
    endpoint_registrations: Mapping[str, AttentionStoreEndpointRegistration] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AttentionStoreEndpointBinding:
    """One independently named page or checkpoint storage contract."""

    name: str
    role: AttentionStoreEndpointRole
    store: Any
    namespace: str
    layout_id: str
    bytes_per_block: int
    payload_fields: Sequence[str]
    registration_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize and validate the endpoint contract."""
        object.__setattr__(self, "payload_fields", tuple(self.payload_fields))
        if self.registration_name is None:
            object.__setattr__(self, "registration_name", self.name)
        if not self.name:
            raise ValueError("AttentionStore endpoint name must not be empty")
        if not self.registration_name:
            raise ValueError(
                "AttentionStore endpoint registration name must not be empty"
            )
        if not self.namespace:
            raise ValueError(f"AttentionStore endpoint {self.name} has no namespace")
        if self.bytes_per_block <= 0:
            raise ValueError(
                f"AttentionStore endpoint {self.name} has invalid "
                f"bytes_per_block={self.bytes_per_block}"
            )
        if not self.payload_fields:
            raise ValueError(
                f"AttentionStore endpoint {self.name} has no payload fields"
            )


@dataclass(frozen=True)
class AttentionStoreBinding:
    """Runtime stores and endpoint schemas selected for one attention plan."""

    plan_name: str
    cache_store: Any
    endpoints: Sequence[AttentionStoreEndpointBinding]
    endpoint_registrations: Mapping[str, AttentionStoreEndpointRegistration] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Normalize and validate the endpoint collection."""
        if not self.plan_name:
            raise ValueError("AttentionStore cache plan name must not be empty")
        endpoints = tuple(self.endpoints)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(
            self, "endpoint_registrations", dict(self.endpoint_registrations)
        )
        names = [endpoint.name for endpoint in endpoints]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate AttentionStore endpoint names: {names}")
        page_count = sum(
            endpoint.role is AttentionStoreEndpointRole.PAGE for endpoint in endpoints
        )
        checkpoint_count = sum(
            endpoint.role is AttentionStoreEndpointRole.CHECKPOINT
            for endpoint in endpoints
        )
        if page_count != 1:
            raise ValueError(
                "AttentionStore binding requires exactly one public page endpoint, "
                f"got {page_count}"
            )
        if checkpoint_count > 1:
            raise ValueError(
                "AttentionStore binding supports at most one checkpoint endpoint, "
                f"got {checkpoint_count}"
            )

    @property
    def page_endpoint(self) -> AttentionStoreEndpointBinding:
        """Return the binding's page endpoint."""
        return next(
            endpoint
            for endpoint in self.endpoints
            if endpoint.role is AttentionStoreEndpointRole.PAGE
        )

    @property
    def checkpoint_endpoint(self) -> Optional[AttentionStoreEndpointBinding]:
        """Return the optional checkpoint endpoint."""
        return next(
            (
                endpoint
                for endpoint in self.endpoints
                if endpoint.role is AttentionStoreEndpointRole.CHECKPOINT
            ),
            None,
        )

    @property
    def namespace(self) -> str:
        """Return the public page namespace."""
        return self.page_endpoint.namespace

    @property
    def layout_id(self) -> str:
        """Return a stable composite identifier for all endpoint layouts."""
        if len(self.endpoints) == 1:
            return self.endpoints[0].layout_id
        return ":".join(
            f"{endpoint.name}={endpoint.layout_id}" for endpoint in self.endpoints
        )

    @property
    def bytes_per_block(self) -> int:
        """Return the page endpoint's bytes per block."""
        return self.page_endpoint.bytes_per_block


@dataclass(frozen=True)
class AttentionStoreWriteOwner:
    """Identify the scheduler work that owns an asynchronous AS write.

    ``maintenance_round`` is the scheduler/cache lifecycle coordinate.  It is
    deliberately not called a checkpoint: KDA checkpoints are model-state
    snapshots and are unrelated to write retirement.
    """

    maintenance_round: int
    pp_batch_plan_id: Optional[int] = None
    mb_id: Optional[int] = None
    rid: Optional[str] = None

    def __post_init__(self) -> None:
        if self.maintenance_round < 0:
            raise ValueError("AttentionStore maintenance round must be non-negative")
        if self.pp_batch_plan_id is not None and self.pp_batch_plan_id < 0:
            raise ValueError("AttentionStore PP batch plan ID must be non-negative")
        if self.mb_id is not None and self.mb_id < 0:
            raise ValueError("AttentionStore microbatch ID must be non-negative")


@dataclass
class AttentionStoreWriteContext:
    """Capture request state across a local cache commit."""

    req: Any
    token_ids: Sequence[int]
    snapshot: torch.Tensor
    is_insert: bool
    prepared_write: Any = None
    owner: Optional[AttentionStoreWriteOwner] = None
