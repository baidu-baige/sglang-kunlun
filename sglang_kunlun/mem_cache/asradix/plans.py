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
"""AttentionStore cache capability composition and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

from .checkpoint_binding import (
    create_kda_binding,
    create_spa_binding,
    create_spag_binding,
    describe_spa_endpoint_schema,
    describe_spag_endpoint_schema,
)
from .checkpoint_coordinator import (
    KDACacheCoordinator,
    SPACacheCoordinator,
    SPAGCacheCoordinator,
)
from .coordinator import AttentionStoreCoordinator
from .page_binding import create_dsa_binding, create_mha_binding, create_mla_binding
from .page_coordinator import FullPageCacheCoordinator
from .types import AttentionStoreBinding, AttentionStoreInitContext


def _empty_endpoint_schema(_kv_cache: Any) -> Mapping[str, int]:
    return {}


@dataclass(frozen=True)
class AttentionStoreCachePlan:
    """Compose endpoint binding and cache lifecycle capabilities."""

    name: str
    coordinator_type: type[AttentionStoreCoordinator]
    binding_factory: Callable[[AttentionStoreInitContext], AttentionStoreBinding]
    tree_components: tuple[ComponentType, ...]
    schema_factory: Callable[[Any], Mapping[str, int]] = _empty_endpoint_schema

    def __post_init__(self) -> None:
        """Validate the cache plan contract."""
        if not self.name:
            raise ValueError("AttentionStore cache plan name must not be empty")
        if (
            not self.tree_components
            or self.tree_components[0] is not ComponentType.FULL
        ):
            raise ValueError("AttentionStore plans must start with the FULL component")


_ATTENTION_STORE_CACHE_PLANS = {
    "MHA": AttentionStoreCachePlan(
        name="MHA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_mha_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "MLA": AttentionStoreCachePlan(
        name="MLA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_mla_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "DSA": AttentionStoreCachePlan(
        name="DSA",
        coordinator_type=FullPageCacheCoordinator,
        binding_factory=create_dsa_binding,
        tree_components=(ComponentType.FULL,),
    ),
    "SPA": AttentionStoreCachePlan(
        name="SPA",
        coordinator_type=SPACacheCoordinator,
        binding_factory=create_spa_binding,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        schema_factory=describe_spa_endpoint_schema,
    ),
    # SPA-Generic: DeepSeek-V4.1 ratio-1/2 compressed KV. Same tree components as
    # SPA, but every page block carries its own SWA window, so there is no
    # separate checkpoint endpoint and the coordinator differs too.
    "SPAG": AttentionStoreCachePlan(
        name="SPAG",
        coordinator_type=SPAGCacheCoordinator,
        binding_factory=create_spag_binding,
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        schema_factory=describe_spag_endpoint_schema,
    ),
    "KDA": AttentionStoreCachePlan(
        name="KDA",
        coordinator_type=KDACacheCoordinator,
        binding_factory=create_kda_binding,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
    ),
}


def resolve_attention_store_plan(model_config, kv_cache) -> AttentionStoreCachePlan:
    """Resolve a cache plan from cache capabilities, not model names."""
    from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(kv_cache, DeepSeekV4TokenToKVPool):
        # Dispatch by the compressed-KV ratios the pool actually holds, not by
        # model name: V4 has c4/c128 (fixed SPA layout); V4.1 has ratio-1/2 and
        # no c4/c128, which needs the generic SPAG segment layout.
        ratios = set(getattr(kv_cache, "kv_pools", {}) or {})
        if ratios & {4, 128}:
            plan_name = "SPA"
        elif ratios & {1, 2}:
            plan_name = "SPAG"
        else:
            plan_name = "SPA"
    elif isinstance(kv_cache, HybridLinearKVPool) and getattr(
        kv_cache.mamba_pool, "replayssm_is_kda", False
    ):
        plan_name = "KDA"
    elif is_deepseek_dsa(model_config.hf_config):
        plan_name = "DSA"
    elif model_config.attention_arch == AttentionArch.MLA:
        plan_name = "MLA"
    else:
        plan_name = "MHA"
    return _ATTENTION_STORE_CACHE_PLANS[plan_name]
