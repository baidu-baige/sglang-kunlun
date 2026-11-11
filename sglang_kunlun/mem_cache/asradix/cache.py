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
"""AttentionStore-backed unified radix cache construction."""

from __future__ import annotations

import atexit
import hashlib
import logging
import struct
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from .coordinator import AttentionStoreCoordinator
from .plans import AttentionStoreCachePlan
from .types import (
    AttentionStoreBinding,
    AttentionStoreInitContext,
    AttentionStoreWriteOwner,
    AS_BLOCK_SIZE,
)

logger = logging.getLogger(__name__)


class AttentionStoreIntegrationMixin:
    """Connect UnifiedRadixCache to its selected AttentionStore plan."""

    _as_coordinator: Optional[AttentionStoreCoordinator] = None
    _as_plan: Optional[AttentionStoreCachePlan] = None

    @property
    def attention_store_block_size(self) -> int:
        """Return the token granularity used by the active AS cache plan."""
        if self._as_plan is not None and self._as_plan.name in ("KDA", "MLA"):
            return AS_BLOCK_SIZE
        return self.page_size

    _AS_RESULT_FIELDS = {
        "write": "written_blocks",
        "match": "matched_blocks",
        "load_back": "loaded_blocks",
    }

    def __init__(
        self,
        params: CacheInitParams,
        server_args,
        plan: Optional[AttentionStoreCachePlan] = None,
    ) -> None:
        """Initialize the local cache and its AttentionStore plan."""
        if plan is None:
            raise ValueError(f"{type(self).__name__} requires a cache plan")
        self.configure_attention_store(plan)
        self.binding: Optional[AttentionStoreBinding] = None
        self.ascli = None
        # getattr, not params.disable_finished_insert: this engine is built on
        # community sglang, whose CacheInitParams never carried that field (it
        # is an aiak dsv41-lineage addition consumed by radix_cache /
        # pure_swa_radix_cache). Its upstream default is False, and the other
        # read of it further down is already defensive, so behaviour is unchanged.
        self.disable_finished_insert = getattr(params, "disable_finished_insert", False)
        self._disagg_mode = server_args.disaggregation_mode
        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_group is not None
            else 1
        )
        super().__init__(params)
        self.pp_world_size = (
            torch.distributed.get_world_size(group=self.pp_group)
            if self.pp_group is not None
            else 1
        )
        self.pp_rank = (
            torch.distributed.get_rank(group=self.pp_group)
            if self.pp_group is not None
            else 0
        )
        attention_group_sizes = [self.tp_world_size]
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None:
                attention_group_sizes.append(
                    torch.distributed.get_world_size(group=group)
                )
        self.as_world_size = max(attention_group_sizes)
        atexit.register(self._shutdown_attention_store)

    def _log_as_result(
        self,
        operation: str,
        *,
        rid: str,
        requested: int,
        result: int,
        started: float,
    ) -> None:
        result_field = self._AS_RESULT_FIELDS[operation]
        logger.info(
            "AS %s: cache_plan=%s rid=%s requested_blocks=%s "
            "%s=%s status=ok cost_ms=%.2f",
            operation,
            self.attention_store_plan.name,
            rid,
            requested,
            result_field,
            result,
            (time.perf_counter() - started) * 1000,
        )

    def _log_as_failure(
        self,
        operation: str,
        *,
        rid: str,
        requested: int,
        error: Exception,
        started: Optional[float] = None,
    ) -> None:
        result_field = self._AS_RESULT_FIELDS[operation]
        cost_ms = (
            (time.perf_counter() - started) * 1000 if started is not None else -1.0
        )
        logger.warning(
            "AS %s: cache_plan=%s rid=%s requested_blocks=%s "
            "%s=-1 status=error cost_ms=%.2f error=%s",
            operation,
            self.attention_store_plan.name,
            rid,
            requested,
            result_field,
            cost_ms,
            error,
        )

    def _as_min(self, value: int) -> int:
        """Reduce a runtime cache decision within the current PP stage."""
        tensor = torch.tensor(value, dtype=torch.int64)
        self._all_reduce_attn_groups(tensor, torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def _as_topology_min(self, value: int) -> int:
        """Reduce one cache decision across attention shards and all PP stages."""
        tensor = torch.tensor(self._as_min(value), dtype=torch.int64)
        if self.pp_world_size > 1:
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.pp_group,
            )
        return int(tensor.item())

    def _as_topology_max(self, value: int) -> int:
        """Return the largest cache decision across attention shards and PP."""
        tensor = torch.tensor(value, dtype=torch.int64)
        self._all_reduce_attn_groups(tensor, torch.distributed.ReduceOp.MAX)
        if self.pp_world_size > 1:
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.pp_group,
            )
        return int(tensor.item())

    def sync_attention_store_read_abort(self, req) -> bool:
        """Return the stage-wide AttentionStore read-abort decision.

        PP stages do not call request admission in lockstep, so this path must
        not issue a PP collective.
        """
        abort = torch.tensor(
            int(bool(getattr(req, "_as_cache_read_abort_reason", None))),
            dtype=torch.int64,
        )
        self._all_reduce_attn_groups(abort, torch.distributed.ReduceOp.MAX)
        return bool(abort.item())

    def _tp_min(self, value: int) -> int:
        """Reduce a write-retirement value across local attention shards."""
        return self._as_min(value)

    def _take_attention_store_load_context(
        self, params: InitLoadBackParams
    ) -> tuple[Any, Any, Any]:
        anchor = params.best_match_node
        req = params.req
        load_info = getattr(req, "_as_load_info", None) if req is not None else None
        if req is not None and hasattr(req, "_as_load_info"):
            del req._as_load_info
        return anchor, load_info, req

    @staticmethod
    def _pipeline_prefix_digest(block_hash_ids) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack(">Q", len(block_hash_ids)))
        for block_hash_id in block_hash_ids:
            digest.update(struct.pack(">Q", int(block_hash_id)))
        return digest.hexdigest()

    def _pipeline_generation_key(self) -> tuple:
        if self.binding is None:
            return ()
        return tuple(
            sorted(
                (
                    registration.group_id,
                    registration.endpoint_name,
                    int(registration.generation),
                )
                for registration in self.binding.endpoint_registrations.values()
            )
        )

    def _validate_authoritative_match_decision(
        self, req, hashes, requested: int
    ) -> int:
        decision = getattr(req, "_as_pp_match_decision", None)
        if not isinstance(decision, dict):
            raise RuntimeError(
                f"missing PP0 AttentionStore match decision for rid={req.rid}"
            )
        expected_digest = self._pipeline_prefix_digest(hashes)
        expected_generation = self._pipeline_generation_key()
        if (
            decision.get("version") != 1
            or decision.get("rid") != req.rid
            or int(decision.get("requested", -1)) != requested
            or decision.get("hash_digest") != expected_digest
            or tuple(decision.get("generation", ())) != expected_generation
        ):
            raise RuntimeError(
                "invalid PP0 AttentionStore match decision: "
                f"rid={req.rid} requested={requested} "
                f"digest={expected_digest} generation={expected_generation} "
                f"decision={decision}"
            )
        matched = int(decision.get("matched", -1))
        if matched < 0 or matched > requested:
            raise RuntimeError(
                f"invalid PP0 AttentionStore match count rid={req.rid}: {matched}"
            )
        return matched

    def _match_checkpointed_page_prefix(
        self,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Match page prefixes whose terminal hash has a separate checkpoint."""
        req = params.req
        if (
            req is None
            or req.ever_read_as
            or not req.as_block_hashs
            or self.ascli is None
        ):
            return None

        block_size = self.attention_store_block_size
        max_hit_len = len(params.key) // block_size * block_size
        joint_pp_match = self.pp_world_size > 1
        local_hit_len = len(local_result.device_indices)
        aligned_local_hit_len = local_hit_len // block_size * block_size
        if local_hit_len != aligned_local_hit_len:
            local_result = super().match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        params.key.token_ids,
                        params.key.extra_key,
                        is_bigram=params.key.is_bigram,
                        limit=aligned_local_hit_len,
                    ),
                    cow_mamba=params.cow_mamba,
                    req=req,
                )
            )
            local_hit_len = len(local_result.device_indices)
        if local_hit_len == max_hit_len and not joint_pp_match:
            return None

        start_block = local_hit_len // block_size
        # Every PP stage must submit the same absolute prefix to its local Agent.
        # Submitting only the suffix after a stage-local HBM hit would preserve
        # the very PP-stage divergence this path is meant to eliminate.
        match_start_block = 0 if joint_pp_match else start_block
        max_end_block = max_hit_len // block_size
        hashes = req.as_block_hashs.hash_ids[match_start_block:max_end_block]
        # A checkpoint match is valid only for the hash at its returned endpoint.
        # Apply the request's prefix limit before matching so a later checkpoint
        # cannot be matched and then relabeled as an earlier prefix by truncation.
        if not hashes:
            return zero_match_result(self, local_result) if joint_pp_match else None

        requested = len(hashes)
        started = time.perf_counter()
        match_error = None
        authoritative_pp_match = (
            joint_pp_match
            and self.pp_rank != 0
            and bool(getattr(req, "_as_pp_match_decision_required", False))
        )
        if authoritative_pp_match:
            # Later PP stages consume PP0's keyed decision. They must not create
            # a second storage snapshot while independently rebuilding the batch.
            match_count = self._validate_authoritative_match_decision(
                req, hashes, requested
            )
        else:
            # The public SDK deadline encloses the Agent coordinator budget,
            # which in turn encloses its two bounded peer-RPC rounds.
            match_timeout = 3.0 if joint_pp_match else 1.0
            try:
                # A pipeline-registered Agent endpoint already probes all members,
                # computes the common terminal checkpoint, and pins both payloads.
                match_count = self.ascli.match(hashes, timeout=match_timeout)
            except Exception as exc:
                match_error = exc
                match_count = 0
        # The pipeline Agent match is already joint across all PP members.
        # Dynamic request matching is not lockstep across PP stages, so only
        # reduce across the attention ranks in this stage.
        match_count = max(0, min(int(self._as_min(match_count)), requested))
        if joint_pp_match and self.pp_rank == 0:
            req._as_pp_match_decision = {
                "version": 1,
                "rid": req.rid,
                "requested": requested,
                "matched": match_count,
                "hash_digest": self._pipeline_prefix_digest(hashes),
                "generation": self._pipeline_generation_key(),
            }
        if match_error is None:
            self._log_as_result(
                "match",
                rid=req.rid,
                requested=requested,
                result=match_count,
                started=started,
            )
        else:
            self._log_as_failure(
                "match",
                rid=req.rid,
                requested=requested,
                error=match_error,
                started=started,
            )
        if match_count <= 0:
            return zero_match_result(self, local_result) if joint_pp_match else None

        terminal_block = match_count if joint_pp_match else start_block + match_count
        if terminal_block <= start_block:
            if terminal_block == start_block:
                return local_result
            terminal_tokens = terminal_block * block_size
            capped_params = MatchPrefixParams(
                key=RadixKey(
                    params.key.token_ids,
                    params.key.extra_key,
                    is_bigram=params.key.is_bigram,
                    limit=terminal_tokens,
                ),
                cow_mamba=params.cow_mamba,
                req=req,
            )
            # The capped Full-KV path can still stop before the negotiated
            # endpoint when no exact local Mamba/KDA checkpoint exists there.
            # Continue through the normal AS load path for the remaining gap.
            local_result = super().match_prefix(capped_params)
            local_hit_len = len(local_result.device_indices)
            start_block = local_hit_len // block_size
            if terminal_block == start_block:
                return local_result
            if terminal_block < start_block:
                raise RuntimeError(
                    "local cache match exceeded the AttentionStore PP terminal "
                    f"after capping: terminal_block={terminal_block} "
                    f"local_block={start_block}"
                )

        # A page match can end anywhere, but a plan whose state lives only at
        # certain page boundaries can only restore up to the last such boundary.
        restorable_block = self.attention_store_coordinator.floor_remote_terminal(
            self, terminal_block
        )
        if restorable_block != terminal_block:
            logger.debug(
                "AS match floored to a restorable boundary: cache_plan=%s rid=%s "
                "matched_blocks=%s restorable_blocks=%s local_blocks=%s",
                self.attention_store_plan.name,
                req.rid,
                terminal_block,
                restorable_block,
                start_block,
            )
        if restorable_block <= start_block:
            # Keep the local hit intact rather than capping it down to the AS
            # boundary: HBM already holds more than the store can hand back.
            return zero_match_result(self, local_result) if joint_pp_match else None
        terminal_block = restorable_block

        read_hashes = req.as_block_hashs.hash_ids[start_block:terminal_block]
        read_count = terminal_block - start_block

        anchor = local_result.last_device_node
        req._as_load_info = {
            "rid": req.rid,
            "start_block": start_block,
            "block_count": read_count,
            "hashes": read_hashes,
            "remote_addrs": (req.as_max_hit_host_ip or [] if self.pp_size <= 1 else []),
        }
        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=anchor,
            last_host_node=anchor,
            best_match_node=anchor,
            host_hit_length=read_count * block_size,
        )

    def configure_attention_store(self, plan: AttentionStoreCachePlan) -> None:
        """Attach an AttentionStore plan and its payload coordinator."""
        if self._as_plan is not None:
            raise RuntimeError("AttentionStore cache plan is already configured")
        self._as_plan = plan
        self._as_coordinator = plan.coordinator_type()

    @property
    def attention_store_coordinator(self) -> AttentionStoreCoordinator:
        """Return the configured write coordinator."""
        if self._as_coordinator is None:
            raise RuntimeError("AttentionStore coordinator is not configured")
        return self._as_coordinator

    @property
    def attention_store_plan(self) -> AttentionStoreCachePlan:
        """Return the configured cache plan."""
        if self._as_plan is None:
            raise RuntimeError("AttentionStore cache plan is not configured")
        return self._as_plan

    def create_attention_store_binding(
        self, context: AttentionStoreInitContext
    ) -> AttentionStoreBinding:
        """Create the endpoint binding selected by the cache plan."""
        return self.attention_store_plan.binding_factory(context)

    def ref_as_binding(self, binding: AttentionStoreBinding) -> None:
        """Attach a validated runtime binding and its cache store."""
        if binding.plan_name != self.attention_store_plan.name:
            raise ValueError(
                "AttentionStore binding plan mismatch: "
                f"cache={self.attention_store_plan.name} "
                f"binding={binding.plan_name}"
            )
        self.binding = binding
        self.ascli = binding.cache_store

    def needs_req_in_match(self) -> bool:
        """Return whether prefix matching requires request context."""
        return True

    def _is_prefill_role(self) -> bool:
        return self._disagg_mode in ("null", "prefill")

    def ready_to_load_host_cache(self) -> int:
        """Report that AttentionStore loads need no host-cache readiness wait."""
        return 0

    def set_attention_store_write_owner(
        self, owner: Optional[AttentionStoreWriteOwner]
    ) -> None:
        """Set the owner for scheduler-side cache commits in this round."""
        self._as_active_write_owner = owner
        self.attention_store_coordinator._owner_hint = owner

    @contextmanager
    def attention_store_write_context(
        self, owner: Optional[AttentionStoreWriteOwner]
    ) -> Iterator[None]:
        """Associate cache commits made in this scope with scheduler work."""
        previous_owner = getattr(self, "_as_active_write_owner", None)
        coordinator = self.attention_store_coordinator
        previous_hint = coordinator._owner_hint
        self.set_attention_store_write_owner(owner)
        try:
            yield
        finally:
            self._as_active_write_owner = previous_owner
            coordinator._owner_hint = previous_hint

    def _as_request_write_owner(self, req):
        """Combine current creation round with the request's batch identity."""
        active = getattr(self, "_as_active_write_owner", None)
        request_owner = getattr(req, "_as_write_owner", None)
        owner = active or request_owner
        if active is not None and request_owner is not None:
            owner = replace(
                active,
                pp_batch_plan_id=(
                    active.pp_batch_plan_id
                    if active.pp_batch_plan_id is not None
                    else request_owner.pp_batch_plan_id
                ),
                mb_id=active.mb_id if active.mb_id is not None else request_owner.mb_id,
                rid=req.rid,
            )
        elif owner is not None and owner.rid is None:
            owner = replace(owner, rid=req.rid)
        return owner

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs) -> None:
        """Commit a finished request locally and schedule its remote write."""
        owner = self._as_request_write_owner(req)
        with self.attention_store_write_context(owner):
            coordinator = self.attention_store_coordinator
            effective_is_insert = (
                is_insert
                and not self.disable
                and not getattr(self, "disable_finished_insert", False)
            )
            context = coordinator.capture_finished(self, req, effective_is_insert)
            try:
                if context is not None:
                    context.owner = owner
                    coordinator.before_local_commit(self, context)
                super().cache_finished_req(req, effective_is_insert, **kwargs)
                if context is not None:
                    if context.prepared_write is not None:
                        context.prepared_write.owner = context.owner
                    coordinator.after_local_commit(self, context)
            except Exception:
                if context is not None:
                    coordinator.abort_local_commit(self, context)
                raise

    def cache_unfinished_req(self, req, chunked=False) -> None:
        """Commit an unfinished request locally and schedule its remote write."""
        owner = self._as_request_write_owner(req)
        with self.attention_store_write_context(owner):
            coordinator = self.attention_store_coordinator
            context = coordinator.capture_unfinished(self, req)
            try:
                if context is not None:
                    context.owner = owner
                    coordinator.before_local_commit(self, context)
                super().cache_unfinished_req(req, chunked=chunked)
                if context is not None:
                    if context.prepared_write is not None:
                        context.prepared_write.owner = context.owner
                    coordinator.after_local_commit(self, context)
            except Exception:
                if context is not None:
                    coordinator.abort_local_commit(self, context)
                raise

    def check_ascache_events(self, maintenance_round: Optional[int] = None) -> None:
        """Retire writes from earlier cache maintenance rounds."""
        self.attention_store_coordinator.check_events(self, maintenance_round)

    def drain_attention_store_writes(self) -> None:
        """Wait for all pending AttentionStore writes."""
        self.attention_store_coordinator.drain(self)

    def reset(self) -> None:
        """Drain remote writes before resetting the local cache."""
        self.attention_store_coordinator.before_reset(self)
        super().reset()

    def sanity_check(self) -> None:
        """Drain remote writes before checking local cache invariants."""
        self.attention_store_coordinator.before_sanity_check(self)
        super().sanity_check()

    def _shutdown_attention_store(self) -> None:
        self.attention_store_coordinator.shutdown(self)
        if self.ascli is not None:
            try:
                self.ascli.stop()
            except Exception as exc:
                logger.warning(
                    "Failed to stop AttentionStore: cache_plan=%s error=%s",
                    self.attention_store_plan.name,
                    exc,
                )
            self.ascli = None
            self.binding = None


class AttentionStoreUnifiedRadixCache(
    AttentionStoreIntegrationMixin, UnifiedRadixCache
):
    """Component-driven UnifiedRadixCache with an AttentionStore L2."""

    def __init__(
        self,
        params: CacheInitParams,
        server_args,
        plan: Optional[AttentionStoreCachePlan] = None,
    ) -> None:
        """Select the component set declared by the AttentionStore plan."""
        if plan is None:
            raise ValueError(f"{type(self).__name__} requires a cache plan")
        params.tree_components = plan.tree_components
        super().__init__(params=params, server_args=server_args, plan=plan)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Extend a local component-consistent match through its coordinator."""
        if params.req is not None and hasattr(params.req, "_as_load_info"):
            del params.req._as_load_info
        local_result = super().match_prefix(params)
        remote = self.attention_store_coordinator.extend_match(
            self,
            params,
            local_result,
        )
        return remote if remote is not None else local_result

    def init_load_back(self, params: InitLoadBackParams) -> Tuple[torch.Tensor, Any]:
        """Delegate type-specific payload restoration to the plan coordinator."""
        new_indices, restored_node = self.attention_store_coordinator.init_load_back(
            self, params
        )
        if (
            self.pp_world_size > 1
            and params.host_hit_length > 0
            and len(new_indices) != params.host_hit_length
        ):
            reason = (
                "AttentionStore PP read length mismatch: "
                f"expected={params.host_hit_length} actual={len(new_indices)}"
            )
            params.req._as_cache_read_abort_reason = reason
            logger.error("%s rid=%s", reason, params.req.rid)
            return params.req.prefix_indices[:0], self.root_node
        return new_indices, restored_node


def create_attention_store_cache(
    plan: AttentionStoreCachePlan,
    *,
    params: CacheInitParams,
    server_args,
):
    """Create the component-driven UnifiedRadixCache for an AS plan."""
    return AttentionStoreUnifiedRadixCache(
        params=params,
        server_args=server_args,
        plan=plan,
    )
