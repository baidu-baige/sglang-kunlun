"""Atomic Kunlun MTP token, KV-length, metadata, and graph contract hooks."""

from __future__ import annotations

import contextlib
import inspect

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def accepted_prefix_indices(accept_lens, tokens_per_request):
    """Flatten accepted prefixes from fixed-width per-request speculative rows."""
    columns = torch.arange(
        tokens_per_request, device=accept_lens.device, dtype=accept_lens.dtype
    )
    rows = (
        torch.arange(
            accept_lens.numel(), device=accept_lens.device, dtype=accept_lens.dtype
        )
        * tokens_per_request
    )
    return (rows[:, None] + columns)[columns < accept_lens[:, None]].to(torch.int64)


def packed_rows_fit_fixed_width(accepted_lengths, width: int) -> bool:
    """Return whether a linear packed copy preserves fixed-width request slots."""
    return all(length == width for length in accepted_lengths[:-1])


def can_run_draft_extend_graph(cuda_graph_runner, forward_batch) -> bool:
    """Reject ragged accepted-only Draft Extend at every graph boundary."""
    return bool(
        cuda_graph_runner
        and not getattr(forward_batch, "_kunlun_ragged_draft_extend", False)
        and cuda_graph_runner.can_run_graph(forward_batch)
    )


@plugin_hook(
    "sglang.srt.speculative.base_spec_worker.EagleDraftWorkerBase.prepare_for_draft_extend",
    type=HookType.REPLACE,
)
def prepare_for_draft_extend_kunlun(
    self,
    draft_extend_input,
    batch,
    predict,
    num_draft_tokens,
    draft_model_runner,
    cuda_graph_runner,
):
    """Build fixed-width graph metadata or correct ragged eager metadata."""
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.utils.async_probe import maybe_detect_oob
    from sglang.srt.utils.common import is_npu

    batch_size = len(batch.seq_lens)
    accept_lens = draft_extend_input.num_accept_tokens
    accepted_only = accept_lens is not None and (
        predict.shape[0] != batch_size * num_draft_tokens
    )
    batch.spec_info = draft_extend_input
    batch.input_ids = predict
    maybe_detect_oob(
        predict,
        0,
        batch.model_config.vocab_size,
        "v2 prepare_for_draft_extend input_ids",
    )

    gpu_only = batch.seq_lens_cpu is None
    accepted_lengths_cpu = None
    if accepted_only:
        accepted_lengths_cpu = (
            accept_lens.detach().to(device="cpu", dtype=torch.int32).tolist()
        )

    if gpu_only:
        batch.prefix_lens = batch.seq_lens.to(torch.int32)
        batch.extend_lens = (
            accept_lens.to(torch.int32)
            if accepted_only
            else torch.full(
                (batch_size,),
                num_draft_tokens,
                dtype=torch.int32,
                device=batch.seq_lens.device,
            )
        )
    else:
        batch.prefix_lens = batch.seq_lens_cpu.tolist()
        batch.extend_lens = (
            accepted_lengths_cpu
            if accepted_only
            else [num_draft_tokens] * batch_size
        )
    batch.extend_num_tokens = (
        predict.shape[0] if accepted_only else batch_size * num_draft_tokens
    )
    batch.forward_mode = (
        ForwardMode.IDLE
        if batch.forward_mode.is_idle()
        else ForwardMode.DRAFT_EXTEND_V2
    )
    batch.capture_hidden_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )

    forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
    increment = (
        accept_lens.to(forward_batch.seq_lens.dtype)
        if accepted_only
        else num_draft_tokens
    )
    forward_batch.seq_lens = forward_batch.seq_lens + increment
    # Linear graph-buffer copies preserve request slots only while every
    # non-final packed request remains full-width.
    packed_layout_safe = not accepted_only or packed_rows_fit_fixed_width(
        accepted_lengths_cpu, num_draft_tokens
    )
    forward_batch._kunlun_ragged_draft_extend = bool(
        accepted_only and not packed_layout_safe
    )

    if accepted_only:
        forward_batch.extend_seq_lens_cpu = accepted_lengths_cpu
        forward_batch.extend_prefix_lens_cpu = (
            batch.seq_lens.detach().to(device="cpu", dtype=torch.int32).tolist()
            if gpu_only
            else list(batch.prefix_lens)
        )
        draft_extend_input.extend_seq_lens_cpu = list(accepted_lengths_cpu)
        draft_extend_input.extend_seq_lens_tensor = forward_batch.extend_seq_lens

    if gpu_only:
        if accepted_only:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens.detach().to(
                device="cpu", dtype=forward_batch.seq_lens.dtype
            )
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
        else:
            forward_batch.extend_seq_lens_cpu = [num_draft_tokens] * batch_size
    else:
        cpu_increment = (
            torch.tensor(batch.extend_lens, dtype=forward_batch.seq_lens_cpu.dtype)
            if accepted_only
            else num_draft_tokens
        )
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + cpu_increment
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())

    can_graph = can_run_draft_extend_graph(cuda_graph_runner, forward_batch)
    forward_batch._kunlun_can_run_draft_extend_graph = can_graph
    if not batch.forward_mode.is_idle() and not can_graph:
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        if not is_npu() or can_graph:
            forward_batch.mark_forward_metadata_ready()
    return forward_batch


@plugin_hook(
    "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner."
    "EAGLEDraftExtendCudaGraphRunner.can_run_graph",
    type=HookType.AROUND,
)
def reject_ragged_draft_extend_graph_kunlun(original_fn, self, forward_batch):
    """Defensively reject accepted-only ragged rows in the graph runner itself."""
    if getattr(forward_batch, "_kunlun_ragged_draft_extend", False):
        return False
    return original_fn(self, forward_batch)


@plugin_hook(
    "sglang.srt.speculative.eagle_info_v2.EagleDraftInputV2Mixin.prepare_for_decode",
    type=HookType.REPLACE,
)
def prepare_for_decode_kunlun(self, batch):
    """Synchronously publish current/next KV lengths before mapping writes."""
    import sglang.srt.speculative.eagle_info_v2 as upstream
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    batch.maybe_evict_swa()
    batch_size = batch.batch_size()
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()
    page_size = batch.token_to_kv_pool_allocator.page_size
    reserve = upstream.get_alloc_reserve_per_decode()
    current_lengths = [0] * batch_size
    next_lengths = [0] * batch_size
    num_needed_tokens = 0
    for index, request in enumerate(batch.reqs):
        current = request.kv_allocated_len
        next_length = max(current, request.kv_committed_len + reserve)
        current_lengths[index] = current
        next_lengths[index] = next_length
        num_needed_tokens += next_length - current
        request.kv_allocated_len = next_length
        request.decode_batch_idx += 1
        request.kv_committed_len += 1

    current_cpu = torch.tensor(current_lengths, dtype=torch.int32, device="cpu")
    next_cpu = torch.tensor(next_lengths, dtype=torch.int32, device="cpu")
    if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(next_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, (
            f"spec v2 allocation {max_alloc_len} exceeds req_to_token width "
            f"{row_width}; page_size={page_size}"
        )

    current_device = current_cpu.to(device=batch.device)
    next_device = next_cpu.to(device=batch.device)
    if page_size == 1:
        out_cache_loc = upstream.alloc_token_slots(
            batch.tree_cache, num_needed_tokens
        )
    else:
        last_loc = upstream.get_last_loc(
            batch.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            current_device,
        )
        out_cache_loc = upstream.alloc_paged_token_slots_extend(
            batch.tree_cache,
            current_device,
            current_cpu,
            next_device,
            next_cpu,
            last_loc,
            num_needed_tokens,
        )
    assign_req_to_token_pool_func(
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        current_device,
        next_device,
        out_cache_loc,
        batch_size,
    )


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._capture_cuda_graphs",
    type=HookType.REPLACE,
)
def capture_cuda_graphs_kunlun(self):
    """Version-pinned graph admission including Kunlun DSV4 without rebinding."""
    import sglang.srt.speculative.eagle_worker_v2 as upstream
    from sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend import (
        KunlunDeepseekV4AttnBackend,
    )

    self.cuda_graph_runner = None
    self.cuda_graph_runner_for_draft_extend = None
    if upstream.check_cuda_graph_backend(upstream.Phase.DECODE, upstream.Backend.DISABLED):
        return
    if self.server_args.model_impl == "mindspore":
        return

    device_to_draft_runner = {
        "npu": upstream.EAGLEDraftNpuGraphRunner,
        "cuda": upstream.EAGLEDraftCudaGraphRunner,
        "musa": upstream.EAGLEDraftCudaGraphRunner,
    }
    if self.speculative_num_steps > 1:
        self.cuda_graph_runner = device_to_draft_runner[self.target_worker.device](self)

    device_to_extend_runner = {
        "npu": upstream.EAGLEDraftExtendNpuGraphRunner,
        "cuda": upstream.EAGLEDraftExtendCudaGraphRunner,
        "musa": upstream.EAGLEDraftCudaGraphRunner,
    }
    supports_hip_aiter = False
    if upstream._is_hip:
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        supports_hip_aiter = isinstance(
            self.draft_attn_backend, AiterMultiStepDraftBackend
        )

    supported_backend_classes = tuple(
        backend_class
        for backend_class in (
            getattr(upstream, "TritonAttnBackend", None),
            getattr(upstream, "TRTLLMMLABackend", None),
            getattr(upstream, "TRTLLMHAAttnBackend", None),
            getattr(upstream, "TokenspeedMLABackend", None),
            getattr(upstream, "FlashInferAttnBackend", None),
            KunlunDeepseekV4AttnBackend,
        )
        if isinstance(backend_class, type)
    )
    supports_cuda_extend = (
        upstream._is_cuda or upstream._is_musa
    ) and isinstance(self.draft_extend_attn_backend, supported_backend_classes)
    if self.draft_extend_attn_backend and (
        upstream._is_npu or supports_cuda_extend or supports_hip_aiter
    ):
        self.cuda_graph_runner_for_draft_extend = device_to_extend_runner[
            self.target_worker.device
        ](self)


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward",
    type=HookType.AROUND,
)
def draft_forward_position_kunlun(original_fn, self, forward_batch):
    """Advance draft positions before model forward while preserving final state."""
    source = inspect.getsource(original_fn)
    position_update = source.find("forward_batch.positions.add_(1)")
    model_forward = source.find("self.draft_runner.forward(forward_batch)")
    if 0 <= position_update < model_forward:
        return original_fn(self, forward_batch)

    original_forward = self.draft_runner.forward

    def next_position_forward(inner_batch, *args, **kwargs):
        inner_batch.positions.add_(1)
        try:
            return original_forward(inner_batch, *args, **kwargs)
        finally:
            inner_batch.positions.sub_(1)

    self.draft_runner.forward = next_position_forward
    try:
        return original_fn(self, forward_batch)
    finally:
        self.draft_runner.forward = original_forward


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.verify",
    type=HookType.AFTER,
)
def publish_verify_stride_kunlun(result, self, batch):
    """Carry the verify-time fixed row width to scheduler result processing."""
    result.speculative_num_draft_tokens = self.speculative_num_draft_tokens
    return result


@plugin_hook(
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor._resolve_spec_v2_tokens",
    type=HookType.REPLACE,
)
def resolve_spec_v2_tokens_kunlun(self, result, batch):
    """Resolve each request from its fixed-width result row without leakage."""
    assert result.next_token_ids.is_cpu
    assert result.accept_lens.is_cpu

    next_token_ids = result.next_token_ids.tolist()
    accept_lens = result.accept_lens.tolist()
    result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)
    result.num_correct_drafts_per_req_cpu = [length - 1 for length in accept_lens]

    on_verify_complete = getattr(self.model_worker, "on_verify_complete_cpu", None)
    if on_verify_complete is not None:
        on_verify_complete(
            result.num_correct_drafts_per_req_cpu,
            batch_size=len(batch.reqs),
        )

    stride = getattr(result, "speculative_num_draft_tokens", None)
    assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"
    assert stride > 0
    assert len(next_token_ids) >= len(batch.reqs) * stride

    predict_tokens = []
    for index, request in enumerate(batch.reqs):
        row = next_token_ids[index * stride : (index + 1) * stride]
        accepted_tokens = row[: accept_lens[index]]

        if request.is_retracted:
            pass
        elif request.finished():
            if not batch.spec_algorithm.is_dflash():
                request.kv_committed_len -= 1
        else:
            if request.grammar is not None:
                accepted_tokens = self._accept_grammar_tokens(request, accepted_tokens)

            num_accepted_tokens = len(accepted_tokens)
            if batch.spec_algorithm.is_dflash():
                request.kv_committed_len += num_accepted_tokens
            else:
                request.kv_committed_len += num_accepted_tokens - 1
            request.spec_verify_ct += 1

            num_correct_drafts = result.num_correct_drafts_per_req_cpu[index]
            request.spec_num_correct_drafts += num_correct_drafts
            request.update_spec_correct_drafts_histogram(num_correct_drafts)

        predict_tokens.append(accepted_tokens)
    return predict_tokens


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode",
    type=HookType.REPLACE,
)
def draft_extend_for_decode_kunlun(self, batch, batch_result):
    """Compact accepted prefixes and run ragged inputs only through eager."""
    import sglang.srt.speculative.eagle_worker_v2 as upstream

    if self.server_args.disable_overlap_schedule:
        accepted = accepted_prefix_indices(
            batch_result.accept_lens, self.speculative_num_draft_tokens
        )
        hidden_states = batch_result.logits_output.hidden_states
        hidden_states = (
            hidden_states.index_select(0, accepted)
            if hidden_states is not None
            else None
        )
        out_cache_loc = batch.out_cache_loc.index_select(0, accepted)
        select_index = torch.cumsum(batch_result.accept_lens, dim=0) - 1
        next_token_ids = batch_result.next_token_ids.index_select(
            0, accepted
        ).to(torch.int64)
    else:
        hidden_states = batch_result.logits_output.hidden_states
        out_cache_loc = batch.out_cache_loc
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

    draft_extend_input = upstream.EagleDraftExtendInput(
        hidden_states=hidden_states,
        num_correct_drafts=batch_result.accept_lens - 1,
        num_accept_tokens=batch_result.accept_lens,
        num_tokens_per_req=self.speculative_num_draft_tokens,
        num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
    )
    batch.out_cache_loc = out_cache_loc
    with self.plan_stream_ctx:
        forward_batch = self.prepare_for_draft_extend(
            draft_extend_input,
            batch,
            next_token_ids,
            self.speculative_num_draft_tokens,
            self.draft_runner,
            self.cuda_graph_runner_for_draft_extend,
        )
    if self.plan_stream:
        torch.get_device_module(self.device).current_stream().wait_stream(
            self.plan_stream
        )

    can_graph = getattr(
        forward_batch, "_kunlun_can_run_draft_extend_graph", None
    )
    if can_graph is None:
        can_graph = can_run_draft_extend_graph(
            self.cuda_graph_runner_for_draft_extend, forward_batch
        )
    manager = self.draft_runner.canary_manager
    canary_context = (
        upstream.context_tuple(
            manager.with_ops_outside_graph(
                single_forward_indices=[0],
                maybe_inaccurate_forward_batch=forward_batch,
            ),
            manager.with_active_single_forward_manager(0),
        )
        if manager is not None
        else contextlib.nullcontext()
    )
    with canary_context:
        output = (
            self.cuda_graph_runner_for_draft_extend.execute(forward_batch)
            if can_graph
            else self.draft_runner.forward(forward_batch).logits_output
        )
    upstream.maybe_detect_nan(
        output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_graph})",
    )
    upstream.maybe_detect_inf(
        output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_graph})",
    )

    output.next_token_logits = output.next_token_logits[select_index]
    if output.hidden_states is not None:
        output.hidden_states = output.hidden_states[select_index]
    if self.server_args.speculative_use_rejection_sampling:
        probs = upstream.renorm_draft_probs(
            output.next_token_logits, batch.sampling_info, True
        )
        topk_p, topk_index = upstream.fast_sample(probs, num_samples=1)
        draft_probs = probs
    elif self.topk == 1 and not upstream._is_hip:
        topk_index = torch.argmax(output.next_token_logits, dim=-1, keepdim=True)
        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
        draft_probs = None
    else:
        probs = upstream.renorm_draft_probs(
            output.next_token_logits,
            batch.sampling_info,
            self.server_args.speculative_use_rejection_sampling,
        )
        topk_p, topk_index = upstream.fast_topk(probs, self.topk, dim=-1)
        draft_probs = None

    next_draft_input = batch_result.next_draft_input
    next_draft_input.topk_p = topk_p
    next_draft_input.topk_index = topk_index
    next_draft_input.hidden_states = output.hidden_states
    if self.server_args.speculative_use_rejection_sampling:
        next_draft_input.draft_probs = draft_probs
