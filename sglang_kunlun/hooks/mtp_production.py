"""Atomic Kunlun MTP token, KV-length, metadata, and graph contract hooks."""

from __future__ import annotations

import contextlib
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

# Default off: compacting to the real accept length makes
# kpool_write_tail_and_maybe_compress see bn % num_draft_tokens != 0 (e.g. bn=2,
# ndt=4); both the upstream host wrapper and the new op require divisibility, so
# the compacted path silently skips / breaks the DSA index-cache write on P800.
_MTP_ACCEPTED_PREFIX_COMPACT = (
    os.environ.get("SGLANG_KUNLUN_MTP_ACCEPTED_PREFIX_COMPACT", "0") == "1"
)


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
    """Return whether packed rows exactly match fixed-width graph request slots."""
    return all(length == width for length in accepted_lengths)


def can_run_draft_extend_graph(cuda_graph_runner, forward_batch) -> bool:
    """Reject ragged accepted-only Draft Extend at every graph boundary."""
    return bool(
        cuda_graph_runner
        and not getattr(forward_batch, "_kunlun_ragged_draft_extend", False)
        and cuda_graph_runner.can_run_graph(forward_batch)
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

    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=batch.capture_hidden_mode,
        return_hidden_states_before_norm=False,
    )
    increment = (
        accept_lens.to(forward_batch.seq_lens.dtype)
        if accepted_only
        else num_draft_tokens
    )
    forward_batch.seq_lens = forward_batch.seq_lens + increment
    # The captured graph and DSV4 LoD use exactly num_draft_tokens queries per
    # request, so any shortened accepted-only row must use ragged eager metadata.
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

    def _capture_shapes() -> tuple:
        return (
            upstream.get_exec().graph.cuda_graph_config.decode.backend,
            upstream.get_batch_sizes_to_capture(self.draft_runner)[0],
        )

    def _account(phase: str, tic: float, before_mem: float) -> None:
        """Mirror upstream's per-phase graph accounting.

        Without this the startup ``cuda_graph={...}`` line reports
        ``draft_decode=0.00`` even when the graph was captured, which reads as
        "ran eager" and has already sent one perf investigation down the wrong
        path.
        """
        after_mem = upstream.get_available_gpu_memory(self.device, self.gpu_id)
        elapsed = upstream.time.perf_counter() - tic
        self._specialized_graph_memory_usage[phase] = (
            self._specialized_graph_memory_usage.get(phase, 0.0)
            + before_mem
            - after_mem
        )
        self._specialized_graph_time_usage[phase] = (
            self._specialized_graph_time_usage.get(phase, 0.0) + elapsed
        )
        upstream.log_info_on_rank0(
            upstream.logger,
            f"Capture {phase.replace('_', ' ')} CUDA graph end. "
            f"elapsed={elapsed:.2f} s, "
            f"mem usage={(before_mem - after_mem):.2f} GB, "
            f"avail mem={after_mem:.2f} GB.",
        )

    device_to_draft_runner = {
        "npu": upstream.EAGLEDraftNpuGraphRunner,
        "cuda": upstream.EAGLEDraftCudaGraphRunner,
        "musa": upstream.EAGLEDraftCudaGraphRunner,
    }
    if self.speculative_num_steps > 1:
        decode_backend, capture_bs = _capture_shapes()
        tic = upstream.time.perf_counter()
        before_mem = upstream.get_available_gpu_memory(self.device, self.gpu_id)
        upstream.log_info_on_rank0(
            upstream.logger,
            f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
            f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
            f"avail mem={before_mem:.2f} GB",
        )
        self.cuda_graph_runner = device_to_draft_runner[self.target_worker.device](self)
        _account("draft_decode", tic, before_mem)

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

    # Upstream admits DeepseekSparseAttnBackend under _is_cuda
    # (eagle_worker_v2.py:493-497) and GLM-5-Next's draft-extend backend
    # (KunlunDSAAttnBackend) derives from it -- but DSA must stay OFF here.
    # `draft_extend_for_decode_kunlun` below replaces upstream's
    # `_draft_extend_for_decode` and never sets
    # `next_draft_input.dsa_topk_indices`, while upstream's graph path reads the
    # DSA index-share seed out of
    # `cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture`
    # (eagle_worker_v2.py:1094-1101, 1149-1150). With the graph on, draft decode
    # step 0 attends with an unseeded indexer top-k: measured accept_len 3.619 ->
    # 2.274 (step-1 hit rate 0.779 -> 0.166) for a 4.4% iteration-time gain, i.e.
    # 21.08 -> 11.06 tok/s. Re-enable only together with seed propagation in the
    # replacement hook.
    dsa_backend_cls = None

    supported_backend_classes = tuple(
        backend_class
        for backend_class in (
            getattr(upstream, "TritonAttnBackend", None),
            getattr(upstream, "TRTLLMMLABackend", None),
            getattr(upstream, "TRTLLMHAAttnBackend", None),
            getattr(upstream, "TokenspeedMLABackend", None),
            getattr(upstream, "FlashInferAttnBackend", None),
            KunlunDeepseekV4AttnBackend,
            dsa_backend_cls,
        )
        if isinstance(backend_class, type)
    )
    supports_cuda_extend = (
        upstream._is_cuda or upstream._is_musa
    ) and isinstance(self.draft_extend_attn_backend, supported_backend_classes)
    if (
        self.draft_extend_attn_backend
        and not upstream.envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
        and (upstream._is_npu or supports_cuda_extend or supports_hip_aiter)
    ):
        decode_backend, capture_bs = _capture_shapes()
        tic = upstream.time.perf_counter()
        before_mem = upstream.get_available_gpu_memory(self.device, self.gpu_id)
        upstream.log_info_on_rank0(
            upstream.logger,
            f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
            f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
            f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
        )
        self.cuda_graph_runner_for_draft_extend = device_to_extend_runner[
            self.target_worker.device
        ](self)
        _account("draft_extend", tic, before_mem)


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.verify",
    type=HookType.AFTER,
)
def publish_verify_stride_kunlun(result, self, batch, grammar_barrier=None):
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

        if request.is_retracted or request.finished():
            # Nothing to settle: no worker pre-claims the bonus, so
            # kv_committed_len already holds the committed prefix.
            pass
        else:
            if request.grammar is not None:
                accepted_tokens = self._accept_grammar_tokens(request, accepted_tokens)

            # Commit the full accepted run (drafts + bonus). Upstream keeps
            # kv_committed_len honest; eagle_prepare_for_decode rounds the
            # speculative reserve up from it, so any lag strands whole pages.
            num_accepted_tokens = len(accepted_tokens)
            request.kv_committed_len += num_accepted_tokens
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
    """Draft extend over the full tree width by default, matching upstream.

    Compacting each request down to its accepted prefix (the upstream DSV4
    behaviour, reachable via ``SGLANG_KUNLUN_MTP_ACCEPTED_PREFIX_COMPACT=1``)
    saves ``bs * ndt - sum(accept_lens)`` draft tokens but leaves the batch
    ragged while ``num_tokens_per_req`` still claims the full width. Every kpool
    consumer downstream is built for the fixed width -- the DSA write plan is
    created with ``num_draft_tokens=speculative_num_draft_tokens`` and
    ``write_start = seq_lens - num_draft_tokens`` (``dsa_backend.py:1103``), and
    both the op and the upstream host wrapper require
    ``key.size(0) % num_draft_tokens == 0`` (``kpool_fp8_index.py:1830``) -- so
    the compacted batch skips the draft model's index-cache write instead.
    Upstream keeps the whole tree width for exactly this reason
    (``eagle_worker_v2.py:974``) and only picks the last accepted row per request
    via ``select_index``.
    """
    import sglang.srt.speculative.eagle_worker_v2 as upstream

    if _MTP_ACCEPTED_PREFIX_COMPACT and self.server_args.disable_overlap_schedule:
        # Upstream DSV4 path: compact each request to its accepted prefix.
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
        next_token_ids = batch_result.next_token_ids.index_select(0, accepted).to(
            torch.int64
        )
    else:
        # GLM/P800 default: keep the full tree width (see module-level flag note).
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
        forward_batch = prepare_for_draft_extend_kunlun(
            self,
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
