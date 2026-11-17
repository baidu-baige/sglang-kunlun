"""Separate DSPARK graphs for all-greedy requests, retaining sampling state.

Only the second capture suppresses Markov Philox. Both graphs keep step logits,
the default greedy_draft=False path, collectives and the block offset increment.
Replay selection uses staged CPU metadata, never capture-time tensor contents.
"""

from contextvars import ContextVar
import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)
_SAMPLER = "sglang.srt.speculative.dspark_components.dspark_draft_sampler"
_BACKEND = (
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend."
    "FullCudaGraphBackend"
)
_capture_sampler = ContextVar("kunlun_dspark_greedy_capture_sampler", default=None)
_markov_greedy = ContextVar("kunlun_dspark_greedy_markov", default=False)


@plugin_hook(f"{_SAMPLER}.maybe_build_draft_sampler", type=HookType.AFTER)
def remember_sampler(result, *, draft_model, **kwargs):
    """Record the draft sampler on the draft model after it is built."""
    draft_model._kunlun_dspark_sampler = result
    if result is not None:
        # Unknown until stage_sampling_params has actually completed.
        result._kunlun_all_greedy = False
    return result


@plugin_hook(
    f"{_SAMPLER}.DsparkDraftSampler.stage_sampling_params", type=HookType.AFTER
)
def remember_sampling_mode(result, self, *, bs, sampling_info):
    """Mark whether the staged sampling params are for an all-greedy batch."""
    # Identity checks deliberately reject tensors and unknown metadata.
    self._kunlun_all_greedy = sampling_info is None or (
        getattr(sampling_info, "is_all_greedy", None) is True
    )
    return result


@plugin_hook(f"{_SAMPLER}.DsparkDraftSampler.__call__", type=HookType.AROUND)
def greedy_sampler_capture(original_fn, self, *args, **kwargs):
    """Flag markov-Philox suppression while the greedy sampler capture runs."""
    token = _markov_greedy.set(
        _capture_sampler.get() is self and self.greedy_draft is False
    )
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        _markov_greedy.reset(token)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.markov_sample.MarkovSampleStep.execute",
    type=HookType.AROUND,
)
def greedy_markov_capture(original_fn, cls, **kwargs):
    """Keep logits_out on the torch path during greedy markov capture."""
    # HookRegistry applies REPLACE first, so original_fn is the existing Kunlun
    # dispatcher. Keeping logits_out forces its torch path, not native Triton.
    if _markov_greedy.get() and kwargs.get("logits_out") is not None:
        kwargs["has_sampled_rows"] = False
    return original_fn(cls, **kwargs)


def _eligible_sampler(backend):
    """Return the DSpark sampler eligible for greedy graph capture, else None."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.runtime_context import get_parallel

    runner = getattr(backend, "_cuda_graph_runner", None)
    if not isinstance(runner, DecodeCudaGraphRunner):
        return None
    mr = runner.model_runner
    algo = getattr(mr, "spec_algorithm", None)
    if (
        getattr(mr, "is_draft_worker", False) is not True
        or not callable(getattr(algo, "is_dspark", None))
        or not algo.is_dspark()
        or any(
            getattr(runner, flag, False)
            for flag in (
                "enable_torch_compile",
                "enable_two_batch_overlap",
                "enable_pdmux",
                "enable_profile_cuda_graph",
            )
        )
    ):
        return None
    model = getattr(mr, "model", None)
    sampler = getattr(model, "_kunlun_dspark_sampler", None)
    if (
        sampler is None
        or getattr(sampler, "model", None) is not model
        or getattr(sampler, "greedy_draft", None) is not False
    ):
        return None
    try:
        parallel = get_parallel()
        if (
            parallel.attn_tp_group.world_size != 1
            or parallel.attn_tp_size != 1
            or parallel.enable_dp_lm_head is not True
            or getattr(model.lm_head, "use_attn_tp_group", None) is not True
        ):
            return None
    except (AttributeError, RuntimeError, AssertionError):
        # Unpublished runtime / unsupported topology must keep the normal graph.
        return None
    return sampler


@plugin_hook(f"{_BACKEND}.capture_one", type=HookType.AROUND)
def capture_greedy_graph(original_fn, self, shape_key, *args, **kwargs):
    """Capture an extra all-greedy graph alongside the normal one."""
    sampler = _eligible_sampler(self)
    graphs = getattr(self, "_kunlun_greedy_graphs", None)
    outputs = getattr(self, "_kunlun_greedy_outputs", None)
    if graphs is not None:
        # Recapture must not leave an alternate graph for obsolete inputs.
        old = graphs.pop(shape_key, None)
        outputs.pop(shape_key, None)
        if old is not None:
            old.reset()
    result = original_fn(self, shape_key, *args, **kwargs)
    if sampler is None:
        return result
    if graphs is None:
        graphs = self._kunlun_greedy_graphs = {}
        outputs = self._kunlun_greedy_outputs = {}

    normal_graph = self._graphs[shape_key]
    normal_output = self._outputs[shape_key]
    # Extra warmups/capture must not advance the startup RNG counter. Each
    # subsequent replay still advances it through upstream draft_markov_block.
    offset = sampler._noise.offset
    saved_offset = offset.clone()
    token = _capture_sampler.set(sampler)
    try:
        # Same shape, pool, stream, precarve and output reuse policy. Upstream
        # capture_one has no dedup check; it overwrites only these two entries.
        original_fn(self, shape_key, *args, **kwargs)
        graphs[shape_key] = self._graphs[shape_key]
        outputs[shape_key] = self._outputs[shape_key]
    finally:
        _capture_sampler.reset(token)
        self._graphs[shape_key] = normal_graph
        self._outputs[shape_key] = normal_output
        offset.copy_(saved_offset)
    logger.info("DSPARK all-greedy graph captured: %s", shape_key)
    return result


@plugin_hook(f"{_BACKEND}.replay", type=HookType.AROUND)
def replay_greedy_graph(original_fn, self, shape_key, *args, **kwargs):
    """Replay the all-greedy graph variant when the batch is all-greedy."""
    sampler = _eligible_sampler(self)
    graphs = getattr(self, "_kunlun_greedy_graphs", {})
    if (
        sampler is None
        or getattr(sampler, "_kunlun_all_greedy", False) is not True
        or shape_key not in graphs
    ):
        return original_fn(self, shape_key, *args, **kwargs)
    normal_graph = self._graphs[shape_key]
    normal_output = self._outputs[shape_key]
    self._graphs[shape_key] = graphs[shape_key]
    self._outputs[shape_key] = self._kunlun_greedy_outputs[shape_key]
    try:
        # Preserve upstream graph_pool_replay_scope and any existing wrappers.
        return original_fn(self, shape_key, *args, **kwargs)
    finally:
        self._graphs[shape_key] = normal_graph
        self._outputs[shape_key] = normal_output


@plugin_hook(f"{_BACKEND}.cleanup", type=HookType.AROUND)
def cleanup_greedy_graphs(original_fn, self, *args, **kwargs):
    """Reset and drop all cached all-greedy graphs, then run upstream cleanup."""
    graphs = getattr(self, "_kunlun_greedy_graphs", {})
    for graph in graphs.values():
        graph.reset()
    graphs.clear()
    getattr(self, "_kunlun_greedy_outputs", {}).clear()
    return original_fn(self, *args, **kwargs)
