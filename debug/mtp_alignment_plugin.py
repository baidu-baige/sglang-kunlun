"""Optional target-side MTP alignment hook registration."""
from __future__ import annotations

import importlib
import logging
import os

import torch

from mtp_alignment_common import emit, has_generation_context
from mtp_alignment_hooks import register_target

logger = logging.getLogger(__name__)
_GRAPH_OUTPUTS_ATTR = "_dsv4_alignment_graph_outputs"
_GRAPH_EVENT_COUNT_ATTR = "_dsv4_alignment_graph_event_count"
_MODEL_OUTPUTS_ATTR = "_DSV4_DECODE_LAYER_OUTPUTS"
_ACTIVE_GRAPH_CAPTURE = False
_RUNNER_TARGET = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
    "DecodeCudaGraphRunner"
)


def _graph_operator_probe_enabled() -> bool:
    return bool(os.environ.get("DSV4_ALIGNMENT_PROBE_DIR")) and (
        os.environ.get("DSV4_ALIGNMENT_GRAPH_OPERATOR_PROBE") == "1"
        or os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1"
        or os.environ.get("DSV4_DECODE_LAYER_DUMP") == "1"
    )


def _model_outputs() -> dict[str, torch.Tensor]:
    model_module = importlib.import_module("sglang.srt.models.deepseek_v4")
    outputs = getattr(model_module, _MODEL_OUTPUTS_ATTR, None)
    if outputs is None:
        outputs = {}
        setattr(model_module, _MODEL_OUTPUTS_ATTR, outputs)
    return outputs


def _probe_layers() -> set[int]:
    return {
        int(layer_id)
        for layer_id in os.environ.get("DSV4_DECODE_PROBE_LAYERS", "2").split(",")
        if layer_id
    }


def _probe_capture_sizes() -> set[int]:
    return {
        int(size)
        for size in os.environ.get(
            "DSV4_ALIGNMENT_GRAPH_PROBE_CAPTURE_SIZES", "4"
        ).split(",")
        if size
    }


def _capture_graph_shape(
    original_fn,
    self,
    size,
    forward,
    stream_idx=None,
    variant_label=None,
):
    if not _graph_operator_probe_enabled():
        return original_fn(self, size, forward, stream_idx, variant_label)

    global _ACTIVE_GRAPH_CAPTURE

    model_module = importlib.import_module("sglang.srt.models.deepseek_v4")
    had_model_outputs = hasattr(model_module, _MODEL_OUTPUTS_ATTR)
    saved_model_outputs = getattr(model_module, _MODEL_OUTPUTS_ATTR, None)
    graph_outputs: dict[str, torch.Tensor] = {}
    setattr(model_module, _MODEL_OUTPUTS_ATTR, graph_outputs)
    should_capture = (
        int(size) in _probe_capture_sizes()
        and not bool(getattr(self.model_runner, "is_draft_worker", False))
    )
    saved_active_capture = _ACTIVE_GRAPH_CAPTURE
    saved_verify_flag = os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP")
    saved_decode_flag = os.environ.get("DSV4_DECODE_LAYER_DUMP")
    _ACTIVE_GRAPH_CAPTURE = should_capture
    os.environ["DSV4_MTP_VERIFY_LAYER_DUMP"] = "0"
    os.environ["DSV4_DECODE_LAYER_DUMP"] = "0"
    try:
        result = original_fn(self, size, forward, stream_idx, variant_label)
    finally:
        _ACTIVE_GRAPH_CAPTURE = saved_active_capture
        if had_model_outputs:
            setattr(model_module, _MODEL_OUTPUTS_ATTR, saved_model_outputs)
        else:
            delattr(model_module, _MODEL_OUTPUTS_ATTR)
        for name, value in (
            ("DSV4_MTP_VERIFY_LAYER_DUMP", saved_verify_flag),
            ("DSV4_DECODE_LAYER_DUMP", saved_decode_flag),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not should_capture:
        return result

    shape_key = self._make_graph_key(size, stream_idx, variant_label)
    outputs_by_key = getattr(self, _GRAPH_OUTPUTS_ATTR, None)
    if outputs_by_key is None:
        outputs_by_key = {}
        setattr(self, _GRAPH_OUTPUTS_ATTR, outputs_by_key)
    outputs_by_key[shape_key] = graph_outputs
    return result


def _capture_all_verify_widths(
    original_fn,
    backend,
    *,
    layer,
    forward_batch,
    core,
    q_3d,
    win_indices,
    extra_indices,
    kv_lens,
):
    suppress_native_probe = _graph_operator_probe_enabled()
    saved_verify_flag = os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP")
    saved_decode_flag = os.environ.get("DSV4_DECODE_LAYER_DUMP")
    if suppress_native_probe:
        os.environ["DSV4_MTP_VERIFY_LAYER_DUMP"] = "0"
        os.environ["DSV4_DECODE_LAYER_DUMP"] = "0"
    try:
        result = original_fn(
            backend,
            layer=layer,
            forward_batch=forward_batch,
            core=core,
            q_3d=q_3d,
            win_indices=win_indices,
            extra_indices=extra_indices,
            kv_lens=kv_lens,
        )
    finally:
        if suppress_native_probe:
            for name, value in (
                ("DSV4_MTP_VERIFY_LAYER_DUMP", saved_verify_flag),
                ("DSV4_DECODE_LAYER_DUMP", saved_decode_flag),
            ):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    layer_id = getattr(layer, "layer_id", -1)
    probe_prefix = f"layer{layer_id}"
    already_captured = getattr(backend, "_dsv4_decode_probe_prefix", None)
    should_capture = (
        _ACTIVE_GRAPH_CAPTURE
        and layer_id in _probe_layers()
        and str(getattr(forward_batch, "forward_mode", None))
        == "ForwardMode.TARGET_VERIFY"
        and torch.cuda.is_current_stream_capturing()
        and already_captured != probe_prefix
    )
    if not should_capture:
        return result

    outputs = _model_outputs()
    backend._dsv4_decode_probe_prefix = probe_prefix
    outputs[f"{probe_prefix}_attn_q"] = q_3d.clone()
    outputs[f"{probe_prefix}_win_indices"] = win_indices.clone()
    outputs[f"{probe_prefix}_com_indices"] = extra_indices.clone()
    outputs[f"{probe_prefix}_kv_lens"] = kv_lens.clone()
    for name, value in (
        ("input_ids", forward_batch.input_ids),
        ("positions", forward_batch.positions),
        ("seq_lens", forward_batch.seq_lens),
        ("req_pool_indices", forward_batch.req_pool_indices),
        ("scheduler_out_cache_loc", forward_batch.out_cache_loc),
    ):
        if isinstance(value, torch.Tensor):
            outputs[f"{probe_prefix}_cache_{name}"] = value.clone()
    for name in (
        "raw_out_loc",
        "swa_out_cache_loc",
        "c4_out_loc",
        "page_table",
        "swa_page_indices",
        "c4_sparse_page_indices",
        "c4_sparse_raw_indices",
    ):
        value = getattr(core, name, None)
        if isinstance(value, torch.Tensor):
            outputs[f"{probe_prefix}_cache_{name}"] = value.clone()
    return result


def _capture_indexer_logits(original_fn, backend, site, scope):
    result = original_fn(backend, site, scope)
    should_capture = (
        site == "indexer.operator_outputs"
        and _ACTIVE_GRAPH_CAPTURE
        and bool(scope.get("is_target_verify"))
        and int(scope.get("layer_id", -1)) == 2
        and torch.cuda.is_current_stream_capturing()
    )
    if should_capture:
        outputs = _model_outputs()
        for name in ("q", "logits"):
            value = scope.get(name)
            if isinstance(value, torch.Tensor):
                outputs[f"layer2_indexer_operator_{name}"] = value.clone()
    return result


def _capture_graphs(original_fn, self, *args, **kwargs):
    setattr(self, _GRAPH_OUTPUTS_ATTR, {})
    setattr(self, _GRAPH_EVENT_COUNT_ATTR, 0)
    return original_fn(self, *args, **kwargs)


def _bounded_graph_outputs(self, forward_batch) -> dict[str, torch.Tensor]:
    outputs_by_key = getattr(self, _GRAPH_OUTPUTS_ATTR, {})
    outputs = outputs_by_key.get(getattr(self, "_replay_graph_key", None), {})
    graph_bs = int(getattr(self, "bs", forward_batch.batch_size))
    tokens_per_request = int(getattr(self, "num_tokens_per_bs", 1))
    max_requests = int(os.environ.get("DSV4_ALIGNMENT_GRAPH_PROBE_REQUESTS", "4"))
    request_rows = min(int(forward_batch.batch_size), max_requests)
    max_numel = int(os.environ.get("DSV4_ALIGNMENT_GRAPH_PROBE_MAX_NUMEL", "10000000"))
    bounded = {
        "physical_batch_size": torch.tensor(graph_bs, dtype=torch.int64),
        "logical_batch_size": torch.tensor(
            int(forward_batch.batch_size), dtype=torch.int64
        ),
        "tokens_per_request": torch.tensor(tokens_per_request, dtype=torch.int64),
        "is_draft_worker": torch.tensor(
            int(bool(getattr(self.model_runner, "is_draft_worker", False))),
            dtype=torch.int64,
        ),
    }
    for name, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        selected = value
        if value.ndim:
            if value.shape[0] == graph_bs * tokens_per_request:
                selected = value[: request_rows * tokens_per_request]
            elif value.shape[0] == graph_bs:
                selected = value[:request_rows]
        if selected.numel() <= max_numel:
            bounded[name] = selected
    return bounded


def _emit_graph_replay(original_fn, self, forward_batch, *args, **kwargs):
    result = original_fn(self, forward_batch, *args, **kwargs)
    max_events = int(os.environ.get("DSV4_ALIGNMENT_GRAPH_PROBE_MAX_EVENTS", "4"))
    event_count = int(getattr(self, _GRAPH_EVENT_COUNT_ATTR, 0))
    should_emit = (
        _graph_operator_probe_enabled()
        and not bool(getattr(self.model_runner, "is_draft_worker", False))
        and int(getattr(self, "bs", -1)) in _probe_capture_sizes()
        and event_count < max_events
        and has_generation_context()
        and str(getattr(forward_batch, "forward_mode", None))
        == "ForwardMode.TARGET_VERIFY"
    )
    if not should_emit:
        return result

    graph_outputs = _bounded_graph_outputs(self, forward_batch)
    if len(graph_outputs) <= 4:
        logger.warning(
            "MTP graph operator probe found no tensors for replay key %r",
            getattr(self, "_replay_graph_key", None),
        )
        return result
    try:
        emit(
            "verify_graph_operator.post",
            batch=forward_batch,
            result=graph_outputs,
        )
        setattr(self, _GRAPH_EVENT_COUNT_ATTR, event_count + 1)
    except Exception:
        logger.exception("MTP graph operator probe failed after replay")
    return result


def _register_graph_operator_probes() -> None:
    if not _graph_operator_probe_enabled():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        f"{_RUNNER_TARGET}.capture",
        _capture_graphs,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_RUNNER_TARGET}.capture_one_shape",
        _capture_graph_shape,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_RUNNER_TARGET}.execute",
        _emit_graph_replay,
        HookType.AROUND,
    )
    HookRegistry.register(
        "debug.dsv4_backend_probes.capture_decode_layer_inputs",
        _capture_all_verify_widths,
        HookType.AROUND,
    )
    HookRegistry.register(
        "debug.dsv4_backend_probes.dsv4_probe",
        _capture_indexer_logits,
        HookType.AROUND,
    )
    logger.warning("MTP CUDA Graph operator probes registered")


register_target()
_register_graph_operator_probes()
