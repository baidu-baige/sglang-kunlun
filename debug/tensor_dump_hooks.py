"""Diagnostic-only tensor dump hooks for Kunlun alignment runs."""

from __future__ import annotations

import logging
import os
import sys
from types import MethodType

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _register_layer_boundary_hooks(
    tensor_dumper, model, dump_layers, top_level_module_name, layers_module_name
):
    top_level_module = model._modules.get(top_level_module_name)
    assert top_level_module is not None, (
        f"model should have a module named {top_level_module_name}"
    )
    layers_module = top_level_module._modules.get(layers_module_name)
    assert layers_module is not None, (
        f"{top_level_module_name} should have a module named {layers_module_name}"
    )

    def hidden_state_hook(tensor_name):
        def hook(_module, _inputs, output):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            tensor_dumper.add_tensor(tensor_name, hidden_states)

        return hook

    selected_layers = None if dump_layers is None else set(dump_layers)
    for name, module in layers_module._modules.items():
        if module is None or not name.isdigit():
            continue
        layer_id = int(name)
        if selected_layers is not None and layer_id not in selected_layers:
            continue
        tensor_name = f"{top_level_module_name}.{layers_module_name}.{name}"
        module.register_forward_hook(hidden_state_hook(tensor_name))


def _register_input_capture_hooks(tensor_dumper, model, module_suffix):
    if not module_suffix:
        return

    import torch

    for module_name, module in model.named_modules():
        if not module_name.endswith(module_suffix):
            continue

        def capture_inputs(_module, inputs, _output, name=module_name):
            for index, item in enumerate(inputs):
                if isinstance(item, (torch.Tensor, tuple, list)):
                    tensor_dumper.add_tensor(f"{name}.input.{index}", item)

        module.register_forward_hook(capture_inputs)


def _register_parent_io_hook(tensor_dumper, model, module_suffix):
    if not module_suffix:
        return

    import torch

    for module_name, module in model.named_modules():
        if not module_name.endswith(module_suffix):
            continue

        def capture_parent_io(
            _module, inputs, kwargs, output, name=module_name
        ):
            for index, item in enumerate(inputs):
                if isinstance(item, (torch.Tensor, tuple, list)):
                    tensor_dumper.add_tensor(f"{name}.input.{index}", item)
            for key, item in kwargs.items():
                if isinstance(item, (torch.Tensor, tuple, list)):
                    tensor_dumper.add_tensor(f"{name}.input.{key}", item)
            for key in ("weight", "weight_scale", "weight_scale_inv", "bias"):
                item = getattr(_module, key, None)
                if isinstance(item, torch.Tensor):
                    tensor_dumper.add_tensor(f"{name}.param.{key}", item)
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            tensor_dumper.add_tensor(name, hidden_states)

        module.register_forward_hook(capture_parent_io, with_kwargs=True)


def _register_dsv4_attention_chain(tensor_dumper):
    if os.getenv("TENSOR_DUMP_DSV4_ATTN_CHAIN", "0") != "1":
        return None

    import torch

    ops = getattr(torch.ops, "xspeedgate_ops", None)
    if ops is None:
        raise RuntimeError("xspeedgate_ops is required for the DSV4 attention chain")

    state = {"compressed_attention": 0, "einsum_tgd_grd_tgr": 0}

    def add_tensor(name, value):
        if isinstance(value, torch.Tensor):
            tensor_dumper.add_tensor(f"dsv4_attn.layer0.{name}", value)

    def gather_cache_rows(cache, indices):
        if indices.numel() == 0:
            return cache.new_empty((*indices.shape, *cache.shape[1:]))
        valid = indices >= 0
        flat_indices = (
            indices.reshape(-1)
            .clamp(min=0, max=cache.shape[0] - 1)
            .long()
        )
        rows = cache.index_select(0, flat_indices).reshape(
            *indices.shape, *cache.shape[1:]
        )
        mask = valid.reshape(*indices.shape, *((1,) * (cache.ndim - 1)))
        return torch.where(mask, rows, torch.zeros_like(rows))

    original_attention = ops.compressed_attention

    def compressed_attention_with_dump(*args, **kwargs):
        capture = state["compressed_attention"] == 0
        state["compressed_attention"] += 1
        if capture:
            input_names = {
                0: "compressed_attention.input.q",
                2: "compressed_attention.input.win_indices",
                4: "compressed_attention.input.extra_indices",
                8: "compressed_attention.input.q_lod_cpu",
                9: "compressed_attention.input.q_lod",
                10: "compressed_attention.input.kv_lens_cpu",
                11: "compressed_attention.input.kv_lens",
                17: "compressed_attention.input.attn_sink",
            }
            for index, name in input_names.items():
                if index < len(args):
                    add_tensor(name, args[index])
            if len(args) > 2:
                add_tensor(
                    "compressed_attention.input.win_cache_rows",
                    gather_cache_rows(args[1], args[2]),
                )
            if len(args) > 4:
                add_tensor(
                    "compressed_attention.input.extra_cache_rows",
                    gather_cache_rows(args[3], args[4]),
                )
            for index, name in {
                12: "softmax_scale",
                13: "causal",
                14: "max_window_size",
                15: "compress_ratio",
                16: "compressed_topk",
            }.items():
                if index < len(args) and args[index] is not None:
                    add_tensor(
                        f"compressed_attention.contract.{name}",
                        torch.as_tensor(args[index]),
                    )
        result = original_attention(*args, **kwargs)
        if capture:
            for index, name in {
                5: "compressed_attention.output.out",
                6: "compressed_attention.output.max_logits",
                7: "compressed_attention.output.lse",
            }.items():
                if index < len(args):
                    add_tensor(name, args[index])
            add_tensor("compressed_attention.output.return", result)
        return result

    original_einsum = ops.einsum_tgd_grd_tgr

    def einsum_with_dump(*args, **kwargs):
        capture = state["einsum_tgd_grd_tgr"] == 0
        state["einsum_tgd_grd_tgr"] += 1
        if capture:
            if args:
                add_tensor("wo_a_einsum.input.o", args[0])
            if len(args) > 1:
                add_tensor("wo_a_einsum.input.weight", args[1])
        result = original_einsum(*args, **kwargs)
        if capture:
            add_tensor("wo_a_einsum.output", result)
            tensor_dumper.add_tensor("dsv4_attn.layer0", result)
        return result

    ops.compressed_attention = compressed_attention_with_dump
    ops.einsum_tgd_grd_tgr = einsum_with_dump

    def reset():
        state["compressed_attention"] = 0
        state["einsum_tgd_grd_tgr"] = 0

    return reset


def _register_dsv4_selected_attention_chain(tensor_dumper, model):
    if os.getenv("TENSOR_DUMP_DSV4_SELECTED_ATTN_CHAIN", "0") != "1":
        return None

    import torch

    layer_id = int(os.getenv("TENSOR_DUMP_DSV4_SELECTED_ATTN_LAYER", "0"))
    layers = model.model.layers
    if not 0 <= layer_id < len(layers):
        raise ValueError(
            f"TENSOR_DUMP_DSV4_SELECTED_ATTN_LAYER={layer_id} is outside "
            f"the decoder layer range [0, {len(layers)})"
        )
    attn = layers[layer_id].self_attn
    if getattr(attn, "compress_ratio", None) != 4:
        raise ValueError(
            f"Selected DSV4 attention layer {layer_id} is not C4: "
            f"compress_ratio={getattr(attn, 'compress_ratio', None)}"
        )
    prefix = f"dsv4_selected_attn.layer{layer_id}"
    state = {"active": 0}
    capture_cache_rows = (
        os.getenv("TENSOR_DUMP_DSV4_SELECTED_ATTN_CACHE_ROWS", "0") == "1"
    )
    lightweight = (
        os.getenv("TENSOR_DUMP_DSV4_SELECTED_ATTN_LIGHTWEIGHT", "0") == "1"
    )
    deferred = (
        os.getenv("TENSOR_DUMP_DSV4_SELECTED_ATTN_DEFERRED", "0") == "1"
    )
    lightweight_names = {
        "compressed_attention.input.q",
        "compressed_attention.input.win_indices",
        "compressed_attention.input.extra_indices",
        "compressed_attention.input.q_lod_cpu",
        "compressed_attention.input.q_lod",
        "compressed_attention.input.kv_lens_cpu",
        "compressed_attention.input.kv_lens",
        "compressed_attention.input.attn_sink",
        "compressed_attention.output.out",
        "compressed_attention.output.max_logits",
        "compressed_attention.output.lse",
    }

    def add_tensor(name, value):
        if lightweight and not (
            name in lightweight_names
            or name.startswith("compressed_attention.contract.")
        ):
            return
        if isinstance(value, torch.Tensor):
            add = (
                tensor_dumper.add_tensor_deferred
                if deferred
                else tensor_dumper.add_tensor
            )
            add(f"{prefix}.{name}", value)

    def first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            return next(
                (item for item in value if isinstance(item, torch.Tensor)),
                None,
            )
        return None

    def capture_module_io(name):
        def hook(module, inputs, kwargs, output):
            if state["active"] == 0:
                return
            for index, item in enumerate(inputs):
                add_tensor(f"{name}.input.{index}", item)
            for key, item in kwargs.items():
                add_tensor(f"{name}.input.{key}", item)
            for key in ("weight", "weight_scale", "weight_scale_inv", "bias"):
                add_tensor(f"{name}.param.{key}", getattr(module, key, None))
            wkv_gate = getattr(module, "wkv_gate", None)
            if wkv_gate is not None:
                add_tensor(
                    f"{name}.wkv_gate.param.weight",
                    getattr(wkv_gate, "weight", None),
                )
            add_tensor(f"{name}.output", first_tensor(output))

        return hook

    modules = {
        "q_a": getattr(attn, "wq_a", None),
        "qkv_a": getattr(attn, "wqkv_a", None),
        "q_norm": getattr(attn, "q_norm", None),
        "q_b": getattr(attn, "wq_b", None),
        "kv": getattr(attn, "wkv", None),
        "kv_norm": getattr(attn, "kv_norm", None),
        "core_compressor": getattr(attn, "compressor", None),
        "wo_b": getattr(attn, "wo_b", None),
    }
    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        raise ValueError(f"Selected DSV4 C4 layer {layer_id} has no indexer")
    modules.update(
        {
            "indexer": indexer,
            "indexer.q_b": getattr(indexer, "wq_b", None),
            "indexer.weights_proj": getattr(indexer, "weights_proj", None),
            "indexer.compressor": getattr(indexer, "compressor", None),
        }
    )
    for name, module in modules.items():
        if module is not None:
            module.register_forward_hook(capture_module_io(name), with_kwargs=True)

    def capture_compressor_score(name, compressor):
        if compressor is None or not hasattr(compressor, "compute_kv_score"):
            return
        original = compressor.compute_kv_score

        def with_dump(self, x, *args, **kwargs):
            capture = state["active"] > 0
            if capture:
                add_tensor(f"{name}.compute_kv_score.input.x", x)
                add_tensor(
                    f"{name}.compute_kv_score.input.weight",
                    self.wkv_gate.weight,
                )
            result = original(x, *args, **kwargs)
            if capture:
                add_tensor(f"{name}.compute_kv_score.output", result)
            return result

        compressor.compute_kv_score = MethodType(with_dump, compressor)

    capture_compressor_score("core_compressor", modules["core_compressor"])
    capture_compressor_score(
        "indexer.compressor", modules["indexer.compressor"]
    )

    def capture_method(name):
        original = getattr(attn, name)

        def with_dump(self, *args, **kwargs):
            result = original(*args, **kwargs)
            if state["active"]:
                add_tensor(f"{name}.output", first_tensor(result))
            return result

        setattr(attn, name, MethodType(with_dump, attn))

    for method_name in ("_compute_q_a", "_compute_q_b"):
        capture_method(method_name)

    def capture_indexer_state(_module, _inputs, kwargs, _output):
        if state["active"] == 0:
            return
        forward_batch = kwargs.get("forward_batch")
        backend = kwargs.get("attn_backend")
        if backend is None and forward_batch is not None:
            backend = getattr(forward_batch, "attn_backend", None)
        metadata = getattr(backend, "forward_metadata", None)
        core = getattr(metadata, "core_metadata", None)
        add_tensor(
            "indexer.output.c4_sparse_page_indices",
            getattr(core, "c4_sparse_page_indices", None),
        )

    indexer.register_forward_hook(capture_indexer_state, with_kwargs=True)

    ops = getattr(torch.ops, "xspeedgate_ops", None)
    if ops is None:
        raise RuntimeError("xspeedgate_ops is required for selected DSV4 attention")

    def gather_cache_rows(cache, indices):
        if indices.numel() == 0:
            return cache.new_empty((*indices.shape, *cache.shape[1:]))
        valid = indices >= 0
        flat_indices = (
            indices.reshape(-1)
            .clamp(min=0, max=cache.shape[0] - 1)
            .long()
        )
        rows = cache.index_select(0, flat_indices).reshape(
            *indices.shape, *cache.shape[1:]
        )
        mask = valid.reshape(*indices.shape, *((1,) * (cache.ndim - 1)))
        return torch.where(mask, rows, torch.zeros_like(rows))

    original_attention = ops.compressed_attention

    def compressed_attention_with_dump(*args, **kwargs):
        capture = state["active"] > 0
        if capture:
            for index, name in {
                0: "compressed_attention.input.q",
                2: "compressed_attention.input.win_indices",
                4: "compressed_attention.input.extra_indices",
                8: "compressed_attention.input.q_lod_cpu",
                9: "compressed_attention.input.q_lod",
                10: "compressed_attention.input.kv_lens_cpu",
                11: "compressed_attention.input.kv_lens",
                17: "compressed_attention.input.attn_sink",
            }.items():
                if index < len(args):
                    add_tensor(name, args[index])
            if capture_cache_rows and len(args) > 2:
                add_tensor(
                    "compressed_attention.input.win_cache_rows",
                    gather_cache_rows(args[1], args[2]),
                )
            if capture_cache_rows and len(args) > 4:
                add_tensor(
                    "compressed_attention.input.extra_cache_rows",
                    gather_cache_rows(args[3], args[4]),
                )
            for index, name in {
                12: "softmax_scale",
                13: "causal",
                14: "max_window_size",
                15: "compress_ratio",
                16: "compressed_topk",
            }.items():
                if index < len(args) and args[index] is not None:
                    add_tensor(
                        f"compressed_attention.contract.{name}",
                        torch.as_tensor(args[index]),
                    )
        result = original_attention(*args, **kwargs)
        if capture:
            for index, name in {
                5: "compressed_attention.output.out",
                6: "compressed_attention.output.max_logits",
                7: "compressed_attention.output.lse",
            }.items():
                if index < len(args):
                    add_tensor(name, args[index])
            add_tensor("compressed_attention.output.return", result)
        return result

    original_einsum = ops.einsum_tgd_grd_tgr

    def einsum_with_dump(*args, **kwargs):
        capture = state["active"] > 0
        if capture:
            if args:
                add_tensor("wo_a_einsum.input.o", args[0])
            if len(args) > 1:
                add_tensor("wo_a_einsum.input.weight", args[1])
        result = original_einsum(*args, **kwargs)
        if capture:
            add_tensor("wo_a_einsum.output", result)
        return result

    ops.compressed_attention = compressed_attention_with_dump
    ops.einsum_tgd_grd_tgr = einsum_with_dump

    original_forward = attn.forward

    def forward_with_dump(self, *args, **kwargs):
        outermost = state["active"] == 0
        if outermost:
            state["active"] = 1
            for index, item in enumerate(args):
                add_tensor(f"input.{index}", item)
            for key, item in kwargs.items():
                add_tensor(f"input.{key}", item)
            add_tensor("param.attn_sink", self.attn_sink)
        try:
            result = original_forward(*args, **kwargs)
            if outermost:
                add_tensor("output", first_tensor(result))
            return result
        finally:
            if outermost:
                state["active"] = 0

    attn.forward = MethodType(forward_with_dump, attn)

    def reset():
        state["active"] = 0

    return reset


def _register_dsv4_layer0_block_chain(tensor_dumper, model):
    if os.getenv("TENSOR_DUMP_DSV4_LAYER0_BLOCK_CHAIN", "0") != "1":
        return None

    import torch

    selected_rank = os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_RANK")
    if selected_rank is not None:
        rank_marker = f"TP{int(selected_rank)}_"
        if not os.path.basename(tensor_dumper.get_dump_dir()).startswith(rank_marker):
            return None

    layer_id = int(os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER", "0"))
    layers = model.model.layers
    if not 0 <= layer_id < len(layers):
        raise ValueError(
            f"TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER={layer_id} is outside "
            f"the decoder layer range [0, {len(layers)})"
        )
    layer = layers[layer_id]
    prefix = f"dsv4_block.layer{layer_id}"
    parameter_rank = os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_PARAMETER_RANK")
    capture_parameters = (
        os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_PARAMETERS", "0") == "1"
        and (
            parameter_rank is None
            or os.path.basename(tensor_dumper.get_dump_dir()).startswith(
                f"TP{int(parameter_rank)}_"
            )
        )
    )
    capture_linear_internals = (
        os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_LINEAR_INTERNALS", "0") == "1"
    )
    mlp_max_passes = int(os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_MLP_PASSES", "0"))
    state = {
        "hc_pre": 0,
        "hc_post": 0,
        "parameter_modules": set(),
        "shared_gate_up_active": 0,
        "linear_internals_captured": False,
        "w8a8_modules_captured": set(),
        "attention_backend_captured": False,
        "mlp_active": 0,
        "mlp_passes": 0,
    }

    def add_tensor(name, value):
        if isinstance(value, torch.Tensor):
            tensor_dumper.add_tensor(f"{prefix}.{name}", value)

    def capture_module_io(name):
        def hook(_module, inputs, kwargs, output):
            if (
                name == "self_attn.wqkv_a"
                and os.getenv("TENSOR_DUMP_DSV4_SYNCHRONIZE_WQKV_A", "0") == "1"
            ):
                torch.cuda.synchronize()
            for index, item in enumerate(inputs):
                add_tensor(f"{name}.input.{index}", item)
            for key, item in kwargs.items():
                add_tensor(f"{name}.input.{key}", item)
            value = output[0] if isinstance(output, (tuple, list)) else output
            add_tensor(f"{name}.output", value)
            if (
                capture_parameters
                and name == "mlp.shared_experts.gate_up_proj"
                and name not in state["parameter_modules"]
            ):
                state["parameter_modules"].add(name)
                for parameter_name, parameter in _module.named_parameters(
                    recurse=False
                ):
                    add_tensor(f"{name}.param.{parameter_name}", parameter)
                for buffer_name, buffer in _module.named_buffers(recurse=False):
                    add_tensor(f"{name}.buffer.{buffer_name}", buffer)
                for attribute in (
                    "weight",
                    "weight_scale",
                    "weight_scale_inv",
                    "input_scale",
                    "input_scale_inv",
                    "bias",
                ):
                    add_tensor(f"{name}.param.{attribute}", getattr(_module, attribute, None))

        return hook

    for name in (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    ):
        getattr(layer, name).register_forward_hook(
            capture_module_io(name), with_kwargs=True
        )
    self_attn = layer.self_attn

    def capture_attention_stage(name, value):
        add_tensor(f"self_attn.stage.{name}", value)

    self_attn._dsv4_tensor_dump_stage_callback = capture_attention_stage
    for name, attribute in (
        ("self_attn.wqkv_a", "wqkv_a"),
        ("self_attn.wq_a", "wq_a"),
        ("self_attn.q_norm", "q_norm"),
        ("self_attn.wq_b", "wq_b"),
        ("self_attn.wkv", "wkv"),
        ("self_attn.kv_norm", "kv_norm"),
        ("self_attn.wo_a", "wo_a"),
        ("self_attn.wo_b", "wo_b"),
    ):
        module = getattr(self_attn, attribute, None)
        if module is not None:
            module.register_forward_hook(
                capture_module_io(name), with_kwargs=True
            )

    w8a8_module_name = os.getenv("TENSOR_DUMP_DSV4_BLOCK_CHAIN_W8A8_MODULE")
    if w8a8_module_name is not None:
        if w8a8_module_name == "self_attn.wo_b":
            w8a8_module = self_attn.wo_b
        elif w8a8_module_name == "mlp.shared_experts.gate_up_proj":
            w8a8_module = layer.mlp.shared_experts.gate_up_proj
        else:
            raise ValueError(
                "unsupported TENSOR_DUMP_DSV4_BLOCK_CHAIN_W8A8_MODULE="
                f"{w8a8_module_name}"
            )

        def capture_w8a8_module(name, value):
            if w8a8_module_name in state["w8a8_modules_captured"]:
                return
            add_tensor(f"{w8a8_module_name}.{name}", value)
            if name == "matmul.output":
                state["w8a8_modules_captured"].add(w8a8_module_name)

        w8a8_module._dsv4_tensor_dump_callback = capture_w8a8_module
        w8a8_quant_method = getattr(w8a8_module, "quant_method", None)
        original_w8a8_apply = getattr(w8a8_quant_method, "apply", None)

        def w8a8_apply_with_dump(method_self, linear_layer, x, bias=None):
            if linear_layer is not w8a8_module:
                return original_w8a8_apply(linear_layer, x, bias=bias)
            import kunlun_ops

            original_quant2d = kunlun_ops.quant2d
            original_matmul = kunlun_ops.matmul

            def quant2d_with_dump(*args, **kwargs):
                result = original_quant2d(*args, **kwargs)
                add_tensor("self_attn.wo_b.quant2d.input.x", args[0])
                add_tensor("self_attn.wo_b.quant2d.output.x_q", args[1])
                add_tensor("self_attn.wo_b.quant2d.output.x_scale", args[2])
                return result

            def matmul_with_dump(*args, **kwargs):
                add_tensor("self_attn.wo_b.matmul.input.x_q", args[0])
                add_tensor("self_attn.wo_b.matmul.input.weight", args[1])
                result = original_matmul(*args, **kwargs)
                add_tensor("self_attn.wo_b.matmul.output", args[2])
                return result

            kunlun_ops.quant2d = quant2d_with_dump
            kunlun_ops.matmul = matmul_with_dump
            try:
                return original_w8a8_apply(linear_layer, x, bias=bias)
            finally:
                kunlun_ops.quant2d = original_quant2d
                kunlun_ops.matmul = original_matmul

        if w8a8_quant_method is not None:
            w8a8_quant_method.apply = MethodType(
                w8a8_apply_with_dump, w8a8_quant_method
            )

        original_wo_b_forward = w8a8_module.forward

        def wo_b_forward_with_dump(self, *args, **kwargs):
            import kunlun_ops

            original_quant2d = kunlun_ops.quant2d
            original_matmul = kunlun_ops.matmul
            original_f_linear = torch.nn.functional.linear

            def linear_with_dump(input_tensor, weight, bias=None):
                result = original_f_linear(input_tensor, weight, bias)
                add_tensor("self_attn.wo_b.f_linear.input.x", input_tensor)
                add_tensor("self_attn.wo_b.f_linear.input.weight", weight)
                add_tensor("self_attn.wo_b.f_linear.output", result)
                return result

            def quant2d_with_dump(*call_args, **call_kwargs):
                result = original_quant2d(*call_args, **call_kwargs)
                x = call_args[0] if call_args else call_kwargs["x"]
                x_q = call_args[1] if len(call_args) > 1 else call_kwargs["x_q"]
                x_scale = call_args[2] if len(call_args) > 2 else call_kwargs["x_scale"]
                add_tensor("self_attn.wo_b.quant2d.input.x", x)
                add_tensor("self_attn.wo_b.quant2d.output.x_q", x_q)
                add_tensor("self_attn.wo_b.quant2d.output.x_scale", x_scale)
                return result

            def matmul_with_dump(*call_args, **call_kwargs):
                x_q = call_args[0] if call_args else call_kwargs["x_q"]
                weight = call_args[1] if len(call_args) > 1 else call_kwargs["weight"]
                result = original_matmul(*call_args, **call_kwargs)
                output = call_args[2] if len(call_args) > 2 else call_kwargs["out"]
                add_tensor("self_attn.wo_b.matmul.input.x_q", x_q)
                add_tensor("self_attn.wo_b.matmul.input.weight", weight)
                add_tensor("self_attn.wo_b.matmul.output", output)
                return result

            kunlun_ops.quant2d = quant2d_with_dump
            kunlun_ops.matmul = matmul_with_dump
            torch.nn.functional.linear = linear_with_dump
            try:
                return original_wo_b_forward(*args, **kwargs)
            finally:
                kunlun_ops.quant2d = original_quant2d
                kunlun_ops.matmul = original_matmul
                torch.nn.functional.linear = original_f_linear

        w8a8_module.forward = MethodType(
            wo_b_forward_with_dump, w8a8_module
        )

    original_self_attn_forward = self_attn.forward

    def self_attn_forward_with_dump(self, *args, **kwargs):
        forward_batch = kwargs.get("forward_batch")
        if forward_batch is None and len(args) > 2:
            forward_batch = args[2]
        attn_backend = getattr(forward_batch, "attn_backend", None)
        if attn_backend is None:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            try:
                attn_backend = get_attn_backend()
            except AssertionError:
                attn_backend = None
        original_operator_callback = getattr(
            attn_backend,
            "_dsv4_tensor_dump_compressed_attention_callback",
            None,
        )

        def capture_compressed_attention_operator(name, value):
            add_tensor(f"self_attn.compressed_attention.{name}", value)

        if attn_backend is not None:
            attn_backend._dsv4_tensor_dump_compressed_attention_callback = (
                capture_compressed_attention_operator
            )
        original_backend_forward = getattr(attn_backend, "forward", None)
        capture_backend = (
            not state["attention_backend_captured"]
            and callable(original_backend_forward)
        )
        if capture_backend:

            def backend_forward_with_dump(*backend_args, **backend_kwargs):
                for index, name in enumerate(("q", "k", "v")):
                    if index < len(backend_args):
                        add_tensor(
                            f"self_attn.backend.input.{name}",
                            backend_args[index],
                        )
                    elif name in backend_kwargs:
                        add_tensor(
                            f"self_attn.backend.input.{name}",
                            backend_kwargs[name],
                        )
                result = original_backend_forward(
                    *backend_args, **backend_kwargs
                )
                value = result[0] if isinstance(result, (tuple, list)) else result
                add_tensor("self_attn.backend.output", value)
                state["attention_backend_captured"] = True
                return result

            attn_backend.forward = backend_forward_with_dump
        try:
            return original_self_attn_forward(*args, **kwargs)
        finally:
            if capture_backend:
                attn_backend.forward = original_backend_forward
            if attn_backend is not None:
                if original_operator_callback is None:
                    delattr(
                        attn_backend,
                        "_dsv4_tensor_dump_compressed_attention_callback",
                    )
                else:
                    attn_backend._dsv4_tensor_dump_compressed_attention_callback = (
                        original_operator_callback
                    )

    self_attn.forward = MethodType(self_attn_forward_with_dump, self_attn)

    for name, attribute in (
        ("mlp.gate_up_proj", "gate_up_proj"),
        ("mlp.act_fn", "act_fn"),
        ("mlp.down_proj", "down_proj"),
    ):
        module = getattr(layer.mlp, attribute, None)
        if module is not None:
            module.register_forward_hook(capture_module_io(name), with_kwargs=True)

    def capture_moe_module_io(name):
        def hook(module, inputs, kwargs, output):
            for index, item in enumerate(inputs):
                add_tensor(f"{name}.input.{index}", item)
            for key, item in kwargs.items():
                add_tensor(f"{name}.input.{key}", item)
            if isinstance(output, torch.Tensor):
                add_tensor(f"{name}.output", output)
            elif hasattr(output, "_fields"):
                for field in output._fields:
                    add_tensor(f"{name}.output.{field}", getattr(output, field))
            elif isinstance(output, (tuple, list)):
                for index, item in enumerate(output):
                    add_tensor(f"{name}.output.{index}", item)
            if name == "mlp.gate":
                for key in ("weight", "e_score_correction_bias"):
                    add_tensor(f"{name}.param.{key}", getattr(module, key, None))

        return hook

    for name, attribute in (
        ("mlp.gate", "gate"),
        ("mlp.topk", "topk"),
        ("mlp.experts", "experts"),
        ("mlp.shared_experts", "shared_experts"),
    ):
        module = getattr(layer.mlp, attribute, None)
        if module is not None:
            module.register_forward_hook(
                capture_moe_module_io(name), with_kwargs=True
            )

    shared_experts = getattr(layer.mlp, "shared_experts", None)
    if shared_experts is not None:
        shared_gate_up = getattr(shared_experts, "gate_up_proj", None)
        if capture_linear_internals and shared_gate_up is not None:

            def capture_shared_gate_up_linear(name, value):
                if state["linear_internals_captured"]:
                    return
                add_tensor(f"mlp.shared_experts.gate_up_proj.{name}", value)
                if name == "matmul.output":
                    state["linear_internals_captured"] = True

            shared_gate_up._dsv4_tensor_dump_callback = (
                capture_shared_gate_up_linear
            )

        for name, attribute in (
            ("mlp.shared_experts.gate_up_proj", "gate_up_proj"),
            ("mlp.shared_experts.act_fn", "act_fn"),
            ("mlp.shared_experts.down_proj", "down_proj"),
        ):
            module = getattr(shared_experts, attribute, None)
            if module is not None:
                module.register_forward_hook(
                    capture_module_io(name), with_kwargs=True
                )

    experts = getattr(layer.mlp, "experts", None)
    if experts is not None:

        def capture_moe_kernel(name, value):
            if state["mlp_active"]:
                add_tensor(f"mlp.experts.w8a8.{name}", value)

        experts._dsv4_moe_tensor_dump_callback = capture_moe_kernel
        for descendant in experts.modules():
            descendant._dsv4_moe_tensor_dump_callback = capture_moe_kernel

    original_mlp_forward = layer.mlp.forward

    dsv2_module = sys.modules.get("sglang.srt.models.deepseek_v2")
    original_tp_all_reduce = getattr(
        dsv2_module, "tensor_model_parallel_all_reduce", None
    )

    def tp_all_reduce_with_dump(value, *args, **kwargs):
        if state["mlp_active"]:
            add_tensor("mlp.collective.pre_all_reduce", value)
        result = original_tp_all_reduce(value, *args, **kwargs)
        if state["mlp_active"]:
            add_tensor("mlp.collective.post_all_reduce", result)
        return result

    def mlp_forward_with_dump(self, *args, **kwargs):
        capture = mlp_max_passes <= 0 or state["mlp_passes"] < mlp_max_passes
        state["mlp_passes"] += 1
        if not capture:
            return original_mlp_forward(*args, **kwargs)
        state["mlp_active"] += 1
        if original_tp_all_reduce is not None:
            previous_tp_all_reduce = dsv2_module.tensor_model_parallel_all_reduce
            dsv2_module.tensor_model_parallel_all_reduce = tp_all_reduce_with_dump
        try:
            return original_mlp_forward(*args, **kwargs)
        finally:
            if original_tp_all_reduce is not None:
                dsv2_module.tensor_model_parallel_all_reduce = previous_tp_all_reduce
            state["mlp_active"] -= 1

    layer.mlp.forward = MethodType(mlp_forward_with_dump, layer.mlp)

    def capture_layer_io(_module, inputs, kwargs, output):
        for index, item in enumerate(inputs):
            add_tensor(f"input.{index}", item)
        for key, item in kwargs.items():
            add_tensor(f"input.{key}", item)
        value = output[0] if isinstance(output, (tuple, list)) else output
        add_tensor("output", value)

    layer.register_forward_hook(capture_layer_io, with_kwargs=True)

    original_hc_pre = layer.hc_pre

    def hc_pre_with_dump(self, *args, **kwargs):
        call_name = "attn" if state["hc_pre"] == 0 else "ffn"
        state["hc_pre"] += 1
        for index, name in enumerate(("x", "fn", "scale", "base")):
            if index < len(args):
                add_tensor(f"hc_pre.{call_name}.input.{name}", args[index])
        norm = kwargs.get("norm")
        if norm is not None:
            add_tensor(f"hc_pre.{call_name}.input.norm_weight", norm.weight)
        result = original_hc_pre(*args, **kwargs)
        for value, name in zip(result[:3], ("y", "post", "comb")):
            add_tensor(f"hc_pre.{call_name}.output.{name}", value)
        try:
            norm_fused = result[3]
        except (IndexError, TypeError):
            norm_fused = False
        add_tensor(
            f"hc_pre.{call_name}.output.norm_fused",
            torch.as_tensor(norm_fused, dtype=torch.uint8),
        )
        return result

    layer.hc_pre = MethodType(hc_pre_with_dump, layer)
    original_hc_post = layer.hc_post

    def hc_post_with_dump(self, *args, **kwargs):
        call_name = "attn" if state["hc_post"] == 0 else "ffn"
        state["hc_post"] += 1
        for index, name in enumerate(("x", "residual", "post", "comb")):
            if index < len(args):
                add_tensor(f"hc_post.{call_name}.input.{name}", args[index])
        result = original_hc_post(*args, **kwargs)
        add_tensor(f"hc_post.{call_name}.output", result)
        return result

    layer.hc_post = MethodType(hc_post_with_dump, layer)

    def reset():
        state["hc_pre"] = 0
        state["hc_post"] = 0
        state["shared_gate_up_active"] = 0
        state["mlp_active"] = 0

    return reset


def _combine_resets(*resets):
    active = tuple(reset for reset in resets if reset is not None)
    if not active:
        return None

    def reset_all():
        for reset in active:
            reset()

    return reset_all


def _wrap_root_forward(tensor_dumper, model, reset_dsv4_probes=None):
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    original_forward = model.forward

    def forward_with_tensor_dump(self, *args, **kwargs):
        if os.getenv("TENSOR_DUMP_SYNCHRONIZE_AT_FORWARD_START", "0") == "1":
            torch.cuda.synchronize()
            logger.info("Tensor dump forward-start device synchronization complete")
        if reset_dsv4_probes is not None:
            reset_dsv4_probes()
        for index, item in enumerate(args):
            if isinstance(item, (torch.Tensor, ForwardBatch)):
                tensor_dumper.add_tensor(f"__root__.input.{index}", item)
        for name, item in kwargs.items():
            if isinstance(item, (torch.Tensor, ForwardBatch)):
                tensor_dumper.add_tensor(f"__root__.input.{name}", item)
        output = original_forward(*args, **kwargs)
        tensor_dumper.add_tensor("__root__", output)
        tensor_dumper.dump_current_tensors()
        return output

    model.forward = MethodType(forward_with_tensor_dump, model)
    model._tensor_dump_root_forward_wrapped = True


@plugin_hook(
    "sglang.srt.debug_utils.tensor_dump_forward_hook."
    "register_forward_hook_for_model",
    type=HookType.AROUND,
)
def register_layer_boundary_tensor_dump_kunlun(
    original_fn, model, dump_dir, dump_layers, tp_size, tp_rank, pp_rank
):
    """Register boundary-only hooks without modifying upstream SGLang."""
    if os.getenv("TENSOR_DUMP_LAYER_BOUNDARIES_ONLY", "0") != "1":
        tensor_dumper = original_fn(
            model, dump_dir, dump_layers, tp_size, tp_rank, pp_rank
        )
        _register_input_capture_hooks(
            tensor_dumper,
            model,
            os.getenv("TENSOR_DUMP_INPUT_MODULE_SUFFIX", ""),
        )
        _register_parent_io_hook(
            tensor_dumper,
            model,
            os.getenv("TENSOR_DUMP_PARENT_MODULE_SUFFIX", ""),
        )
        reset_dsv4_probes = _combine_resets(
            _register_dsv4_attention_chain(tensor_dumper),
            _register_dsv4_selected_attention_chain(tensor_dumper, model),
            _register_dsv4_layer0_block_chain(tensor_dumper, model),
        )
        if (
            os.getenv("TENSOR_DUMP_ROOT_FLUSH", "0") == "1"
            and not getattr(model, "_tensor_dump_root_forward_wrapped", False)
        ):
            _wrap_root_forward(tensor_dumper, model, reset_dsv4_probes)
        return tensor_dumper

    from sglang.srt.debug_utils.tensor_dump_forward_hook import TensorDumper

    tensor_dumper = TensorDumper(
        dump_dir, dump_layers, tp_size, tp_rank, pp_rank
    )
    top_level_module_name = os.getenv("TENSOR_DUMP_TOP_LEVEL_MODULE_NAME", "model")
    layers_module_name = os.getenv("TENSOR_DUMP_LAYERS_MODULE_NAME", "layers")
    _register_layer_boundary_hooks(
        tensor_dumper,
        model,
        dump_layers,
        top_level_module_name,
        layers_module_name,
    )
    reset_dsv4_probes = _combine_resets(
        _register_dsv4_attention_chain(tensor_dumper),
        _register_dsv4_selected_attention_chain(tensor_dumper, model),
        _register_dsv4_layer0_block_chain(tensor_dumper, model),
    )
    _wrap_root_forward(tensor_dumper, model, reset_dsv4_probes)
    return tensor_dumper


def _decode_layer_alias_enabled():
    return os.getenv("DSV4_DECODE_LAYER_ALIAS_DUMP", "0") == "1"


def prepare_decode_layer_aliases_kunlun(result, self, size, stream_idx=None):
    if not _decode_layer_alias_enabled():
        return result
    batch_size = int(os.getenv("DSV4_DECODE_LAYER_ALIAS_BATCH_SIZE", "1"))
    if size == batch_size:
        import torch

        forward_batch = result[0]
        config = self.model_runner.model_config
        hc_mult = getattr(config.hf_text_config, "hc_mult", 1)
        num_tokens = size * self.num_tokens_per_bs
        layer_shape = (
            config.num_hidden_layers,
            num_tokens,
            hc_mult,
            config.hidden_size,
        )
        logits_storage = self.buffers.next_token_logits_buffer
        if (
            self.pp_size != 1
            or not self.capture_forward_mode.is_decode()
            or forward_batch.spec_info is not None
            or num_tokens != size
            or self.model_runner.dtype != torch.float16
            or logits_storage.dtype != torch.float32
            or not logits_storage.is_contiguous()
        ):
            raise RuntimeError(
                "DSV4 decode layer dump requires PP1 ordinary FP16 decode "
                "with contiguous FP32 logits storage"
            )
        active_fp32 = num_tokens * logits_storage.shape[1]
        needed_fp16 = (
            config.num_hidden_layers
            * num_tokens
            * hc_mult
            * config.hidden_size
        )
        needed_fp32 = (needed_fp16 + 1) // 2
        if active_fp32 + needed_fp32 > logits_storage.numel():
            raise RuntimeError(
                "DSV4 decode layer dump does not fit the inactive logits tail"
            )
        forward_batch._dsv4_decode_layer_buffer = (
            logits_storage.reshape(-1)
            .narrow(0, active_fp32, needed_fp32)
            .view(torch.float16)[:needed_fp16]
            .reshape(layer_shape)
        )
        self._dsv4_decode_layer_capture_batch = forward_batch
    else:
        self._dsv4_decode_layer_capture_batch = None
    return result


def capture_decode_layer_alias_kunlun(
    result,
    self,
    positions,
    hidden_states,
    input_ids,
    forward_batch,
    input_ids_global,
    *args,
    **kwargs,
):
    if not hasattr(forward_batch, "_dsv4_decode_layer_buffer"):
        return result
    if getattr(self, "use_fused_mhc_post_pre", False):
        raise RuntimeError(
            "DSV4 decode layer dump requires non-fused MHC layer boundaries"
        )
    exact_num_tokens = int(
        os.getenv(
            "DSV4_DECODE_LAYER_ALIAS_EXACT_NUM_TOKENS",
            os.getenv("DSV4_DECODE_LAYER_ALIAS_BATCH_SIZE", "1"),
        )
    )
    output_hidden_states = result[0]
    if output_hidden_states.shape[0] == exact_num_tokens:
        forward_batch._dsv4_decode_layer_buffer[
            self.layer_id, : output_hidden_states.shape[0]
        ].copy_(output_hidden_states)
    return result


def retain_decode_layer_aliases_kunlun(
    result,
    self,
    size,
    forward,
    stream_idx=None,
    variant_label=None,
):
    batch_size = int(os.getenv("DSV4_DECODE_LAYER_ALIAS_BATCH_SIZE", "1"))
    capture_batch = getattr(self, "_dsv4_decode_layer_capture_batch", None)
    if size != batch_size or capture_batch is None or not hasattr(
        capture_batch, "_dsv4_decode_layer_buffer"
    ):
        return result
    if not hasattr(self, "_dsv4_decode_layer_buffers_by_graph"):
        self._dsv4_decode_layer_buffers_by_graph = {}
    graph_key = self._make_graph_key(size, stream_idx, variant_label)
    self._dsv4_decode_layer_buffers_by_graph[graph_key] = (
        capture_batch._dsv4_decode_layer_buffer
    )
    self._dsv4_decode_layer_capture_batch = None
    return result


def dump_decode_layer_aliases_kunlun(
    result, self, forward_batch, pp_proxy_tensors=None
):
    if not _decode_layer_alias_enabled() or not forward_batch.forward_mode.is_decode():
        return result
    dump_dir = os.getenv("DSV4_DECODE_LAYER_ALIAS_DUMP_DIR")
    seq_len = int(os.getenv("DSV4_DECODE_LAYER_ALIAS_SEQ_LEN", "16715"))
    batch_size = int(os.getenv("DSV4_DECODE_LAYER_ALIAS_BATCH_SIZE", "1"))
    if (
        not dump_dir
        or self.raw_bs != batch_size
        or getattr(self, "_dsv4_decode_layer_alias_dumped", False)
        or forward_batch.seq_lens_cpu is None
        or any(
            int(value) != seq_len
            for value in forward_batch.seq_lens_cpu[:batch_size]
        )
    ):
        return result

    layer_buffer = getattr(self, "_dsv4_decode_layer_buffers_by_graph", {}).get(
        self._replay_graph_key
    )
    if layer_buffer is None:
        raise RuntimeError(
            f"Missing DSV4 layer buffer for graph {self._replay_graph_key!r}"
        )

    import torch

    from sglang.srt.distributed import get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        {
            "input_ids": forward_batch.input_ids.detach().cpu(),
            "positions": forward_batch.positions.detach().cpu(),
            "seq_lens": forward_batch.seq_lens.detach().cpu(),
            "layers": {
                layer_id: layer_buffer[
                    layer_id, : self.raw_num_token
                ].detach().cpu()
                for layer_id in range(layer_buffer.shape[0])
            },
        },
        os.path.join(
            dump_dir,
            f"decode_alias_0514_rank{rank}_seq{seq_len}.pt",
        ),
    )
    self._dsv4_decode_layer_alias_dumped = True
    return result


# The four decode-alias hooks are registered only when
# DSV4_DECODE_LAYER_ALIAS_DUMP=1. capture_decode_layer_alias_kunlun is an AFTER
# hook on DeepseekV4DecoderLayer.forward, so registering it unconditionally
# would put a wrapper frame on every layer of every decode step just to reach a
# disabled body.
#
# register_layer_boundary_tensor_dump_kunlun above keeps its decorator: it wraps
# a setup function that only runs when --debug-tensor-dump-output-folder is
# given, its enable condition is a server argument that is not known at
# registration time, and it is not on any per-token path.
_DECODE_ALIAS_HOOKS = (
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.capture_prepare",
        prepare_decode_layer_aliases_kunlun,
    ),
    (
        "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.forward",
        capture_decode_layer_alias_kunlun,
    ),
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.capture_one_shape",
        retain_decode_layer_aliases_kunlun,
    ),
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.execute",
        dump_decode_layer_aliases_kunlun,
    ),
)


def install_decode_layer_alias_hooks() -> int:
    if not _decode_layer_alias_enabled():
        return 0
    # Imported lazily: the unit tests stub sglang.srt.plugins.hook_registry with
    # a module that only provides HookType and plugin_hook, so importing
    # HookRegistry at module scope would break collection.
    from sglang.srt.plugins.hook_registry import HookRegistry

    for target, hook in _DECODE_ALIAS_HOOKS:
        HookRegistry.register(target, hook, HookType.AFTER)
    logger.warning(
        "DSV4 decode-alias hooks installed on %d targets because "
        "DSV4_DECODE_LAYER_ALIAS_DUMP=1; this adds a wrapper frame to every "
        "decoder forward",
        len(_DECODE_ALIAS_HOOKS),
    )
    return len(_DECODE_ALIAS_HOOKS)


install_decode_layer_alias_hooks()
