import ast
from collections import namedtuple
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "sglang" / "python" / "sglang" / "srt"


def load_with_fake_registry(relative_path, module_name):
    registered = {}
    registry = types.ModuleType("sglang.srt.plugins.hook_registry")

    class HookType:
        REPLACE = "replace"
        AROUND = "around"
        AFTER = "after"

    def plugin_hook(target, type=None):
        def decorate(fn):
            registered[target] = (type, fn)
            return fn

        return decorate

    class HookRegistry:
        """Records conditional registrations the same way plugin_hook does."""

        @staticmethod
        def register(target, fn, type=None):
            registered[target] = (type, fn)

    registry.HookType = HookType
    registry.HookRegistry = HookRegistry
    registry.plugin_hook = plugin_hook
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    old = sys.modules.get("sglang.srt.plugins.hook_registry")
    sys.modules["sglang.srt.plugins.hook_registry"] = registry
    try:
        spec.loader.exec_module(module)
    finally:
        if old is None:
            sys.modules.pop("sglang.srt.plugins.hook_registry", None)
        else:
            sys.modules["sglang.srt.plugins.hook_registry"] = old
    return module, registered


class DSV4MHCSpeculativeContractTest(unittest.TestCase):
    def test_w8a8_moe_restores_058_fp16_clamp_contract(self):
        module, _registered = load_with_fake_registry(
            "sglang_kunlun/hooks/layers/quantization/w8a8_int8.py",
            "contract_w8a8_int8",
        )
        values = torch.tensor([-12.0, -7.0, 0.0, 7.0, 12.0], dtype=torch.float16)
        with mock.patch.dict(os.environ, {"SGLANG_FP16_LIMIT_IN_MOE": "7"}):
            actual = module._clamp_fp16_moe_output(values)

        expected = torch.tensor([-7.0, -7.0, 0.0, 7.0, 7.0], dtype=torch.float16)
        self.assertTrue(torch.equal(actual, expected))

        bf16_values = values.to(torch.bfloat16)
        self.assertIs(module._clamp_fp16_moe_output(bf16_values), bf16_values)

    def test_base_model_hc_head_matches_058_fp32_sequence(self):
        from sglang_kunlun.models.deepseek_v4 import hc_head_kunlun

        model = types.SimpleNamespace(norm_eps=1e-6, hc_eps=1e-4)
        x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
        x = (x / 31).to(torch.bfloat16)
        hc_fn = torch.linspace(-0.25, 0.25, 4 * 32).reshape(4, 32)
        hc_scale = torch.linspace(0.5, 1.0, 4)
        hc_base = torch.linspace(-0.1, 0.1, 4)

        flattened = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            flattened.square().mean(-1, keepdim=True) + model.norm_eps
        )
        mixes = torch.nn.functional.linear(flattened, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + model.hc_eps
        expected = torch.sum(
            pre.unsqueeze(-1) * flattened.view(x.shape), dim=1
        ).to(x.dtype)

        actual = hc_head_kunlun(model, x, hc_fn, hc_scale, hc_base)
        self.assertTrue(torch.equal(actual, expected))

    def test_nextn_hc_head_matches_058_fp32_sequence(self):
        from sglang_kunlun.models.deepseek_v4_nextn import hc_head_kunlun

        model = types.SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-4)
        x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
        x = (x / 29).to(torch.bfloat16)
        hc_fn = torch.linspace(-0.3, 0.2, 4 * 32).reshape(4, 32)
        hc_scale = torch.linspace(0.4, 1.1, 4)
        hc_base = torch.linspace(-0.2, 0.15, 4)

        flattened = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            flattened.square().mean(-1, keepdim=True) + model.rms_norm_eps
        )
        mixes = torch.nn.functional.linear(flattened, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + model.hc_eps
        expected = torch.sum(
            pre.unsqueeze(-1) * flattened.view(x.shape), dim=1
        ).to(x.dtype)

        actual = hc_head_kunlun(model, x, hc_fn, hc_scale, hc_base)
        self.assertTrue(torch.equal(actual, expected))
        empty = hc_head_kunlun(
            model, x[:0], hc_fn, hc_scale, hc_base
        )
        self.assertEqual(empty.shape, (0, x.shape[-1]))

    def test_mqa_init_matches_requested_wo_a_parameter_dtype(self):
        fake_kernel_ops = types.ModuleType("sglang_kunlun.kernels.kernel_ops")
        fake_kernel_ops.dsv4_mqa_wo_a_einsum_kunlun = mock.Mock()
        fake_server_args = types.ModuleType("sglang.srt.server_args")
        requested_dtype = {"value": "float16"}
        fake_server_args.get_global_server_args = lambda: types.SimpleNamespace(
            dtype=requested_dtype["value"]
        )
        fake_unquant = types.ModuleType(
            "sglang.srt.layers.quantization.unquant"
        )

        class UnquantizedLinearMethod:
            pass

        fake_unquant.UnquantizedLinearMethod = UnquantizedLinearMethod
        fake_rope = types.ModuleType("sglang.srt.layers.deepseek_v4_rope")
        fake_rope.precompute_freqs_cis = mock.Mock()
        fake_upstream_model = types.ModuleType("sglang.srt.models.deepseek_v4")
        fake_upstream_model.get_rope_config = lambda _config: (10000, None)
        with mock.patch.dict(
            sys.modules,
            {
                fake_kernel_ops.__name__: fake_kernel_ops,
                fake_server_args.__name__: fake_server_args,
                fake_unquant.__name__: fake_unquant,
                fake_rope.__name__: fake_rope,
                fake_upstream_model.__name__: fake_upstream_model,
            },
        ):
            module, registered = load_with_fake_registry(
                "sglang_kunlun/models/deepseek_v4_precision.py",
                "contract_deepseek_v4_precision",
            )

            target = "sglang.srt.models.deepseek_v4.MQALayer.__init__"
            self.assertEqual(registered[target][0], "after")
            config = types.SimpleNamespace(rope_scaling=None)
            for dtype_name, expected_dtype in (
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
            ):
                with self.subTest(dtype=dtype_name):
                    requested_dtype["value"] = dtype_name
                    weight = torch.nn.Parameter(
                        torch.ones((1, 2, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                    loader_marker = object()
                    weight.weight_loader = loader_marker
                    weight.input_dim = 2
                    weight.output_dim = 1
                    owner = types.SimpleNamespace(
                        compress_ratio=0,
                        wo_a=types.SimpleNamespace(
                            weight=weight,
                            quant_method=UnquantizedLinearMethod(),
                            params_dtype=torch.bfloat16,
                        ),
                    )
                    result = module.initialize_mqa_rope_policy_kunlun(
                        None, owner, config
                    )
                    self.assertIsNone(result)
                    self.assertIs(owner.wo_a.weight, weight)
                    self.assertIs(weight.weight_loader, loader_marker)
                    self.assertEqual(weight.input_dim, 2)
                    self.assertEqual(weight.output_dim, 1)
                    self.assertEqual(owner.wo_a.params_dtype, expected_dtype)
                    self.assertEqual(weight.dtype, expected_dtype)
                    weight.data.copy_(
                        torch.full(weight.shape, 2, dtype=torch.bfloat16)
                    )
                    self.assertEqual(weight.dtype, expected_dtype)
                    self.assertTrue(torch.equal(weight, torch.full_like(weight, 2)))

            requested_dtype["value"] = "float16"
            fp8_weight = torch.nn.Parameter(
                torch.ones((1, 2, 4), dtype=torch.bfloat16),
                requires_grad=False,
            )
            fp8_wo_a = types.SimpleNamespace(
                weight=fp8_weight,
                quant_method=object(),
                params_dtype=torch.bfloat16,
                weight_scale_inv=torch.ones(1),
            )
            fp8_owner = types.SimpleNamespace(compress_ratio=0, wo_a=fp8_wo_a)
            module.initialize_mqa_rope_policy_kunlun(None, fp8_owner, config)
            self.assertEqual(fp8_weight.dtype, torch.bfloat16)

    def test_c4_indexer_and_compressor_match_requested_parameter_dtypes(self):
        fake_kernel_ops = types.ModuleType("sglang_kunlun.kernels.kernel_ops")
        fake_kernel_ops.dsv4_mqa_wo_a_einsum_kunlun = mock.Mock()
        fake_server_args = types.ModuleType("sglang.srt.server_args")
        requested_dtype = {"value": "float16"}
        fake_server_args.get_global_server_args = lambda: types.SimpleNamespace(
            dtype=requested_dtype["value"]
        )
        fake_unquant = types.ModuleType(
            "sglang.srt.layers.quantization.unquant"
        )

        class UnquantizedLinearMethod:
            pass

        fake_unquant.UnquantizedLinearMethod = UnquantizedLinearMethod
        with mock.patch.dict(
            sys.modules,
            {
                fake_kernel_ops.__name__: fake_kernel_ops,
                fake_server_args.__name__: fake_server_args,
                fake_unquant.__name__: fake_unquant,
            },
        ):
            module, registered = load_with_fake_registry(
                "sglang_kunlun/models/deepseek_v4_precision.py",
                "contract_deepseek_v4_c4_parameter_dtype",
            )

            indexer_target = (
                "sglang.srt.layers.attention.dsv4.indexer.C4Indexer.__init__"
            )
            compressor_target = (
                "sglang.srt.layers.attention.dsv4.compressor.Compressor.__init__"
            )
            self.assertEqual(registered[indexer_target][0], "after")
            self.assertEqual(registered[compressor_target][0], "after")

            for dtype_name, expected_dtype in (
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
            ):
                with self.subTest(dtype=dtype_name):
                    requested_dtype["value"] = dtype_name
                    wq_b_weight = torch.nn.Parameter(
                        torch.ones((4, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                    weights_proj_weight = torch.nn.Parameter(
                        torch.ones((4, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                    wkv_gate_weight = torch.nn.Parameter(
                        torch.ones((4, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                    weights = (
                        wq_b_weight,
                        weights_proj_weight,
                        wkv_gate_weight,
                    )
                    metadata_markers = []
                    for index, weight in enumerate(weights):
                        marker = object()
                        weight.weight_loader = marker
                        weight.input_dim = index
                        weight.output_dim = index + 1
                        metadata_markers.append(marker)
                    indexer = types.SimpleNamespace(
                        wq_b=types.SimpleNamespace(
                            weight=wq_b_weight,
                            quant_method=UnquantizedLinearMethod(),
                            params_dtype=torch.bfloat16,
                        ),
                        weights_proj=types.SimpleNamespace(
                            weight=weights_proj_weight,
                            quant_method=UnquantizedLinearMethod(),
                            params_dtype=torch.bfloat16,
                        ),
                    )
                    compressor = types.SimpleNamespace(
                        wkv_gate=types.SimpleNamespace(
                            weight=wkv_gate_weight,
                            quant_method=UnquantizedLinearMethod(),
                            params_dtype=torch.bfloat16,
                        )
                    )

                    indexer_result = module.initialize_c4_indexer_parameter_dtype_kunlun(
                        None, indexer
                    )
                    compressor_result = (
                        module.initialize_compressor_parameter_dtype_kunlun(
                            None, compressor
                        )
                    )

                    self.assertIsNone(indexer_result)
                    self.assertIsNone(compressor_result)
                    self.assertIs(indexer.wq_b.weight, wq_b_weight)
                    self.assertIs(indexer.weights_proj.weight, weights_proj_weight)
                    self.assertIs(compressor.wkv_gate.weight, wkv_gate_weight)
                    self.assertEqual(indexer.wq_b.params_dtype, expected_dtype)
                    self.assertEqual(indexer.weights_proj.params_dtype, expected_dtype)
                    self.assertEqual(compressor.wkv_gate.params_dtype, expected_dtype)
                    self.assertEqual(wq_b_weight.dtype, expected_dtype)
                    self.assertEqual(weights_proj_weight.dtype, expected_dtype)
                    self.assertEqual(wkv_gate_weight.dtype, expected_dtype)
                    for index, (weight, marker) in enumerate(
                        zip(weights, metadata_markers)
                    ):
                        self.assertIs(weight.weight_loader, marker)
                        self.assertEqual(weight.input_dim, index)
                        self.assertEqual(weight.output_dim, index + 1)
                        weight.data.copy_(
                            torch.full(weight.shape, 2, dtype=torch.bfloat16)
                        )
                        self.assertEqual(weight.dtype, expected_dtype)
                        self.assertTrue(
                            torch.equal(weight, torch.full_like(weight, 2))
                        )

            requested_dtype["value"] = "float16"
            quantized_linears = []
            for scale_name in (None, "weight_scale", "weight_scale_inv"):
                weight = torch.nn.Parameter(
                    torch.ones((4, 4), dtype=torch.bfloat16),
                    requires_grad=False,
                )
                linear = types.SimpleNamespace(
                    weight=weight,
                    quant_method=object(),
                    params_dtype=torch.bfloat16,
                )
                if scale_name is not None:
                    setattr(linear, scale_name, torch.ones(1))
                quantized_linears.append((linear, weight))
            indexer = types.SimpleNamespace(
                wq_b=quantized_linears[0][0],
                weights_proj=quantized_linears[1][0],
            )
            compressor = types.SimpleNamespace(
                wkv_gate=quantized_linears[2][0]
            )

            module.initialize_c4_indexer_parameter_dtype_kunlun(None, indexer)
            module.initialize_compressor_parameter_dtype_kunlun(None, compressor)

            for linear, weight in quantized_linears:
                self.assertIs(linear.weight, weight)
                self.assertEqual(linear.params_dtype, torch.bfloat16)
                self.assertEqual(weight.dtype, torch.bfloat16)

    def test_mhc_sinkhorn_hook_preserves_shapes_dtype_and_empty(self):
        fake_kunlun = types.ModuleType("kunlun_ops")
        fake_kunlun.mhc_split_sinkhorn = mock.Mock()
        with mock.patch.dict(sys.modules, {"kunlun_ops": fake_kunlun}):
            module, registered = load_with_fake_registry(
                "sglang_kunlun/hooks/layers/mhc.py", "contract_mhc"
            )
        target = "sglang.srt.layers.mhc.hc_split_sinkhorn"
        self.assertIn(target, registered)
        mixes = torch.empty((0, 1, 24), dtype=torch.bfloat16)
        pre, post, comb = module.hc_split_sinkhorn_kunlun(
            mixes, torch.empty(3), torch.empty(24)
        )
        self.assertEqual(pre.shape, (0, 1, 4))
        self.assertEqual(post.shape, (0, 1, 4))
        self.assertEqual(comb.shape, (0, 1, 4, 4))
        self.assertEqual(pre.dtype, mixes.dtype)
        fake_kunlun.mhc_split_sinkhorn.assert_not_called()

    def test_selected_attention_tensor_dump_scopes_requested_c4_layer(self):
        module, _registered = load_with_fake_registry(
            "debug/tensor_dump_hooks.py",
            "contract_debug_tensor_dump_selected_attention",
        )

        class FakeIndexer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wq_b = torch.nn.Identity()
                self.weights_proj = torch.nn.Identity()
                self.compressor = torch.nn.Identity()

            def forward(self, **_kwargs):
                return None

        class FakeAttention(torch.nn.Module):
            def __init__(self, compress_ratio):
                super().__init__()
                self.compress_ratio = compress_ratio
                self.attn_sink = torch.nn.Parameter(torch.ones(4))
                self.q_norm = torch.nn.Identity()
                self.wq_b = torch.nn.Identity()
                self.wo_b = torch.nn.Identity()
                self.compressor = torch.nn.Identity()
                self.indexer = FakeIndexer() if compress_ratio == 4 else None

            def _compute_q_a(self, x):
                return x

            def _compute_q_b(self, x):
                return x

            def forward(self, x, positions):
                del positions
                return self.wo_b(self._compute_q_b(self.q_norm(x)))

        class DecoderLayer(torch.nn.Module):
            def __init__(self, compress_ratio):
                super().__init__()
                self.self_attn = FakeAttention(compress_ratio)

        class FakeTensorDumper:
            def __init__(self):
                self.tensors = {}

            def add_tensor(self, name, value):
                self.tensors[name] = value.detach().clone()

        fake_ops = types.SimpleNamespace(
            compressed_attention=lambda *args, **kwargs: None,
            einsum_tgd_grd_tgr=lambda *args, **kwargs: None,
        )
        layers = torch.nn.ModuleList(
            [DecoderLayer(0) for _ in range(12)] + [DecoderLayer(4)]
        )
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        dumper = FakeTensorDumper()
        with mock.patch.object(torch.ops, "xspeedgate_ops", fake_ops), mock.patch.dict(
            os.environ,
            {
                "TENSOR_DUMP_DSV4_SELECTED_ATTN_CHAIN": "1",
                "TENSOR_DUMP_DSV4_SELECTED_ATTN_LAYER": "12",
            },
            clear=True,
        ):
            reset = module._register_dsv4_selected_attention_chain(dumper, model)

        value = torch.ones((1, 4), dtype=torch.float16)
        positions = torch.arange(1)
        layers[12].self_attn(x=value, positions=positions)
        reset()

        self.assertIn("dsv4_selected_attn.layer12.input.x", dumper.tensors)
        self.assertIn("dsv4_selected_attn.layer12.q_norm.output", dumper.tensors)
        self.assertIn("dsv4_selected_attn.layer12._compute_q_b.output", dumper.tensors)
        self.assertIn("dsv4_selected_attn.layer12.wo_b.output", dumper.tensors)
        self.assertIn("dsv4_selected_attn.layer12.output", dumper.tensors)
        self.assertFalse(
            any(key.startswith("dsv4_selected_attn.layer0.") for key in dumper.tensors)
        )

    def test_block_chain_tensor_dump_selects_requested_layer_and_prefix(self):
        module, _registered = load_with_fake_registry(
            "debug/tensor_dump_hooks.py",
            "contract_debug_tensor_dump_block_chain",
        )

        class DecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = torch.nn.Identity()
                self.self_attn = torch.nn.Identity()
                self.post_attention_layernorm = torch.nn.Identity()
                self.mlp = torch.nn.Identity()

            def hc_pre(self, x, fn, scale, base, norm=None):
                return x, fn, scale, False

            def hc_post(self, x, residual, post, comb):
                return x

            def forward(self, hidden_states):
                return hidden_states

        class FakeTensorDumper:
            def __init__(self):
                self.tensors = {}

            def add_tensor(self, name, value):
                self.tensors[name] = value.detach().clone()

        layers = torch.nn.ModuleList([DecoderLayer() for _ in range(13)])
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        dumper = FakeTensorDumper()
        with mock.patch.dict(
            os.environ,
            {
                "TENSOR_DUMP_DSV4_LAYER0_BLOCK_CHAIN": "1",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER": "12",
            },
            clear=True,
        ):
            reset = module._register_dsv4_layer0_block_chain(dumper, model)

        self.assertEqual(len(layers[0]._forward_hooks), 0)
        self.assertEqual(len(layers[12]._forward_hooks), 1)
        self.assertEqual(len(layers[12].input_layernorm._forward_hooks), 1)

        value = torch.ones((1, 4), dtype=torch.float16)
        layers[12].input_layernorm(value)
        layers[12].hc_pre(value, value, value, value)
        reset()
        layers[12].hc_pre(value, value, value, value)
        layers[12](value)

        self.assertIn("dsv4_block.layer12.input_layernorm.output", dumper.tensors)
        self.assertIn("dsv4_block.layer12.hc_pre.attn.output.y", dumper.tensors)
        self.assertNotIn("dsv4_block.layer12.hc_pre.ffn.output.y", dumper.tensors)
        self.assertIn("dsv4_block.layer12.output", dumper.tensors)
        self.assertFalse(any(key.startswith("dsv4_block.layer0.") for key in dumper.tensors))

        with mock.patch.dict(
            os.environ,
            {
                "TENSOR_DUMP_DSV4_LAYER0_BLOCK_CHAIN": "1",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER": "13",
            },
            clear=True,
        ), self.assertRaisesRegex(
            ValueError,
            "TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER=13 is outside",
        ):
            module._register_dsv4_layer0_block_chain(dumper, model)

    def test_block_chain_tensor_dump_captures_moe_stages(self):
        module, _registered = load_with_fake_registry(
            "debug/tensor_dump_hooks.py",
            "contract_debug_tensor_dump_moe_chain",
        )
        TopKOutput = namedtuple(
            "TopKOutput", ("topk_weights", "topk_ids", "router_logits")
        )

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(4, dtype=torch.float16))
                self.e_score_correction_bias = torch.zeros(4, dtype=torch.float16)

            def forward(self, hidden_states):
                return hidden_states @ self.weight.T

        class TopK(torch.nn.Module):
            def forward(self, hidden_states, router_logits):
                return TopKOutput(
                    torch.ones((hidden_states.shape[0], 1)),
                    torch.zeros((hidden_states.shape[0], 1), dtype=torch.int64),
                    router_logits,
                )

        class Experts(torch.nn.Module):
            def forward(self, hidden_states, _topk_output):
                return hidden_states + 1

        fake_kunlun_ops = types.ModuleType("kunlun_ops")

        def quant2d(x, x_q, x_scale, force_sdnn=False):
            self.assertTrue(force_sdnn)
            x_q.copy_(x.to(torch.int8))
            x_scale.fill_(1)

        def matmul(x_q, weight, out, **_kwargs):
            out.copy_(x_q.to(out.dtype) @ weight.to(out.dtype).T)

        fake_kunlun_ops.quant2d = quant2d
        fake_kunlun_ops.matmul = matmul

        class LinearIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, 4))
                self.weight_scale = torch.nn.Parameter(torch.ones(4))

            def forward(self, hidden_states):
                callback = getattr(self, "_dsv4_tensor_dump_callback", None)
                x_q = torch.empty_like(hidden_states, dtype=torch.int8)
                x_scale = torch.empty(
                    hidden_states.shape[0], dtype=torch.float32
                )
                out = torch.empty_like(hidden_states)
                if callback is not None:
                    callback("quant2d.input.x", hidden_states)
                fake_kunlun_ops.quant2d(
                    hidden_states, x_q, x_scale, force_sdnn=True
                )
                if callback is not None:
                    callback("quant2d.output.x_q", x_q)
                    callback("quant2d.output.x_scale", x_scale)
                    callback("matmul.input.x_q", x_q)
                    callback("matmul.input.weight", self.weight)
                    callback("matmul.input.x_pc_max", x_scale)
                    callback("matmul.input.w_pc_max", self.weight_scale)
                fake_kunlun_ops.matmul(
                    x_q,
                    self.weight,
                    out,
                    x_pc_max=x_scale,
                    w_pc_max=self.weight_scale,
                )
                if callback is not None:
                    callback("matmul.output", out)
                return out

        class SelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wo_b = LinearIdentity()

            def forward(self, hidden_states):
                return self.wo_b(hidden_states)

        class SharedExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = LinearIdentity()
                self.act_fn = torch.nn.Identity()
                self.down_proj = torch.nn.Identity()

            def forward(self, hidden_states):
                hidden_states = self.gate_up_proj(hidden_states)
                hidden_states = self.act_fn(hidden_states)
                return self.down_proj(hidden_states) + 2

        class MoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()
                self.topk = TopK()
                self.experts = Experts()
                self.shared_experts = SharedExperts()

            def forward(self, hidden_states):
                return hidden_states

        class DecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = torch.nn.Identity()
                self.self_attn = SelfAttention()
                self.post_attention_layernorm = torch.nn.Identity()
                self.mlp = MoE()

            def hc_pre(self, x, fn, scale, base, norm=None):
                return x, fn, scale, False

            def hc_post(self, x, residual, post, comb):
                return x

            def forward(self, hidden_states):
                return hidden_states

        class FakeTensorDumper:
            def __init__(self):
                self.tensors = {}

            def add_tensor(self, name, value):
                self.tensors[name] = value.detach().clone()

            def get_dump_dir(self):
                return "/tmp/TP1_PP0_Rank1_pid1"

        layer = DecoderLayer()
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=torch.nn.ModuleList([layer]))
        )
        dumper = FakeTensorDumper()
        with mock.patch.dict(
            os.environ,
            {
                "TENSOR_DUMP_DSV4_LAYER0_BLOCK_CHAIN": "1",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_LAYER": "0",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_LINEAR_INTERNALS": "1",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_W8A8_MODULE": "self_attn.wo_b",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_PARAMETER_RANK": "1",
                "TENSOR_DUMP_DSV4_BLOCK_CHAIN_PARAMETERS": "1",
            },
            clear=True,
        ), mock.patch.dict(
            sys.modules, {"kunlun_ops": fake_kunlun_ops}, clear=False
        ):
            module._register_dsv4_layer0_block_chain(dumper, model)

        hidden_states = torch.ones((2, 4), dtype=torch.float16)
        layer.self_attn(hidden_states)
        router_logits = layer.mlp.gate(hidden_states)
        topk_output = layer.mlp.topk(hidden_states, router_logits)
        layer.mlp.experts(hidden_states, topk_output)
        layer.mlp.shared_experts(hidden_states)

        prefix = "dsv4_block.layer0.mlp"
        for suffix in (
            "gate.output",
            "gate.param.weight",
            "gate.param.e_score_correction_bias",
            "topk.output.topk_weights",
            "topk.output.topk_ids",
            "topk.output.router_logits",
            "experts.output",
            "shared_experts.output",
            "shared_experts.gate_up_proj.output",
            "shared_experts.gate_up_proj.param.weight",
            "shared_experts.gate_up_proj.param.weight_scale",
            "shared_experts.gate_up_proj.quant2d.input.x",
            "shared_experts.gate_up_proj.quant2d.output.x_q",
            "shared_experts.gate_up_proj.quant2d.output.x_scale",
            "shared_experts.gate_up_proj.matmul.input.x_q",
            "shared_experts.gate_up_proj.matmul.input.weight",
            "shared_experts.gate_up_proj.matmul.input.x_pc_max",
            "shared_experts.gate_up_proj.matmul.input.w_pc_max",
            "shared_experts.gate_up_proj.matmul.output",
            "shared_experts.act_fn.output",
            "shared_experts.down_proj.output",
        ):
            self.assertIn(f"{prefix}.{suffix}", dumper.tensors)

        wo_b_prefix = "dsv4_block.layer0.self_attn.wo_b"
        for suffix in (
            "quant2d.input.x",
            "quant2d.output.x_q",
            "quant2d.output.x_scale",
            "matmul.input.x_q",
            "matmul.input.weight",
            "matmul.input.x_pc_max",
            "matmul.input.w_pc_max",
            "matmul.output",
        ):
            self.assertIn(f"{wo_b_prefix}.{suffix}", dumper.tensors)

    def test_decode_layer_alias_uses_fixed_graph_buffer(self):
        # The decode-alias hooks are only registered when the dump is enabled,
        # so that no wrapper frame sits on DeepseekV4DecoderLayer.forward in a
        # normal serving run.
        with mock.patch.dict(
            os.environ, {"DSV4_DECODE_LAYER_ALIAS_DUMP": "1"}, clear=False
        ):
            module, registered = load_with_fake_registry(
                "debug/tensor_dump_hooks.py",
                "contract_debug_tensor_dump_decode_buffer",
            )
        prepare_target = (
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
            "DecodeCudaGraphRunner.capture_prepare"
        )
        layer_target = "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.forward"
        retain_target = (
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
            "DecodeCudaGraphRunner.capture_one_shape"
        )
        self.assertEqual(registered[prepare_target][0], "after")
        self.assertEqual(registered[layer_target][0], "after")
        self.assertEqual(registered[retain_target][0], "after")

        forward_batch = types.SimpleNamespace(spec_info=None)
        config = types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(hc_mult=4),
            num_hidden_layers=43,
            hidden_size=8,
        )
        logits_storage = torch.zeros((24, 32), dtype=torch.float32)
        runner = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                model_config=config,
                dtype=torch.float16,
                device=torch.device("cpu"),
            ),
            buffers=types.SimpleNamespace(
                next_token_logits_buffer=logits_storage,
            ),
            pp_size=1,
            capture_forward_mode=types.SimpleNamespace(is_decode=lambda: True),
            num_tokens_per_bs=1,
            _make_graph_key=lambda size, stream_idx, variant_label: (
                size,
                stream_idx,
                variant_label,
            ),
        )
        env = {
            "DSV4_DECODE_LAYER_ALIAS_DUMP": "1",
            "DSV4_DECODE_LAYER_ALIAS_BATCH_SIZE": "1",
            "DSV4_DECODE_LAYER_ALIAS_EXACT_NUM_TOKENS": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            prepare_result = (forward_batch, object())
            returned_prepare = module.prepare_decode_layer_aliases_kunlun(
                prepare_result, runner, 1
            )
            output = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
            layer = types.SimpleNamespace(
                layer_id=7,
                use_fused_mhc_post_pre=False,
            )
            layer_result = (output, None, None, None)
            returned_layer = module.capture_decode_layer_alias_kunlun(
                layer_result,
                layer,
                None,
                None,
                None,
                forward_batch,
                None,
            )
            capture_result = object()
            returned_capture = module.retain_decode_layer_aliases_kunlun(
                capture_result, runner, 1, None
            )

        self.assertIs(returned_prepare, prepare_result)
        self.assertEqual(forward_batch._dsv4_decode_layer_buffer.shape, (43, 1, 4, 8))
        self.assertEqual(
            forward_batch._dsv4_decode_layer_buffer.untyped_storage().data_ptr(),
            logits_storage.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(logits_storage[0], torch.zeros_like(logits_storage[0])))
        self.assertTrue(
            torch.equal(forward_batch._dsv4_decode_layer_buffer[7], output)
        )
        self.assertFalse(hasattr(forward_batch, "_dsv4_decode_layer_aliases"))
        self.assertIs(returned_layer, layer_result)
        self.assertIs(returned_capture, capture_result)
        self.assertIsNone(runner._dsv4_decode_layer_capture_batch)
        self.assertIs(
            runner._dsv4_decode_layer_buffers_by_graph[(1, None, None)],
            forward_batch._dsv4_decode_layer_buffer,
        )

    def test_boundary_tensor_dump_hook_is_gated_and_flushes_root_forward(self):
        module, registered = load_with_fake_registry(
            "debug/tensor_dump_hooks.py", "contract_debug_tensor_dump"
        )
        target = (
            "sglang.srt.debug_utils.tensor_dump_forward_hook."
            "register_forward_hook_for_model"
        )
        self.assertEqual(registered[target][0], "around")

        original = mock.Mock(return_value="upstream-dumper")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                module.register_layer_boundary_tensor_dump_kunlun(
                    original, object(), "/tmp/dump", [0], 1, 0, 0
                ),
                "upstream-dumper",
            )
        original.assert_called_once()

        class DecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

            def forward(self, hidden_states):
                return self.proj(hidden_states), None, None, None

        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([DecoderLayer(), DecoderLayer()])

            def forward(self, hidden_states):
                for layer in self.layers:
                    hidden_states = layer(hidden_states=hidden_states)[0]
                return hidden_states

        class CausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

            def forward(self, hidden_states):
                return self.model(hidden_states)

        class FakeTensorDumper:
            def __init__(self, *args):
                self.tensors = {}
                self.dump_count = 0

            def add_tensor(self, name, value):
                self.tensors[name] = value.detach().clone()

            def dump_current_tensors(self):
                self.dump_count += 1

            def _dump_hook(self, name, do_dump):
                self.assert_do_dump_false = not do_dump

                def hook(_module, _inputs, output):
                    self.add_tensor(name, output)

                return hook

        dump_module = types.ModuleType(
            "sglang.srt.debug_utils.tensor_dump_forward_hook"
        )
        dump_module.TensorDumper = FakeTensorDumper
        forward_batch_module = types.ModuleType(
            "sglang.srt.model_executor.forward_batch_info"
        )
        forward_batch_module.ForwardBatch = type("ForwardBatch", (), {})
        model = CausalLM()
        inp = torch.randn(2, 4)
        reset_attention_chain = mock.Mock()
        with mock.patch.dict(
            sys.modules,
            {
                dump_module.__name__: dump_module,
                forward_batch_module.__name__: forward_batch_module,
            },
        ), mock.patch.dict(
            "os.environ", {"TENSOR_DUMP_LAYER_BOUNDARIES_ONLY": "1"}, clear=True
        ), mock.patch.object(
            module,
            "_register_dsv4_attention_chain",
            return_value=reset_attention_chain,
        ) as register_attention_chain:
            dumper = module.register_layer_boundary_tensor_dump_kunlun(
                original, model, "/tmp/dump", [0, 1], 1, 0, 0
            )
            result = model(inp)

        register_attention_chain.assert_called_once_with(dumper)
        reset_attention_chain.assert_called_once_with()
        self.assertEqual(dumper.dump_count, 1)
        self.assertEqual(
            set(dumper.tensors),
            {"model.layers.0", "model.layers.1", "__root__.input.0", "__root__"},
        )
        self.assertTrue(torch.equal(dumper.tensors["model.layers.1"], result))
        self.assertNotIn("model.layers.0.proj", dumper.tensors)

        normal_model = CausalLM()
        normal_dumper = FakeTensorDumper()
        normal_original = mock.Mock(return_value=normal_dumper)
        with mock.patch.dict(
            sys.modules,
            {forward_batch_module.__name__: forward_batch_module},
        ), mock.patch.dict(
            "os.environ",
            {
                "TENSOR_DUMP_PARENT_MODULE_SUFFIX": "model.layers.0",
                "TENSOR_DUMP_ROOT_FLUSH": "1",
            },
            clear=True,
        ):
            returned = module.register_layer_boundary_tensor_dump_kunlun(
                normal_original, normal_model, "/tmp/dump", [0], 1, 0, 0
            )
            normal_result = normal_model(inp)

        self.assertIs(returned, normal_dumper)
        self.assertEqual(normal_dumper.dump_count, 1)
        self.assertTrue(
            torch.equal(
                normal_dumper.tensors["model.layers.0.input.hidden_states"], inp
            )
        )
        self.assertIn("model.layers.0", normal_dumper.tensors)
        self.assertTrue(torch.equal(normal_dumper.tensors["__root__"], normal_result))

    def test_dsv4_half_cache_migration_uses_plugin_hooks(self):
        hook_files = (
            ROOT
            / "sglang_kunlun/hooks/layers/attention/nsa/index_buf_accessor_v4.py",
            ROOT / "sglang_kunlun/hooks/layers/attention/nsa/quant_k_cache_v4.py",
        )
        targets = set()
        for path in hook_files:
            source = path.read_text()
            self.assertNotIn("apply_index_buf_accessor_v4_patch", source)
            self.assertNotIn("apply_quant_k_cache_v4_patch", source)
            tree = ast.parse(source)
            definitions = (
                node
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            )
            for definition in definitions:
                for decorator in definition.decorator_list:
                    if not (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "plugin_hook"
                    ):
                        continue
                    target = decorator.args[0] if decorator.args else next(
                        keyword.value
                        for keyword in decorator.keywords
                        if keyword.arg == "target"
                    )
                    hook_type = next(
                        keyword.value
                        for keyword in decorator.keywords
                        if keyword.arg == "type"
                    )
                    self.assertEqual(
                        (hook_type.value.id, hook_type.attr), ("HookType", "REPLACE")
                    )
                    targets.add(target.value)

        self.assertEqual(
            targets,
            {
                "sglang.srt.layers.attention.dsv4.index_buf_accessor.NopeFp8RopeBf16Pack",
                "sglang.srt.layers.attention.dsv4.index_buf_accessor._set_k_and_s_triton",
                "sglang.srt.layers.attention.dsv4.quant_k_cache.quant_to_nope_fp8_rope_bf16_pack_triton",
            },
        )
        nsa_init = (
            ROOT / "sglang_kunlun/hooks/layers/attention/nsa/__init__.py"
        ).read_text()
        self.assertIn("from . import index_buf_accessor_v4", nsa_init)
        self.assertIn("from . import quant_k_cache_v4", nsa_init)

    def test_model_hooks_target_upstream_methods_and_hc_pre_is_quadruple(self):
        source = (ROOT / "sglang_kunlun/models/deepseek_v4.py").read_text()
        tree = ast.parse(source)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        returns = [n for n in ast.walk(functions["hc_pre_kunlun"]) if isinstance(n, ast.Return)]
        self.assertTrue(
            all(isinstance(n.value, ast.Tuple) and len(n.value.elts) == 4 for n in returns)
        )
        hook_targets = {}
        for name in ("hc_pre_kunlun", "hc_post_kunlun"):
            decorators = functions[name].decorator_list
            self.assertEqual(len(decorators), 1)
            decorator = decorators[0]
            self.assertIsInstance(decorator, ast.Call)
            self.assertEqual(decorator.func.id, "plugin_hook")
            hook_targets[name] = decorator.args[0].value
            hook_type = next(k for k in decorator.keywords if k.arg == "type").value
            self.assertEqual((hook_type.value.id, hook_type.attr), ("HookType", "REPLACE"))
        self.assertEqual(
            hook_targets,
            {
                "hc_pre_kunlun": "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
                "hc_post_kunlun": "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
            },
        )
        self.assertNotIn("class DeepseekV4", source)

        nextn = (ROOT / "sglang_kunlun/models/deepseek_v4_nextn.py").read_text()
        self.assertIn("DeepseekV4ModelNextN.hc_head", nextn)
        self.assertIn('x.new_empty((0, x.shape[-1]))', nextn)
        self.assertNotIn("class DeepseekV4", nextn)

    def test_hook_signatures_match_0514_symbols(self):
        upstream_model = ast.parse((UPSTREAM / "models/deepseek_v4.py").read_text())
        decoder = next(
            n for n in upstream_model.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4DecoderLayer"
        )
        upstream = {
            n.name: [a.arg for a in n.args.args]
            for n in decoder.body
            if isinstance(n, ast.FunctionDef) and n.name in {"hc_pre", "hc_post"}
        }
        plugin = ast.parse((ROOT / "sglang_kunlun/models/deepseek_v4.py").read_text())
        hooks = {
            n.name.removesuffix("_kunlun"): [a.arg for a in n.args.args]
            for n in plugin.body
            if isinstance(n, ast.FunctionDef) and n.name in {"hc_pre_kunlun", "hc_post_kunlun"}
        }
        self.assertEqual(hooks, upstream)

    def test_speculative_imports_both_worker_hooks_and_upstream_supports_steps1_draft2(self):
        init = (ROOT / "sglang_kunlun/hooks/speculative/__init__.py").read_text()
        self.assertIn("from . import eagle_worker_v2", init)
        self.assertIn("from . import multi_layer_eagle_worker_v2", init)

        eagle = (UPSTREAM / "speculative/eagle_worker_v2.py").read_text()
        multi = (UPSTREAM / "speculative/multi_layer_eagle_worker_v2.py").read_text()
        expected = "self.speculative_num_draft_tokens == self.speculative_num_steps + 1"
        self.assertIn(expected, eagle)
        self.assertIn(expected, multi)
        self.assertIn("if self.speculative_num_steps == 1:", multi)

    def test_function_call_and_grammar_are_not_reimplemented(self):
        plugin_sources = "\n".join(
            p.read_text() for p in (ROOT / "sglang_kunlun").rglob("*.py")
            if p.name != "sitecustomize.py"
        )
        self.assertNotIn("generate_token_bitmask", plugin_sources)
        self.assertNotIn("function_call", plugin_sources.lower())


if __name__ == "__main__":
    unittest.main()
