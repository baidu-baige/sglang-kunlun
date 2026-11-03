"""CPU contracts for checkpoint-free channel INT4 MoE index placeholders."""

import ast
from enum import Enum
from pathlib import Path
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
CT = ROOT / "sglang_kunlun/hooks/layers/quantization/compressed_tensors"
INDEX_WIDTHS = {
    "w13_weight_g_idx": 16,
    "w2_weight_g_idx": 16,
    "w13_g_idx_sort_indices": 16,
    "w2_g_idx_sort_indices": 16,
}
CHECKPOINT_NAMES = {
    "w13_weight_packed", "w2_weight_packed",
    "w13_weight_scale", "w2_weight_scale",
    "w13_weight_shape", "w2_weight_shape",
}


class Strategy(str, Enum):
    CHANNEL = "channel"
    GROUP = "group"


def set_weight_attrs(tensor, attrs):
    for name, value in attrs.items():
        setattr(tensor, name, value)


def load_cpu_schemes():
    """Execute production classes/helpers, excluding accelerator package imports."""
    ns = {
        "torch": torch,
        "QuantizationStrategy": Strategy,
        "CompressionFormat": types.SimpleNamespace(
            pack_quantized=types.SimpleNamespace(value="pack-quantized")
        ),
        "WNA16_SUPPORTED_BITS": [4],
        "CompressedTensorsMoEScheme": object,
        "set_weight_attrs": set_weight_attrs,
        "INT4_SIGNED_SCALE_MULT": 7.0,
    }
    for path, names in (
        (CT / "utils.py", {"check_standard_routed_moe_layer", "scale_to_kernel_max"}),
        (CT / "schemes/compressed_tensors_wna16_moe.py",
         {"KunlunCompressedTensorsWNA16MoE"}),
        (CT / "schemes/compressed_tensors_w4a8_int8_moe.py",
         {"KunlunCompressedTensorsW4A8Int8MoE"}),
    ):
        tree = ast.parse(path.read_text())
        body = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(body) == len(names)
        module = ast.Module(body=body, type_ignores=[])
        exec(compile(module, str(path), "exec"), ns)
    return ns


class MoEIndexCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_cpu_schemes()

    def create(self, *, w4a8=True, strategy=Strategy.CHANNEL, actorder=None):
        weights = types.SimpleNamespace(
            num_bits=4, strategy=strategy, group_size=8,
            actorder=actorder, symmetric=True,
        )
        name = ("KunlunCompressedTensorsW4A8Int8MoE" if w4a8
                else "KunlunCompressedTensorsWNA16MoE")
        scheme = self.ns[name]({
            "weights": weights, "format": "pack-quantized",
            "input_activations": types.SimpleNamespace(num_bits=8, dynamic=True),
        })
        layer = torch.nn.Module()
        layer.moe_tp_size = 2
        loader = mock.Mock()
        with torch.device("cpu"):
            scheme.create_weights(layer, 2, 16, 16, torch.bfloat16, weight_loader=loader)
        return scheme, layer, loader

    def test_buffers_initialized_and_loader_interface_preserved(self):
        for w4a8 in (False, True):
            for strategy in (Strategy.CHANNEL, "channel"):
                with self.subTest(w4a8=w4a8, strategy=strategy):
                    _, layer, loader = self.create(w4a8=w4a8, strategy=strategy)
                    self.assertEqual(set(dict(layer.named_parameters())), CHECKPOINT_NAMES)
                    self.assertEqual(set(layer.state_dict()), CHECKPOINT_NAMES)
                    self.assertEqual(set(dict(layer.named_buffers())), set(INDEX_WIDTHS))
                    for name, width in INDEX_WIDTHS.items():
                        tensor = getattr(layer, name)
                        self.assertEqual(tensor.shape, (2, width))
                        self.assertEqual(tensor.dtype, torch.int32)
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertFalse(tensor.requires_grad)
                        self.assertEqual(torch.count_nonzero(tensor).item(), 0)
                        self.assertIs(tensor.weight_loader, loader)
                        self.assertTrue(tensor.is_transposed)
                        self.assertEqual(tensor.quant_method, strategy)

    def test_40_layer_missing_set_and_real_weight_negative_controls(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([self.create()[1] for _ in range(40)])
        params = dict(model.named_parameters())
        loaded = {f"layers.{i}.{name}" for i in range(40) for name in CHECKPOINT_NAMES}
        self.assertEqual(params.keys() - loaded, set())
        self.assertEqual(len(dict(model.named_buffers())), 160)
        # Exactly the loader's named_parameters minus loaded_params contract:
        # every real weight, scale and checkpoint shape remains detectable.
        for missing in loaded:
            self.assertEqual(params.keys() - (loaded - {missing}), {missing})

    def test_other_configs_retain_checkpoint_parameters(self):
        for strategy, actorder in (
            (Strategy.CHANNEL, "static"),
            (Strategy.CHANNEL, "group"),
            (Strategy.GROUP, None),
            (Strategy.GROUP, "group"),
        ):
            with self.subTest(strategy=strategy, actorder=actorder):
                _, layer, _ = self.create(strategy=strategy, actorder=actorder)
                self.assertEqual(
                    set(dict(layer.named_parameters())), CHECKPOINT_NAMES | set(INDEX_WIDTHS)
                )
                self.assertEqual(dict(layer.named_buffers()), {})

    def test_postprocess_discards_poisoned_indices_and_preserves_repack(self):
        for w4a8 in (False, True):
            results = []
            for poison in (0, torch.iinfo(torch.int32).max):
                scheme, layer, _ = self.create(w4a8=w4a8)
                with torch.no_grad():
                    for name, param in layer.named_parameters():
                        param.copy_(torch.arange(param.numel()).reshape(param.shape))
                    for name in INDEX_WIDTHS:
                        getattr(layer, name).fill_(poison)
                before = {name: p.clone() for name, p in layer.named_parameters()}
                scheme.process_weights_after_loading(layer)
                for name in INDEX_WIDTHS:
                    self.assertFalse(hasattr(layer, name))
                self.assertEqual(dict(layer.named_buffers()), {})
                for prefix in ("w13", "w2"):
                    weight = getattr(layer, f"{prefix}_weight_packed")
                    original = before[f"{prefix}_weight_packed"]
                    expected = original.transpose(1, 2).contiguous().view(torch.uint8)
                    expected = (expected ^ 0x88).view(torch.int8)
                    torch.testing.assert_close(weight, expected)
                    scale = getattr(layer, f"{prefix}_weight_scale")
                    expected_scale = before[f"{prefix}_weight_scale"].transpose(1, 2).float() * 7
                    torch.testing.assert_close(scale, expected_scale)
                results.append(layer.state_dict())
            for name in results[0]:
                torch.testing.assert_close(results[0][name], results[1][name])

    def test_forward_does_not_read_index_placeholders(self):
        scheme, layer, _ = self.create()
        for p in layer.parameters():
            p.data.fill_(1)
        scheme.process_weights_after_loading(layer)
        # Run the real non-empty forward body; only accelerator operations are
        # replaced. Missing placeholder attributes would fail any direct read.
        x = torch.ones(3, 16)
        dispatch = types.SimpleNamespace(
            hidden_states=x,
            topk_output=types.SimpleNamespace(
                topk_weights=torch.ones(3, 1), topk_ids=torch.zeros(3, 1)
            ),
        )
        fc = mock.Mock(side_effect=[torch.ones(3, 1, 32), torch.ones(3, 1, 16)])
        ops = types.ModuleType("kunlun_ops")
        ops.swiglu = lambda gate_up, out: out.fill_(1)
        dispatcher = types.ModuleType("sglang.srt.layers.moe.token_dispatcher")
        dispatcher.StandardCombineInput = types.SimpleNamespace
        with mock.patch.dict("sys.modules", {
            "kunlun_ops": ops,
            "xspeedgate_ops": types.ModuleType("xspeedgate_ops"),
            "sglang.srt.layers.moe.token_dispatcher": dispatcher,
        }), mock.patch.dict(self.ns, {
            "moe_pre_sorted": lambda *a, **kw: (x, torch.arange(3), torch.tensor([0, 3])),
            "moe_fc": fc,
            "moe_post": lambda *a, **kw: x.clone(),
        }):
            result = scheme.apply_weights(layer, dispatch)
        torch.testing.assert_close(result.hidden_states, x)
        self.assertEqual(fc.call_count, 2)
        for call, prefix in zip(fc.call_args_list, ("w13", "w2")):
            self.assertIs(call.args[1], getattr(layer, f"{prefix}_weight_packed"))
            self.assertIs(call.args[2], getattr(layer, f"{prefix}_weight_scale"))
            self.assertTrue(call.kwargs["quantize_input"])


if __name__ == "__main__":
    unittest.main()
