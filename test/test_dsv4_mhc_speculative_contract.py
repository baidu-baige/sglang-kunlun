import ast
import importlib.util
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

    def plugin_hook(target, type=None):
        def decorate(fn):
            registered[target] = (type, fn)
            return fn

        return decorate

    registry.HookType = HookType
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

    def test_model_hooks_target_upstream_methods_and_hc_pre_is_quadruple(self):
        source = (ROOT / "sglang_kunlun/models/deepseek_v4.py").read_text()
        tree = ast.parse(source)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        returns = [n for n in ast.walk(functions["hc_pre_kunlun"]) if isinstance(n, ast.Return)]
        self.assertTrue(
            all(isinstance(n.value, ast.Tuple) and len(n.value.elts) == 4 for n in returns)
        )
        self.assertIn("DeepseekV4DecoderLayer.hc_pre", source)
        self.assertIn("DeepseekV4DecoderLayer.hc_post", source)
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
