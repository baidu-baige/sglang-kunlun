"""CPU-only V4.1 router contract tests; no plugin/XPU initialization."""
import ast
import itertools
import math
from pathlib import Path
import sys
import types
import unittest
from typing import Optional
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


def load_gate():
    path = Path(__file__).resolve().parents[1] / "sglang_kunlun/kernels/kernel_ops.py"
    names = {"_moe_gate_sqrtsoftplus_log1p", "moe_fused_gate_dsv4_kunlun"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    for node in nodes:
        node.decorator_list = []
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def scalar_reference(scores, bias, topk, **kw):
    """Independent scalar selection + torch softplus activation oracle."""
    values = F.softplus(scores.float(), beta=1, threshold=20).sqrt()
    shared = kw.get("num_fused_shared_experts", 0)
    routed = topk - shared
    scale = kw.get("routed_scaling_factor", 1.0)
    scale = 1.0 if scale is None else scale
    out_w, out_i = [], []
    for row in range(scores.shape[0]):
        if row >= int(kw.get("num_token_non_padded", scores.shape[0])):
            out_w.append([0.0] * topk)
            out_i.append([-1] * topk)
            continue
        row_bias = bias
        if kw.get("input_ids") is not None:
            if int(kw["input_ids"][row]) == kw["bias_alt_token_id"]:
                row_bias = kw["bias_alt"]
        # Round additions to fp32 before scalar sorting, as upstream does.
        keys = (values[row] + row_bias.float()).tolist()
        keys = [math.inf if math.isnan(v) else v for v in keys]
        selected = sorted(range(len(keys)), key=lambda i: (-keys[i], i))[:routed]
        w = values[row, selected]
        total = w.sum()
        w = torch.cat((w, (total / scale).repeat(shared)))
        if kw.get("renormalize", True):
            eps = kw.get("renormalize_epsilon", 0.0)
            norm = total + eps if eps > 0 else (total if total > 0 else 1.0)
            w = w / norm
        if kw.get("apply_routed_scaling_factor_on_output", False):
            w = w * scale
        out_w.append(w.tolist())
        out_i.append(selected + list(range(scores.shape[1], scores.shape[1] + shared)))
    return (torch.tensor(out_w, dtype=torch.float32).reshape(-1, topk),
            torch.tensor(out_i, dtype=torch.int32).reshape(-1, topk))


class GateTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_gate()
        self.gate = self.ns["moe_fused_gate_dsv4_kunlun"]

    def check(self, scores, bias, topk, **kw):
        w, ids = self.gate(
            scores, bias, topk, scoring_func="sqrtsoftplus",
            sqrtsoftplus_log1p=True, **kw,
        )
        expected_w, expected_ids = scalar_reference(scores, bias, topk, **kw)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(w, expected_w, rtol=3e-6, atol=0, equal_nan=True)
        self.assertEqual(w.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(w.shape, (scores.shape[0], topk))
        self.assertTrue(w.is_contiguous() and ids.is_contiguous())
        return w, ids

    def test_threshold_and_negative_precision(self):
        x = torch.tensor([[-80., -60., -40., -20., 0., 19.999998, 20., 20.000002, 100.]])
        w, ids = self.check(x, torch.zeros(9), 9, renormalize=False)
        self.assertTrue((w > 0).all())
        # Independent double scalar oracle, avoiding an implementation-copy formula.
        for slot, expert in enumerate(ids[0].tolist()):
            v = float(x[0, expert])
            expected = math.sqrt(v if v > 20 else math.log1p(math.exp(v)))
            self.assertAlmostEqual(float(w[0, slot]) / expected, 1., delta=3e-6)
        # At -100 exp is subnormal: separate exp/log1p and fused softplus can
        # differ by subnormal ULPs. Allow one ULP per exp/log1p operation
        # in pre-sqrt units, not a broad relative tolerance.
        tiny, _ = self.gate(torch.tensor([[-100.]]), torch.zeros(1), 1,
                            scoring_func="sqrtsoftplus", sqrtsoftplus_log1p=True,
                            renormalize=False)
        self.assertGreater(float(tiny[0, 0]), 0.)
        self.assertLessEqual(abs(float(tiny[0, 0]) ** 2 - math.exp(-100)), 2 * 2 ** -149)

    def test_nan_infinity_and_ties(self):
        scores = torch.tensor([
            [0., float("nan"), float("inf"), float("nan"), 0., -float("inf")],
            [-float("inf")] * 6,
            [0.] * 6,
        ])
        _, ids = self.check(scores, torch.zeros(6), 6, renormalize=False)
        self.assertEqual(ids[0].tolist(), [1, 2, 3, 0, 4, 5])
        self.assertEqual(ids[1].tolist(), list(range(6)))
        self.check(torch.zeros(1, 6), torch.tensor([0., float("nan"), 1., 1., 0., 0.]), 4)
        self.check(torch.zeros(1, 6), torch.full((6,), -float("inf")), 6)

    def test_contract_combinations_and_strides(self):
        gen = torch.Generator().manual_seed(410522)
        for dtype, renorm, apply_scale, shared, eps in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16),
            (False, True), (False, True), (0, 2), (0., 1e-20, .25),
        ):
            scores = torch.randn(6, 18, generator=gen).to(dtype)[:, ::2]
            bias = torch.randn(18, generator=gen).to(dtype)[::2]
            alt = torch.randn(18, generator=gen).to(dtype)[::2]
            token_ids = torch.tensor([7, 0, 3, 0, 7, 0, 3, 0, 7, 0, 3, 0])[::2]
            with self.subTest(dtype=dtype, renorm=renorm, scale=apply_scale, shared=shared, eps=eps):
                self.check(
                    scores, bias, 4 + shared,
                    bias_alt=alt, input_ids=token_ids, bias_alt_token_id=7,
                    renormalize=renorm, renormalize_epsilon=eps,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=apply_scale,
                    num_fused_shared_experts=shared,
                    num_token_non_padded=torch.tensor(4),
                )

    def test_empty_padding_zero_sum_and_ignored_alt(self):
        for rows in (0, 3):
            for limit in (0, 2, 3):
                self.check(torch.full((rows, 5), -float("inf")), torch.zeros(5), 4,
                           num_fused_shared_experts=1, num_token_non_padded=torch.tensor(limit),
                           routed_scaling_factor=None)
        for eps in (0., -1., 1e-20):
            self.check(torch.full((2, 5), -1000.), torch.zeros(5), 3,
                       num_fused_shared_experts=1, renormalize_epsilon=eps)
        self.check(torch.zeros(2, 5), torch.zeros(5), 1, bias_alt=torch.full((5,), math.nan))
        self.check(torch.full((3, 5), math.nan), torch.zeros(5), 4,
                   num_fused_shared_experts=1, num_token_non_padded=torch.tensor(0))

    def test_validation_and_packed_rejection(self):
        scores, bias = torch.zeros(2, 5), torch.zeros(5)
        for kw in ({"input_ids": torch.tensor([1, 2])},
                   {"input_ids": torch.tensor([1]), "bias_alt": bias, "bias_alt_token_id": 1}):
            with self.assertRaises(AssertionError):
                self.gate(scores, bias, 2, scoring_func="sqrtsoftplus", sqrtsoftplus_log1p=True, **kw)
        packed = torch.full((2, 2), 123, dtype=torch.int32)
        for flag in (False, True):
            with self.assertRaisesRegex(NotImplementedError, "packed_out"):
                self.gate(scores, bias, 2, sqrtsoftplus_log1p=flag, packed_out=packed)
        self.assertTrue((packed == 123).all())

    def test_dsv41_384_experts_top6(self):
        gen = torch.Generator().manual_seed(3846)
        scores = torch.randn(17, 384, generator=gen)
        bias = torch.randn(384, generator=gen).bfloat16()
        alt = torch.randn(384, generator=gen).bfloat16()
        for ids in (None, torch.arange(17) % 2):
            self.check(scores, bias, 6, bias_alt=alt, input_ids=ids,
                       bias_alt_token_id=1, renormalize_epsilon=1e-20,
                       routed_scaling_factor=2.5,
                       apply_routed_scaling_factor_on_output=True)

    def test_legacy_native_and_k3_dispatch(self):
        native = Mock()
        radix = Mock(return_value=("radix_weights", "radix_ids"))
        self.ns["_moe_route_radix_usable"] = Mock(return_value=True)
        self.ns["moe_route_radix_kunlun"] = radix
        with patch.dict(sys.modules, {"kunlun_ops": types.SimpleNamespace(moe_fused_gate_dsv4=native)}):
            scores, bias = torch.zeros(2, 5), torch.zeros(5)
            self.gate(scores, bias, 2, scoring_func="sqrtsoftplus")
            self.assertEqual(native.call_count, 1)
            args = native.call_args.args
            self.assertEqual(args[5:], ("sqrtsoftplus", 0, True, 1.0, False))
            self.assertEqual(self.gate(scores, bias, 2), ("radix_weights", "radix_ids"))
            self.assertEqual(radix.call_args.kwargs, {"sorted": False})
            self.assertEqual(native.call_count, 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
