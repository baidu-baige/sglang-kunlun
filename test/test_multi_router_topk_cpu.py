"""CPU-only ERNIE 5.5 multi-router top-k hook tests; no plugin/XPU init.

The hook body is extracted from source with the decorator stripped, so the
module-level ``sglang_kunlun.hooks.layers`` imports (and the XPU-only plugin
initialization) never run here.
"""
import ast
import logging
from pathlib import Path
import types
import unittest

import torch


def load_hook(kunlun_ops, force_supported=False):
    path = (
        Path(__file__).resolve().parents[1]
        / "sglang_kunlun/hooks/layers/moe/topk.py"
    )
    names = {"_multi_router_kernel_supported", "multi_router_topk_kunlun"}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == 2
    for node in nodes:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "kunlun_ops": kunlun_ops,
        "logger": logging.getLogger("multi_router_topk_cpu"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    if force_supported:
        # The dispatch tests below run on CPU tensors; this stub isolates the
        # hook's call/fallback structure from the device gate. The gate itself
        # is covered by the support-check tests.
        namespace["_multi_router_kernel_supported"] = lambda *args, **kwargs: True
    return namespace


class CudaMeta:
    """CPU tensor that the support check sees as an XPU tensor.

    Only metadata is read by the check, so delegating everything except
    ``device`` keeps the CPU test free of XPU allocator dependencies.
    """

    def __init__(self, tensor, device=None):
        self._tensor = tensor
        self.device = torch.device("cuda") if device is None else device

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def oracle(logits, bias, scale, topk, renormalize, num_routers):
    """Eager reference for finite, gap-separated inputs."""
    tokens, width = logits.shape
    experts = width // num_routers
    scores = (
        torch.sigmoid(logits.to(torch.float32).view(tokens, num_routers, experts))
        .sum(dim=1)
    )
    ids = torch.topk(scores + bias.to(torch.float32), topk, dim=1).indices
    weights = torch.gather(scores, 1, ids)
    if renormalize:
        denom = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.where(denom != 0, denom, torch.ones_like(denom))
    weights = weights * scale.to(torch.float32)[ids]
    return weights, ids.to(torch.int32)


def make_inputs(tokens=7, experts=16, routers=2, dtype=torch.bfloat16, seed=20261120):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((tokens, experts * routers), generator=generator).to(dtype)
    bias = torch.randperm(experts, generator=generator).float() * (routers + 1)
    scale = torch.randn(experts, generator=generator)
    return logits, bias, scale


class FakeKernel:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result
        self.error = error

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class MultiRouterTopkHookTest(unittest.TestCase):
    def check_fallback(self, logits, bias, scale, topk, renormalize, routers):
        """The kernel must not be called; the eager path must match the oracle."""
        kernel = FakeKernel()
        namespace = load_hook(types.SimpleNamespace(multi_router_topk=kernel))
        weights, ids = namespace["multi_router_topk_kunlun"](
            logits, bias, scale, topk, renormalize, num_routers=routers
        )
        self.assertEqual(kernel.calls, [])
        expected_weights, expected_ids = oracle(
            logits, bias, scale, topk, renormalize, routers
        )
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)
        return weights, ids

    def test_dispatches_to_kernel_and_returns_its_outputs(self):
        sentinel = (
            torch.zeros(7, 4, dtype=torch.float32),
            torch.zeros(7, 4, dtype=torch.int32),
        )
        kernel = FakeKernel(result=sentinel)
        namespace = load_hook(
            types.SimpleNamespace(multi_router_topk=kernel), force_supported=True
        )
        logits, bias, scale = make_inputs()
        result = namespace["multi_router_topk_kunlun"](
            logits, bias, scale, 4, True, num_routers=2
        )
        self.assertIs(result, sentinel)
        self.assertEqual(len(kernel.calls), 1)
        args, kwargs = kernel.calls[0]
        self.assertEqual(args, (logits, bias, scale, 4, True))
        self.assertEqual(kwargs, {"num_routers": 2})

    def test_falls_back_when_the_kernel_raises(self):
        logits, bias, scale = make_inputs()
        kernel = FakeKernel(error=RuntimeError("ret=1"))
        namespace = load_hook(
            types.SimpleNamespace(multi_router_topk=kernel), force_supported=True
        )
        with self.assertLogs("multi_router_topk_cpu", level="WARNING") as logs:
            weights, ids = namespace["multi_router_topk_kunlun"](
                logits, bias, scale, 4, True, num_routers=2
            )
        self.assertIn("multi_router_topk_fused_fallback", logs.output[0])
        expected_weights, expected_ids = oracle(logits, bias, scale, 4, True, 2)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            weights, expected_weights, rtol=1e-5, atol=1e-6
        )

    def test_falls_back_without_the_op(self):
        namespace = load_hook(types.SimpleNamespace())
        logits, bias, scale = make_inputs()
        weights, ids = namespace["multi_router_topk_kunlun"](
            logits, bias, scale, 4, True, num_routers=2
        )
        expected_weights, expected_ids = oracle(logits, bias, scale, 4, True, 2)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            weights, expected_weights, rtol=1e-5, atol=1e-6
        )

    def test_falls_back_on_cpu_tensors(self):
        self.check_fallback(*make_inputs(), 4, False, 2)

    def test_falls_back_on_non_power_of_two_experts(self):
        logits, bias, scale = make_inputs(experts=3)
        self.check_fallback(logits, bias, scale, 3, True, 2)

    def test_falls_back_on_inner_stride(self):
        logits, bias, scale = make_inputs(experts=32)
        sliced = logits[:, ::2]
        self.assertEqual(sliced.stride(1), 2)
        self.check_fallback(sliced, bias[:16], scale[:16], 4, True, 2)

    def test_falls_back_on_row_overlap(self):
        logits, bias, scale = make_inputs()
        expanded = logits[:1].expand(7, -1)
        self.assertEqual(expanded.stride(0), 0)
        self.check_fallback(expanded, bias, scale, 4, True, 2)

    def test_falls_back_on_non_contiguous_bias(self):
        logits, bias, scale = make_inputs(experts=16)
        self.check_fallback(logits, bias.repeat_interleave(2)[::2], scale, 4, True, 2)

    def test_zero_denominator_renormalize(self):
        # scores are all zero, so the denominator stays 1 (no division by zero).
        logits = torch.full((3, 16), float("-inf"), dtype=torch.bfloat16)
        bias = torch.arange(8).float()
        scale = torch.ones(8)
        namespace = load_hook(types.SimpleNamespace())
        weights, ids = namespace["multi_router_topk_kunlun"](
            logits, bias, scale, 3, True, num_routers=2
        )
        torch.testing.assert_close(weights, torch.zeros(3, 3), rtol=0, atol=0)
        torch.testing.assert_close(
            ids,
            torch.tensor([[7, 6, 5]] * 3, dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_fp16_and_fp32_inputs(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                logits, bias, scale = make_inputs(dtype=dtype)
                self.check_fallback(logits, bias, scale, 4, True, 2)

    def test_zero_tokens(self):
        logits, bias, scale = make_inputs(tokens=0)
        namespace = load_hook(types.SimpleNamespace())
        weights, ids = namespace["multi_router_topk_kunlun"](
            logits, bias, scale, 4, False, num_routers=2
        )
        self.assertEqual(tuple(weights.shape), (0, 4))
        self.assertEqual(tuple(ids.shape), (0, 4))
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)


class MultiRouterSupportCheckTest(unittest.TestCase):
    """``_multi_router_kernel_supported`` mirrors the kernel's input contract."""

    def setUp(self):
        self.kernel = FakeKernel()
        self.supported = load_hook(
            types.SimpleNamespace(multi_router_topk=self.kernel)
        )["_multi_router_kernel_supported"]

    def assertSupported(self, logits, bias, scale, topk, routers):
        device = torch.device("cuda")
        self.assertTrue(
            self.supported(
                CudaMeta(logits, device),
                CudaMeta(bias, device),
                CudaMeta(scale, device),
                topk,
                True,
                routers,
            )
        )

    def assertUnsupported(self, logits, bias, scale, topk, routers):
        device = torch.device("cuda")
        self.assertFalse(
            self.supported(
                CudaMeta(logits, device),
                CudaMeta(bias, device),
                CudaMeta(scale, device),
                topk,
                True,
                routers,
            )
        )

    def test_accepts_the_eb5_shape(self):
        logits, bias, scale = make_inputs(tokens=1, experts=512, routers=2)
        self.assertSupported(logits, bias, scale, 10, 2)
        for dtype in (torch.float16, torch.float32):
            logits, bias, scale = make_inputs(experts=512, dtype=dtype)
            self.assertSupported(logits, bias, scale, 10, 2)

    def test_rejects_without_the_op(self):
        supported = load_hook(types.SimpleNamespace())[
            "_multi_router_kernel_supported"
        ]
        logits, bias, scale = make_inputs()
        device = torch.device("cuda")
        self.assertFalse(
            supported(
                CudaMeta(logits, device),
                CudaMeta(bias, device),
                CudaMeta(scale, device),
                4,
                True,
                2,
            )
        )

    def test_rejects_cpu_tensors(self):
        logits, bias, scale = make_inputs()
        self.assertFalse(self.supported(logits, bias, scale, 4, True, 2))

    def test_rejects_non_power_of_two_experts(self):
        logits, bias, scale = make_inputs(experts=3)
        self.assertUnsupported(logits, bias, scale, 3, 2)
        logits, bias, scale = make_inputs(experts=24, routers=1)
        self.assertUnsupported(logits, bias, scale, 4, 1)

    def test_rejects_value_mismatches(self):
        logits, bias, scale = make_inputs(experts=16)
        self.assertUnsupported(logits, bias, scale, 0, 2)
        self.assertUnsupported(logits, bias, scale, 17, 2)
        self.assertUnsupported(logits, bias, scale, 4, 0)
        self.assertUnsupported(logits, bias, scale, 4, 3)
        self.assertUnsupported(logits, bias[:8], scale, 4, 2)
        self.assertUnsupported(logits, bias, scale[:8], 4, 2)
        self.assertUnsupported(logits, bias, scale, torch.tensor(4), 2)
        self.assertUnsupported(logits, bias, scale, True, 2)

    def test_rejects_unsupported_layouts(self):
        logits, bias, scale = make_inputs(experts=32)
        sliced = logits[:, ::2]  # inner stride 2, 16 experts per router
        self.assertEqual(sliced.stride(1), 2)
        self.assertUnsupported(sliced, bias[:16], scale[:16], 4, 2)
        self.assertUnsupported(logits[:1].expand(7, -1), bias, scale, 4, 2)
        logits, bias, scale = make_inputs(experts=16)
        self.assertUnsupported(
            logits, bias.repeat_interleave(2)[::2], scale, 4, 2
        )
        self.assertUnsupported(logits, bias, scale.to(torch.float64), 4, 2)

    def test_rejects_mismatched_devices(self):
        logits, bias, scale = make_inputs()
        d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
        self.assertFalse(
            self.supported(
                CudaMeta(logits, d0),
                CudaMeta(bias, d1),
                CudaMeta(scale, d0),
                4,
                True,
                2,
            )
        )

    def test_rejects_non_bool_renormalize(self):
        logits, bias, scale = make_inputs()
        device = torch.device("cuda")
        self.assertFalse(
            self.supported(
                CudaMeta(logits, device),
                CudaMeta(bias, device),
                CudaMeta(scale, device),
                4,
                torch.tensor(True),
                2,
            )
        )


if __name__ == "__main__":
    unittest.main()