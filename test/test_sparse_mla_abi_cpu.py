"""Protect both wheel ABIs and old-wheel sink/empty-row semantics."""
import ast
import inspect
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch


def load_adapter():
    path = Path(__file__).resolve().parents[1] / (
        "sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py"
    )
    names = {"flash_mla_sparse_fwd_kunlun", "_sparse_mla_allocates_outputs"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = {"torch": torch, "inspect": inspect}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["flash_mla_sparse_fwd_kunlun"]


class SparseMLAABITest(unittest.TestCase):
    def setUp(self):
        self.invoke = load_adapter()
        self.q = torch.zeros(2, 3, 4)
        self.kv = torch.ones(3, 1, 4)
        self.indices = torch.tensor([[[0, 1, 2]], [[-1, -1, -1]]], dtype=torch.int32)

    def test_out_buffer_abi_normalizes_sink_and_clears_unwritten_row(self):
        def native(q, kv, indices, o, max_logits, lse, **kwargs):
            self.assertEqual(indices.shape, (2, 3))
            self.assertIsNone(kwargs["attn_sink"])
            self.assertFalse(kwargs["is_causal"])
            self.assertEqual(kwargs["kv_lora_rank"], 4)
            self.assertEqual(kwargs["rope_head_dim"], 0)
            o[0].fill_(2)
            o[1].fill_(float("nan"))
            max_logits.fill_(0)
            lse[0].fill_(torch.log(torch.tensor(3.)).item())
            lse[1].fill_(float("nan"))
            return 0

        module = types.SimpleNamespace(flash_mla_sparse_fwd=native)
        with patch.dict("sys.modules", {"kunlun_ops": module}):
            sink = torch.full((3,), torch.log(torch.tensor(3.)).item())
            out, maxima, lse = self.invoke(
                self.q, self.kv, self.indices, 0.5, d_v=4,
                attn_sink=sink, topk_length=torch.tensor([3, 0]),
            )
        torch.testing.assert_close(out[0], torch.ones(3, 4))
        torch.testing.assert_close(out[1], torch.zeros(3, 4))
        torch.testing.assert_close(lse[0], torch.full((3,), torch.log(torch.tensor(6.)).item()))
        torch.testing.assert_close(lse[1], sink)
        torch.testing.assert_close(maxima, sink.expand(2, -1))

    def test_allocating_abi_keeps_native_contract(self):
        expected = (torch.randn(2, 3, 4), torch.randn(2, 3), torch.randn(2, 3), None)

        def native(q, kv, indices, scale, d_v=512, **kwargs):
            self.assertEqual(indices.shape, (2, 1, 3))
            self.assertEqual(d_v, 4)
            self.assertFalse(kwargs["is_causal"])
            return expected

        with patch.dict("sys.modules", {"kunlun_ops": types.SimpleNamespace(flash_mla_sparse_fwd=native)}):
            actual = self.invoke(self.q, self.kv, self.indices, 0.5, d_v=4)
        for item, reference in zip(actual, expected):
            self.assertIs(item, reference)

    def test_native_failure_is_not_silently_accepted(self):
        def native(**kwargs):
            return 7

        with patch.dict("sys.modules", {"kunlun_ops": types.SimpleNamespace(flash_mla_sparse_fwd=native)}):
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                self.invoke(self.q, self.kv, self.indices, 0.5, d_v=4)


if __name__ == "__main__":
    unittest.main()
