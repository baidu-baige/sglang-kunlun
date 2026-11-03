"""Byte-level indexer contracts; no plugin import or accelerator initialization."""

import ast
import math
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "sglang_kunlun/kernels/dsv41_low_ratio.py"


def load_functions():
    names = {"quantize_fp4_indexer_tensor_kunlun", "store_fp4_index_k_cache_kunlun"}
    nodes = [
        n for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )] + nodes, type_ignores=[])
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), ns)
    return types.SimpleNamespace(**ns)


def reference(x, rne):
    """Independent nearest-grid oracle with scalar log2 scales and byte packing."""
    blocks = x.float().reshape(-1, 4, 32).numpy()
    payload = np.zeros((len(blocks), 64), dtype=np.uint8)
    exps = np.zeros((len(blocks), 4), dtype=np.uint8)
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    for row, groups in enumerate(blocks):
        codes = []
        for group, values in enumerate(groups):
            sf = max(np.float32(np.max(np.abs(values)) / np.float32(6)), np.float32(1e-4))
            exponent = min(max(math.ceil(math.log2(float(sf))) + 127, 1), 254)
            exps[row, group] = exponent
            scaled = values / np.float32(2.0 ** (exponent - 127))
            for value in scaled:
                distances = np.abs(grid - min(abs(value), 6))
                candidates = np.flatnonzero(distances == distances.min())
                code = int(candidates[0])
                if rne and len(candidates) == 2:
                    code = next(int(c) for c in candidates if c % 2 == 0)
                codes.append(code | (8 if value < 0 and code else 0))
        payload[row] = np.array(codes[::2]) | (np.array(codes[1::2]) << 4)
    return torch.from_numpy(payload.view(np.int8)), torch.from_numpy(exps.copy().view("<i4").reshape(-1))


class FP4IndexerCPU(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()

    def check_quant(self, x, rne):
        actual = self.ns.quantize_fp4_indexer_tensor_kunlun(x, rne)
        expected = reference(x, rne)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
            self.assertTrue(a.is_contiguous())
        self.assertEqual(actual[0].dtype, torch.int8)
        self.assertEqual(actual[1].dtype, torch.int32)
        return actual

    def test_all_midpoints_neighbors_and_negative_zero(self):
        mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        values = torch.cat((
            torch.nextafter(mids, torch.full_like(mids, -math.inf)),
            mids, torch.nextafter(mids, torch.full_like(mids, math.inf)),
            torch.tensor([0., -0., 6.]),
        ))
        for sign in (1, -1):
            x = torch.zeros(1, 4, 32)
            x[:, :, :len(values)] = sign * values
            for rne in (False, True):
                self.check_quant(x.reshape(1, 128), rne)
        x = torch.zeros(1, 128)
        x[0, :8] = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.])
        self.assertEqual(self.check_quant(x, False)[0][0, :4].tolist(), [16, 50, 84, 118])
        self.assertEqual(self.check_quant(x, True)[0][0, :4].tolist(), [32, 66, 100, 118])

    def test_scale_floor_powers_and_signed_scale_word(self):
        for maximum in (0., 1e-9, .0006, .0007, 6., 12., 2.0**120):
            x = torch.full((2, 128), maximum)
            x[1] *= -1
            for rne in (False, True):
                self.check_quant(x, rne)
        zero_payload, zero_sf = self.check_quant(torch.zeros(1, 128), True)
        self.assertEqual(zero_sf.item(), 0x72727272)
        self.assertFalse(zero_payload.any())
        for amax in (torch.nextafter(torch.tensor(6.), torch.tensor(0.)),
                     torch.tensor(6.), torch.nextafter(torch.tensor(6.), torch.tensor(7.))):
            self.check_quant(amax.expand(1, 128), True)

    def test_random_dtypes_noncontiguous_and_empty(self):
        rng = torch.Generator().manual_seed(20261030)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = (torch.randn(3, 2, 256, generator=rng) * 17).to(dtype)[..., ::2]
            for rne in (False, True):
                self.check_quant(x, rne)
                self.check_quant(torch.empty(0, 2, 128, dtype=dtype), rne)

    def test_chunk_boundary(self):
        x = torch.zeros(4097, 128)
        x[4095:] = torch.arange(256).reshape(2, 128)
        q, s = self.ns.quantize_fp4_indexer_tensor_kunlun(x, True)
        eq, es = reference(x[4095:], True)
        torch.testing.assert_close(q[4095:], eq, rtol=0, atol=0)
        torch.testing.assert_close(s[4095:], es, rtol=0, atol=0)
        self.assertTrue(torch.all(s[:4095] == 0x72727272))

    def test_cache_layout_invalid_slots_page_stride_and_rne(self):
        for page_size in (1, 4, 64):
            for rne in (False, True):
                storage = torch.full((3, page_size * 68 + 9), 173, dtype=torch.uint8)
                cache = storage[:, :page_size * 68]
                loc = torch.tensor([0, -1, page_size, 3 * page_size - 1,
                                    3 * page_size, 2**40], dtype=torch.int64)
                x = torch.arange(6 * 128, dtype=torch.float32).reshape(6, 128) / 64
                expected = storage.clone()
                payload, sf = reference(x, rne)
                for row in (2, 3):
                    page, slot = divmod(int(loc[row]), page_size)
                    expected[page, slot * 64:(slot + 1) * 64] = payload[row].view(torch.uint8)
                    start = page_size * 64 + slot * 4
                    expected[page, start:start + 4] = sf[row:row + 1].view(torch.uint8)
                self.ns.store_fp4_index_k_cache_kunlun(x, cache, loc, page_size=page_size, rne=rne)
                torch.testing.assert_close(storage, expected, rtol=0, atol=0)
                self.ns.store_fp4_index_k_cache_kunlun(
                    x[:0], cache, loc[:0], page_size=page_size, rne=rne)
                torch.testing.assert_close(storage, expected, rtol=0, atol=0)

    def test_missing_ops_and_native_dispatch(self):
        x = torch.ones(1, 128, dtype=torch.bfloat16)
        with mock.patch.object(torch.ops, "xspeedgate_ops", types.SimpleNamespace()):
            self.check_quant(x, True)
        quant = mock.Mock(return_value=(torch.zeros(1, 64, dtype=torch.int8),
                                       torch.zeros(1, dtype=torch.int32)))
        store = mock.Mock()
        ops = types.SimpleNamespace(quantize_fp4_indexer_tensor=quant,
                                    store_fp4_index_k_cache=store)
        with mock.patch.object(torch.ops, "xspeedgate_ops", ops):
            self.check_quant(x, True)  # CPU must not call accelerator ops.
            quant.assert_not_called()
            with mock.patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True):
                self.ns.store_fp4_index_k_cache_kunlun(
                    x, torch.zeros(2, 68, dtype=torch.uint8),
                    torch.tensor([1]), page_size=1, rne=True)
            quant.assert_called_once_with(x, True)
            store.assert_called_once()

    def test_partial_native_installations(self):
        x = torch.arange(128, dtype=torch.bfloat16).reshape(1, 128)
        expected = reference(x, True)
        for has_quant, has_store in ((False, False), (True, False), (False, True)):
            quant = mock.Mock(return_value=expected)
            store = mock.Mock()
            ops = types.SimpleNamespace()
            if has_quant:
                ops.quantize_fp4_indexer_tensor = quant
            if has_store:
                ops.store_fp4_index_k_cache = store
            cache = torch.full((2, 68), 173, dtype=torch.uint8)
            with mock.patch.object(torch.ops, "xspeedgate_ops", ops), mock.patch.object(
                torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True
            ):
                self.ns.store_fp4_index_k_cache_kunlun(
                    x, cache, torch.tensor([1], dtype=torch.int32), page_size=1, rne=True)
            self.assertEqual(quant.call_count, int(has_quant))
            self.assertEqual(store.call_count, int(has_store))
            if not has_store:
                torch.testing.assert_close(cache[1, :64], expected[0][0].view(torch.uint8))
                torch.testing.assert_close(cache[1, 64:], expected[1].view(torch.uint8))
                self.assertTrue(torch.all(cache[0] == 173))

    def test_store_chunk_boundary_and_strided_locations(self):
        n = 4097
        x = torch.zeros(n, 128)
        x[-1] = 6
        loc = torch.arange(2 * n, dtype=torch.int64)[::2] + 1
        cache = torch.full((2 * n, 68), 173, dtype=torch.uint8)
        self.ns.store_fp4_index_k_cache_kunlun(x, cache, loc, page_size=1)
        self.assertTrue(torch.all(cache[::2] == 173))
        self.assertTrue(torch.all(cache[1:-1:2, :64] == 0))
        self.assertTrue(torch.all(cache[1:-1:2, 64:] == 114))
        self.assertTrue(torch.all(cache[-1, :64] == 0x77))
        self.assertTrue(torch.all(cache[-1, 64:] == 127))


if __name__ == "__main__":
    unittest.main()
