# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch CPU reference tests. Not executed on the hung development machine."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


SOURCE = (Path(__file__).resolve().parents[1] / "sglang_kunlun"
          / "kernels/dspark_metadata_torch.py")
SPEC = importlib.util.spec_from_file_location("dspark_metadata_torch_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs(lengths=(0, 1, 128), window=128, block=5, align=64, dtype=torch.int64):
    bs = len(lengths)
    return dict(
        req_to_token=torch.arange(bs * (window + 8), dtype=dtype).reshape(bs, window + 8),
        full_to_swa_mapping=torch.arange(bs * (window + 8) + bs * block, dtype=dtype) + 10,
        req_pool_indices_per_request=torch.arange(bs, dtype=dtype),
        offsets=torch.arange(window, dtype=dtype).expand(bs, window).clone(),
        invalid=torch.zeros((bs, window), dtype=torch.bool),
        out_loc=torch.arange(bs * block, dtype=dtype) + bs * (window + 8),
        context_lens=torch.tensor(lengths, dtype=dtype),
        block_size=block, swa_window=window, page_index_aligned_size=align,
    )


def reference(x):
    """Independent scalar layout oracle, including invalid window slots."""
    width = ((x["swa_window"] + x["block_size"] + x["page_index_aligned_size"] - 1)
             // x["page_index_aligned_size"] * x["page_index_aligned_size"])
    pages, topk = [], []
    for i, cl in enumerate(x["context_lens"].tolist()):
        row = []
        for j in range(x["swa_window"] - cl, x["swa_window"]):
            if x["invalid"][i, j]:
                row.append(-1)
            else:
                full = x["req_to_token"][x["req_pool_indices_per_request"][i], x["offsets"][i, j]]
                row.append(int(x["full_to_swa_mapping"][full]))
        for j in range(x["block_size"]):
            full = x["out_loc"][i * x["block_size"] + j]
            row.append(int(x["full_to_swa_mapping"][full]))
        row += [-1] * (width - len(row))
        pages.extend([row] * x["block_size"])
        topk.extend([cl + x["block_size"]] * x["block_size"])
    return (torch.tensor(pages, dtype=torch.int32).reshape(-1, width),
            torch.tensor(topk, dtype=torch.int32))


class MetadataTorchTest(unittest.TestCase):
    def test_safe_take_zero_index_never_leaks_and_contiguous_source_is_not_copied(self):
        values = torch.tensor([[777, -1], [22, 33]], dtype=torch.int64)
        indices = torch.tensor([[0, -1, 4], [1, 2, 3]], dtype=torch.int64)
        active = torch.tensor([[True, True, True], [True, False, True]])
        original = values.clone()
        original_reshape = torch.Tensor.reshape
        def shared_reshape(source, *shape):
            result = original_reshape(source, *shape)
            if source is values:
                self.assertEqual(result.data_ptr(), values.data_ptr())
            return result
        with patch.object(torch, "cat", side_effect=AssertionError("must not append source")), \
             patch.object(torch.Tensor, "reshape", shared_reshape):
            selected, valid = MODULE._safe_take(values, indices, active, False, "test")
        torch.testing.assert_close(selected, torch.tensor([[777, -1, -1], [-1, -1, 33]]))
        torch.testing.assert_close(valid, torch.tensor([[True, False, False], [True, False, True]]))
        torch.testing.assert_close(values, original)
        with self.assertRaisesRegex(ValueError, "test"):
            MODULE._safe_take(values, indices, active, True, "test")

    def test_safe_take_empty_source_never_gathers_and_still_validates(self):
        for dtype in (torch.int32, torch.int64):
            values = torch.empty((0, 2), dtype=dtype)
            indices = torch.tensor([[0, -1], [99, 0]], dtype=torch.int64)
            active = torch.tensor([[True, False], [True, False]])
            with patch.object(torch, "cat", side_effect=AssertionError("must not append source")), \
                 patch.object(torch.Tensor, "reshape", side_effect=AssertionError("must not gather")):
                selected, valid = MODULE._safe_take(values, indices, active, False, "empty")
                inactive_result, inactive_valid = MODULE._safe_take(
                    values, indices, torch.zeros_like(active), True, "empty",
                )
                with self.assertRaisesRegex(ValueError, "empty"):
                    MODULE._safe_take(values, indices, active, True, "empty")
            for result, mask in ((selected, valid), (inactive_result, inactive_valid)):
                torch.testing.assert_close(result, values.new_full(indices.shape, -1))
                torch.testing.assert_close(mask, torch.zeros_like(active))
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, values.device)

    def run_impl(self, x, validate=True):
        with patch.object(MODULE, "_can_validate_values", return_value=validate):
            return MODULE.build_dspark_swa_page_indices(**x)

    def assert_reference(self, x):
        actual = self.run_impl(x)
        for got, expected in zip(actual, reference(x)):
            torch.testing.assert_close(got, expected)
            self.assertEqual(got.dtype, torch.int32)
            self.assertEqual(got.device, x["offsets"].device)
            self.assertTrue(got.is_contiguous())

    def test_layout_integer_types_blocks_and_alignment(self):
        for dtype in (torch.int32, torch.int64):
            for window, block, align in ((128, 5, 64), (7, 3, 4), (1, 1, 1), (0, 2, 8)):
                with self.subTest(dtype=dtype, window=window, block=block):
                    self.assert_reference(inputs((0, min(1, window), window), window, block, align, dtype))
        x = inputs((128,) * 16)
        pages, topk = self.run_impl(x)
        self.assertEqual(pages.shape, (80, 192))
        self.assertEqual(topk.shape, (80,))

    def test_invalid_padding_and_final_mapping_minus_one(self):
        x = inputs((0, 1, 128))
        x["req_pool_indices_per_request"][0] = -999
        x["offsets"][0] = -999
        x["offsets"][1, :-1] = 999999
        x["invalid"][2, -1] = True
        x["offsets"][2, -1] = -999
        x["full_to_swa_mapping"][x["out_loc"][0]] = -1
        x["full_to_swa_mapping"][x["req_to_token"][1, -9]] = -1
        self.assert_reference(x)

    def test_mixed_integer_types_and_noncontiguous_inputs(self):
        x = inputs((1, 7), window=7, block=3, align=8)
        x["req_to_token"] = x["req_to_token"].t().contiguous().t()
        x["req_pool_indices_per_request"] = x["req_pool_indices_per_request"].to(torch.int32)
        x["context_lens"] = x["context_lens"].to(torch.int32)
        x["offsets"] = x["offsets"].to(torch.int32)
        self.assert_reference(x)

    def test_active_bad_indices_raise_eager_and_mask_capture(self):
        mutations = (
            ("req_pool_indices_per_request", (0,), -1),
            ("req_pool_indices_per_request", (0,), 999),
            ("offsets", (0, 127), -1),
            ("offsets", (0, 127), 999),
            ("offsets", (0, 127), torch.iinfo(torch.int64).max),
            ("req_to_token", (0, 127), -1),
            ("req_to_token", (0, 127), 999999),
            ("out_loc", (0,), -1),
            ("out_loc", (0,), 999999),
        )
        for name, index, value in mutations:
            with self.subTest(name=name, value=value):
                x = inputs((1,))
                x[name][index] = value
                with self.assertRaisesRegex(ValueError, "invalid active"):
                    self.run_impl(x)
                pages, topk = self.run_impl(x, validate=False)
                target = 1 if name == "out_loc" else 0
                self.assertEqual(int(pages[0, target]), -1)
                self.assertEqual(int(topk[0]), 6)

    def test_bad_context_lengths(self):
        for value in (-1, 129, torch.iinfo(torch.int64).max):
            x = inputs((value,))
            with self.assertRaisesRegex(ValueError, "context_lens"):
                self.run_impl(x)
            pages, topk = self.run_impl(x, False)
            self.assertTrue(torch.all(pages == -1))
            self.assertTrue(torch.all(topk == 0))

    def test_empty_batch_sources_and_zero_context(self):
        self.assert_reference(inputs(()))
        x = inputs((0,))
        x["req_to_token"] = torch.empty((0, 0), dtype=torch.int64)
        self.assert_reference(x)
        x["full_to_swa_mapping"] = torch.empty(0, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "block full_to_swa_mapping"):
            self.run_impl(x)
        pages, _ = self.run_impl(x, False)
        self.assertTrue(torch.all(pages == -1))
        x = inputs((1,))
        x["req_to_token"] = torch.empty((0, 0), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "req_to_token row/offset"):
            self.run_impl(x)
        pages, _ = self.run_impl(x, False)
        self.assertEqual(int(pages[0, 0]), -1)
        x = inputs(())
        x["full_to_swa_mapping"] = torch.empty(0, dtype=torch.int64)
        self.assert_reference(x)

    def test_shape_dtype_validation(self):
        for name, value in (
            ("invalid", torch.zeros((3, 1), dtype=torch.bool)),
            ("invalid", torch.zeros((3, 128), dtype=torch.int64)),
            ("context_lens", torch.zeros((3, 1), dtype=torch.int64)),
            ("offsets", torch.zeros((3, 128), dtype=torch.float32)),
            ("out_loc", torch.zeros(1, dtype=torch.int64)),
            ("full_to_swa_mapping", torch.zeros((1, 1), dtype=torch.int64)),
            ("block_size", 0),
            ("page_index_aligned_size", 0),
        ):
            x = inputs()
            x[name] = value
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                self.run_impl(x)
        x = inputs()
        x["req_to_token"] = torch.empty(x["req_to_token"].shape, dtype=torch.int64, device="meta")
        with self.assertRaisesRegex(ValueError, "device"):
            self.run_impl(x)

    def test_capture_branch_never_reads_host_values(self):
        x = inputs((0, 1, 128))
        x["out_loc"][0] = -1
        def forbidden(*args, **kwargs):
            raise AssertionError("host tensor read in capture path")
        with patch.object(torch.Tensor, "__bool__", forbidden), \
             patch.object(torch.Tensor, "item", forbidden), \
             patch.object(torch.Tensor, "cpu", forbidden), \
             patch.object(torch.Tensor, "tolist", forbidden), \
             patch.object(torch.Tensor, "numpy", forbidden):
            self.run_impl(x, False)


if __name__ == "__main__":
    unittest.main()
