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
"""Hardware-free tests for the DSV4 attention-metadata dispatch hook."""

import sys
import types
import unittest
from unittest import mock

from sglang_kunlun.hooks.kernels import dsv4_attn_metadata as hook

NEW_LAYOUT = "sglang.kernels.ops.attention.dsv4.attn_metadata_kernels"
OLD_LAYOUT = "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels"


class ForceTorchDispatchTest(unittest.TestCase):
    def test_reports_not_cuda_even_when_original_says_cuda(self):
        """A Kunlun tensor reports is_cuda True; the verdict must still be False."""
        original = mock.Mock(return_value=True)

        result = hook.force_torch_metadata_dispatch(original, "a", k=1)

        self.assertIs(result, False)
        original.assert_called_once_with("a", k=1)

    def test_reports_not_cuda_when_original_says_not_cuda(self):
        self.assertIs(
            hook.force_torch_metadata_dispatch(mock.Mock(return_value=False)),
            False,
        )

    def test_preserves_upstream_validation(self):
        """Upstream asserts on a call with no tensor argument; keep that."""
        boom = mock.Mock(side_effect=AssertionError("no tensor argument"))

        with self.assertRaises(AssertionError):
            hook.force_torch_metadata_dispatch(boom, 1, 2)


class ResolveModuleTest(unittest.TestCase):
    def test_candidates_cover_both_known_layouts(self):
        self.assertEqual(hook._CANDIDATE_MODULES, (NEW_LAYOUT, OLD_LAYOUT))

    def test_prefers_an_already_imported_module(self):
        with mock.patch.dict(
            sys.modules, {OLD_LAYOUT: types.ModuleType(OLD_LAYOUT)}, clear=False
        ):
            sys.modules.pop(NEW_LAYOUT, None)
            self.assertEqual(hook._resolve_module(), OLD_LAYOUT)

    def test_prefers_the_newer_layout_when_both_are_present(self):
        with mock.patch.dict(
            sys.modules,
            {
                NEW_LAYOUT: types.ModuleType(NEW_LAYOUT),
                OLD_LAYOUT: types.ModuleType(OLD_LAYOUT),
            },
            clear=False,
        ):
            self.assertEqual(hook._resolve_module(), NEW_LAYOUT)

    def test_returns_none_when_no_candidate_exists(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in hook._CANDIDATE_MODULES:
                sys.modules.pop(name, None)
            with mock.patch.object(hook.importlib.util, "find_spec", return_value=None):
                self.assertIsNone(hook._resolve_module())

    def test_survives_an_unimportable_candidate(self):
        """An SGLang build without the DSV4 kernels must not break startup."""
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in hook._CANDIDATE_MODULES:
                sys.modules.pop(name, None)
            with mock.patch.object(
                hook.importlib.util,
                "find_spec",
                side_effect=ImportError("no triton"),
            ):
                self.assertIsNone(hook._resolve_module())


if __name__ == "__main__":
    unittest.main()
