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


class HookRegistrationTest(unittest.TestCase):
    def test_candidates_cover_both_known_layouts(self):
        self.assertEqual(hook._CANDIDATE_MODULES, (NEW_LAYOUT, OLD_LAYOUT))

    def test_registers_a_module_the_build_ships(self):
        with (
            mock.patch.object(hook.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(hook, "plugin_hook") as plugin_hook,
        ):
            self.assertIs(hook._hook_if_present(NEW_LAYOUT), True)

        plugin_hook.assert_called_once_with(
            f"{NEW_LAYOUT}._inputs_on_cuda", type=hook.HookType.AROUND
        )
        plugin_hook.return_value.assert_called_once_with(
            hook.force_torch_metadata_dispatch
        )

    def test_skips_a_module_the_build_lacks(self):
        """Registering an absent path would log a traceback at every start."""
        with (
            mock.patch.object(hook.importlib.util, "find_spec", return_value=None),
            mock.patch.object(hook, "plugin_hook") as plugin_hook,
        ):
            self.assertIs(hook._hook_if_present(OLD_LAYOUT), False)

        plugin_hook.assert_not_called()

    def test_survives_an_unimportable_candidate(self):
        """An SGLang build without the DSV4 kernels must not break startup."""
        with (
            mock.patch.object(
                hook.importlib.util, "find_spec", side_effect=ImportError("no triton")
            ),
            mock.patch.object(hook, "plugin_hook") as plugin_hook,
        ):
            self.assertIs(hook._hook_if_present(OLD_LAYOUT), False)

        plugin_hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
