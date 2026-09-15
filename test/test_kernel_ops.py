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
import sys
import types
import unittest
from unittest import mock


class KernelOpsTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("sglang.fake_kernel_source", None)
        sys.modules.pop("sglang.fake_kernel_user", None)

    def test_install_patches_imported_bindings_automatically(self):
        from sglang_kunlun.kernels import kernel_ops

        def original_kernel(*args, **kwargs):
            return "original"

        source_module = types.ModuleType("sglang.fake_kernel_source")
        source_module.fake_kernel = original_kernel
        user_module = types.ModuleType("sglang.fake_kernel_user")
        user_module.fake_kernel = original_kernel
        sys.modules[source_module.__name__] = source_module
        sys.modules[user_module.__name__] = user_module

        def replacement_kernel(*args, **kwargs):
            return "replacement"

        spec = kernel_ops.KernelSpec(source_module.__name__, "fake_kernel", replacement_kernel)
        with mock.patch.object(kernel_ops, "_TRITON_OPS", {(source_module.__name__, "fake_kernel"): spec}), mock.patch.object(
            kernel_ops, "_JIT_OPS", {}
        ):
            kernel_ops.install()

        self.assertIs(source_module.fake_kernel.impl, replacement_kernel)
        self.assertIs(user_module.fake_kernel.impl, replacement_kernel)
        self.assertEqual(user_module.fake_kernel[lambda meta: (1,)](), "replacement")

    def test_direct_call_style_patches_function_without_launcher(self):
        from sglang_kunlun.kernels import kernel_ops

        def original_kernel():
            return "original"

        source_module = types.ModuleType("sglang.fake_kernel_source")
        source_module.fake_kernel = original_kernel
        sys.modules[source_module.__name__] = source_module

        def replacement_kernel():
            return "replacement"

        spec = kernel_ops.KernelSpec(
            source_module.__name__,
            "fake_kernel",
            replacement_kernel,
            {"call_style": "direct"},
        )
        with mock.patch.object(kernel_ops, "_TRITON_OPS", {(source_module.__name__, "fake_kernel"): spec}), mock.patch.object(
            kernel_ops, "_JIT_OPS", {}
        ):
            kernel_ops.install()

        self.assertIs(source_module.fake_kernel, replacement_kernel)
        self.assertEqual(source_module.fake_kernel(), "replacement")


if __name__ == "__main__":
    unittest.main()
