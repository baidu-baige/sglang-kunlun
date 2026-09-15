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
import importlib
import sys
import unittest
from unittest import mock


class KunlunPreShimTest(unittest.TestCase):
    """Validate Kunlun bootstrap compatibility patches."""

    def test_pre_shim_forces_triton_backend_and_stubs_cuda(self):
        """The pre-shim exposes CUDA-compatible device metadata."""
        from sglang_kunlun.bootstrap import _kunlun_pre_shim

        try:
            sys.modules["torch_xmlir"] = mock.Mock()
            _kunlun_pre_shim()

            import triton.runtime.driver as driver_config

            self.assertEqual(driver_config.active.get_current_target().backend, "cuda")

            import torch

            self.assertEqual(torch.cuda.get_device_name(0), "Kunlun-XPU")
            self.assertEqual(torch.cuda.get_device_capability(0), (8, 0))
        finally:
            sys.modules.pop("torch_xmlir", None)

    def test_platform_import_runs_pre_shim_before_srt_import(self):
        """Platform import runs the pre-shim before importing SGLang SRT."""
        sys.modules.pop("sglang_kunlun.platform", None)
        with mock.patch("sglang_kunlun._kunlun_pre_shim") as pre_shim:
            importlib.import_module("sglang_kunlun.platform")

        pre_shim.assert_called_once()

    # def test_float16_kv_patch_is_available_to_general_plugin(self):
    #     """The general plugin installs the float16 KV compatibility patch."""
    #     import argparse
    #     import types

    #     from sglang_kunlun.bootstrap.pre_shim import patch_float16_kv_cache

    #     class FakeServerArgs:
    #         """Provide the ServerArgs methods patched by the plugin."""

    #         @staticmethod
    #         def add_cli_args(parser):
    #             """Register the baseline KV cache argument choices."""
    #             parser.add_argument("--kv-cache-dtype", choices=["auto", "bf16"])

    #         def _set_default_dsa_kv_cache_dtype(self, major, quantization):
    #             """Record use of the unpatched DSA dtype implementation."""
    #             self.kv_cache_dtype = "original"

    #     class FakeModelRunner:
    #         """Provide the model runner method patched by the plugin."""

    #         def configure_kv_cache_dtype(self):
    #             """Record use of the unpatched model runner implementation."""
    #             self.kv_cache_dtype = "original"

    #     seen_dtypes = []
    #     raise_from_defaults = [False]

    #     def apply_deepseek_v4_defaults(server_args, model_arch):
    #         """Capture the dtype observed by the original defaults hook."""
    #         seen_dtypes.append(server_args.kv_cache_dtype)
    #         if raise_from_defaults[0]:
    #             raise RuntimeError("defaults failed")

    #     fake_torch = types.ModuleType("torch")
    #     fake_torch.float16 = object()
    #     server_args_module = types.ModuleType("sglang.srt.server_args")
    #     server_args_module.ServerArgs = FakeServerArgs
    #     model_runner_module = types.ModuleType("sglang.srt.model_executor.model_runner")
    #     model_runner_module.ModelRunner = FakeModelRunner
    #     model_executor_module = types.ModuleType("sglang.srt.model_executor")
    #     model_executor_module.model_runner = model_runner_module
    #     deepseek_v4_module = types.ModuleType("sglang.srt.arg_groups.deepseek_v4_hook")
    #     deepseek_v4_module.apply_deepseek_v4_defaults = apply_deepseek_v4_defaults
    #     arg_groups_module = types.ModuleType("sglang.srt.arg_groups")
    #     arg_groups_module.deepseek_v4_hook = deepseek_v4_module

    #     modules = {
    #         "torch": fake_torch,
    #         "sglang.srt.server_args": server_args_module,
    #         "sglang.srt.model_executor": model_executor_module,
    #         "sglang.srt.model_executor.model_runner": model_runner_module,
    #         "sglang.srt.arg_groups": arg_groups_module,
    #         "sglang.srt.arg_groups.deepseek_v4_hook": deepseek_v4_module,
    #     }
    #     with mock.patch.dict(sys.modules, modules):
    #         import concurrent.futures

    #         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #             list(executor.map(lambda _: patch_float16_kv_cache(), range(8)))
    #         patch_float16_kv_cache()

    #     parser = argparse.ArgumentParser()
    #     FakeServerArgs.add_cli_args(parser)
    #     action = next(a for a in parser._actions if a.dest == "kv_cache_dtype")
    #     self.assertEqual(action.choices, ["auto", "bf16", "float16", "half"])

    #     server_args = FakeServerArgs()
    #     server_args.kv_cache_dtype = "half"
    #     server_args._set_default_dsa_kv_cache_dtype(8, None)
    #     self.assertEqual(server_args.kv_cache_dtype, "float16")

    #     defaults_args = types.SimpleNamespace(kv_cache_dtype="float16")
    #     deepseek_v4_module.apply_deepseek_v4_defaults(defaults_args, "DeepseekV4")
    #     self.assertEqual(seen_dtypes, ["bfloat16"])
    #     self.assertEqual(defaults_args.kv_cache_dtype, "float16")

    #     raise_from_defaults[0] = True
    #     defaults_args.kv_cache_dtype = "half"
    #     with self.assertRaisesRegex(RuntimeError, "defaults failed"):
    #         deepseek_v4_module.apply_deepseek_v4_defaults(defaults_args, "DeepseekV4")
    #     self.assertEqual(defaults_args.kv_cache_dtype, "half")

    #     runner = FakeModelRunner()
    #     runner.server_args = types.SimpleNamespace(kv_cache_dtype="float16")
    #     runner.configure_kv_cache_dtype()
    #     self.assertIs(runner.kv_cache_dtype, fake_torch.float16)


if __name__ == "__main__":
    unittest.main()
