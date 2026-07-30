import builtins
import importlib
import sys
import unittest
from unittest import mock


class KunlunPreShimTest(unittest.TestCase):
    def test_pre_shim_forces_triton_backend_and_stubs_cuda(self):
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
        sys.modules.pop("sglang_kunlun.platform", None)
        with mock.patch("sglang_kunlun._kunlun_pre_shim") as pre_shim:
            importlib.import_module("sglang_kunlun.platform")

        pre_shim.assert_called_once()

    def test_tilelang_shared_library_failure_becomes_optional_import(self):
        from sglang_kunlun.bootstrap.pre_shim import (
            _patch_optional_tilelang_import,
        )

        original_import = builtins.__import__

        def import_with_broken_tilelang(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tilelang" or name.startswith("tilelang."):
                raise OSError("libz3.so.4.15: cannot open shared object file")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_with_broken_tilelang
        try:
            _patch_optional_tilelang_import()
            with self.assertRaises(ModuleNotFoundError):
                builtins.__import__("tilelang")
            self.assertIs(builtins.__import__("sys"), sys)
        finally:
            builtins.__import__ = original_import


if __name__ == "__main__":
    unittest.main()
