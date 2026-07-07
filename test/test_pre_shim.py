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


if __name__ == "__main__":
    unittest.main()
