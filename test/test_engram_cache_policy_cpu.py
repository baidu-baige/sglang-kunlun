"""The launcher's cache-eviction opt-out must reach the host-table helper."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


class EngramCachePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / (
            "sglang_kunlun/hooks/models/dsv41_engram_int8.py"
        )
        node = next(
            n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name == "drop_checkpoint_page_cache_kunlun"
        )
        node.decorator_list = []
        namespace = {"os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls.invoke = staticmethod(namespace[node.name])

    def test_explicit_opt_out_never_evicts_checkpoint(self):
        for value in ("0", "false", "False"):
            callback = Mock(side_effect=AssertionError("must not evict"))
            with patch.dict(os.environ, {"SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE": value}):
                self.assertEqual(self.invoke(callback), (0, 0))
            callback.assert_not_called()

    def test_default_and_opt_in_preserve_upstream_result(self):
        key = "SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE"
        for value in (None, "1", "true"):
            callback = Mock(return_value=(8, 1024))
            with patch.dict(os.environ):
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
                self.assertEqual(self.invoke(callback), (8, 1024))
            callback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
