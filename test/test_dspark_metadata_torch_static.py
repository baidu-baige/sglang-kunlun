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
"""Standard-library-only AST/contract tests; intentionally not executed here."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1] / "sglang_kunlun"


class MetadataTorchStaticTest(unittest.TestCase):
    def test_safe_take_does_not_append_source_and_handles_empty_before_gather(self):
        tree = ast.parse((ROOT / "kernels/dspark_metadata_torch.py").read_text())
        take = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_safe_take")
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ("cat", "contiguous", "clone")
                             for n in ast.walk(take)))
        empty = next(n for n in take.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(empty.test), "size == 0")
        self.assertIsInstance(empty.body[-1], ast.Return)
        self.assertFalse(any(isinstance(n, ast.Subscript) for n in ast.walk(empty)))
        check_position = next(i for i, n in enumerate(take.body)
                              if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                              and isinstance(n.value.func, ast.Name)
                              and n.value.func.id == "_check_active")
        self.assertLess(check_position, take.body.index(empty))
        source = ast.unparse(take)
        self.assertIn("safe = torch.where(valid, indices, 0)", source)
        self.assertIn("return (torch.where(valid, selected, -1), valid)", source)

    def test_single_replace_and_classmethod_forwarding(self):
        path = ROOT / "hooks/speculative/dspark_metadata_torch.py"
        tree = ast.parse(path.read_text())
        registrations, calls = [], []
        def register(target, type):
            def decorate(fn):
                registrations.append((target, type, fn))
                return fn
            return decorate
        ns = {
            "plugin_hook": register,
            "HookType": SimpleNamespace(REPLACE="replace"),
            "build_dspark_swa_page_indices": lambda *a, **k: calls.append((a, k)),
        }
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), ns)
        self.assertEqual(len(registrations), 1)
        target, kind, fn = registrations[0]
        self.assertTrue(target.endswith("BuildDsparkSwaPageIndices.execute"))
        self.assertEqual(kind, "replace")
        class Owner:
            execute = classmethod(fn)
        Owner.execute(offsets="sentinel")
        self.assertEqual(calls, [((), {"offsets": "sentinel"})])
        init = (ROOT / "hooks/speculative/__init__.py").read_text()
        self.assertIn("from . import dspark_metadata_torch", init)

    def test_capture_guard_is_conservative_and_independent_of_diagnostics(self):
        tree = ast.parse((ROOT / "kernels/dspark_metadata_torch.py").read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_can_validate_values")
        guard.body = [n for n in guard.body if not isinstance(n, ast.ImportFrom)]
        for framework in (False, True):
            for answer in (False, True, None, 0, RuntimeError("unknown")):
                def query():
                    if isinstance(answer, Exception):
                        raise answer
                    return answer
                ns = {"is_graph_capturing": lambda: framework,
                      "torch": SimpleNamespace(cuda=SimpleNamespace(
                          is_current_stream_capturing=query))}
                exec(compile(ast.Module(body=[guard], type_ignores=[]), "<guard>", "exec"), ns)
                self.assertEqual(ns["_can_validate_values"](SimpleNamespace(type="cuda")),
                                 not framework and answer is False)
        source = ast.unparse(tree)
        self.assertNotIn("environ", source)
        self.assertNotIn("triton", source)
        for name in ("item", "cpu", "tolist", "numpy", "synchronize", "nonzero"):
            self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == name
                                 for n in ast.walk(tree)), name)
        bool_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "bool"]
        self.assertEqual(len(bool_calls), 1)
        check = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_check_active")
        self.assertIn("validate and bool", ast.unparse(check))


if __name__ == "__main__":
    unittest.main()
