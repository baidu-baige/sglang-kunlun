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
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaOnlyContractTest(unittest.TestCase):
    def test_runtime_code_does_not_call_torch_xpu(self):
        offenders = []
        for path in (ROOT / "sglang_kunlun").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or node.attr != "xpu":
                    continue
                value = node.value
                if isinstance(value, ast.Name) and value.id == "torch":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(value, ast.Name) and value.id == "_torch":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], "torch.xpu is forbidden on Kunlun: " + ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
