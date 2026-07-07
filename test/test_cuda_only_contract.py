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
