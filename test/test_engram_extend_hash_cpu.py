"""Independent scalar oracle for the eager extend entry; no accelerator imports.

The fixture and arithmetic helper are read from the designated upstream checkout.
Only their function ASTs are loaded, so importing this test cannot start plugins.
"""

import ast
import inspect
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/home/ljj/dsv4_1_v0522/code_sglang_0522/sglang")
SOURCE = ROOT / "sglang_kunlun/kernels/dsv41_low_ratio.py"
TARGET = "sglang.kernels.ops.embeddings.engram_hash"


def load_functions(path, names, namespace=None, cpu=False):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
        if cpu:
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and child.value == "cuda":
                    child.value = "cpu"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"torch": torch} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[future] + nodes, type_ignores=[])
    ), str(path), "exec"), ns)
    return ns


fixture = load_functions(
    UPSTREAM / "test/manual/quantization/test_engram_extend_hash.py",
    {"fixture", "clone_inputs"}, cpu=True,
)
production = load_functions(
    SOURCE, {"engram_hash_ids_kunlun", "engram_hash_extend_and_commit_kunlun"}
)
arithmetic = ModuleType("sglang.srt.layers.engram")
arithmetic.compute_engram_hash_ids = load_functions(
    UPSTREAM / "python/sglang/srt/layers/engram.py", {"compute_engram_hash_ids"}
)["compute_engram_hash_ids"]
candidate = production["engram_hash_extend_and_commit_kunlun"]


def oracle(kw):
    """Python integer hash and list history update, independent of torch hash."""
    ids = kw["input_ids"].tolist()
    positions = kw["positions"].tolist()
    old = kw["history"].tolist()
    committed = kw["commit_history"].clone()
    lens, starts, slots = (kw[k].tolist() for k in ("lengths", "starts", "req_slots"))
    mult, primes, offsets = (kw[k].tolist() for k in ("multipliers", "primes", "offsets"))
    token_map = kw["token_map"].tolist()
    n, layers, heads = len(mult[0]), len(mult), len(primes[0][0])
    rows = [r for r, length in enumerate(lens) for _ in range(length)]
    out, predecessors = [], []
    for t in range(len(ids)):
        real = t < kw["num_real"]
        r = rows[t] if real else 0
        off = t - starts[r] if real else 0
        tokens, compressed, blocked = [], [], False
        for s in range(n):
            if not real:
                tok = 0
            elif s <= off:
                tok = ids[t - s]
            else:
                hrow = slots[r] if kw["history_via_slots"] else r
                tok = old[hrow][n - 1 + off - s]
            if kw["image_token_id"] is not None and tok >= kw["mm_pad_shift"]:
                tok = kw["image_token_id"]
            blocked |= (not real or positions[t] < s or
                        (kw["image_token_id"] is not None and tok == kw["image_token_id"]))
            tokens.append(tok)
            compressed.append(kw["pad_id"] if blocked else token_map[tok])
        predecessors.append(tokens)
        layer_hashes = []
        for layer in range(layers):
            rolling, values = 0, []
            for s in range(n):
                # Match signed int64 multiply/XOR even on overflow.
                product = (compressed[s] * mult[layer][s]) & ((1 << 64) - 1)
                rolling ^= product
                signed = rolling if rolling < (1 << 63) else rolling - (1 << 64)
                if s:
                    values.extend(
                        signed % primes[layer][s - 1][h] + offsets[layer][(s - 1) * heads + h]
                        for h in range(heads)
                    )
            layer_hashes.append(values)
        out.append(layer_hashes)
    if ids:
        for r, length in enumerate(lens):
            last = min(max(starts[r] + length - 1, 0), len(ids) - 1)
            if length and (kw["out_cache_loc"] is None or kw["out_cache_loc"][last] != 0):
                committed[slots[r]] = torch.tensor(predecessors[last][:n - 1][::-1])
    return torch.tensor(out, dtype=torch.int64).reshape(len(ids), layers, (n - 1) * heads), committed


class ExtendHashCPU(unittest.TestCase):
    def check_case(self, lengths, **options):
        kw = fixture["fixture"](lengths, **options)
        if kw["num_real"] > 1:
            kw["out_cache_loc"][kw["num_real"] - 1] = 0
        for _ in range(2):
            expected, history = oracle(kw)
            before = kw["history"].clone()
            with patch.dict("sys.modules", {arithmetic.__name__: arithmetic}):
                actual = candidate(**kw)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(kw["commit_history"], history, rtol=0, atol=0)
            if not kw["history_via_slots"]:
                torch.testing.assert_close(kw["history"], before, rtol=0, atol=0)
            kw["out_cache_loc"].fill_(1)

    def test_upstream_matrix_and_empty_batch(self):
        for lengths in ([8192], [1], [2], [0, 3, 0, 5, 0], [3, 1, 7],
                        [1] * 64, [0, 0], []):
            for explicit_history in (False, True):
                for image in (False, True):
                    with self.subTest(lengths=lengths, explicit=explicit_history, image=image):
                        self.check_case(lengths, padding=7,
                                        explicit_history=explicit_history, image=image)

    def test_no_tokens(self):
        for lengths in ([], [0], [0, 0]):
            self.check_case(lengths, explicit_history=True)

    def test_prefill_boundaries(self):
        for length in (8, 129, 16401):
            self.check_case([length], padding=3, image=True)

    def test_int64_metadata_no_mask_and_overflow(self):
        for explicit in (False, True):
            kw = fixture["fixture"]([5, 0, 9], dtype=torch.int64, explicit_history=explicit)
            kw["out_cache_loc"] = None
            kw["multipliers"].fill_(2**62 + 1)
            expected, history = oracle(kw)
            with patch.dict("sys.modules", {arithmetic.__name__: arithmetic}):
                actual = candidate(**kw)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(kw["commit_history"], history, rtol=0, atol=0)

    def test_inactive_slots_and_hash_failure(self):
        kw = fixture["fixture"]([0, 3, 0], padding=7)
        # Use int64 to retain the deliberately invalid inactive slot.
        kw["req_slots"] = torch.tensor([-1, 1, 2**40], dtype=torch.int64)
        expected, history = oracle(kw)
        with patch.dict("sys.modules", {arithmetic.__name__: arithmetic}):
            actual = candidate(**kw)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(kw["commit_history"], history, rtol=0, atol=0)
        before = kw["commit_history"].clone()
        with patch.dict(candidate.__globals__, engram_hash_ids_kunlun=unittest.mock.Mock(
            side_effect=RuntimeError("hash failed")
        )):
            with self.assertRaisesRegex(RuntimeError, "hash failed"):
                candidate(**kw)
        torch.testing.assert_close(kw["commit_history"], before, rtol=0, atol=0)

    def test_signature_and_single_registration(self):
        upstream = load_functions(
            UPSTREAM / "python/sglang/kernels/ops/embeddings/engram_hash.py",
            {"engram_hash_extend_and_commit"},
        )["engram_hash_extend_and_commit"]
        self.assertEqual(inspect.signature(candidate), inspect.signature(upstream))
        registrations = []
        for path in (ROOT / "sglang_kunlun").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "register_jit_op" and len(node.args) >= 2:
                        args = node.args[:2]
                        if all(isinstance(a, ast.Constant) for a in args):
                            if [a.value for a in args] == [TARGET, "engram_hash_extend_and_commit"]:
                                registrations.append(path)
        self.assertEqual(registrations, [SOURCE])


if __name__ == "__main__":
    unittest.main()
