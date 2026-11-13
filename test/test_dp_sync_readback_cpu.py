"""The fast DP-sync readback returns upstream's values, with upstream's types.

Loads two things by AST, so importing this test starts no plugin and needs no
accelerator: the plugin's ``read_sync_fields`` and the eleven statements
upstream writes in ``MLPSyncBatchInfo.all_gather``. The oracle is upstream's
own text, re-executed, so a column index or a reduction that changes on either
side fails here rather than in a scheduler.
"""

import ast
import hashlib
from pathlib import Path
import re
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "sglang_kunlun/hooks/managers/scheduler_components/dp_attn.py"
UPSTREAM = (
    ROOT.parent / "sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py"
)

# dp_attn.py:288-300 -- the statements the hook replaces, as the oracle.
ORACLE_SOURCE = """
def oracle(tp0_info_cpu):
    return (
        tp0_info_cpu[:, 0].tolist(),
        tp0_info_cpu[:, 1].tolist(),
        bool(tp0_info_cpu[:, 2].min()),
        bool(tp0_info_cpu[:, 3].max()),
        tp0_info_cpu[:, 5].tolist(),
        bool(tp0_info_cpu[:, 6].min()),
        int(tp0_info_cpu[:, 7].max()),
        bool(tp0_info_cpu[:, 8].min()),
        bool(tp0_info_cpu[:, 9].max()),
        tp0_info_cpu[:, 10].tolist(),
    )
"""

FIELDS = (
    "global_num_tokens",
    "global_num_tokens_for_logprob",
    "can_run_decode_cuda_graph",
    "is_extend_in_batch",
    "forward_modes",
    "can_run_prefill_cuda_graph",
    "prefill_cuda_graph_max_prefix_len",
    "can_run_draft_cuda_graph",
    "any_health_check",
    "global_attention_pairs",
)
def load_function(path, name):
    """Execute one top-level function from *path*, without importing its module."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"{name} not found exactly once in {path}"
    nodes[0].decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future] + nodes, type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[name]


candidate = load_function(PLUGIN, "read_sync_fields")
_oracle_ns = {}
exec(compile(ORACLE_SOURCE, "<oracle>", "exec"), _oracle_ns)
oracle = _oracle_ns["oracle"]


def make_block(dp_size, attn_tp, seed, low=0, high=4096):
    """``global_info_tensor.cpu()[:, 0, :]`` at the real shape and dtype."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randint(
        low, high, (dp_size, attn_tp, 11), dtype=torch.int64, generator=g
    )
    for col in (2, 3, 6, 8, 9):
        full[:, :, col] = torch.randint(
            0, 2, (dp_size, attn_tp), dtype=torch.int64, generator=g
        )
    return full[:, 0, :]
class TestDPSyncReadbackValues(unittest.TestCase):
    def assert_same(self, block):
        got = candidate(block)
        want = oracle(block)
        self.assertEqual(len(got), len(want))
        for name, g, w in zip(FIELDS, got, want):
            with self.subTest(field=name):
                self.assertEqual(g, w)
                self.assertIs(type(g), type(w))
                if isinstance(w, list):
                    self.assertTrue(all(type(x) is int for x in g))

    def test_random_shapes(self):
        for dp_size in (1, 2, 8, 32):
            for attn_tp in (1, 4, 8):
                for seed in range(4):
                    with self.subTest(dp=dp_size, tp=attn_tp, seed=seed):
                        self.assert_same(make_block(dp_size, attn_tp, seed))

    def test_single_row(self):
        self.assert_same(make_block(1, 1, 7))

    def test_zero_column(self):
        for col in range(11):
            block = make_block(8, 4, 11).clone()
            block[:, col] = 0
            with self.subTest(zeroed=col):
                self.assert_same(block)

    def test_one_zero_entry_per_column(self):
        """A single 0 flips every min-reduced graph vote."""
        for col in range(11):
            for row in (0, 3, 7):
                block = make_block(8, 1, 13).clone()
                block[:, col] = 5
                block[row, col] = 0
                with self.subTest(col=col, row=row):
                    self.assert_same(block)

    def test_negative_values(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self.assert_same(make_block(8, 4, 100 + seed, low=-4096, high=4096))

    def test_negative_min_is_truthy(self):
        """``bool(min())`` is not ``all()``: a column of 0 and -1 has a truthy
        min and a falsy ``all``, and the hook must follow upstream's min."""
        block = make_block(8, 1, 17).clone()
        block[:, 2] = 0
        block[0, 2] = -1
        self.assertTrue(oracle(block)[2])
        self.assertFalse(all(block[:, 2].tolist()))
        self.assert_same(block)

    def test_int32_max_prefix_len(self):
        block = make_block(8, 4, 23).clone()
        block[:, 7] = torch.iinfo(torch.int32).max
        block[:, 0] = 0
        self.assert_same(block)
def upstream_all_gather_source():
    tree = ast.parse(UPSTREAM.read_text())
    (cls,) = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MLPSyncBatchInfo"
    ]
    (fn,) = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "all_gather"
    ]
    lines = UPSTREAM.read_text().splitlines()
    return "\n".join(lines[fn.lineno - 1 : fn.end_lineno])


def normalized_fingerprint(src):
    body = "\n".join(line.rstrip() for line in src.splitlines()).strip()
    return hashlib.sha1(body.encode()).hexdigest()[:16]


class TestDPSyncReadbackContract(unittest.TestCase):
    """The fork is only safe while upstream's body is the forked one."""

    def test_fingerprint_matches_upstream(self):
        recorded = re.search(
            r'_UPSTREAM_FINGERPRINT = "([0-9a-f]+)"', PLUGIN.read_text()
        )
        self.assertIsNotNone(recorded, "fingerprint constant missing")
        self.assertEqual(
            recorded.group(1),
            normalized_fingerprint(upstream_all_gather_source()),
            "upstream all_gather changed: re-port the hook, then update "
            "_UPSTREAM_FINGERPRINT",
        )

    def test_oracle_is_upstream_text(self):
        """Every statement the oracle models is still in upstream's body."""
        upstream = upstream_all_gather_source()
        for expr in (
            "self.global_num_tokens = tp0_info_cpu[:, 0].tolist()",
            "self.global_num_tokens_for_logprob = tp0_info_cpu[:, 1].tolist()",
            "self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())",
            "self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())",
            "self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())",
            "self.prefill_cuda_graph_max_prefix_len = int(tp0_info_cpu[:, 7].max())",
            "self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 8].min())",
            "self.any_health_check = bool(tp0_info_cpu[:, 9].max())",
            "self.global_attention_pairs = tp0_info_cpu[:, 10].tolist()",
            "tp0_info_cpu[:, 5].tolist()",
            "self.tp0_info_cpu = tp0_info_cpu",
        ):
            with self.subTest(statement=expr):
                self.assertIn(expr, upstream)
    def test_target_hooked_once(self):
        """Two hooks on one target are forbidden by the registry."""
        target = (
            "sglang.srt.managers.scheduler_components.dp_attn."
            "MLPSyncBatchInfo.all_gather"
        )
        hits = [
            path
            for path in (ROOT / "sglang_kunlun").rglob("*.py")
            if "MLPSyncBatchInfo.all_gather" in path.read_text()
        ]
        self.assertEqual([p.name for p in hits], [PLUGIN.name])
        self.assertIn(target.split("dp_attn.")[1], PLUGIN.read_text())

    def test_tail_assignments_cover_every_field(self):
        """The hook assigns all ten values, in ``read_sync_fields``' order."""
        plugin_src = PLUGIN.read_text()
        attrs = [f for f in FIELDS if f != "forward_modes"]
        for attr in attrs:
            with self.subTest(attr=attr):
                self.assertIn(f"self.{attr},", plugin_src)
        order = [plugin_src.index(f"self.{a},") for a in attrs]
        self.assertEqual(order, sorted(order))


if __name__ == "__main__":
    unittest.main(verbosity=2)




