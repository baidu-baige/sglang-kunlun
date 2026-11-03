"""CPU contracts for the 0.5.22 migration, without importing accelerator plugins."""

import ast
import logging
import math
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "sglang_kunlun/kernels/kernel_ops.py"
LOW = ROOT / "sglang_kunlun/kernels/dsv41_low_ratio.py"
BACKEND = ROOT / "sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py"


def functions(path, *names, **extra):
    """Execute actual function bodies with only their CPU dependencies."""
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            found[node.name] = node
    assert set(found) == set(names), set(names) - set(found)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + list(found.values()), type_ignores=[]
    )
    ns = {"torch": torch, "math": math, "os": os, "logger": logging.getLogger(__name__)}
    ns.update(extra)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return types.SimpleNamespace(**ns)


class MigrationCPUContracts(unittest.TestCase):
    def test_swa_rounding_and_clear_fuzz(self):
        ns = functions(KERNEL, "_trunc_div", "get_and_clear_swa_pages_kernel_kunlun")
        rng = torch.Generator().manual_seed(941)
        for _ in range(250):
            mapping = torch.randint(-1, 33, (32,), generator=rng, dtype=torch.int64)
            old = mapping.clone()
            reps = torch.tensor([1, 5, 13, 25])
            pages = torch.full((6,), -99, dtype=torch.int64)
            peers = torch.zeros(6, dtype=torch.bool)
            valid = torch.ones(6, dtype=torch.bool)
            ns.get_and_clear_swa_pages_kernel_kunlun(
                reps, mapping, pages, peers, valid, 4, 1, 4, True, 32, 4
            )
            for i, rep in enumerate(reps.tolist()):
                self.assertEqual(pages[i].item(), math.trunc(old[rep].item() / 4))
                expected = old[rep] > 0 and all(
                    value <= 0 or math.trunc(value / 4) == math.trunc(old[rep].item() / 4)
                    for value in old[rep // 4 * 4:rep // 4 * 4 + 4].tolist()
                )
                self.assertEqual(valid[i].item(), bool(expected))
                self.assertTrue(torch.all(mapping[rep // 4 * 4:rep // 4 * 4 + 4] == 0))
            self.assertEqual(pages[4:].tolist(), [-99, -99])

    def test_engram_history_long_commit_and_untouched_rows(self):
        fn = functions(LOW, "_engram_commit_history_torch")._engram_commit_history_torch
        history = torch.arange(15, dtype=torch.int32).reshape(5, 3)
        original = history.clone()
        verify = torch.arange(18, dtype=torch.int64).reshape(3, 6) + 100
        slots, lengths = torch.tensor([3, 0, 2]), torch.tensor([0, 4, 6])
        fn(history, verify, slots, lengths)
        for i, (slot, length) in enumerate(zip(slots.tolist(), lengths.tolist())):
            expected = (original[slot].tolist() + verify[i].tolist())[length:length + 3]
            self.assertEqual(history[slot].tolist(), expected)
        torch.testing.assert_close(history[[1, 4]], original[[1, 4]])

    def test_e4m3_rounding_matches_cpu_dtype(self):
        ns = functions(LOW, "_fp32_pow2", "_round_to_e4m3_rne")
        values = torch.linspace(2**-9, 448, 10001, dtype=torch.float32)
        torch.testing.assert_close(
            ns._round_to_e4m3_rne(values), values.to(torch.float8_e4m3fn).float(),
            rtol=0, atol=0,
        )

    def test_padded_kv_pages_and_out_alias(self):
        ns = functions(
            KERNEL, "dsv4_dequantize_k_cache_paged_kunlun",
            _assert_v4_kv_layout=lambda *args: None, DSV4_KV_DIM=512,
        )
        cache = torch.arange(3 * 1536, dtype=torch.float32).reshape(3, 1536)
        ids = torch.tensor([0, 3, 4])
        expected = torch.stack([cache[0, :512], cache[1, 512:1024], cache[2, :512]])[:, None]
        backing = torch.full((5, 1, 512), -1.0)
        out = backing[1:4]
        result = ns.dsv4_dequantize_k_cache_paged_kunlun(cache, ids, 2, out=out)
        self.assertIs(result, out)
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.all(backing[[0, 4]] == -1))

    def test_moe_presort_negative_routes(self):
        path = ROOT / "sglang_kunlun/hooks/layers/quantization/compressed_tensors/utils.py"
        fn = functions(path, "_moe_pre_sorted_torch")._moe_pre_sorted_torch
        x = torch.arange(12).reshape(4, 3)
        routes = torch.tensor([[2, -1], [0, 2], [1, 0], [-1, 1]])
        sorted_x, positions, lod = fn(x, routes, 3, index_have_neg=True)
        self.assertEqual(lod.tolist(), [0, 2, 4, 6])
        for lane, expert in enumerate(routes.flatten().tolist()):
            if expert < 0:
                self.assertEqual(positions[lane].item(), -1)
            else:
                row = positions[lane].item()
                self.assertTrue(lod[expert] <= row < lod[expert + 1])
                torch.testing.assert_close(sorted_x[row], x[lane // 2])

    def test_memfd_fallback(self):
        path = ROOT / "sglang_kunlun/compat.py"
        fn = functions(path, "memfd_create").memfd_create
        with mock.patch.object(os, "memfd_create", None, create=True):
            fd = fn("migration-cpu")
            try:
                os.write(fd, b"contract")
                os.lseek(fd, 0, os.SEEK_SET)
                self.assertEqual(os.read(fd, 8), b"contract")
            finally:
                os.close(fd)

    def test_overlay_restored_on_exception(self):
        path = ROOT / "sglang_kunlun/hooks/arg_groups/dsv41_relax.py"
        ns = functions(
            path, "_enabled", "_push_overlay", "_pop_overlay", "_run_relaxed",
            _MARKER_SOURCE="test", _RELAX_FIELDS={"dp_size": 1}, _LOGGED=set(),
        )
        previous = ("user", {"dp_size": 8})
        args = types.SimpleNamespace(_resolved_overrides=[previous])
        def fail(value):
            self.assertEqual(value._resolved_overrides[-1][1]["dp_size"], 1)
            raise ValueError("expected")
        with mock.patch.dict(os.environ, {"MIGRATION_TEST_RELAX": "1"}):
            with self.assertRaises(ValueError):
                ns._run_relaxed(fail, args, "MIGRATION_TEST_RELAX", "test")
        self.assertEqual(args._resolved_overrides, [previous])

    def test_spec_padding_lcm(self):
        path = ROOT / "sglang_kunlun/hooks/model_executor/forward_batch_info.py"
        context = types.ModuleType("sglang.srt.runtime_context")
        context.get_parallel = lambda: types.SimpleNamespace(attn_tp_size=8)
        fn = functions(path, "_align_global_num_tokens", _LOGGED_ALIGNMENT=True)._align_global_num_tokens
        batch = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(num_tokens_per_req=6),
            global_num_tokens_cpu=[6, 18, 192],
            forward_mode=types.SimpleNamespace(is_decode=lambda: True),
        )
        with mock.patch.dict("sys.modules", {"sglang.srt.runtime_context": context}):
            fn(batch)
        self.assertEqual(batch.global_num_tokens_cpu, [24, 24, 192])

    def test_protected_target_files_unchanged(self):
        for rel in (
            "sglang_kunlun/hooks/layers/cp/bcg.py",
            "sglang_kunlun/hooks/layers/attention/nsa/kunlun_dsa_indexer.py",
            "dockerfile/Dockerfile", "dockerfile/install.sh",
        ):
            expected = subprocess.check_output(["git", "show", f"1ce0288:{rel}"], cwd=ROOT)
            self.assertEqual((ROOT / rel).read_bytes(), expected, rel)

    def test_precision_file_only_gates_q_head_norm(self):
        """`deepseek_v4_precision.py` only allows this one change: the Q per-head norm switch.

        This file was originally in the "byte-for-byte unchanged" list. Dual-machine PD
        verification found that `compute_q_b_kunlun` unconditionally passes `self.eps` to
        `fused_q_norm_rope`, so V4.1 (`config.q_head_norm = False`) gets an extra per-head
        RMSNorm applied to Q every layer, pinning the q RMS to 1.0 and making attention
        logits uniformly small; on long prompts the model only repeats nearby content. Keep
        the drift protection here: apart from this one line (and the comment explaining it),
        the file must match 1ce0288 exactly.
        """
        import difflib

        rel = "sglang_kunlun/hooks/models/deepseek_v4_precision.py"
        base = subprocess.check_output(
            ["git", "show", f"1ce0288:{rel}"], cwd=ROOT
        ).decode()
        current = (ROOT / rel).read_text()
        diff = [
            line
            for line in difflib.unified_diff(
                base.splitlines(), current.splitlines(), lineterm="", n=0
            )
            if line[:1] in "+-" and not line.startswith(("---", "+++"))
        ]
        removed = [line[1:].strip() for line in diff if line.startswith("-")]
        added = [line[1:].strip() for line in diff if line.startswith("+")]
        self.assertEqual(
            removed,
            ["fused_q_norm_rope(q, local_q_out, self.eps, self.freqs_cis, positions)"],
        )
        self.assertEqual(
            [line for line in added if not line.startswith("#")],
            [
                'norm_eps = self.eps if getattr(self, "q_head_norm", False) else None',
                "fused_q_norm_rope(q, local_q_out, norm_eps, self.freqs_cis, positions)",
            ],
        )

    def test_bcg_live_rows_and_causality_preserved(self):
        text = BACKEND.read_text()
        self.assertIn("_dsv4_topk_torch_fallback(", text)
        self.assertIn("use_causal = all(int(positions[int(_i)]) < 3", text)
        self.assertNotIn(".core_attn_metadata", text)
        fn = functions(BACKEND, "_low_ratio_index_topk_sm90_decode_native")
        # The fallback must never synchronize a device tensor during capture.
        source = ast.get_source_segment(text, next(
            n for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.FunctionDef) and n.name == "_low_ratio_index_topk_sm90_decode_native"
        ))
        calls = [
            n.func.attr for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertNotIn("item", calls)
        self.assertTrue(callable(fn._low_ratio_index_topk_sm90_decode_native))

    def test_decode_topk_matches_dense_reference_with_partial_chunk(self):
        fn = functions(
            BACKEND, "_low_ratio_index_topk_sm90_decode_native",
            _low_ratio_index_bmm_enabled=lambda: True,
        )._low_ratio_index_topk_sm90_decode_native
        torch.manual_seed(42)
        q = torch.randn(2, 2, 4)
        keys = torch.randn(20, 4)
        weights = torch.randn(2, 2)
        output = torch.full((2, 3), -99, dtype=torch.int32)
        table = torch.arange(20).reshape(2, 10)
        indexer = types.SimpleNamespace(
            index_topk=3, is_candidate_source=False, uses_candidates=False,
            queries=lambda *args: q, head_weights=lambda x: weights,
        )
        layer = types.SimpleNamespace(
            compress_ratio=1, indexer=indexer, freqs_cis=torch.zeros(10), layer_id=0,
        )
        backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(
                core_metadata=types.SimpleNamespace(
                    sparse_page_indices=lambda r: output,
                    sparse_raw_indices=lambda r: None,
                ),
                c1_indexer_metadata=types.SimpleNamespace(max_compressed_seq_len=7),
            ),
            token_to_kv_pool=types.SimpleNamespace(
                get_low_ratio_index_k_dequant=lambda layer, slots: keys[slots.long()]
            ),
            req_to_token=table,
        )
        pos = torch.tensor([2, 6])
        with mock.patch.dict(os.environ, {"DSV4_LOW_RATIO_INDEX_CHUNK": "4"}):
            fn(backend, layer, torch.zeros(2, 1), None, torch.tensor([0, 1]), pos)
        for row, length in enumerate([3, 7]):
            scores = (q[row] @ keys[table[row, :length]].T).relu()
            scores = (scores * weights[row, :, None]).sum(0)
            expected = scores.topk(3).indices.sort().values + row * 10
            torch.testing.assert_close(output[row].long(), expected)

    def test_prefill_cache_rejects_inplace_position_updates(self):
        path = ROOT / "sglang_kunlun/hooks/layers/attention/dsv41_torch_indexer.py"
        fn = functions(path, "_prep", _CACHE_ATTR="_kunlun_lri_prep")._prep
        backend = types.SimpleNamespace(forward_metadata=object())
        inputs = types.SimpleNamespace(
            req_rows=torch.tensor([0, 0]), positions=torch.tensor([1, 2]), compress_ratio=1,
        )
        table = torch.arange(10).reshape(1, 10)
        first = fn(backend, inputs, table)
        self.assertIs(fn(backend, inputs, table), first)
        inputs.positions.add_(1)
        second = fn(backend, inputs, table)
        self.assertIsNot(first, second)
        self.assertEqual(second[0][0], 4)


if __name__ == "__main__":
    unittest.main()
