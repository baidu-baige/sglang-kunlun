"""CPU regressions for the DSV4.1 review fixes, without accelerator imports.

Run directly with Python or unittest discovery. The MM reference is loaded only
from the designated upstream checkout at baa9befa, never from another install.
"""

import ast
import gc
import logging
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "sglang_kunlun/hooks"
UPSTREAM = Path("/home/ljj/dsv4_1_v0522/code_sglang_0522/sglang")
UPSTREAM_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"


def _load_cpu(path, names):
    """Load actual production definitions and literal constants without plugins.

    Args:
        path: Production source file.
        names: Top-level functions or classes needed by a regression group.

    Returns:
        An isolated namespace containing the selected implementation.
    """
    tree = ast.parse(path.read_text())
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
    if found != set(names):
        raise AssertionError(f"Missing production definitions: {set(names) - found}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {"torch": torch, "os": os, "logger": logging.getLogger(__name__)}
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class ReviewRegressionsCPU(unittest.TestCase):
    def test_identity_ids(self):
        """Arbitrary row counts must not retain tensors or overwrite live IDs."""
        ids_fn = _load_cpu(
            HOOKS / "models/dsv41_engram_int8.py", {"_identity_ids"}
        )._identity_ids
        refs = []
        live = ids_fn(17, "cpu")
        for n in list(range(257)) + [4096, 1, 4097, 0]:
            ids = ids_fn(n, torch.device("cpu"))
            torch.testing.assert_close(ids, torch.arange(n, dtype=torch.int32))
            refs.append(weakref.ref(ids))
            del ids
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs), "identity tensors remain cached")
        torch.testing.assert_close(live, torch.arange(17, dtype=torch.int32))
        ids_fn(17, "cpu").fill_(-1)
        torch.testing.assert_close(live, torch.arange(17, dtype=torch.int32))

    def test_cp_mm_contract(self):
        """Use the authoritative upstream method to check CP validation and IDs."""
        upstream_file = "python/sglang/srt/models/deepseek_v4.py"
        self.assertEqual(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
            ).strip(),
            UPSTREAM_COMMIT,
        )
        reference = subprocess.check_output(
            ["git", "show", f"{UPSTREAM_COMMIT}:{upstream_file}"], cwd=UPSTREAM
        )
        path = UPSTREAM / upstream_file
        self.assertEqual(path.read_bytes(), reference, "upstream reference has local edits")
        ns = _load_cpu(
            HOOKS / "model_executor/cp_mm_embeds.py",
            {"_has_mm_items", "cp_extend_with_mm_embeds", "_with_mm_ids_remapped"},
        )
        cls = next(n for n in ast.parse(reference).body if isinstance(n, ast.ClassDef)
                   and n.name == "DeepseekV4ForCausalLM")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "prepare_language_model_inputs")
        method.decorator_list = []
        namespace = {"torch": torch, "MM_PAD_SHIFT_VALUE": ns._MM_PAD_SHIFT_VALUE}
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        prepare = namespace["prepare_language_model_inputs"]
        model = SimpleNamespace(vision=object(), config=SimpleNamespace(image_token_id=42))
        model.prepare_language_model_inputs = prepare.__get__(model)
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        ids = torch.tensor([7, ns._MM_PAD_SHIFT_VALUE + 3])
        embeds = torch.ones(2, 3)
        for vision, mode, mm, supplied in (
            (True, "extend", [object()], True),
            (True, "extend", [object()], False),
            (True, "extend", [None], True),
            (False, "extend", [object()], True),
            (True, "decode", [object()], True),
            (True, "verify", [object()], True),
        ):
            with self.subTest(vision=vision, mode=mode, mm=mm, supplied=supplied):
                model.vision = object() if vision else None
                model._prepare_mm_embeddings = Mock(return_value=embeds)
                batch = SimpleNamespace(
                    input_ids=ids, mm_inputs=mm,
                    forward_mode=SimpleNamespace(
                        is_decode=lambda: mode == "decode",
                        is_target_verify=lambda: mode == "verify",
                        is_decode_or_idle=lambda: mode in ("decode", "idle"),
                    ),
                )
                kwargs = {"input_embeds": embeds if supplied else None}
                original = Mock(return_value="result")
                if vision and mode == "extend" and mm[0] is not None and supplied:
                    with self.assertRaisesRegex(
                        ValueError, "^Cannot combine input_embeds and image inputs$"
                    ):
                        ns.cp_extend_with_mm_embeds(original, runner, batch, kwargs)
                    original.assert_not_called()
                    model._prepare_mm_embeddings.assert_not_called()
                else:
                    expected_ids, expected_embeds = prepare(
                        model, ids, batch, kwargs["input_embeds"]
                    )
                    original.side_effect = lambda _, fb, kw: (
                        fb.input_ids.clone(), kw["input_embeds"]
                    )
                    actual_ids, actual_embeds = ns.cp_extend_with_mm_embeds(
                        original, runner, batch, kwargs
                    )
                    torch.testing.assert_close(actual_ids, expected_ids)
                    self.assertIs(actual_embeds, expected_embeds)
                    original.side_effect = RuntimeError("model failed")
                    with self.assertRaisesRegex(RuntimeError, "model failed"):
                        ns.cp_extend_with_mm_embeds(original, runner, batch, kwargs)
                self.assertIs(batch.input_ids, ids)
                self.assertIs(kwargs["input_embeds"], embeds if supplied else None)

    def test_engram_stage(self):
        """Check fresh rows, failure gates, capture lifetime and hash errors."""
        ns = _load_cpu(
            HOOKS / "models/dsv41_engram_stage.py",
            {"EngramHostStager", "engram_stage_capture", "engram_stage_execute"},
        )
        stager = ns.EngramHostStager()
        stager._ready = True
        stager._cols, stager._dim, stager._max_rows = 2, 3, 8
        embed = SimpleNamespace(
            weight=torch.arange(24, dtype=torch.int8).view(8, 3),
            weight_scale=torch.full((8, 1), 0.5, dtype=torch.bfloat16),
            _stage_slot=0,
        )
        stager._layers = [(0, embed), (1, embed)]
        stager._stage = [torch.full((8, 3), -99, dtype=torch.bfloat16) for _ in range(2)]
        stager._pinned = [torch.empty_like(t) for t in stager._stage]
        runner = SimpleNamespace(model_runner=object(), _kunlun_engram_stager=stager)
        mode = SimpleNamespace(
            is_decode=Mock(return_value=True), is_target_verify=Mock(return_value=False)
        )
        batch = SimpleNamespace(
            forward_mode=mode, input_ids=torch.tensor([5, 6]),
            req_pool_indices=torch.tensor([0, 1]),
            spec_info=SimpleNamespace(draft_token_num=2),
        )
        hashes = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        stager._hash_without_commit = Mock(return_value=hashes)
        original = Mock(return_value="replayed")
        pointers = [t.data_ptr() for t in stager._stage]
        self.assertEqual(ns.engram_stage_execute(original, runner, batch), "replayed")
        for slot in range(2):
            expected = embed.weight[hashes[:, slot].reshape(-1)].float() * 0.5
            torch.testing.assert_close(stager._stage[slot][:4], expected.bfloat16())
        self.assertEqual(stager._prepared_tokens, 2)
        stager._hash_without_commit.return_value = hashes.flip(0)
        ns.engram_stage_execute(original, runner, batch)
        self.assertEqual(pointers, [t.data_ptr() for t in stager._stage])
        torch.testing.assert_close(
            stager._stage[0][:4],
            (embed.weight[hashes.flip(0)[:, 0].reshape(-1)].float() * 0.5).bfloat16(),
        )
        for failure in ("not_ready", "oversized", "mode", "block",
                        "unequal", "no_hash", "bad_shape", "hash_error", "copy_error",
                        "partial_copy"):
            with self.subTest(failure=failure):
                stager._ready = failure != "not_ready"
                batch.input_ids = torch.tensor(
                    list(range(5)) if failure == "oversized" else [5, 6]
                )
                batch.req_pool_indices = torch.tensor([0, 1])
                mode.is_decode.return_value = failure not in ("mode", "block", "unequal")
                mode.is_target_verify.return_value = failure in ("block", "unequal")
                batch.spec_info.draft_token_num = 0 if failure == "block" else 2
                stager._hash_without_commit = Mock(
                    return_value=None if failure == "no_hash" else
                    hashes[:1] if failure == "bad_shape" else hashes,
                    side_effect=RuntimeError("hash failure") if failure == "hash_error" else None,
                )
                gather = stager._gather_into_pinned
                if failure == "copy_error":
                    stager._gather_into_pinned = Mock(side_effect=RuntimeError("copy failure"))
                elif failure == "partial_copy":
                    stager._gather_into_pinned = Mock(
                        side_effect=[None, RuntimeError("second layer")]
                    )
                original.reset_mock()
                with self.assertRaises(RuntimeError):
                    ns.engram_stage_execute(original, runner, batch)
                original.assert_not_called()
                self.assertEqual(stager._prepared_tokens, 0)
                stager._gather_into_pinned = gather
        # DP idle participation replays a 0-token IDLE batch to join the MoE
        # all-to-all; it must be admitted, and must not expose the previous
        # step's rows to that replay.
        stager._ready = True
        idle_mode = SimpleNamespace(
            is_decode=Mock(return_value=False), is_target_verify=Mock(return_value=False)
        )
        idle_batch = SimpleNamespace(
            forward_mode=idle_mode,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            spec_info=None,
        )
        stager._hash_without_commit = Mock(side_effect=AssertionError("hashed an idle batch"))
        original.reset_mock()
        self.assertEqual(ns.engram_stage_execute(original, runner, idle_batch), "replayed")
        original.assert_called_once()
        self.assertEqual(stager._prepared_tokens, 0)
        self.assertEqual(pointers, [t.data_ptr() for t in stager._stage])
        for staged in stager._stage:
            self.assertTrue(torch.all(staged == 0).item())
        mode.is_decode.return_value = False
        mode.is_target_verify.return_value = True
        batch.req_pool_indices = torch.tensor([0])
        stager._hash_without_commit = Mock(return_value=hashes)
        self.assertEqual(ns.engram_stage_execute(original, runner, batch), "replayed")
        stager._hash_without_commit.assert_called_once_with(batch, ns.MODE_VERIFY, 2)
        stager.begin_capture()
        view = stager.lookup(embed, torch.zeros(2, 2, dtype=torch.int64))
        self.assertEqual(view.data_ptr(), pointers[0])
        with self.assertRaises(RuntimeError):
            stager.lookup(embed, torch.zeros(5, 2, dtype=torch.int64))
        stager.end_capture()
        self.assertIsNone(stager.lookup(embed, torch.zeros(2, 2)))
        self.assertFalse(stager.ensure_alloc(None, 5))
        self.assertTrue(stager.ensure_alloc(None, 4))
        capture_runner = SimpleNamespace(model_runner=object(), max_num_token=4)
        with patch.dict(ns.engram_stage_capture.__globals__, get_stager=Mock(return_value=stager)):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                ns.engram_stage_capture(
                    Mock(side_effect=RuntimeError("capture failed")), capture_runner
                )
        self.assertFalse(stager._in_capture)
        self.assertIs(capture_runner._kunlun_engram_stager, stager)
        with patch.dict(ns.engram_stage_execute.__globals__, get_stager=Mock(return_value=None)):
            self.assertEqual(ns.engram_stage_execute(original, capture_runner, batch), "replayed")
        original.reset_mock()
        capture_runner.ragged_verify_mode = True
        with self.assertRaises(RuntimeError):
            ns.engram_stage_execute(original, capture_runner, batch)
        original.assert_not_called()
        kernel_module = ModuleType("sglang.kernels.ops.embeddings.engram_hash")
        kernel_module.engram_hash_ids = Mock(return_value=(hashes, None))
        stager._hasher = SimpleNamespace(
            history=torch.zeros(2, 3, dtype=torch.int64), token_map=None,
            multipliers=None, primes=None, offsets=None, pad_id=0, image_token_id=42,
        )
        batch.positions = torch.tensor([1, 2])
        history = stager._hasher.history.clone()
        with patch.dict("sys.modules", {kernel_module.__name__: kernel_module}):
            result = ns.EngramHostStager._hash_without_commit(stager, batch, ns.MODE_VERIFY, 2)
            self.assertIs(result, hashes)
            call = kernel_module.engram_hash_ids.call_args.kwargs
            self.assertEqual((call["mode"], call["block"]), (ns.MODE_VERIFY, 2))
            self.assertIs(call["history"], stager._hasher.history)
            self.assertEqual(call["num_real"], 2)
            self.assertEqual(call["mm_pad_shift"], ns.MM_PAD_SHIFT_VALUE)
            torch.testing.assert_close(history, stager._hasher.history)
            kernel_module.engram_hash_ids.side_effect = ValueError("kernel failed")
            with self.assertRaises(RuntimeError) as caught:
                ns.EngramHostStager._hash_without_commit(stager, batch, ns.MODE_DECODE, 1)
            self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
