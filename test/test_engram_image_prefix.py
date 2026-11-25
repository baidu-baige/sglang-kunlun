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
"""CPU window/mask regression without importing SGLang, models, or device ops.

Run with ``python -I test/test_engram_image_prefix.py``. Add
``--engram-source /path/to/sglang/srt/layers/engram.py`` to check hash IDs
against that runtime's pure Torch helper, extracted without module imports.
Device graph replay remains a separate hardware check.
"""

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "sglang_kunlun/hooks/models/deepseek_v4.py"
ENGRAM_SOURCE = None
IMAGE = 63
MM_PAD_SHIFT = 1_000_000


def _functions(path, names):
    tree = ast.parse(path.read_text(), filename=str(path))
    return ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )


def _scalar_hash(tokens, blocked, hasher):
    result = []
    for window, mask in zip(tokens.tolist(), blocked.tolist()):
        compressed = [
            hasher.pad_id if stop else int(hasher.token_map[token])
            for token, stop in zip(window, mask)
        ]
        layers = []
        for layer, multipliers in enumerate(hasher.multipliers.tolist()):
            rolling = compressed[0] * multipliers[0]
            values = []
            for shift in range(1, len(window)):
                rolling ^= compressed[shift] * multipliers[shift]
                for head, prime in enumerate(hasher.primes[layer, shift - 1].tolist()):
                    col = (shift - 1) * hasher.primes.shape[-1] + head
                    values.append(rolling % prime + int(hasher.offsets[layer, col]))
            layers.append(values)
        result.append(layers)
    return torch.tensor(result, dtype=torch.int64)


class EngramImagePrefixTest(unittest.TestCase):
    def setUp(self):
        self.registrations = {}

        def plugin_hook(target, type):
            def register(fn):
                self.registrations[target] = (type, fn)
                return fn
            return register

        self.compute = mock.Mock(return_value=torch.tensor([-1]))
        if ENGRAM_SOURCE is not None:
            scope = {"torch": torch}
            exec(compile(_functions(ENGRAM_SOURCE, {"compute_engram_hash_ids"}),
                         str(ENGRAM_SOURCE), "exec"), scope)
            self.compute.side_effect = scope["compute_engram_hash_ids"]
        self.precompute = SimpleNamespace(
            active=mock.Mock(return_value=False),
            dummy_hash_ids=mock.Mock(return_value=torch.tensor([-2])),
            dummy_verify_tokens=mock.Mock(return_value=torch.tensor([-3])),
        )
        self.scope = {
            "torch": torch, "SimpleNamespace": SimpleNamespace,
            "plugin_hook": plugin_hook,
            "HookType": SimpleNamespace(REPLACE="replace", AROUND="around"),
            "MODE_DECODE": 0, "MODE_VERIFY": 1, "MM_PAD_SHIFT_VALUE": MM_PAD_SHIFT,
            "compute_engram_hash_ids": self.compute,
            "engram_precompute": self.precompute,
        }
        names = {"engram_torch_hash_ids_kunlun", "_engram_fake_self",
                 "engram_hash_ids_native", "engram_hash_ids_and_commit_native"}
        exec(compile(_functions(HOOK, names), str(HOOK), "exec"), self.scope)
        kind, fn = self.registrations[
            "sglang.srt.layers.engram.EngramHasher._torch_hash_ids"
        ]
        self.assertEqual(kind, "replace")
        hasher_type = type("EngramHasher", (), {"_torch_hash_ids": fn})
        self.scope["EngramHasher"] = hasher_type
        self.hasher = hasher_type()
        self.hasher.max_ngram_size = 4
        self.hasher.image_token_id = IMAGE
        self.hasher.pad_id = 2
        # A nonidentity token map catches accidental use of raw IDs or raw PAD.
        self.hasher.token_map = (torch.arange(64) * 7 + 3) % 61
        self.hasher.multipliers = torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])
        self.hasher.primes = torch.tensor([
            [[101, 103], [107, 109], [113, 127]],
            [[131, 137], [139, 149], [151, 157]],
        ])
        flat = self.hasher.primes.flatten(1)
        self.hasher.offsets = flat.cumsum(-1) - flat

    def _check(self, mode, sequences, histories, positions, padding=0):
        ids = sum(sequences, [])
        num_real = len(ids)
        row = [r for r, seq in enumerate(sequences) for _ in seq]
        starts = [sum(map(len, sequences[:r])) for r in range(len(sequences))]
        expected_tokens, expected_mask = [], []
        for r, seq in enumerate(sequences):
            for offset in range(len(seq)):
                past = histories[r] + seq[:offset + 1]
                window = list(reversed(past[-4:]))
                if self.hasher.image_token_id is not None:
                    window = [IMAGE if token >= MM_PAD_SHIFT else token for token in window]
                first_image = window.index(IMAGE) if IMAGE in window else 4
                pos = positions[len(expected_tokens)]
                mask = [s > pos or (self.hasher.image_token_id is not None and s >= first_image)
                        for s in range(4)]
                expected_tokens.append(window)
                expected_mask.append(mask)
        expected_tokens += [[0] * 4] * padding
        expected_mask += [[True] * 4] * padding
        input_ids = torch.tensor(ids + [2] * padding, dtype=torch.int32)
        pos = torch.tensor(positions + [500] * padding, dtype=torch.int32)
        history = torch.tensor(histories, dtype=torch.int32)
        saved_history = history.clone()
        with mock.patch.object(torch.Tensor, "cummax", side_effect=AssertionError("cummax")):
            output, tokens = self.hasher._torch_hash_ids(
                input_ids, pos, mode, history, num_real, len(sequences[0]),
                torch.tensor(row, dtype=torch.int64), torch.tensor(starts),
            )
        torch.testing.assert_close(tokens, torch.tensor(expected_tokens))
        torch.testing.assert_close(history, saved_history)
        args = self.compute.call_args.args
        torch.testing.assert_close(args[0], tokens)
        torch.testing.assert_close(args[1], torch.tensor(expected_mask))
        self.assertEqual(args[2], self.hasher.pad_id)
        for actual, expected in zip(args[3:], (self.hasher.token_map,
                self.hasher.multipliers, self.hasher.primes, self.hasher.offsets)):
            self.assertIs(actual, expected)
        if ENGRAM_SOURCE is None:
            self.assertIs(output, self.compute.return_value)
        else:
            torch.testing.assert_close(output, _scalar_hash(tokens, args[1], self.hasher))
        return input_ids, pos, history, output, tokens

    def test_image_at_each_lookback_and_multimodal_history(self):
        """Images and scheduler multimodal pad IDs cut off every older lookback."""
        for image in (IMAGE, MM_PAD_SHIFT, MM_PAD_SHIFT + 123):
            for col in range(4):
                with self.subTest(image=image, col=col):
                    window = [4, 5, 6, 7]
                    window[col] = image
                    self._check(0, [[window[0]]], [list(reversed(window[1:]))], [100])

    def test_verify_and_extend_boundaries_and_padding(self):
        """Request boundaries, sequence starts, and graph padding retain PAD masks."""
        self._check(1, [[4, IMAGE, 6], [7, 8, 9]],
                    [[10, 11, MM_PAD_SHIFT + 1], [12, 13, 14]], [0, 1, 2, 2, 3, 4], 2)
        self._check(2, [[4], [], [IMAGE, 8, 9, 10]],
                    [[11, 12, 13], [14, 15, 16], [17, 18, 19]], [-1, 0, 1, 2, 3], 3)
        self._check(2, [[]], [[4, 5, 6]], [], 2)

    def test_text_and_special_ids_preserve_token_mapping(self):
        for image_token in (None, IMAGE):
            with self.subTest(image_token=image_token):
                self.hasher.image_token_id = image_token
                self._check(1, [[0, 2, 62], [IMAGE, 2, 0]],
                            [[2, 0, 62], [0, 2, 62]], [0, 1, 2, 30, 31, 32])

    def _native_kwargs(self, history):
        return dict(history=history, token_map=self.hasher.token_map,
                    multipliers=self.hasher.multipliers, primes=self.hasher.primes,
                    offsets=self.hasher.offsets, pad_id=self.hasher.pad_id,
                    image_token_id=IMAGE, mm_pad_shift=MM_PAD_SHIFT)

    def test_native_success_and_verify_fallback(self):
        ids, pos, history, expected, tokens = self._check(
            1, [[4, IMAGE], [6, 7]], [[8, 9, 10], [11, 12, 13]], [2, 3, 0, 1])
        kwargs = self._native_kwargs(history.flip(0))
        kwargs.update(mode=1, block=2, req_slots=torch.tensor([1, 0]))
        op = mock.Mock(return_value=(expected, tokens))
        with mock.patch.object(torch.ops, "xspeedgate_ops", SimpleNamespace(engram_hash_ids=op)):
            self.compute.reset_mock()
            self.assertIs(self.scope["engram_hash_ids_native"](None, ids, pos, **kwargs)[0], expected)
            self.compute.assert_not_called()
            op.side_effect = RuntimeError("native unavailable")
            actual, actual_tokens = self.scope["engram_hash_ids_native"](None, ids, pos, **kwargs)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_tokens, tokens)
        self.compute.assert_called_once()

    def test_decode_fallback_commits_only_live_history(self):
        ids, pos, history, expected, tokens = self._check(
            0, [[4], [IMAGE]], [[8, 9, 10], [11, 12, 13]], [2, 20])
        before = history.clone()
        op = mock.Mock(side_effect=RuntimeError("native unavailable"))
        with mock.patch.object(torch.ops, "xspeedgate_ops", SimpleNamespace(engram_hash_ids=op)):
            actual = self.scope["engram_hash_ids_and_commit_native"](
                None, ids, pos, req_slots=torch.tensor([0, 1]),
                out_cache_loc=torch.tensor([7, 0]), **self._native_kwargs(history))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(history[0], tokens[0, :3].flip(0).to(history.dtype))
        torch.testing.assert_close(history[1], before[1])

    def test_runtime_forward_precompute_refreshes_repaired_hashes(self):
        """Pre-replay host gathers consume fresh IDs from the repaired Torch path."""
        if ENGRAM_SOURCE is None:
            self.skipTest("supply --engram-source to exercise the runtime forward method")
        tree = ast.parse(ENGRAM_SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngramHasher")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        self.scope.update(ForwardBatch=SimpleNamespace, MODE_EXTEND=2,
                          _cuda_kernels=lambda _: False)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(ENGRAM_SOURCE), "exec"),
             self.scope)
        type(self.hasher).__call__ = self.scope["forward"]
        self.hasher.history = torch.tensor([[4, 5, 6]], dtype=torch.int32)
        self.hasher.pad_row = 0
        batch = SimpleNamespace(
            input_ids=torch.tensor([7, 8]), positions=torch.tensor([20, 21]),
            req_pool_indices=torch.tensor([0]),
            forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True),
            spec_info=SimpleNamespace(draft_token_num=2),
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
        )
        buffer = torch.empty((2, 6, 1), dtype=torch.bfloat16)
        embed = mock.Mock(side_effect=lambda ids, _: ids.unsqueeze(-1).to(torch.bfloat16))
        embed._kunlun_engram_precomp = buffer
        state = SimpleNamespace(hasher=self.hasher,
                                layers=[SimpleNamespace(embed=embed, layer_hash_index=0)])
        scope = {"_state": state, "active": lambda: False}
        path = ROOT / "sglang_kunlun/hooks/utils/engram_precompute.py"
        exec(compile(_functions(path, {"precompute"}), str(path), "exec"), scope)
        dp_attention = SimpleNamespace(set_dp_buffer_len=mock.Mock(),
                                       set_is_extend_in_batch=mock.Mock())
        with mock.patch.dict(sys.modules, {"sglang.srt.layers.dp_attention": dp_attention}):
            scope["precompute"](batch, 2)
            first = buffer.clone()
            batch.input_ids[0] = IMAGE
            scope["precompute"](batch, 2)
        self.assertFalse(torch.equal(first, buffer))
        expected = self.hasher(batch.input_ids, batch)[:, 0].unsqueeze(-1).to(torch.bfloat16)
        torch.testing.assert_close(buffer, expected)
        self.precompute.active.assert_not_called()

    def test_precompute_active_keeps_dummy_outputs_and_history(self):
        self.precompute.active.return_value = True
        history = torch.tensor([[4, 5, 6]], dtype=torch.int32)
        before = history.clone()
        ids, pos = torch.tensor([IMAGE]), torch.tensor([50])
        kwargs = self._native_kwargs(history)
        op = mock.Mock(side_effect=AssertionError("native must be skipped"))
        with mock.patch.object(torch.ops, "xspeedgate_ops", SimpleNamespace(engram_hash_ids=op)):
            result = self.scope["engram_hash_ids_native"](None, ids, pos, mode=1, **kwargs)
            self.assertIs(result[0], self.precompute.dummy_hash_ids.return_value)
            self.assertIs(result[1], self.precompute.dummy_verify_tokens.return_value)
            result = self.scope["engram_hash_ids_and_commit_native"](
                None, ids, pos, req_slots=torch.tensor([0]), out_cache_loc=torch.tensor([7]),
                **kwargs)
            self.assertIs(result, self.precompute.dummy_hash_ids.return_value)
        self.compute.assert_not_called()
        op.assert_not_called()
        torch.testing.assert_close(history, before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engram-source", type=Path)
    args, rest = parser.parse_known_args()
    ENGRAM_SOURCE = args.engram_source
    unittest.main(argv=[__file__, *rest])
