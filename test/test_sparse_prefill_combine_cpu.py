"""Scalar-oracle coverage of sparse prefill coordinates, holes and output reuse."""
import ast
from pathlib import Path
import random
import unittest

import torch


def load_function():
    path = Path(__file__).resolve().parents[1] / "sglang_kunlun/kernels/kernel_ops.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "combine_topk_swa_indices_kunlun")
    node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


class SparsePrefillCombineTest(unittest.TestCase):
    def test_coordinates_and_reused_strided_outputs(self):
        combine = load_function()
        rng = random.Random(94122)
        generator = torch.Generator().manual_seed(94122)
        for case in range(100):
            counts = [rng.randrange(5) for _ in range(rng.randrange(1, 5))]
            total = sum(counts)
            n = total + 2  # padding queries must overwrite stale output values
            base = rng.randrange(20)
            starts = [base]
            for count in counts:
                starts.append(starts[-1] + count)
            seq = [rng.randrange(8, 24) for _ in counts]
            gather = [rng.randrange(1, s + 1) for s in seq]
            pos = [rng.randrange(s) for s, count in zip(seq, counts) for _ in range(count)]
            pos += [999, 999]
            topk, window, ratio = rng.randrange(6), rng.randrange(1, 6), rng.randrange(1, 5)
            tops = torch.randint(-1, 12, (n, max(topk, 1)), generator=generator, dtype=torch.int32)
            explicit = case % 2 == 0
            windows = torch.randint(-1, 20, (n, window), generator=generator, dtype=torch.int32)
            lengths = torch.randint(0, window + 1, (n,), generator=generator, dtype=torch.int32)
            cb, sb = [100 * r for r in range(len(seq))], [1000 + 100 * r for r in range(len(seq))]
            width = ((topk + window + 127) // 128) * 128
            output = torch.full((n, width * 2), -99, dtype=torch.int32)[:, ::2]
            out_lens = torch.full((n * 2,), -99, dtype=torch.int32)[::2]
            t = lambda values: torch.tensor(values, dtype=torch.int32)
            actual, actual_lens = combine(
                tops, t(starts), t(pos), t(seq), t(gather), t(cb), t(sb),
                window, ratio, topk, output, out_lens,
                windows if explicit else None, lengths if explicit else None,
            )
            self.assertIs(actual, output)
            self.assertIs(actual_lens, out_lens)
            for row in range(n):
                expected = [-1] * width
                length = 0
                if row < total:
                    request = max(r for r in range(len(counts)) if starts[r] - base <= row)
                    k = min((pos[row] + 1) // ratio, topk)
                    w = int(lengths[row]) if explicit else min(pos[row] + 1, window)
                    for j in range(k):
                        value = int(tops[row, j])
                        expected[j] = value + cb[request] if value >= 0 else -1
                    for j in range(w):
                        if explicit:
                            value = int(windows[row, j])
                            expected[k + j] = value + sb[request] if value >= 0 else -1
                        else:
                            expected[k + j] = (
                                sb[request] + pos[row] - w + 1
                                - (seq[request] - gather[request]) + j
                            )
                    length = k + w
                self.assertEqual(actual[row].tolist(), expected, (case, row))
                self.assertEqual(actual_lens[row].item(), length)

    def test_empty_requests_and_tokens(self):
        combine = load_function()
        empty = torch.empty(0, dtype=torch.int32)
        for rows in (0, 3):
            values, lengths = combine(
                torch.empty((rows, 1), dtype=torch.int32),
                torch.tensor([7], dtype=torch.int32),
                torch.zeros(rows, dtype=torch.int32),
                empty, empty, empty, empty, 128, 1, 0,
            )
            self.assertEqual(values.shape, (rows, 128))
            self.assertTrue((values == -1).all())
            self.assertTrue((lengths == 0).all())


if __name__ == "__main__":
    unittest.main()
