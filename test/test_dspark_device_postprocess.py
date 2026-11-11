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
"""Differential tests without importing the SGLang hook registration runtime."""
import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "sglang_kunlun/hooks/models/deepseek_v4_dspark.py"
NAMES = {
    "reference_verify_argmax",
    "correct_drafts_and_bonus_kunlun",
    "committed_swa_locations_kunlun",
}
tree = ast.parse(SOURCE.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
for node in nodes:
    node.decorator_list = []
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)

HOOK_SOURCE = SOURCE.parents[1] / "speculative/dspark_accept_torch.py"
hook_tree = ast.parse(HOOK_SOURCE.read_text())
hook_nodes = [
    node for node in hook_tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {
        "finalize_accept_lens_kunlun",
        "finalize_accept_lens_direct_kunlun",
        "build_out_tokens_kunlun",
    }
]
for node in hook_nodes:
    node.decorator_list = []
exec(compile(ast.Module(body=hook_nodes, type_ignores=[]), str(HOOK_SOURCE), "exec"), namespace)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Accelerator unavailable")
    yield request.param


@pytest.mark.parametrize("width", [1, 2, 6, 9])
def test_acceptance_against_prefix_reference(device, width):
    generator = torch.Generator().manual_seed(42)
    target = torch.randint(0, 100, (12, width), generator=generator)
    candidates = torch.randint(0, 100, (12, width), generator=generator)
    for row in range(12):
        prefix = row % width
        candidates[row, 1:prefix + 1] = target[row, :prefix]
    matches = candidates[:, 1:] == target[:, :-1]
    expected = matches.int().cumprod(1).sum(1).int()
    expected_bonus = target[torch.arange(12), expected.long()]
    correct, bonus = namespace["correct_drafts_and_bonus_kunlun"](
        candidates=candidates.to(device), target_predict=target.to(device)
    )
    assert correct.device.type == device
    assert bonus.device.type == device
    assert correct.dtype == torch.int32
    assert bonus.dtype == torch.int64
    torch.testing.assert_close(correct.cpu(), expected)
    torch.testing.assert_close(bonus.cpu(), expected_bonus)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_argmax_ties_strides_and_infinities(device, dtype):
    logits = torch.tensor([
        [1, 3, 3, -float("inf")],
        [-5, -2, -9, -3],
        [float("inf"), 0, float("inf"), -1],
    ], dtype=dtype)
    logits = logits.t().contiguous().t()
    result = namespace["reference_verify_argmax"](logits.to(device))
    assert result.device.type == device
    torch.testing.assert_close(result.cpu(), logits.argmax(-1))


def test_argmax_large_vocabulary(device):
    # Exercise the reduction at verify-like dimensions, rather than only
    # testing a tiny row that may select a different backend kernel.
    logits = torch.randn((96, 131072), generator=torch.Generator().manual_seed(7))
    result = namespace["reference_verify_argmax"](logits.to(device))
    torch.testing.assert_close(result.cpu(), logits.argmax(-1))


@pytest.mark.parametrize("bad", [float("nan"), -float("inf")])
def test_argmax_rejects_invalid_rows(device, bad):
    logits = torch.full((2, 4), bad, device=device)
    with pytest.raises(RuntimeError, match="NaN or fully masked"):
        namespace["reference_verify_argmax"](logits)


def test_swa_partial_commits_negative_and_uncommitted_indices(device):
    locations = torch.tensor([[0, 4, -1], [0, 3, -999], [999, 999, 999]])
    lengths = torch.tensor([2, 2, 0])
    mapping = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int64)
    result = namespace["committed_swa_locations_kunlun"](
        locations.to(device), mapping.to(device), lengths.to(device), 3
    )
    assert result.device.type == device
    expected = torch.tensor([[11, 55, -1], [11, 44, -1], [-1, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("length", [-1, 4])
def test_swa_rejects_invalid_lengths(device, length):
    with pytest.raises(ValueError, match="lengths outside"):
        namespace["committed_swa_locations_kunlun"](
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.arange(5, device=device),
            torch.tensor([length], device=device), 3,
        )


@pytest.mark.parametrize("index", [-6, -5, -1, 5])
def test_swa_rejects_committed_out_of_bounds(device, index):
    with pytest.raises(ValueError, match="mapping bounds"):
        namespace["committed_swa_locations_kunlun"](
            torch.tensor([index], device=device),
            torch.arange(5, device=device),
            torch.tensor([1], device=device), 1,
        )


def test_swa_empty_mapping_without_commits(device):
    result = namespace["committed_swa_locations_kunlun"](
        torch.tensor([999, -999], device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        torch.tensor([0], device=device), 2,
    )
    torch.testing.assert_close(result.cpu(), torch.full((2,), -1, dtype=torch.int32))


@pytest.mark.parametrize("direct", [False, True])
def test_accept_finalize_output_chain_preserves_inputs(device, direct):
    from sglang.kernels.ops.speculative.dspark.dspark_accept import FinalizeAcceptLens
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens

    def forbidden(*args, **kwargs):
        raise AssertionError("The torch route must not call Triton/original dispatch")

    batch, width = 16, 6
    # Keep all steps queued until the complete chain has run. Reading each
    # intermediate back to CPU would hide ordering and storage-reuse problems.
    results = []
    with patch.object(FinalizeAcceptLens, "triton", side_effect=forbidden), \
         patch.object(BuildOutTokens, "triton", side_effect=forbidden):
        for step in range(20):
            target_cpu = torch.arange(batch * width).reshape(batch, width) + step * 1000
            candidates_cpu = torch.full((batch, width), -1, dtype=torch.int64)
            expected_correct = (torch.arange(batch) + step) % width
            for row, count in enumerate(expected_correct.tolist()):
                candidates_cpu[row, 1:count + 1] = target_cpu[row, :count]
            candidates = candidates_cpu.to(device)
            target = target_cpu.to(device)
            correct, bonus = namespace["correct_drafts_and_bonus_kunlun"](
                candidates=candidates, target_predict=target,
            )
            prefix = torch.arange(batch, device=device, dtype=torch.int64) + 60000
            trim = torch.zeros_like(correct)
            kwargs = dict(correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix)
            if direct:
                finalized = namespace["finalize_accept_lens_direct_kunlun"](**kwargs)
            else:
                finalized = namespace["finalize_accept_lens_kunlun"](
                    forbidden, FinalizeAcceptLens, **kwargs,
                )
            out = namespace["build_out_tokens_kunlun"](
                forbidden, BuildOutTokens,
                draft_tokens=candidates[:, 1:],
                correct_len=correct, bonus=bonus,
                verify_num_draft_tokens=width, gamma=width - 1,
            )
            results.append((correct, bonus, finalized, out, expected_correct,
                            target_cpu, candidates_cpu))

    for correct, bonus, finalized, out, expected_correct, target_cpu, candidates_cpu in results:
        expected_bonus = target_cpu[torch.arange(batch), expected_correct]
        expected_out = candidates_cpu.clone()
        expected_out[:, :-1] = candidates_cpu[:, 1:]
        expected_out[:, -1] = 0
        expected_out[torch.arange(batch), expected_correct] = expected_bonus
        assert out.device.type == device
        torch.testing.assert_close(correct.cpu(), expected_correct.int())
        torch.testing.assert_close(bonus.cpu(), expected_bonus)
        torch.testing.assert_close(finalized.commit_lens.cpu(), expected_correct.int() + 1)
        torch.testing.assert_close(
            finalized.new_seq_lens.cpu(), torch.arange(batch) + 60001 + expected_correct,
        )
        torch.testing.assert_close(finalized.cap_trim_lens.cpu(), torch.zeros(batch, dtype=torch.int32))
        torch.testing.assert_close(out.cpu(), expected_out)
