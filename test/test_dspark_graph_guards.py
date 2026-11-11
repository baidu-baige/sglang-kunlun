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
# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
"""Exercise guard contracts without loading the model/hook registration runtime."""
import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_kunlun.hooks import graph_state


SOURCE = Path(__file__).resolve().parents[1] / "sglang_kunlun/hooks/models/deepseek_v4_dspark.py"
tree = ast.parse(SOURCE.read_text())
nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {
        "committed_swa_locations_kunlun", "guard_verify_ids", "_guard_verify_ids_impl",
    }
]
for node in nodes:
    node.decorator_list = []
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
swa = namespace["committed_swa_locations_kunlun"]
guard = namespace["guard_verify_ids"]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Accelerator unavailable")
    return request.param


def executor():
    return SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(vocab_size=8))
    )))


def call_guard(ids, original):
    return guard(
        original, executor(), batch="batch", draft_input="draft",
        verify_ids_2d=ids, verify_window="window", sampling_info=None,
    )


def consume_ids(self, **kwargs):
    assert kwargs["batch"] == "batch"
    assert kwargs["draft_input"] == "draft"
    assert kwargs["verify_window"] == "window"
    assert kwargs["sampling_info"] is None
    return kwargs["verify_ids_2d"].clone()


def no_host_reads():
    """Catch implicit scalar reads and explicit device-to-host diagnostics."""
    from contextlib import ExitStack

    stack = ExitStack()
    for name in ("__bool__", "item", "tolist", "cpu"):
        stack.enter_context(patch.object(
            torch.Tensor, name, side_effect=AssertionError(f"host read: {name}")
        ))
    return stack


def test_signature_matches_installed_upstream_and_callsite():
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        pytest.skip("Upstream SGLang not installed")
    root = Path(spec.origin).parent / "srt/speculative/dspark_components"
    upstream = ast.parse((root / "dspark_verify.py").read_text())
    cls = next(n for n in upstream.body if isinstance(n, ast.ClassDef)
               and n.name == "TargetVerifyExecutor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "run_non_compact")
    expected = [n.arg for n in method.args.kwonlyargs]
    actual = inspect.signature(guard)
    assert list(actual.parameters)[2:] == expected
    assert all(actual.parameters[n].kind == inspect.Parameter.KEYWORD_ONLY for n in expected)
    worker = ast.parse((root / "dspark_worker_v2.py").read_text())
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "run_non_compact"]
    assert calls
    assert all(not n.args and {k.arg for k in n.keywords} == set(expected) for n in calls)


@pytest.mark.parametrize("bad_id", [-1, 8, 2**40])
def test_eager_including_before_replay_rejects_without_mutation(device, bad_id):
    ids = torch.tensor([[0, bad_id, 7]], device=device)
    before = ids.clone()
    with patch.object(graph_state, "is_graph_capturing", return_value=False):
        with pytest.raises(ValueError, match="vocabulary bounds"):
            call_guard(ids, lambda *a, **k: pytest.fail("forward must not run"))
    torch.testing.assert_close(ids, before)


def test_capture_masks_ids_in_place_without_host_reads(device):
    ids = torch.tensor([[0, -1, 8, 7]], device=device)
    alias = ids.view(-1)
    with patch.object(graph_state, "is_graph_capturing", return_value=True), no_host_reads():
        result = call_guard(ids, consume_ids)
    torch.testing.assert_close(result.cpu(), torch.tensor([[0, 0, 0, 7]]))
    torch.testing.assert_close(alias.cpu(), torch.tensor([0, 0, 0, 7]))


def test_valid_ids_preserved(device):
    ids = torch.tensor([[0, 7]], dtype=torch.int32, device=device)
    with patch.object(graph_state, "is_graph_capturing", return_value=False):
        result = call_guard(ids, consume_ids)
    torch.testing.assert_close(result, ids)


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("invalid", ["dtype", "rank", "vocab"])
def test_verify_metadata_errors_before_forward(device, capturing, invalid):
    ids = torch.tensor([[1, 2]], device=device)
    owner = executor()
    if invalid == "dtype":
        ids = ids.float()
    elif invalid == "rank":
        ids = ids.flatten()
    else:
        owner.target_worker.model_runner.model_config.hf_text_config.vocab_size = 0
    with patch.object(graph_state, "is_graph_capturing", return_value=capturing):
        with no_host_reads(), pytest.raises(ValueError):
            guard(
                lambda *a, **k: pytest.fail("forward must not run"), owner,
                batch=None, draft_input=None, verify_ids_2d=ids,
                verify_window=None, sampling_info=None,
            )


def test_empty_batch_capture(device):
    with patch.object(graph_state, "is_graph_capturing", return_value=True), no_host_reads():
        result = swa(
            torch.empty(0, dtype=torch.int64, device=device),
            torch.arange(3, device=device),
            torch.empty(0, dtype=torch.int32, device=device), 3,
        )
    assert result.shape == (0,)
    assert result.dtype == torch.int32


def test_capture_swa_masks_indices_and_invalid_length_rows(device):
    loc = torch.tensor([[-1, 5, 2], [0, 1, 2], [0, 1, 2], [4, -999, 999]], device=device)
    lengths = torch.tensor([3, -1, 4, 1], device=device)
    mapping = torch.tensor([10, 20, 30, 40, 50], device=device)
    with patch.object(graph_state, "is_graph_capturing", return_value=True), no_host_reads():
        result = swa(loc, mapping, lengths, 3)
    torch.testing.assert_close(result.cpu(), torch.tensor([
        [-1, -1, 30], [-1, -1, -1], [-1, -1, -1], [50, -1, -1]
    ], dtype=torch.int32))


@pytest.mark.parametrize("capturing", [False, True])
def test_empty_mapping_with_commits(device, capturing):
    loc = torch.tensor([0, -1], device=device)
    mapping = torch.empty(0, dtype=torch.int32, device=device)
    lengths = torch.tensor([1], device=device)
    with patch.object(graph_state, "is_graph_capturing", return_value=capturing):
        if capturing:
            with no_host_reads():
                result = swa(loc, mapping, lengths, 2)
            torch.testing.assert_close(result.cpu(), torch.tensor([-1, -1], dtype=torch.int32))
        else:
            with pytest.raises(ValueError, match="mapping bounds"):
                swa(loc, mapping, lengths, 2)


def test_device_graph_replay_rechecks_changed_inputs():
    if not torch.cuda.is_available():
        pytest.skip("Accelerator unavailable")
    ids = torch.tensor([[1, 2, 7]], device="cuda")
    loc = torch.tensor([0, 1, 2], device="cuda")
    mapping = torch.tensor([10, 20, 30], device="cuda")
    lengths = torch.tensor([3], device="cuda")
    # Warm up on a side stream before recording the exact same device operations.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), patch.object(graph_state, "is_graph_capturing", return_value=True):
        for _ in range(3):
            call_guard(ids, consume_ids)
            swa(loc, mapping, lengths, 3)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with patch.object(graph_state, "is_graph_capturing", return_value=True):
        with torch.cuda.graph(graph, stream=stream), no_host_reads():
            tokens = call_guard(ids, consume_ids)
            slots = swa(loc, mapping, lengths, 3)
    for new_ids, new_loc, length, expected_ids, expected_slots in [
        ([[-1, 8, 7]], [-1, 3, 2], 3, [[0, 0, 7]], [-1, -1, 30]),
        ([[7, 0, 1]], [2, 1, 0], 1, [[7, 0, 1]], [30, -1, -1]),
        ([[1, 2, 3]], [0, 1, 2], 4, [[1, 2, 3]], [-1, -1, -1]),
    ]:
        ids.copy_(torch.tensor(new_ids, device="cuda"))
        loc.copy_(torch.tensor(new_loc, device="cuda"))
        lengths.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(tokens.cpu(), torch.tensor(expected_ids))
        torch.testing.assert_close(slots.cpu(), torch.tensor(expected_slots, dtype=torch.int32))
