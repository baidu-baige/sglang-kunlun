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
"""Run installed upstream contracts on CPU, without importing GPU-only modules."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def upstream():
    spec = importlib.util.find_spec("sglang")
    if spec is None:
        pytest.skip("Upstream SGLang not installed")
    return Path(spec.origin).parent


def load_functions(path, names, owner=None, **namespace):
    """Execute unmodified method bodies; avoid unrelated runtime imports."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), "Upstream contract surface changed"
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ), *nodes],
        type_ignores=[],
    )
    namespace.update(torch=torch, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_paged_allocator_never_hands_out_dummy_page(upstream, page_size):
    methods = load_functions(
        upstream / "srt/mem_cache/allocator/paged.py",
        {"clear", "alloc", "free", "_release_page_ids", "merge_and_sort_free",
         "get_all_free_pages"},
        "PagedTokenToKVPoolAllocator",
    )
    allocator = SimpleNamespace(
        page_size=page_size, num_pages=3, device="cpu",
        debug_mode=False, need_sort=False,
    )
    allocator._release_page_ids = lambda *ids: methods._release_page_ids(allocator, *ids)
    allocator.merge_and_sort_free = lambda: methods.merge_and_sort_free(allocator)
    allocator.get_all_free_pages = lambda: methods.get_all_free_pages(allocator)
    methods.clear(allocator)
    slots = methods.alloc(allocator, 3 * page_size)
    torch.testing.assert_close(slots, torch.arange(page_size, 4 * page_size))
    assert methods.alloc(allocator, page_size) is None
    methods.free(allocator, slots)
    recycled = methods.alloc(allocator, 3 * page_size)
    assert torch.all(recycled >= page_size)
    methods.clear(allocator)
    assert methods.alloc(allocator, page_size)[0] == page_size


@pytest.mark.parametrize("debug", [False, True])
def test_paged_alloc_one_is_not_a_scratch_row(upstream, debug):
    methods = load_functions(
        upstream / "srt/mem_cache/allocator/paged.py",
        {"clear", "alloc"}, "PagedTokenToKVPoolAllocator",
    )
    allocator = SimpleNamespace(page_size=16, num_pages=3, device="cpu", debug_mode=debug)
    methods.clear(allocator)
    before = allocator.free_pages.clone()
    if debug:
        with pytest.raises(AssertionError, match="page-aligned"):
            methods.alloc(allocator, 1)
    else:
        assert methods.alloc(allocator, 1).numel() == 0
    torch.testing.assert_close(allocator.free_pages, before)


@pytest.mark.parametrize("direction", ["up", "down"])
def test_unified_clear_pins_zero_mapping_and_alloc_requires_pages(upstream, direction):
    from torch.profiler import record_function

    methods = load_functions(
        upstream / "srt/mem_cache/allocator/unified_sub_pool.py",
        {"clear", "_reset_watermarks", "alloc"}, "MultiEndedAllocator",
        record_function=record_function,
    )
    allocator = SimpleNamespace(
        grow_direction=direction, min_page_index=1, num_pages=5, device="cpu",
        page_size=16, sub_pool_name="contract", is_id_owner=True,
        virtual_to_physical=torch.full((6,), 99, dtype=torch.int64),
        physical_to_virtual=torch.full((6,), 99, dtype=torch.int64),
        _inverse_history=[], _pending_reuse={}, _pending_reuse_pages_cpu=set(),
    )
    allocator._reset_watermarks = lambda: methods._reset_watermarks(allocator)
    for _ in range(2):
        methods.clear(allocator)
        torch.testing.assert_close(allocator.virtual_to_physical, torch.tensor([0, -1, -1, -1, -1, -1]))
        torch.testing.assert_close(allocator.physical_to_virtual, allocator.virtual_to_physical)
        torch.testing.assert_close(allocator.free_virtual_ids, torch.arange(1, 5))
        assert allocator.watermark_physical == (1 if direction == "up" else 4)
        with pytest.raises(AssertionError, match="multiple of page_size"):
            methods.alloc(allocator, 1)
        # Dirty both maps to ensure clear restores the anchor and sentinel.
        allocator.virtual_to_physical.fill_(99)
        allocator.physical_to_virtual.fill_(99)


@pytest.mark.parametrize("keyword_batch", [False, True])
def test_graph_load_uses_live_locations(upstream, keyword_batch):
    runner_path = upstream / "srt/model_executor/runner/decode_cuda_graph_runner.py"
    view_fn = load_functions(runner_path, {"build_replay_fb_view"}).build_replay_fb_view
    load_batch = load_functions(
        runner_path, {"load_batch"}, "DecodeCudaGraphRunner",
        build_replay_fb_view=view_fn,
    ).load_batch
    mode = SimpleNamespace(is_idle=lambda: False)
    bucket = "verify"
    backend_init = load_functions(
        upstream / "srt/layers/attention/deepseek_v4_backend.py",
        {"init_forward_metadata_out_graph"}, "DeepseekV4AttnBackend",
        _GraphBucket=SimpleNamespace(
            of=lambda _: bucket, DECODE_OR_IDLE="decode",
            TARGET_VERIFY=bucket, DRAFT_EXTEND="extend",
        ),
        ForwardMode=SimpleNamespace(IDLE=object()),
        _get_target_verify_bs=lambda fb: 1,
    ).init_forward_metadata_out_graph
    fb = SimpleNamespace(
        batch_size=1, forward_mode=mode, input_ids=torch.tensor([7, 8]),
        positions=torch.tensor([3, 4]), out_cache_loc=torch.tensor([16, 17]),
        out_cache_loc_virtual=None, seq_lens_cpu=None, seq_lens_sum=5,
        max_seq_len_override=None, spec_info=None,
        needs_forward_metadata_init=lambda: True,
    )
    buffers = SimpleNamespace(
        input_ids=torch.tensor([7, 8, 0, 0]), positions=torch.tensor([3, 4, 0, 0]),
        req_pool_indices=torch.tensor([1, 0]), seq_lens=torch.tensor([5, 1]),
        seq_lens_cpu=None, encoder_lens=None, mamba_track_indices=None,
        # Deliberately distinct: metadata must read fb, not this registry rail.
        out_cache_loc=torch.tensor([80, 81, 0, 0]),
    )
    events = []
    backend = SimpleNamespace(
        needs_cpu_seq_lens=False, is_dspark_draft=False, MAX_SEQ_LEN_FOR_CAPTURE=32,
        online_c128_mtp=SimpleNamespace(
            enabled=lambda: False, prepare_forward=lambda *a, **k: 0,
        ),
        _resolve_verify_layout=lambda *a, **k: None,
        _target_verify_graph_key=lambda **k: (2, 4),
    )

    def make_metadata(**kwargs):
        events.append("metadata")
        return SimpleNamespace(raw_out_loc=kwargs["out_cache_loc"])

    backend.init_forward_metadata_target_verify = make_metadata
    backend.replay_cuda_graph_metadata_from = (
        lambda **kw: setattr(backend, "forward_metadata", kw["temp_metadata"])
    )

    def prepare(view):
        assert view.out_cache_loc is fb.out_cache_loc
        backend_init(backend, view)

    backend.init_forward_metadata_out_graph = prepare
    runner = SimpleNamespace(
        buffers=buffers, ragged_verify_mode=False, deepep_adapter=SimpleNamespace(replay=lambda: None),
        _validate_capture_hidden_mode=lambda _: None, captured_req_width=2,
        require_mlp_tp_gather=False, capture_bs=[2], _pad_to_bucket=lambda *a: 2,
        _capture_graph_size=lambda **k: k["num_tokens"],
        buffer_registry=SimpleNamespace(fill_from=Mock()),
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
            is_draft_worker=False, hisparse_coordinator=None,
        ),
        enable_two_batch_overlap=False, enable_pdmux=False, _metadata_glue=None,
        _replay_attn_backend=lambda: backend, seq_len_fill_value=1,
        capture_forward_mode=mode, is_encoder_decoder=False,
        _resolve_lora_variant=lambda _: None, _resolve_attention_variant=lambda _: None,
        _make_graph_key=lambda *a: a,
    )
    before_live, before_buffers = fb.out_cache_loc.clone(), buffers.out_cache_loc.clone()
    args, kwargs = ((), {"forward_batch": fb}) if keyword_batch else ((fb,), {})
    load_batch(runner, *args, **kwargs)
    assert events == ["metadata"]
    torch.testing.assert_close(fb.out_cache_loc, before_live)
    torch.testing.assert_close(buffers.out_cache_loc, before_buffers)
    torch.testing.assert_close(backend.forward_metadata.raw_out_loc, torch.tensor([16, 17, 0, 0]))


def test_compatibility_module_has_no_allocation_or_hook_registration():
    # Import it without importing the parent package (which registers GPU hooks).
    path = ROOT / "sglang_kunlun/hooks/model_executor/forward_batch_pad.py"
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert {key for key in namespace if not key.startswith("__")} == set()
