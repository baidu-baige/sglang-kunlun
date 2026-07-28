import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import List, Optional
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_DIR = ROOT / "sglang_kunlun" / "hooks" / "layers" / "attention"
MEM_CACHE_DIR = ROOT / "sglang_kunlun" / "hooks" / "mem_cache"
SGLANG_MEM_CACHE_COMMON = (
    ROOT.parent
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "common.py"
)
SGLANG_DECODE_GRAPH_RUNNER = (
    ROOT.parent
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "model_executor"
    / "runner"
    / "decode_cuda_graph_runner.py"
)
MOE_DIR = ROOT / "sglang_kunlun" / "hooks" / "layers" / "moe"


class _ForwardMode:
    def __init__(self, *, decode=False, target_verify=False, draft_extend=False):
        self.decode = decode
        self.target_verify = target_verify
        self.draft_extend = draft_extend

    def is_decode_or_idle(self):
        return self.decode

    def is_extend(self):
        return self.target_verify or self.draft_extend

    def is_target_verify(self):
        return self.target_verify

    def is_draft_extend_v2(self):
        return self.draft_extend


class KunlunDSV4BackendContractTest(unittest.TestCase):
    def test_req_to_token_prefix_pointers_are_created_on_device(self):
        source = SGLANG_MEM_CACHE_COMMON.read_text()
        tree = ast.parse(source, filename=str(SGLANG_MEM_CACHE_COMMON))
        write_cache_indices = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_cache_indices"
        )
        function_source = ast.get_source_segment(source, write_cache_indices)

        self.assertIn("device=req_to_token_pool.device", function_source)
        self.assertNotIn("pin_memory=", function_source)
        self.assertNotIn("non_blocking=True", function_source)

    def test_decode_replay_metadata_uses_runtime_out_cache_loc(self):
        source = SGLANG_DECODE_GRAPH_RUNNER.read_text()
        tree = ast.parse(source, filename=str(SGLANG_DECODE_GRAPH_RUNNER))
        build_view = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_replay_fb_view"
        )
        namespace = {
            "DecodeInputBuffers": object,
            "ForwardBatch": object,
            "ForwardMode": object,
            "SimpleNamespace": types.SimpleNamespace,
        }
        exec(
            compile(
                ast.Module(body=[build_view], type_ignores=[]),
                str(SGLANG_DECODE_GRAPH_RUNNER),
                "exec",
            ),
            namespace,
        )

        runtime_out_cache_loc = torch.tensor([101, 102], dtype=torch.int64)
        stale_buffer_out_cache_loc = torch.tensor([1, 2], dtype=torch.int64)
        forward_batch = types.SimpleNamespace(
            forward_mode="decode",
            seq_lens_sum=2,
            out_cache_loc=runtime_out_cache_loc,
            out_cache_loc_dsv4=None,
            spec_info=None,
        )
        buffers = types.SimpleNamespace(
            input_ids=torch.tensor([11, 12], dtype=torch.int64),
            req_pool_indices=torch.tensor([3, 4], dtype=torch.int64),
            seq_lens=torch.tensor([9, 10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 10], dtype=torch.int64),
            encoder_lens=None,
            out_cache_loc=stale_buffer_out_cache_loc,
        )
        view = namespace["build_replay_fb_view"](
            forward_batch=forward_batch,
            buffers=buffers,
            bs=2,
            raw_bs=2,
            num_tokens=2,
            seq_len_fill_value=1,
            capture_forward_mode="decode",
            is_encoder_decoder=False,
        )

        self.assertIs(view.out_cache_loc, runtime_out_cache_loc)
        self.assertIsNot(view.out_cache_loc, stale_buffer_out_cache_loc)

    def test_allocator_selector_honors_boolean_environment_values(self):
        path = MEM_CACHE_DIR / "kunlun_allocator.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        selector = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_select_alloc_extend_func"
        )
        call = next(node for node in ast.walk(selector) if isinstance(node, ast.Call))
        self.assertIsInstance(call.func, ast.Name)
        self.assertEqual(call.func.id, "get_bool_env_var")
        self.assertEqual(
            [arg.value for arg in call.args],
            ["USE_FAST_ALLOC_EXTEND_KUNLUN", "true"],
        )

        fast_allocator = object()
        kernel_allocator = object()

        def get_bool_env_var(name, default):
            return os.getenv(name, default).lower() in ("true", "1")

        namespace = {
            "get_bool_env_var": get_bool_env_var,
            "_alloc_extend_kunlun_xdnn": fast_allocator,
            "_alloc_extend_kunlun_kernel": kernel_allocator,
        }
        exec(
            compile(ast.Module(body=[selector], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        select_allocator = namespace["_select_alloc_extend_func"]

        for value in ("False", "false", "0"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"USE_FAST_ALLOC_EXTEND_KUNLUN": value}
            ):
                self.assertIs(select_allocator(), kernel_allocator)
        for value in ("True", "true", "1"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"USE_FAST_ALLOC_EXTEND_KUNLUN": value}
            ):
                self.assertIs(select_allocator(), fast_allocator)
        with mock.patch.dict(os.environ, clear=True):
            self.assertIs(select_allocator(), fast_allocator)

    def test_backend_keeps_upstream_graph_api_and_kunlun_multistep(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("KunlunDeepseekV4AttnBackend", classes)
        self.assertIn("KunlunDeepseekV4MultiStepBackend", classes)

        backend = classes["KunlunDeepseekV4AttnBackend"]
        methods = {
            node.name for node in backend.body if isinstance(node, ast.FunctionDef)
        }
        self.assertFalse(
            methods
            & {
                "init_cuda_graph_state",
                "init_forward_metadata_capture_cuda_graph",
                "init_forward_metadata_replay_cuda_graph",
                "init_forward_metadata_in_graph",
            },
            "Kunlun DSV4 must inherit the 0.5.14 graph lifecycle",
        )
        self.assertIn("init_forward_metadata_out_graph", methods)
        out_graph = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_forward_metadata_out_graph"
        )
        out_graph_source = ast.get_source_segment(path.read_text(), out_graph)
        self.assertIn("super().init_forward_metadata_out_graph", out_graph_source)
        self.assertIn("_get_graph_extend_aux", out_graph_source)
        self.assertIn("_refresh_graph_host_lengths", out_graph_source)

        multistep_source = ast.get_source_segment(
            path.read_text(), classes["KunlunDeepseekV4MultiStepBackend"]
        )
        self.assertIn("KunlunDeepseekV4AttnBackend(", multistep_source)
        calls = [
            node.func.id
            for node in ast.walk(classes["KunlunDeepseekV4MultiStepBackend"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertNotIn("DeepseekV4AttnBackend", calls)

    def test_c4_indexer_state_is_explicitly_owned(self):
        backend_path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        backend_source = backend_path.read_text()
        for forbidden in (
            "ContextVar",
            "_C4_PREFILL_FORWARD_BATCH",
            "_C4_ACTIVE_INDEXER_CONTEXT",
            "_MTP_ACTIVE_INDEXER_CONTEXT",
            "_C4_BACKEND_OWNER",
        ):
            self.assertNotIn(forbidden, backend_source)

        backend_tree = ast.parse(backend_source, filename=str(backend_path))
        backend_class = next(
            node
            for node in backend_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "KunlunDeepseekV4AttnBackend"
        )
        init = next(
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        instance_attrs = {
            target.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        self.assertTrue(
            {
                "_attention_decode_aux",
                "_attention_graph_extend_aux",
                "_c4_decode_aux",
            }
            <= instance_attrs
        )

        compute = next(
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_c4_indexer_logits"
        )
        compute_source = ast.get_source_segment(backend_source, compute)
        self.assertIn("backend=self", compute_source)
        self.assertIn("forward_batch=forward_batch", compute_source)
        self.assertIn("c4_indexer=c4_indexer", compute_source)

        indexer_path = (
            ROOT.parent
            / "sglang"
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "attention"
            / "dsv4"
            / "indexer.py"
        )
        indexer_source = indexer_path.read_text()
        indexer_tree = ast.parse(indexer_source, filename=str(indexer_path))
        mixin = next(
            node
            for node in indexer_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "C4IndexerBackendMixin"
        )
        forward = next(
            node
            for node in mixin.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_c4_indexer"
        )
        forward_source = ast.get_source_segment(indexer_source, forward)
        self.assertIn("self._compute_c4_indexer_logits(", forward_source)
        self.assertIn("forward_batch=forward_batch", forward_source)
        self.assertIn("c4_indexer=c4_indexer", forward_source)

    def test_graph_capture_verify_uses_full_compressed_stride(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "KunlunDeepseekV4AttnBackend"
        )
        forward = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        fwd_source = ast.get_source_segment(source, forward)
        # Both graph_extend_mode and decode must trigger full-stride during capture
        self.assertIn("_is_graph_extend_mode(forward_batch.forward_mode)", fwd_source)
        self.assertIn("is_current_stream_capturing()", fwd_source)

    def test_c128_prefill_topk_clamps_only_continuation_chunks(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_clamp_c128_prefill_topk"
        )
        namespace = {"List": List, "Optional": Optional, "torch": torch}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                str(path),
                "exec",
            ),
            namespace,
        )
        clamp = namespace["_clamp_c128_prefill_topk"]

        self.assertEqual(clamp(64, torch.tensor([8192]), [8192], 128), 64)
        self.assertEqual(clamp(128, torch.tensor([16384]), [8192], 128), 64)
        self.assertEqual(clamp(192, torch.tensor([24576]), [8192], 128), 128)
        self.assertEqual(
            clamp(276, torch.tensor([35328, 5632]), [2560, 5632], 128),
            232,
        )
        self.assertEqual(clamp(64, None, [8192], 128), 64)
        self.assertEqual(clamp(64, torch.tensor([8192]), None, 128), 64)

    def test_decode_replay_refreshes_attention_host_lengths_in_place(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helper_names = {
            "_copy_host_lengths_",
            "_is_graph_extend_mode",
            "_seq_lens_cpu_i32",
            "_refresh_graph_host_lengths",
        }
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in helper_names
            ],
            type_ignores=[],
        )
        attention_cache = {}
        graph_extend_cache = {}
        c4_cache = {}
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)

        attention_lens_cpu = torch.ones(2, dtype=torch.int32)
        attention_lens = torch.ones(2, dtype=torch.int32)
        c4_lens_cpu = torch.full((2,), 4, dtype=torch.int32)
        c4_lens = torch.full((2,), 4, dtype=torch.int32)
        attention_ptr = attention_lens_cpu.data_ptr()
        attention_device_ptr = attention_lens.data_ptr()
        c4_ptr = c4_lens_cpu.data_ptr()
        c4_device_ptr = c4_lens.data_ptr()
        attention_cache[2] = (
            torch.arange(3, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            attention_lens_cpu,
            attention_lens,
        )
        c4_cache[(2, torch.device("cpu"), 65536)] = (
            torch.arange(3, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            c4_lens_cpu,
            c4_lens,
        )
        forward_batch = types.SimpleNamespace(
            batch_size=2,
            forward_mode=_ForwardMode(decode=True),
            seq_lens=torch.tensor([16668, 31], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([16668, 31], dtype=torch.int64),
        )

        namespace["_refresh_graph_host_lengths"](
            forward_batch, attention_cache, c4_cache, graph_extend_cache
        )

        self.assertEqual(attention_lens_cpu.data_ptr(), attention_ptr)
        self.assertEqual(attention_lens.data_ptr(), attention_device_ptr)
        self.assertEqual(c4_lens_cpu.data_ptr(), c4_ptr)
        self.assertEqual(c4_lens.data_ptr(), c4_device_ptr)
        self.assertEqual(attention_lens_cpu.tolist(), [16668, 31])
        self.assertEqual(attention_lens.tolist(), [16668, 31])
        self.assertEqual(c4_lens_cpu.tolist(), [16668, 28])
        self.assertEqual(c4_lens.tolist(), [16668, 28])

    def test_verify_and_draft_replay_keep_058_graph_aux_contract(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helper_names = {
            "_copy_host_lengths_",
            "_is_graph_extend_mode",
            "_alloc_graph_extend_aux",
            "_get_graph_extend_aux",
            "_seq_lens_cpu_i32",
            "_refresh_graph_host_lengths",
        }
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in helper_names
            ],
            type_ignores=[],
        )
        attention_cache = {}
        graph_extend_cache = {}
        c4_cache = {}
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)

        for forward_mode in (
            _ForwardMode(target_verify=True),
            _ForwardMode(draft_extend=True),
        ):
            graph_extend_cache.clear()
            aux = namespace["_get_graph_extend_aux"](
                graph_extend_cache, 2, 6, torch.device("cpu")
            )
            q_lod_cpu, q_lod, kv_lens_cpu, kv_lens, query_lens = aux
            pointers = [tensor.data_ptr() for tensor in aux]
            self.assertEqual(q_lod_cpu.tolist(), [0, 3, 6])
            self.assertEqual(q_lod.tolist(), [0, 3, 6])
            self.assertEqual(query_lens.tolist(), [3, 3])

            forward_batch = types.SimpleNamespace(
                batch_size=2,
                forward_mode=forward_mode,
                seq_lens=torch.tensor([2, 31], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([2, 31], dtype=torch.int64),
            )
            namespace["_refresh_graph_host_lengths"](
                forward_batch, attention_cache, c4_cache, graph_extend_cache
            )

            self.assertEqual([tensor.data_ptr() for tensor in aux], pointers)
            self.assertEqual(kv_lens_cpu.tolist(), [2, 31])
            self.assertEqual(kv_lens.tolist(), [3, 31])

    def test_multistep_children_all_use_graph_refresh_hook(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        backend = classes["KunlunDeepseekV4AttnBackend"]
        out_graph = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_forward_metadata_out_graph"
        )
        out_graph_source = ast.get_source_segment(source, out_graph)
        self.assertIn("self._attention_decode_aux", out_graph_source)
        self.assertIn("self._attention_graph_extend_aux", out_graph_source)
        self.assertIn("self._c4_decode_aux", out_graph_source)

        multistep = classes["KunlunDeepseekV4MultiStepBackend"]
        multistep_source = ast.get_source_segment(source, multistep)
        self.assertIn("KunlunDeepseekV4AttnBackend(", multistep_source)
        self.assertNotIn(
            "DeepseekV4AttnBackend(",
            multistep_source.replace("KunlunDeepseekV4AttnBackend(", ""),
        )

        replay = next(
            node
            for node in multistep.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_forward_metadata_out_graph"
        )
        replay_source = ast.get_source_segment(source, replay)
        self.assertIn(
            "self.attn_backends[0].init_forward_metadata_out_graph(inner_fb)",
            replay_source,
        )
        self.assertIn("range(1, self.speculative_num_steps - 1)", replay_source)
        self.assertIn("replay_cuda_graph_metadata_from", replay_source)
        self.assertIn("temp_metadata=temp_metadata", replay_source)

    def test_moe_sqrtsoftplus_calls_058_fused_gate_symbol(self):
        source = (MOE_DIR / "topk.py").read_text()
        tree = ast.parse(source)
        select_experts = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "select_experts_kunlun"
        )
        calls = [
            node
            for node in ast.walk(select_experts)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe_fused_gate"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertIsInstance(keywords["scoring_func"], ast.Name)
        self.assertEqual(keywords["scoring_func"].id, "scoring_func")

        kernel_source = (ROOT / "sglang_kunlun" / "kernels" / "kernel_ops.py").read_text()
        self.assertIn(
            '@register_jit_op("sglang.jit_kernel.moe_fused_gate", "moe_fused_gate")',
            kernel_source,
        )
        self.assertIn("kunlun_ops.moe_fused_gate_dsv4(", kernel_source)

    def test_prefill_metadata_keeps_raw_topk_unallocated_for_058_path(self):
        source = (
            ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        ).read_text()
        tree = ast.parse(source)
        metadata = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "KunlunDSV4AttnMetadata"
        )
        init_related = next(
            node
            for node in metadata.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_flashmla_related"
        )
        assignments = [
            node
            for node in ast.walk(init_related)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "c4_sparse_raw_indices"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIsInstance(assignments[0].value, ast.Constant)
        self.assertIsNone(assignments[0].value.value)

    def test_compressed_attention_matches_058_argument_contract(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compressed_attention"
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            len(calls[0].args),
            18,
            "0.5.8 compressed_attention ends with attn_sink and has no trailing sentinel",
        )
        self.assertEqual([keyword.arg for keyword in calls[0].keywords], ["side_stream"])
        self.assertEqual(
            ast.get_source_segment(source, calls[0].keywords[0].value),
            "torch.cuda.current_stream().cuda_stream",
        )

    def test_target_verify_c4_matches_058_representative_rows(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helper = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_select_c4_target_verify_rows"
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(helper, str(path), "exec"), namespace)

        q = torch.arange(8 * 6, dtype=torch.int8).reshape(8, 1, 2, 3)
        weight = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        seq_lens = torch.arange(100, 108, dtype=torch.int32)
        page_table = torch.arange(8 * 2, dtype=torch.int32).reshape(8, 2)
        selected = namespace["_select_c4_target_verify_rows"](
            q, weight, seq_lens, page_table, num_requests=2
        )
        selected_q, selected_weight, selected_lens, selected_pages, repeat = selected

        self.assertTrue(torch.equal(selected_q, q[[3, 7]]))
        self.assertTrue(torch.equal(selected_weight, weight[[3, 7]]))
        self.assertTrue(torch.equal(selected_lens, seq_lens[[3, 7]]))
        self.assertTrue(torch.equal(selected_pages, page_table[[3, 7]]))
        self.assertEqual(repeat, 4)

        request_lens = torch.tensor([103, 107], dtype=torch.int32)
        request_pages = page_table[[3, 7]]
        selected = namespace["_select_c4_target_verify_rows"](
            q, weight, request_lens, request_pages, num_requests=2
        )
        selected_q, selected_weight, selected_lens, selected_pages, _ = selected
        self.assertTrue(torch.equal(selected_q, q[[0, 1]]))
        self.assertTrue(torch.equal(selected_weight, weight[[0, 1]]))
        self.assertTrue(torch.equal(selected_lens, request_lens))
        self.assertEqual(selected_lens.data_ptr(), request_lens.data_ptr())
        self.assertIs(selected_pages, request_pages)

    def test_contiguous_prefill_matches_058_lod_contract(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_build_c4_prefill_contract"
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)
        forward_batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[3, 2], extend_prefix_lens_cpu=[0, 3]
        )
        contract = namespace["_build_c4_prefill_contract"](
            forward_batch,
            torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int32),
            torch.zeros((5, 2), dtype=torch.int32),
            torch.device("cpu"),
        )
        self.assertEqual(contract["qlod_cpu"].tolist(), [0, 3, 5])
        self.assertEqual(contract["last_rows"].tolist(), [2, 4])
        self.assertEqual(contract["per_req_k_lens"].tolist(), [3, 5])
        self.assertEqual(contract["klod_cpu"].tolist(), [0, 12, 32])
        self.assertEqual(contract["com_k_start_cpu"].tolist(), [0, 0])
        self.assertEqual(contract["max_seq_k"], 20)
        self.assertFalse(contract["use_causal"])

    def test_contiguous_prefill_gathers_058_segmented_cache_layout(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_gather_c4_prefill_kv"
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)
        raw = torch.zeros((3, 64 * 132), dtype=torch.uint8)
        raw[0, : 64 * 128] = 1
        raw[1, : 64 * 128] = 2
        raw[2, : 64 * 128] = 3
        raw[:, 64 * 128 :].view(torch.float32).copy_(
            torch.tensor([[1.0] * 64, [2.0] * 64, [3.0] * 64])
        )
        cache = raw.reshape(3, 64, 1, 132)
        contract = {
            "per_req_k_lens": torch.tensor([65], dtype=torch.int32),
            "last_rows": torch.tensor([0], dtype=torch.int32),
        }
        k, scale = namespace["_gather_c4_prefill_kv"](
            cache,
            torch.tensor([[2, 0]], dtype=torch.int32),
            contract,
            torch.device("cpu"),
        )
        self.assertEqual(k.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(k[:64], torch.full((64, 128), 3, dtype=torch.int8)))
        self.assertTrue(torch.equal(k[64], torch.ones(128, dtype=torch.int8)))
        self.assertTrue(torch.equal(scale[:64], torch.full((64,), 3.0)))
        self.assertEqual(scale[64].item(), 1.0)

    def test_contiguous_prefill_calls_058_operator_contract(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "c4a_mqa_logits"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [keyword.arg for keyword in calls[0].keywords],
            [
                "q", "weight", "k", "k_scale", "logits", "max_seq_q",
                "max_seq_k", "qlod_cpu", "qlod_xpu", "klod_cpu", "klod_xpu",
                "com_k_start_cpu", "com_k_start_xpu", "is_causal",
                "compress_ratio", "clean_logits",
            ],
        )

    def test_decode_c4_aux_is_instance_owned_and_refreshed_in_place(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        helper_names = {
            "_copy_host_lengths_",
            "_seq_lens_cpu_i32",
            "_select_c4_target_verify_rows",
            "_compute_c4_logits_kunlun",
        }
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in helper_names
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch, "os": os}
        exec(compile(helpers, str(path), "exec"), namespace)

        calls = []
        fake_ops = types.ModuleType("kunlun_ops")

        def c4a_paged_mqa_logits(**kwargs):
            calls.append(
                {
                    name: value.clone() if isinstance(value, torch.Tensor) else value
                    for name, value in kwargs.items()
                }
            )
            kwargs["logits"].zero_()

        fake_ops.c4a_paged_mqa_logits = c4a_paged_mqa_logits
        backend = types.SimpleNamespace(
            _c4_decode_aux={},
            speculative_step_id=0,
        )
        c4_indexer = types.SimpleNamespace(layer_id=2)
        cache = torch.zeros((2, 64, 1, 132), dtype=torch.uint8)
        page_table = torch.tensor([[0], [1]], dtype=torch.int32)
        q = torch.zeros((2, 1, 64, 128), dtype=torch.int8)
        weight = torch.zeros((2, 64), dtype=torch.float32)

        with mock.patch.dict(sys.modules, {"kunlun_ops": fake_ops}):
            for raw_lens, c4_lens in (([15, 31], [3, 7]), ([19, 35], [4, 8])):
                forward_batch = types.SimpleNamespace(
                    batch_size=2,
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens=torch.tensor(raw_lens, dtype=torch.int32),
                    seq_lens_cpu=torch.tensor(raw_lens, dtype=torch.int32),
                )
                namespace["_compute_c4_logits_kunlun"](
                    backend=backend,
                    q_fp8=q,
                    kvcache_fp8=cache,
                    weight=weight,
                    seq_lens=torch.tensor(c4_lens, dtype=torch.int32),
                    page_table=page_table,
                    max_seq_len=64,
                    forward_batch=forward_batch,
                    c4_indexer=c4_indexer,
                )

        self.assertEqual(len(backend._c4_decode_aux), 1)
        aux = next(iter(backend._c4_decode_aux.values()))
        self.assertEqual(calls[0]["qlod_cpu"].tolist(), [0, 1, 2])
        self.assertEqual(calls[0]["context_lens_cpu"].tolist(), [12, 28])
        self.assertEqual(calls[0]["context_lens_xpu"].tolist(), [12, 28])
        self.assertTrue(torch.equal(calls[0]["block_table"], page_table))
        self.assertEqual(calls[1]["context_lens_cpu"].tolist(), [16, 32])
        self.assertEqual(calls[1]["context_lens_xpu"].tolist(), [16, 32])
        self.assertEqual(aux[2].tolist(), [16, 32])
        self.assertEqual(aux[3].tolist(), [16, 32])

    def test_hook_import_registers_public_backend_and_draft_factory(self):
        registered = {}

        registry_module = types.ModuleType(
            "sglang.srt.layers.attention.attention_registry"
        )

        def register_attention_backend(name):
            def decorate(factory):
                registered[name] = factory
                return factory

            return decorate

        registry_module.register_attention_backend = register_attention_backend

        module_name = "sglang_kunlun.hooks.layers.attention.attention_registry"
        path = ATTENTION_DIR / "attention_registry.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)

        saved = {
            name: sys.modules.get(name)
            for name in (
                "sglang.srt.layers.attention.attention_registry",
                module_name,
            )
        }
        try:
            sys.modules[
                "sglang.srt.layers.attention.attention_registry"
            ] = registry_module
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

        self.assertIn("kunlun", registered)
        self.assertIn("kunlun_compressed", registered)
        draft_hook = (
            ROOT / "sglang_kunlun" / "hooks" / "speculative" / "draft_utils.py"
        ).read_text()
        self.assertIn('backend_type == "kunlun_compressed"', draft_hook)
        self.assertIn("KunlunDeepseekV4MultiStepBackend(", draft_hook)

    def test_platform_exposes_both_server_choices(self):
        source = (ROOT / "sglang_kunlun" / "platform" / "srt.py").read_text()
        self.assertIn('("kunlun", "kunlun_compressed")', source)

    def test_dsv4_pool_reuses_upstream_pool_selection(self):
        source = (
            ROOT.parent
            / "sglang"
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "model_runner_kv_cache_mixin.py"
        ).read_text()
        self.assertIn("if is_dsv4_model:", source)
        self.assertIn("pool_cls = DeepSeekV4TokenToKVPool", source)
        plugin_sources = "\n".join(
            path.read_text()
            for path in (ROOT / "sglang_kunlun").rglob("*.py")
            if path.name != "sitecustomize.py"
        )
        self.assertNotIn("def init_token_to_kv_pool", plugin_sources)


if __name__ == "__main__":
    unittest.main()
