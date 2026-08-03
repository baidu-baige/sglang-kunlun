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
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        from sglang_kunlun.hooks import production_precision as runtime_precision

        target = "sglang.srt.mem_cache.common.write_cache_indices"
        self.assertTrue(
            any(
                hook_type == HookType.REPLACE
                and hook is runtime_precision.write_cache_indices_kunlun
                for hook_type, hook, _source in HookRegistry._hooks[target]
            )
        )

        launches = []

        class Launcher:
            def __getitem__(self, grid):
                self.grid = grid
                return self.launch

            def launch(self, *args):
                launches.append((self.grid, args))

        prefix_tensors = [
            torch.tensor([11, 12], dtype=torch.int64),
            torch.tensor([21], dtype=torch.int64),
        ]
        req_to_token_pool = types.SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token=torch.zeros((2, 8), dtype=torch.int64),
        )
        inputs = {
            "out_cache_loc": torch.tensor([31, 32, 41], dtype=torch.int64),
            "req_pool_indices_tensor": torch.tensor([0, 1], dtype=torch.int32),
            "req_pool_indices_cpu": torch.tensor([0, 1], dtype=torch.int32),
            "prefix_lens_tensor": torch.tensor([2, 1], dtype=torch.int32),
            "prefix_lens_cpu": torch.tensor([2, 1], dtype=torch.int32),
            "seq_lens_tensor": torch.tensor([4, 2], dtype=torch.int32),
            "seq_lens_cpu": torch.tensor([4, 2], dtype=torch.int32),
            "extend_lens_tensor": torch.tensor([2, 1], dtype=torch.int32),
            "extend_lens_cpu": torch.tensor([2, 1], dtype=torch.int32),
        }
        tensor_factory = torch.tensor
        launcher = Launcher()
        common_module = types.ModuleType("sglang.srt.mem_cache.common")
        common_module.support_triton = lambda _backend: True
        common_module.get_global_server_args = lambda: types.SimpleNamespace(
            attention_backend="triton"
        )
        common_module.write_req_to_token_pool_triton = launcher
        mem_cache_package = types.ModuleType("sglang.srt.mem_cache")
        mem_cache_package.__path__ = []
        mem_cache_package.common = common_module
        with mock.patch.dict(
            sys.modules,
            {
                mem_cache_package.__name__: mem_cache_package,
                common_module.__name__: common_module,
            },
        ), mock.patch.object(
            runtime_precision.torch, "tensor", wraps=tensor_factory
        ) as make_tensor:
            runtime_precision.write_cache_indices_kunlun(
                **inputs,
                prefix_tensors=prefix_tensors,
                req_to_token_pool=req_to_token_pool,
            )

        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][0], (2,))
        prefix_pointers = launches[0][1][2]
        self.assertEqual(prefix_pointers.dtype, torch.uint64)
        self.assertEqual(prefix_pointers.device, req_to_token_pool.device)
        self.assertEqual(
            prefix_pointers.tolist(),
            [tensor.data_ptr() for tensor in prefix_tensors],
        )
        pointer_call = next(
            call
            for call in make_tensor.call_args_list
            if call.kwargs.get("dtype") == torch.uint64
        )
        self.assertEqual(pointer_call.kwargs["device"], req_to_token_pool.device)
        self.assertNotIn("pin_memory", pointer_call.kwargs)
        self.assertNotIn("non_blocking", pointer_call.kwargs)

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

        forward = next(
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_c4_indexer"
        )
        forward_source = ast.get_source_segment(backend_source, forward)
        self.assertIn("self._compute_c4_indexer_logits(", forward_source)
        self.assertIn("forward_batch=forward_batch", forward_source)
        self.assertIn("c4_indexer=c4_indexer", forward_source)
        self.assertNotIn("super().forward_c4_indexer", forward_source)
        self.assertNotIn("capture_probe", forward_source)
        self.assertNotIn("_mtp_tensor_probe", forward_source)

        flag = types.SimpleNamespace(get=lambda: False)
        envs = types.SimpleNamespace(
            SGLANG_OPT_USE_TILELANG_INDEXER=flag,
            SGLANG_OPT_USE_AITER_INDEXER=flag,
            SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=types.SimpleNamespace(
                get=lambda: True
            ),
            SGLANG_TOPK_TRANSFORM_512_TORCH=flag,
            SGLANG_OPT_USE_TOPK_V2=flag,
        )
        namespace = {"torch": torch, "envs": envs}
        exec(
            compile(
                ast.Module(body=[forward], type_ignores=[]),
                str(backend_path),
                "exec",
            ),
            namespace,
        )

        class IndexerMetadata:
            pass

        indexer_module_name = "sglang.srt.layers.attention.dsv4.indexer"
        upstream_indexer = types.ModuleType(indexer_module_name)
        upstream_indexer.PagedIndexerMetadata = IndexerMetadata
        upstream_indexer.is_sm120_supported = lambda: False
        upstream_indexer.fp8_paged_mqa_logits_torch = object()
        upstream_indexer.fp8_paged_mqa_logits_torch_sm120 = object()
        dsv4_package = types.ModuleType("sglang.srt.layers.attention.dsv4")
        dsv4_package.__path__ = []
        dsv4_package.indexer = upstream_indexer

        page_table = torch.zeros((2, 1), dtype=torch.int32)
        indexer_metadata = IndexerMetadata()
        indexer_metadata.c4_seq_lens = torch.tensor([4, 8], dtype=torch.int32)
        indexer_metadata.page_table = page_table
        indexer_metadata.max_c4_seq_len = 64
        indexer_metadata.c4_page_size = 64
        indexer_metadata.topk_metadata = None
        core_metadata = types.SimpleNamespace(
            positions=torch.tensor([0, 1], dtype=torch.int64),
            page_table=page_table,
            c4_sparse_page_indices=torch.full((2, 4), -1, dtype=torch.int32),
            c4_sparse_raw_indices=None,
        )
        q_indexer = torch.zeros((2, 64, 128), dtype=torch.int8)
        weights = torch.zeros((2, 64, 1), dtype=torch.float32)
        cache = torch.zeros((1, 64 * 132), dtype=torch.uint8)
        compute_logits = mock.Mock(return_value=torch.zeros((2, 64)))
        owner = types.SimpleNamespace(
            token_to_kv_pool=object(),
            forward_metadata=types.SimpleNamespace(
                indexer_metadata=indexer_metadata,
                core_metadata=core_metadata,
            ),
            _forward_prepare_normal=mock.Mock(
                return_value=(q_indexer, weights, cache)
            ),
            _compute_c4_indexer_logits=compute_logits,
            debug_use_external_c4_sparse_indices=True,
        )
        c4_indexer = types.SimpleNamespace(use_fp4_indexer=False, layer_id=2)
        forward_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_idle=lambda: False)
        )
        with mock.patch.dict(
            sys.modules,
            {
                dsv4_package.__name__: dsv4_package,
                indexer_module_name: upstream_indexer,
            },
        ):
            namespace["forward_c4_indexer"](
                owner,
                torch.zeros((2, 8)),
                torch.zeros((2, 8)),
                c4_indexer,
                forward_batch,
            )

        compute_logits.assert_called_once()
        kwargs = compute_logits.call_args.kwargs
        self.assertIs(kwargs["forward_batch"], forward_batch)
        self.assertIs(kwargs["c4_indexer"], c4_indexer)
        self.assertIs(kwargs["indexer_metadata"], indexer_metadata)
        self.assertIs(kwargs["core_metadata"], core_metadata)

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

    @staticmethod
    def _load_c4_prefill_contract(cp_ranks=None):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        wanted = {
            "_build_c4_prefill_contract",
            "_dsa_cp_local_extend_lens",
            "_seq_lens_cpu_i32",
        }
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted
            ],
            type_ignores=[],
        )
        namespace = {
            "torch": torch,
            "_dsa_cp_prefill_ranks": lambda _forward_batch: cp_ranks,
        }
        exec(compile(helpers, str(path), "exec"), namespace)
        return namespace["_build_c4_prefill_contract"]

    def test_contiguous_prefill_matches_058_lod_contract(self):
        build_contract = self._load_c4_prefill_contract()
        forward_batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[3, 2], extend_prefix_lens_cpu=[0, 3]
        )
        contract = build_contract(
            forward_batch,
            torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int32),
            torch.zeros((5, 2), dtype=torch.int32),
            torch.device("cpu"),
            5,
        )
        self.assertEqual(contract["qlod_cpu"].tolist(), [0, 3, 5])
        self.assertEqual(contract["last_rows"].tolist(), [2, 4])
        self.assertEqual(contract["per_req_k_lens"].tolist(), [3, 5])
        self.assertEqual(contract["klod_cpu"].tolist(), [0, 12, 32])
        self.assertEqual(contract["com_k_start_cpu"].tolist(), [0, 0])
        self.assertEqual(contract["max_seq_k"], 20)
        self.assertEqual(contract["max_seq_q"], 3)
        self.assertFalse(contract["use_causal"])

    def test_cp_aligned_padding_rows_extend_the_c4_contract(self):
        build_contract = self._load_c4_prefill_contract()
        forward_batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[6], extend_prefix_lens_cpu=[0]
        )
        contract = build_contract(
            forward_batch,
            torch.tensor([[1], [2], [3], [4], [5], [7], [0], [0]], dtype=torch.int32),
            torch.zeros((8, 2), dtype=torch.int32),
            torch.device("cpu"),
            8,
        )
        self.assertEqual(int(contract["qlod_cpu"][-1].item()), 8)
        self.assertEqual(contract["qlod_cpu"].tolist(), [0, 6, 7, 8])
        self.assertEqual(contract["last_rows"].tolist(), [5, 6, 7])
        self.assertEqual(contract["per_req_k_lens"].tolist(), [7, 1, 1])
        self.assertEqual(contract["klod_cpu"].tolist(), [0, 28, 32, 36])
        self.assertEqual(contract["com_k_start_cpu"].tolist(), [0, 0, 0])

    def test_cp_round_robin_prefill_uses_local_windows_and_full_context(self):
        build_contract = self._load_c4_prefill_contract(cp_ranks=(1, 4))
        forward_batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[9, 3],
            extend_prefix_lens_cpu=[0, 0],
            seq_lens_cpu=torch.tensor([40, 24], dtype=torch.int32),
            batch_size=2,
        )
        contract = build_contract(
            forward_batch,
            torch.ones((3, 1), dtype=torch.int32),
            torch.zeros((3, 2), dtype=torch.int32),
            torch.device("cpu"),
            3,
        )
        # cp_rank=1, cp_size=4: request 0 keeps ceil-ish share 2, the spill of
        # one token carries into request 1 which then keeps 1 token.
        self.assertEqual(contract["qlod_cpu"].tolist(), [0, 2, 3])
        self.assertEqual(contract["max_seq_q"], 2)
        self.assertEqual(contract["last_rows"].tolist(), [1, 2])
        # Full request context, not the c4 length at the last local token.
        self.assertEqual(contract["per_req_k_lens"].tolist(), [10, 6])
        self.assertEqual(contract["klod_cpu"].tolist(), [0, 40, 64])
        self.assertFalse(contract["use_causal"])

    def test_cp_round_robin_prefill_builds_per_token_attention_lod(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_make_cp_prefill_lod"
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)
        core_metadata = types.SimpleNamespace(
            seq_lens_casual=torch.tensor([17, 21, 25], dtype=torch.int32)
        )
        q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = namespace["_make_cp_prefill_lod"](
            core_metadata, 4, torch.device("cpu")
        )
        self.assertEqual(q_lod_cpu.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(q_lod.tolist(), q_lod_cpu.tolist())
        self.assertEqual(int(q_lod_cpu[-1].item()), 4)
        self.assertEqual(kv_lens_cpu.tolist(), [17, 21, 25, 1])
        self.assertEqual(kv_lens.tolist(), kv_lens_cpu.tolist())

    def test_cp_local_extend_lens_cover_every_global_token(self):
        path = ATTENTION_DIR / "kunlun_deepseek_v4_backend.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_dsa_cp_local_extend_lens"
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(compile(helpers, str(path), "exec"), namespace)
        split = namespace["_dsa_cp_local_extend_lens"]
        global_lens = [9, 3, 7]
        cp_size = 4
        totals = [sum(split(global_lens, rank, cp_size)) for rank in range(cp_size)]
        self.assertEqual(sum(totals), sum(global_lens))
        self.assertEqual(split(global_lens, 0, 1), global_lens)

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
