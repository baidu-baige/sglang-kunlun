import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_kunlun.hooks import mtp_production as mtp
from sglang_kunlun.hooks import production_precision as runtime
from sglang_kunlun.hooks import ragged_draft_extend as ragged


ROOT = Path(__file__).resolve().parents[1]
MODEL_HOOKS = ROOT / "sglang_kunlun" / "models" / "deepseek_v4_precision.py"
MODEL_PRECISION_SPEC = importlib.util.spec_from_file_location(
    "contract_production_deepseek_v4_precision", MODEL_HOOKS
)
assert MODEL_PRECISION_SPEC is not None and MODEL_PRECISION_SPEC.loader is not None
model_precision = importlib.util.module_from_spec(MODEL_PRECISION_SPEC)
sys.modules[MODEL_PRECISION_SPEC.name] = model_precision
MODEL_PRECISION_SPEC.loader.exec_module(model_precision)
RUNTIME_HOOKS = ROOT / "sglang_kunlun" / "hooks" / "production_precision.py"
MTP_HOOKS = ROOT / "sglang_kunlun" / "hooks" / "mtp_production.py"
BACKEND_HOOKS = (
    ROOT
    / "sglang_kunlun"
    / "hooks"
    / "layers"
    / "attention"
    / "kunlun_deepseek_v4_backend.py"
)
LEGACY_MODEL_HOOKS = ROOT / "sglang_kunlun" / "models" / "deepseek_v4.py"
UPSTREAM_DSV4_MODEL = (
    ROOT.parent / "sglang" / "python" / "sglang" / "srt" / "models" / "deepseek_v4.py"
)


class ProductionPrecisionContractTest(unittest.TestCase):
    def assert_registered(self, target, hook_type, function):
        registrations = HookRegistry._hooks[target]
        self.assertTrue(
            any(
                registered_type == hook_type and registered_hook is function
                for registered_type, registered_hook, _source in registrations
            ),
            f"missing {hook_type} registration for {target}",
        )

    def test_actual_plugin_registrations_and_signatures(self):
        targets = (
            (
                "sglang.srt.layers.attention.dsv4.indexer.C4Indexer.__init__",
                HookType.AFTER,
                model_precision.initialize_c4_indexer_parameter_dtype_kunlun,
                ("result", "self", "args", "kwargs"),
            ),
            (
                "sglang.srt.layers.attention.dsv4.compressor.Compressor.__init__",
                HookType.AFTER,
                model_precision.initialize_compressor_parameter_dtype_kunlun,
                ("result", "self", "args", "kwargs"),
            ),
            (
                "sglang.srt.models.deepseek_v2.MoEGate.forward",
                HookType.AROUND,
                model_precision.moe_gate_forward_half_precision_kunlun,
                (
                    "original_fn",
                    "self",
                    "hidden_states",
                    "gemm_output_zero_allocator",
                    "forward_batch",
                ),
            ),
            (
                "sglang.srt.models.deepseek_v4.MQALayer.__init__",
                HookType.AFTER,
                model_precision.initialize_mqa_rope_policy_kunlun,
                ("result", "self", "config", "args", "kwargs"),
            ),
            (
                "sglang.srt.models.deepseek_v4.MQALayer.forward",
                HookType.REPLACE,
                model_precision.mqa_forward_global_head_layout_kunlun,
                ("self", "x", "positions", "forward_batch", "x_quant"),
            ),
            (
                "sglang.srt.speculative.base_spec_worker."
                "EagleDraftWorkerBase.prepare_for_draft_extend",
                HookType.REPLACE,
                mtp.prepare_for_draft_extend_kunlun,
                (
                    "self",
                    "draft_extend_input",
                    "batch",
                    "predict",
                    "num_draft_tokens",
                    "draft_model_runner",
                    "cuda_graph_runner",
                ),
            ),
            (
                "sglang.srt.managers.scheduler_components.batch_result_processor."
                "SchedulerBatchResultProcessor._resolve_spec_v2_tokens",
                HookType.REPLACE,
                mtp.resolve_spec_v2_tokens_kunlun,
                ("self", "result", "batch"),
            ),
            (
                "sglang_kunlun.hooks.layers.attention."
                "kunlun_deepseek_v4_backend.KunlunDeepseekV4AttnBackend._make_lod",
                HookType.AROUND,
                ragged.make_ragged_draft_extend_lod_kunlun,
                ("original_fn", "self", "forward_batch", "num_queries", "device"),
            ),
        )
        for target, hook_type, function, expected_parameters in targets:
            self.assert_registered(target, hook_type, function)
            self.assertEqual(
                tuple(inspect.signature(function).parameters), expected_parameters
            )

    def test_moe_gate_forward_matches_058_half_contract_and_preserves_fallback(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                hidden_states = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=dtype
                )
                owner = types.SimpleNamespace(
                    is_deepseek_v4=True,
                    weight=torch.tensor(
                        [[2.0, -1.0], [0.5, 3.0]], dtype=dtype
                    ),
                )
                original_fn = mock.Mock(
                    side_effect=AssertionError("DSV4 half path must bypass upstream")
                )

                actual = model_precision.moe_gate_forward_half_precision_kunlun(
                    original_fn, owner, hidden_states
                )

                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, hidden_states @ owner.weight.T, rtol=0, atol=0
                )
                original_fn.assert_not_called()

        fallback = object()
        owner = types.SimpleNamespace(
            is_deepseek_v4=False,
            weight=torch.ones((2, 2), dtype=torch.float16),
        )
        hidden_states = torch.ones((1, 2), dtype=torch.float16)
        allocator = object()
        forward_batch = object()
        original_fn = mock.Mock(return_value=fallback)

        actual = model_precision.moe_gate_forward_half_precision_kunlun(
            original_fn,
            owner,
            hidden_states,
            allocator,
            forward_batch,
        )

        self.assertIs(actual, fallback)
        original_fn.assert_called_once_with(
            owner,
            hidden_states,
            allocator,
            forward_batch,
        )

    def test_mqa_init_matches_requested_wo_a_parameter_dtype(self):
        config = types.SimpleNamespace(rope_scaling=None)

        for requested_dtype, expected_dtype in (
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
        ):
            with self.subTest(requested_dtype=requested_dtype):
                weight = torch.nn.Parameter(
                    torch.ones((1, 2, 4), dtype=torch.bfloat16),
                    requires_grad=False,
                )
                from sglang.srt.layers.quantization.unquant import (
                    UnquantizedLinearMethod,
                )

                owner = types.SimpleNamespace(
                    compress_ratio=0,
                    wo_a=types.SimpleNamespace(
                        weight=weight,
                        quant_method=UnquantizedLinearMethod(),
                        params_dtype=torch.bfloat16,
                    ),
                )
                server_args = types.SimpleNamespace(dtype=requested_dtype)
                with mock.patch(
                    "sglang.srt.server_args.get_global_server_args",
                    return_value=server_args,
                ):
                    result = model_precision.initialize_mqa_rope_policy_kunlun(
                        None, owner, config
                    )

                self.assertIsNone(result)
                self.assertIs(owner.wo_a.weight, weight)
                self.assertEqual(owner.wo_a.params_dtype, expected_dtype)
                self.assertEqual(owner.wo_a.weight.dtype, expected_dtype)

    def test_mqa_forward_uses_deterministic_global_slots_and_local_output(self):
        upstream = types.ModuleType("sglang.srt.models.deepseek_v4")
        upstream.get_attn_tp_context = lambda: types.SimpleNamespace(
            input_scattered=False
        )
        upstream.envs = types.SimpleNamespace(
            SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=types.SimpleNamespace(
                get=lambda: False
            )
        )
        upstream._is_hip = False
        upstream._is_npu = False
        upstream._FP8_WO_A_GEMM = False
        upstream.fused_rope_inplace = mock.Mock()
        upstream.is_in_breakable_cuda_graph = lambda: False
        upstream.get_tensor_model_parallel_world_size = lambda: 2

        class Backend:
            def forward(inner_self, *args, **kwargs):
                inner_self.q = kwargs["q"].clone()
                inner_self.sink = kwargs["attn_sink"].clone()
                return kwargs["q"].clone()

        backend = Backend()
        forward_context = types.ModuleType(
            "sglang.srt.model_executor.forward_context"
        )
        forward_context.get_attn_backend = lambda: backend
        env_gate = types.ModuleType(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate"
        )
        env_gate.is_unified_kv_triton = lambda: False

        owner = types.SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            n_local_heads=2,
            n_heads=4,
            head_dim=1,
            n_local_groups=1,
            o_lora_rank=1,
            attn_sink=torch.tensor([10.0, 20.0, 30.0, 40.0]),
            _attn_sink_local=None,
            alt_streams=None,
            _multi_stream_bs_limit=64,
            dsa_enable_prefill_cp=False,
            compressor=None,
            compress_ratio=4,
            attn_mqa=types.SimpleNamespace(v_head_dim=1, layer_id=0),
            freqs_cis=torch.empty(0),
            qk_rope_head_dim=1,
            wo_a=types.SimpleNamespace(weight=torch.ones((1, 1, 2))),
            wo_b=lambda value: (value, None),
        )

        def forward_prepare(x, positions, forward_batch, attn_backend, q_out, x_quant=None):
            local_q = torch.tensor([[[5.0], [6.0]]])
            q_out.copy_(local_q)
            return local_q, None

        owner._forward_prepare = forward_prepare
        forward_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_extend=lambda: False)
        )
        models_package = types.ModuleType("sglang.srt.models")
        models_package.__path__ = []
        models_package.deepseek_v4 = upstream
        model_executor_package = types.ModuleType("sglang.srt.model_executor")
        model_executor_package.__path__ = []
        model_executor_package.forward_context = forward_context
        layers_package = types.ModuleType("sglang.srt.layers")
        layers_package.__path__ = []
        attention_package = types.ModuleType("sglang.srt.layers.attention")
        attention_package.__path__ = []
        dsv4_package = types.ModuleType("sglang.srt.layers.attention.dsv4")
        dsv4_package.__path__ = []
        unified_package = types.ModuleType(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels"
        )
        unified_package.__path__ = []
        unified_package.env_gate = env_gate
        dsv4_package.unified_kv_kernels = unified_package
        attention_package.dsv4 = dsv4_package
        layers_package.attention = attention_package
        modules = {
            upstream.__name__: upstream,
            models_package.__name__: models_package,
            forward_context.__name__: forward_context,
            model_executor_package.__name__: model_executor_package,
            layers_package.__name__: layers_package,
            attention_package.__name__: attention_package,
            dsv4_package.__name__: dsv4_package,
            unified_package.__name__: unified_package,
            env_gate.__name__: env_gate,
        }
        srt_module = sys.modules["sglang.srt"]
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(srt_module, "models", models_package, create=True),
            mock.patch.object(
                srt_module, "model_executor", model_executor_package, create=True
            ),
            mock.patch.object(srt_module, "layers", layers_package, create=True),
            mock.patch.object(
                model_precision,
                "dsv4_mqa_wo_a_einsum_kunlun",
                side_effect=lambda value, weight: torch.einsum(
                    "tgd,grd->tgr", value, weight
                ),
            ),
        ):
            output = model_precision.mqa_forward_global_head_layout_kunlun(
                owner,
                torch.ones((1, 2)),
                torch.tensor([0]),
                forward_batch,
            )

        self.assertEqual(backend.q.flatten().tolist(), [0.0, 0.0, 5.0, 6.0])
        self.assertEqual(backend.sink.tolist(), [0.0, 0.0, 30.0, 40.0])
        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(output.item(), 11.0)

    def test_ragged_accepted_only_falls_back_to_eager_and_builds_lod(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.utils import async_probe

        source = BACKEND_HOOKS.read_text()
        tree = ast.parse(source, filename=str(BACKEND_HOOKS))
        backend_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "KunlunDeepseekV4AttnBackend"
        )
        make_lod_node = next(
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_make_lod"
        )
        namespace = {
            "torch": torch,
            "_is_graph_extend_mode": lambda _mode: True,
        }
        exec(
            compile(
                ast.Module(body=[make_lod_node], type_ignores=[]),
                str(BACKEND_HOOKS),
                "exec",
            ),
            namespace,
        )
        make_lod = namespace["_make_lod"]

        accept_lens = torch.tensor([1, 3], dtype=torch.int32)
        accepted = mtp.accepted_prefix_indices(accept_lens, 4)
        full_cache_rows = torch.tensor(
            [100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int64
        )
        compact_cache_rows = full_cache_rows.index_select(0, accepted)

        batch = types.SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            spec_info=None,
            input_ids=None,
            model_config=types.SimpleNamespace(vocab_size=1000),
            forward_mode=types.SimpleNamespace(is_idle=lambda: False),
            capture_hidden_mode=None,
            prefix_lens=None,
            extend_lens=None,
            extend_num_tokens=None,
            out_cache_loc=compact_cache_rows,
        )
        draft_extend_input = types.SimpleNamespace(
            num_accept_tokens=accept_lens,
            extend_seq_lens_cpu=None,
            extend_seq_lens_tensor=None,
        )
        backend = types.SimpleNamespace(
            _attention_decode_aux={}, _attention_graph_extend_aux={}
        )
        metadata = {}

        def init_forward_metadata(forward_batch):
            metadata["lod"] = make_lod(
                backend,
                forward_batch,
                num_queries=4,
                device=torch.device("cpu"),
            )
            metadata["cache_rows"] = forward_batch.out_cache_loc.clone()

        runner = types.SimpleNamespace(
            device="cpu",
            spec_algorithm=types.SimpleNamespace(is_standalone=lambda: False),
            attn_backend=types.SimpleNamespace(
                init_forward_metadata=init_forward_metadata
            ),
        )
        graph_runner = types.SimpleNamespace(
            can_run_graph=mock.Mock(return_value=False)
        )

        def init_new(schedule_batch, _runner):
            return types.SimpleNamespace(
                seq_lens=schedule_batch.seq_lens.clone(),
                seq_lens_cpu=None,
                seq_lens_sum=None,
                extend_seq_lens=schedule_batch.extend_lens,
                extend_seq_lens_cpu=None,
                extend_prefix_lens_cpu=None,
                out_cache_loc=schedule_batch.out_cache_loc,
                forward_mode=types.SimpleNamespace(
                    is_decode_or_idle=lambda: False
                ),
                batch_size=2,
                mark_forward_metadata_ready=mock.Mock(),
            )

        with (
            mock.patch.object(ForwardBatch, "init_new", side_effect=init_new),
            mock.patch.object(async_probe, "maybe_detect_oob", return_value=None),
        ):
            forward_batch = mtp.prepare_for_draft_extend_kunlun(
                None,
                draft_extend_input,
                batch,
                torch.tensor([10, 20, 21, 22], dtype=torch.int64),
                4,
                runner,
                graph_runner,
            )

        self.assertTrue(forward_batch._kunlun_ragged_draft_extend)
        self.assertFalse(mtp.can_run_draft_extend_graph(graph_runner, forward_batch))
        graph_runner.can_run_graph.assert_not_called()
        self.assertFalse(forward_batch._kunlun_can_run_draft_extend_graph)
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [1, 3])
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [10, 20])
        self.assertEqual(forward_batch.seq_lens.tolist(), [11, 23])
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [11, 23])
        self.assertEqual(draft_extend_input.extend_seq_lens_cpu, [1, 3])

        q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = metadata["lod"]
        self.assertEqual(q_lod_cpu.tolist(), [0, 1, 4])
        self.assertEqual(q_lod.tolist(), [0, 1, 4])
        self.assertEqual(kv_lens_cpu.tolist(), [11, 23])
        self.assertEqual(kv_lens.tolist(), [11, 23])
        self.assertEqual(metadata["cache_rows"].tolist(), [100, 104, 105, 106])

        metadata.clear()
        graph_runner.can_run_graph.reset_mock()
        graph_runner.can_run_graph.return_value = True
        graph_accept_lens = torch.tensor([4, 3], dtype=torch.int32)
        batch.seq_lens = torch.tensor([10, 20], dtype=torch.int64)
        batch.seq_lens_cpu = None
        batch.out_cache_loc = full_cache_rows[:7]
        draft_extend_input.num_accept_tokens = graph_accept_lens
        draft_extend_input.extend_seq_lens_cpu = None
        draft_extend_input.extend_seq_lens_tensor = None
        with (
            mock.patch.object(ForwardBatch, "init_new", side_effect=init_new),
            mock.patch.object(async_probe, "maybe_detect_oob", return_value=None),
        ):
            graph_batch = mtp.prepare_for_draft_extend_kunlun(
                None,
                draft_extend_input,
                batch,
                torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int64),
                4,
                runner,
                graph_runner,
            )

        self.assertFalse(graph_batch._kunlun_ragged_draft_extend)
        self.assertTrue(graph_batch._kunlun_can_run_draft_extend_graph)
        graph_runner.can_run_graph.assert_called_once_with(graph_batch)
        self.assertEqual(metadata, {})
        self.assertEqual(graph_batch.extend_seq_lens_cpu, [4, 3])
        self.assertEqual(graph_batch.seq_lens.tolist(), [14, 23])

    def test_backend_redetects_ragged_when_custom_marker_is_lost(self):
        draft_extend_mode = types.SimpleNamespace(
            is_draft_extend_v2=lambda: True
        )
        forward_batch = types.SimpleNamespace(
            forward_mode=draft_extend_mode,
            batch_size=4,
            extend_seq_lens_cpu=[1, 2, 1, 2],
        )
        self.assertEqual(
            ragged._ragged_extend_lengths(forward_batch, 6),
            [1, 2, 1, 2],
        )

        forward_batch.batch_size = 3
        forward_batch.extend_seq_lens_cpu = [2, 2, 2]
        self.assertIsNone(ragged._ragged_extend_lengths(forward_batch, 6))

    def test_packed_rows_only_admit_fixed_width_safe_layouts(self):
        self.assertTrue(mtp.packed_rows_fit_fixed_width([1], 4))
        self.assertTrue(mtp.packed_rows_fit_fixed_width([4, 4, 2], 4))
        self.assertFalse(mtp.packed_rows_fit_fixed_width([4, 2, 4], 4))
        self.assertFalse(mtp.packed_rows_fit_fixed_width([1, 2, 1, 2], 4))

    def test_graph_runner_rejects_ragged_without_calling_original(self):
        original = mock.Mock(return_value=True)
        forward_batch = types.SimpleNamespace(_kunlun_ragged_draft_extend=True)
        self.assertFalse(
            mtp.reject_ragged_draft_extend_graph_kunlun(
                original, object(), forward_batch
            )
        )
        original.assert_not_called()

    def test_scheduler_fixed_stride_has_no_cross_request_leakage(self):
        class Request:
            def __init__(self):
                self.is_retracted = False
                self.grammar = None
                self.kv_committed_len = 10
                self.spec_verify_ct = 0
                self.spec_num_correct_drafts = 0
                self.histogram = []

            def finished(self):
                return False

            def update_spec_correct_drafts_histogram(self, value):
                self.histogram.append(value)

        requests = [Request(), Request()]
        result = types.SimpleNamespace(
            next_token_ids=torch.tensor(
                [10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int64
            ),
            accept_lens=torch.tensor([1, 3], dtype=torch.int32),
            speculative_num_draft_tokens=4,
            num_correct_drafts=None,
            num_correct_drafts_per_req_cpu=None,
        )
        model_worker = types.SimpleNamespace(on_verify_complete_cpu=mock.Mock())
        processor = types.SimpleNamespace(model_worker=model_worker)
        batch = types.SimpleNamespace(
            reqs=requests,
            spec_algorithm=types.SimpleNamespace(is_dflash=lambda: False),
        )

        resolved = mtp.resolve_spec_v2_tokens_kunlun(processor, result, batch)

        self.assertEqual(resolved, [[10], [20, 21, 22]])
        self.assertNotIn(13, resolved[1])
        self.assertEqual([request.kv_committed_len for request in requests], [10, 12])
        self.assertEqual([request.histogram for request in requests], [[0], [2]])
        model_worker.on_verify_complete_cpu.assert_called_once_with(
            [0, 2], batch_size=2
        )

    def test_capture_admits_kunlun_backend_without_class_rebinding(self):
        upstream = types.ModuleType("sglang.srt.speculative.eagle_worker_v2")
        upstream.Phase = types.SimpleNamespace(DECODE="decode")
        upstream.Backend = types.SimpleNamespace(DISABLED="disabled")
        upstream.check_cuda_graph_backend = lambda *_args: False
        upstream.EAGLEDraftNpuGraphRunner = type("NpuDraft", (), {})
        upstream.EAGLEDraftCudaGraphRunner = type("CudaDraft", (), {})
        extend_runner = mock.Mock(return_value="kunlun-extend-graph")
        upstream.EAGLEDraftExtendNpuGraphRunner = extend_runner
        upstream.EAGLEDraftExtendCudaGraphRunner = extend_runner
        upstream._is_cuda = True
        upstream._is_musa = False
        upstream._is_npu = False
        upstream._is_hip = False
        upstream.TritonAttnBackend = type("Triton", (), {})
        upstream.TRTLLMMLABackend = type("TRTMLA", (), {})
        upstream.TRTLLMHAAttnBackend = type("TRTHA", (), {})
        upstream.TokenspeedMLABackend = type("Tokenspeed", (), {})
        upstream.FlashInferAttnBackend = type("FlashInfer", (), {})

        backend_module = types.ModuleType(
            "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend"
        )
        backend_class = type("KunlunDeepseekV4AttnBackend", (), {})
        backend_module.KunlunDeepseekV4AttnBackend = backend_class
        backend = backend_class()
        owner = types.SimpleNamespace(
            cuda_graph_runner="stale",
            cuda_graph_runner_for_draft_extend="stale",
            server_args=types.SimpleNamespace(model_impl="torch"),
            speculative_num_steps=1,
            target_worker=types.SimpleNamespace(device="cuda"),
            draft_attn_backend=None,
            draft_extend_attn_backend=backend,
        )
        original_flashinfer = upstream.FlashInferAttnBackend

        speculative_package = types.ModuleType("sglang.srt.speculative")
        speculative_package.__path__ = []
        speculative_package.eagle_worker_v2 = upstream
        layers_package = types.ModuleType("sglang_kunlun.hooks.layers")
        layers_package.__path__ = []
        attention_package = types.ModuleType(
            "sglang_kunlun.hooks.layers.attention"
        )
        attention_package.__path__ = []
        attention_package.kunlun_deepseek_v4_backend = backend_module
        layers_package.attention = attention_package
        srt_module = sys.modules["sglang.srt"]
        hooks_package = sys.modules["sglang_kunlun.hooks"]
        with (
            mock.patch.dict(
                sys.modules,
                {
                    upstream.__name__: upstream,
                    speculative_package.__name__: speculative_package,
                    layers_package.__name__: layers_package,
                    attention_package.__name__: attention_package,
                    backend_module.__name__: backend_module,
                },
            ),
            mock.patch.object(
                srt_module, "speculative", speculative_package, create=True
            ),
            mock.patch.object(hooks_package, "layers", layers_package, create=True),
        ):
            mtp.capture_cuda_graphs_kunlun(owner)

        self.assertEqual(
            owner.cuda_graph_runner_for_draft_extend, "kunlun-extend-graph"
        )
        extend_runner.assert_called_once_with(owner)
        self.assertIs(upstream.FlashInferAttnBackend, original_flashinfer)

    def test_compression_clear_survives_restore_without_double_clear(self):
        class Buffer:
            def __init__(self):
                self.calls = 0

            def clear(self):
                self.calls += 1

        class RestoredOwner:
            def __init__(self):
                self.kv_score_buffer = Buffer()

        class DirtyOwner:
            def __init__(self):
                self.kv_score_buffer = Buffer()
                self.kv_score_buffer.clear()

        restored = RestoredOwner()
        runtime.initialize_non_online_compress_state_kunlun(
            None, restored, 1, 1, False, 4, torch.float32, "cpu", False, 4,
            online=False,
        )
        self.assertEqual(restored.kv_score_buffer.calls, 1)

        dirty = DirtyOwner()
        runtime.initialize_non_online_compress_state_kunlun(
            None, dirty, 1, 1, False, 4, torch.float32, "cpu", False, 4,
            online=False,
        )
        self.assertEqual(dirty.kv_score_buffer.calls, 1)

    def test_draft_position_is_advanced_before_forward(self):
        observed = []

        class Runner:
            def forward(self, batch):
                observed.append(batch.positions.clone())
                return "ok"

        owner = types.SimpleNamespace(draft_runner=Runner())
        batch = types.SimpleNamespace(positions=torch.tensor([7], dtype=torch.int32))

        def original(worker, forward_batch):
            result = worker.draft_runner.forward(forward_batch)
            forward_batch.positions.add_(1)
            return result

        result = mtp.draft_forward_position_kunlun(original, owner, batch)
        self.assertEqual(result, "ok")
        self.assertEqual(observed[0].tolist(), [8])
        self.assertEqual(batch.positions.tolist(), [8])

    def test_backend_monkey_patches_are_exact_plugin_hooks(self):
        source = BACKEND_HOOKS.read_text()
        tree = ast.parse(source, filename=str(BACKEND_HOOKS))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        expected = {
            "_create_paged_compressor_data_kunlun": (
                "sglang.srt.layers.attention.deepseek_v4_backend."
                "create_paged_compressor_data",
                "AROUND",
            ),
            "_generate_compressor_prefill_plan_kunlun": (
                "sglang.jit_kernel.dsv4.compress_old."
                "CompressorPrefillPlan.generate",
                "REPLACE",
            ),
            "_compressor_forward_cuda_kunlun": (
                "sglang.srt.layers.attention.dsv4.compressor."
                "Compressor.forward_cuda",
                "REPLACE",
            ),
        }
        for name, (target, hook_type) in expected.items():
            function = functions[name]
            decorators = [
                decorator
                for decorator in function.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "plugin_hook"
            ]
            self.assertEqual(len(decorators), 1)
            decorator = decorators[0]
            self.assertEqual(ast.literal_eval(decorator.args[0]), target)
            type_arg = next(
                keyword.value
                for keyword in decorator.keywords
                if keyword.arg == "type"
            )
            self.assertEqual(ast.unparse(type_arg), f"HookType.{hook_type}")

        forbidden_assignments = {
            "upstream.create_paged_compressor_data",
            "CompressorPrefillPlan.generate",
            "Compressor.forward_cuda",
        }
        assigned = {
            ast.unparse(target)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
        }
        self.assertTrue(forbidden_assignments.isdisjoint(assigned))

        adapter_node = functions["_create_paged_compressor_data_kunlun"]
        adapter_node.decorator_list = []
        namespace = {}
        exec(
            compile(
                ast.Module(body=[adapter_node], type_ignores=[]),
                str(BACKEND_HOOKS),
                "exec",
            ),
            namespace,
        )
        original = mock.Mock(return_value="paged-data")
        result = namespace["_create_paged_compressor_data_kunlun"](
            original,
            "payload",
            online_state_slot_offset=7,
            keep=True,
        )
        self.assertEqual(result, "paged-data")
        original.assert_called_once_with("payload", keep=True)

    def test_preserved_runtime_contracts_and_no_hot_global_rebinding(self):
        model_source = MODEL_HOOKS.read_text()
        mtp_source = MTP_HOOKS.read_text()
        runtime_source = RUNTIME_HOOKS.read_text()
        legacy_model_source = LEGACY_MODEL_HOOKS.read_text()
        upstream_model_source = UPSTREAM_DSV4_MODEL.read_text()
        backend_source = BACKEND_HOOKS.read_text()

        self.assertNotIn("_dsv4_dump_probe", upstream_model_source)
        self.assertNotIn("_dsv4_module_name", upstream_model_source)
        self.assertNotIn("dsv4_probe_bridge", upstream_model_source)
        self.assertNotIn("_dsv4_dump_probe", model_source)
        self.assertNotIn("_dsv4_dump_probe", legacy_model_source)
        self.assertIn("original_seq_len=0", model_source)
        self.assertIn("local_q_out = torch.empty_like(q)", model_source)
        self.assertIn("kv = self.kv_norm(kv)", model_source)
        self.assertIn("drop_page_margin=False", runtime_source)
        self.assertIn("device=req_to_token_pool.device", runtime_source)
        self.assertIn("current_cpu.to(device=batch.device)", mtp_source)
        self.assertIn("next_cpu.to(device=batch.device)", mtp_source)

        for forbidden in (
            "upstream.get_attn_backend =",
            "upstream.FlashInferAttnBackend =",
            "upstream.fill_bonus_tokens =",
            "module.torch =",
            "model_module.torch =",
        ):
            self.assertNotIn(forbidden, model_source)
            self.assertNotIn(forbidden, mtp_source)
            self.assertNotIn(forbidden, legacy_model_source)
            self.assertNotIn(forbidden, backend_source)

        for source in (model_source, mtp_source, runtime_source):
            for forbidden in (
                "torch.save",
                "logger.",
                "DUMP",
                "PROBE",
                "debug_dumps",
            ):
                self.assertNotIn(forbidden, source)

    def test_registry_imports_all_production_hook_modules(self):
        registry = (ROOT / "sglang_kunlun" / "hooks" / "registry.py").read_text()
        models = (ROOT / "sglang_kunlun" / "models" / "__init__.py").read_text()
        self.assertIn('"sglang_kunlun.hooks.production_precision"', registry)
        self.assertIn('"sglang_kunlun.hooks.mtp_production"', registry)
        self.assertIn('"sglang_kunlun.hooks.ragged_draft_extend"', registry)
        self.assertIn("from . import deepseek_v4_precision", models)


if __name__ == "__main__":
    unittest.main()
