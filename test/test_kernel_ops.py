import importlib
import sys
import types
import unittest
from unittest import mock

import torch


class KernelOpsTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("sglang.fake_kernel_source", None)
        sys.modules.pop("sglang.fake_kernel_user", None)
        sys.modules.pop("sglang.fake_dsv4_source", None)
        sys.modules.pop("sglang.fake_dsv4_user", None)

    def test_install_patches_imported_bindings_automatically(self):
        from sglang_kunlun.kernels import kernel_ops

        def original_kernel(*args, **kwargs):
            return "original"

        source_module = types.ModuleType("sglang.fake_kernel_source")
        source_module.fake_kernel = original_kernel
        user_module = types.ModuleType("sglang.fake_kernel_user")
        user_module.fake_kernel = original_kernel
        sys.modules[source_module.__name__] = source_module
        sys.modules[user_module.__name__] = user_module

        def replacement_kernel(*args, **kwargs):
            return "replacement"

        spec = kernel_ops.KernelSpec(source_module.__name__, "fake_kernel", replacement_kernel)
        with mock.patch.object(kernel_ops, "_TRITON_OPS", {(source_module.__name__, "fake_kernel"): spec}), mock.patch.object(
            kernel_ops, "_JIT_OPS", {}
        ):
            kernel_ops.install()

        self.assertIs(source_module.fake_kernel.impl, replacement_kernel)
        self.assertIs(user_module.fake_kernel.impl, replacement_kernel)
        self.assertEqual(user_module.fake_kernel[lambda meta: (1,)](), "replacement")

    def test_direct_call_style_patches_function_without_launcher(self):
        from sglang_kunlun.kernels import kernel_ops

        def original_kernel():
            return "original"

        source_module = types.ModuleType("sglang.fake_kernel_source")
        source_module.fake_kernel = original_kernel
        sys.modules[source_module.__name__] = source_module

        def replacement_kernel():
            return "replacement"

        spec = kernel_ops.KernelSpec(
            source_module.__name__,
            "fake_kernel",
            replacement_kernel,
            {"call_style": "direct"},
        )
        with mock.patch.object(kernel_ops, "_TRITON_OPS", {(source_module.__name__, "fake_kernel"): spec}), mock.patch.object(
            kernel_ops, "_JIT_OPS", {}
        ):
            kernel_ops.install()

        self.assertIs(source_module.fake_kernel, replacement_kernel)
        self.assertEqual(source_module.fake_kernel(), "replacement")

    def test_dsv4_registration_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        expected_jit = {
            ("sglang.jit_kernel.dsv4.compress_old", "compress_forward"),
            (
                "sglang.jit_kernel.dsv4.compress_old",
                "compress_fused_norm_rope_inplace",
            ),
            ("sglang.jit_kernel.dsv4.attn", "triton_create_paged_compress_data"),
            ("sglang.jit_kernel.dsv4.topk", "topk_transform_512"),
            ("sglang.jit_kernel.dsv4.topk", "topk_transform_512_v2"),
            ("sglang.jit_kernel.dsv4.elementwise", "fused_rope_inplace"),
            ("sglang.jit_kernel.dsv4.elementwise", "fused_q_norm_rope"),
            (
                "sglang.jit_kernel.dsv4.elementwise",
                "fused_q_indexer_rope_hadamard_quant",
            ),
            ("sglang.jit_kernel.dsv4.moe", "hash_topk"),
            ("sglang.jit_kernel.dsv4.gemm", "linear_bf16_fp32"),
            ("sglang.jit_kernel.dsv4.moe", "silu_and_mul_clamp"),
            (
                "sglang.jit_kernel.dsv4.elementwise",
                "fused_k_norm_rope_flashmla",
            ),
            ("sglang.srt.layers.attention.dsv4.indexer", "fused_scale"),
            (
                "sglang.srt.layers.attention.dsv4.metadata_kernel",
                "init_compression_metadata",
            ),
            (
                "sglang.srt.layers.attention.dsv4.quant_k_cache",
                "quant_to_nope_fp8_rope_bf16_pack_triton",
            ),
        }
        self.assertTrue(expected_jit.issubset(kernel_ops.registered_jit_ops()))
        self.assertIn(
            (
                "sglang.srt.layers.attention.dsv4.index_buf_accessor",
                "_set_k_and_s_triton",
            ),
            kernel_ops.registered_triton_ops(),
        )

    def test_dsv4_page_indices_have_deterministic_canonical_order(self):
        from sglang_kunlun.kernels import kernel_ops

        page_indices = torch.tensor(
            [[9, -1, 3, 7, -1], [8, 2, 5, 1, 4]], dtype=torch.int32
        )
        pointer = page_indices.data_ptr()

        kernel_ops._canonicalize_dsv4_page_indices_(page_indices)

        self.assertEqual(page_indices.data_ptr(), pointer)
        self.assertEqual(
            page_indices.tolist(),
            [[3, 7, 9, -1, -1], [1, 2, 4, 5, 8]],
        )

    def test_dsv4_topk_uses_exact_graph_safe_selection(self):
        from sglang_kunlun.kernels import kernel_ops

        scores = torch.tensor(
            [[0.0, 4.0, 1.0, 3.0, 2.0, -1.0]], dtype=torch.float32
        )
        page_table = torch.tensor([[10]], dtype=torch.int32)
        output = torch.empty((1, 4), dtype=torch.int32)
        pointer = output.data_ptr()

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "topk_transform",
            side_effect=AssertionError("fused top-k must not run"),
        ):
            kernel_ops.dsv4_topk_transform_512_kunlun(
                scores,
                torch.tensor([5], dtype=torch.int32),
                page_table,
                output,
                64,
            )

        self.assertEqual(output.data_ptr(), pointer)
        self.assertEqual(output.tolist(), [[641, 642, 643, 644]])

        kernel_ops.dsv4_topk_transform_512_kunlun(
            scores,
            torch.tensor([2], dtype=torch.int32),
            page_table,
            output,
            64,
        )
        self.assertEqual(output.tolist(), [[640, 641, -1, -1]])

    def test_dsv4_stale_binding_install_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        def upstream_compress():
            return "upstream"

        source = types.ModuleType("sglang.fake_dsv4_source")
        source.compress_forward = upstream_compress
        user = types.ModuleType("sglang.fake_dsv4_user")
        user.compress_forward = upstream_compress
        sys.modules[source.__name__] = source
        sys.modules[user.__name__] = user
        spec = kernel_ops.KernelSpec(
            source.__name__,
            "compress_forward",
            kernel_ops.dsv4_compress_forward_kunlun,
        )

        with mock.patch.object(
            kernel_ops, "_JIT_OPS", {(source.__name__, "compress_forward"): spec}
        ), mock.patch.object(kernel_ops, "_TRITON_OPS", {}):
            kernel_ops.install()

        self.assertIs(source.compress_forward, kernel_ops.dsv4_compress_forward_kunlun)
        self.assertIs(user.compress_forward, kernel_ops.dsv4_compress_forward_kunlun)

    def test_dsv4_compress_prefill_matches_058_wrapper_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        kv_score_buffer = torch.randn(4, 4, 256)
        kv_score_input = torch.randn(4, 512)
        ape = torch.randn(4, 256)
        indices = torch.arange(4, dtype=torch.int32)
        extra_data = torch.arange(8, dtype=torch.int32)
        plan = types.SimpleNamespace(
            is_decode=False,
            compress_ratio=4,
            compress_plan=torch.arange(6, dtype=torch.int32),
            write_plan=torch.arange(3, dtype=torch.int32),
        )
        expected = torch.arange(4 * 128, dtype=torch.float32).view(4, 128)

        def reference_operator(*args):
            args[2].copy_(expected)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "compress_forward_fast",
            side_effect=reference_operator,
        ) as op:
            result = kernel_ops.dsv4_compress_forward_kunlun(
                kv_score_buffer,
                kv_score_input,
                ape,
                indices,
                plan=plan,
                extra_data=extra_data,
                head_dim=128,
                compress_ratio=4,
            )

        self.assertIs(op.call_args.args[0], kv_score_buffer)
        self.assertIs(op.call_args.args[1], kv_score_input)
        self.assertIs(op.call_args.args[2], result)
        self.assertIs(op.call_args.args[3], ape)
        self.assertIs(op.call_args.args[4], indices)
        self.assertIs(op.call_args.args[5], plan.compress_plan)
        self.assertIs(op.call_args.args[6], plan.write_plan)
        self.assertIs(op.call_args.args[7], extra_data)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_dsv4_quant_k_cache_matches_058_bf16_then_fp16_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        values = torch.tensor(
            [[0.2325735986, -0.6675894856] + [0.0] * 510],
            dtype=torch.float32,
        )
        pack = kernel_ops.dsv4_quant_k_cache_kunlun(values)
        expected = values.to(torch.bfloat16).to(torch.float16)

        self.assertEqual(pack.k_nope_fp8.dtype, torch.float16)
        self.assertTrue(pack.k_nope_fp8.is_contiguous())
        self.assertTrue(torch.equal(pack.k_nope_fp8, expected))
        self.assertFalse(torch.equal(pack.k_nope_fp8, values.to(torch.float16)))

    def test_dsv4_set_k_and_s_calls_058_xspeedgate_writer(self):
        from types import SimpleNamespace

        from sglang_kunlun.kernels import kernel_ops

        buf = torch.empty((2, 8 * 512), dtype=torch.float16)
        loc = torch.tensor([-1, 15], dtype=torch.int64)
        k_value = torch.randn((2, 512), dtype=torch.float16)
        pack = SimpleNamespace(k_nope_fp8=k_value)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "set_k_and_s_v4",
        ) as set_k_and_s:
            kernel_ops.dsv4_set_k_and_s_kunlun(buf, loc, pack, page_size=8)

        set_k_and_s.assert_called_once()
        actual_buf, actual_loc, actual_k, actual_page_size = (
            set_k_and_s.call_args.args
        )
        self.assertIs(actual_buf, buf)
        self.assertIs(actual_k, k_value)
        self.assertEqual(actual_page_size, 8)
        torch.testing.assert_close(
            actual_loc, torch.tensor([0, 15], dtype=torch.int64), rtol=0, atol=0
        )

    def test_dsv4_mapping_writer_matches_058_raw_location_contract(self):
        from types import SimpleNamespace

        from sglang_kunlun.kernels import kernel_ops

        buf = torch.empty((2, 8 * 512), dtype=torch.float16)
        raw_loc_base = torch.tensor([3, 99, 7, 99], dtype=torch.int64)
        raw_loc = raw_loc_base[::2]
        self.assertFalse(raw_loc.is_contiguous())
        mapping = torch.arange(32, dtype=torch.int32)
        k_value = torch.randn((2, 512), dtype=torch.float16)
        pack = SimpleNamespace(k_nope_fp8=k_value)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "set_k_and_s_v4_with_mapping",
        ) as set_k_and_s:
            kernel_ops.dsv4_set_k_and_s_with_mapping_kunlun(
                buf, raw_loc, mapping, pack, page_size=8
            )

        set_k_and_s.assert_called_once()
        actual_buf, actual_raw_loc, actual_mapping, actual_k, actual_page_size = (
            set_k_and_s.call_args.args
        )
        self.assertIs(actual_buf, buf)
        self.assertTrue(actual_raw_loc.is_contiguous())
        self.assertEqual(actual_raw_loc.dtype, torch.int32)
        expected_raw_loc = torch.as_tensor(raw_loc, dtype=torch.int32)
        torch.testing.assert_close(actual_raw_loc, expected_raw_loc, rtol=0, atol=0)
        self.assertIs(actual_mapping, mapping)
        self.assertIs(actual_k, k_value)
        self.assertEqual(actual_page_size, 8)

    def test_dsv4_create_paged_compress_matches_058_formula(self):
        from sglang_kunlun.kernels import kernel_ops

        compress_ratio = 4
        swa_page_size = 8
        ring_size = 16
        req_pool_indices = torch.tensor([1, 2], dtype=torch.int32)
        seq_lens = torch.tensor([13, 18], dtype=torch.int64)
        extend_seq_lens = torch.tensor([7, 9], dtype=torch.int64)
        req_to_token = torch.arange(3 * 32, dtype=torch.int32).view(3, 32)
        full_to_swa = torch.arange(3 * 32 + 32, dtype=torch.int32) * 3

        def reference_operator(**kwargs):
            rid = kwargs["req_pool_indices"].to(torch.int64)
            seq = kwargs["seq_lens"].to(torch.int64)
            extend = kwargs["extend_seq_lens"].to(torch.int64)
            req_table = kwargs["req_to_token"]
            mapping = kwargs["full_to_swa_index_mapping"]
            prefix = seq - extend
            write_pos = ((seq - 1) // compress_ratio) * compress_ratio
            load_pos = ((prefix - 1) // compress_ratio) * compress_ratio
            positions = torch.stack(
                (load_pos - compress_ratio, load_pos,
                 write_pos - compress_ratio, write_pos), dim=1
            ).clamp_min(0)
            full_loc = req_table[rid[:, None], positions]
            swa_loc = mapping[full_loc]
            state_loc = (
                (swa_loc // kwargs["swa_page_size"]) * kwargs["ring_size"]
                + swa_loc % kwargs["ring_size"]
            ) // compress_ratio
            write_loc = state_loc[:, 1].to(torch.int32)
            extra_data = torch.cat(
                (state_loc[:, 2:3], state_loc[:, 0:1],
                 state_loc[:, 3:4], write_pos[:, None].to(torch.int32)),
                dim=1,
            ).to(torch.int32)
            return write_loc, extra_data

        expected = reference_operator(
            req_pool_indices=req_pool_indices.to(torch.int64),
            seq_lens=seq_lens.to(torch.int32).contiguous(),
            extend_seq_lens=extend_seq_lens.to(torch.int32).contiguous(),
            req_to_token=req_to_token,
            full_to_swa_index_mapping=full_to_swa.to(torch.int64),
            swa_page_size=swa_page_size,
            ring_size=ring_size,
        )
        op = mock.Mock(side_effect=lambda **kwargs: reference_operator(**kwargs))
        with mock.patch.object(
            torch.ops.xspeedgate_ops, "create_paged_compress_data", op
        ):
            actual = kernel_ops.dsv4_create_paged_compress_data_kunlun(
                compress_ratio=compress_ratio,
                is_overlap=True,
                swa_page_size=swa_page_size,
                ring_size=ring_size,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                req_to_token=req_to_token,
                full_to_swa_index_mapping=full_to_swa,
            )

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        kwargs = op.call_args.kwargs
        self.assertEqual(kwargs["req_pool_indices"].dtype, torch.int64)
        self.assertEqual(kwargs["seq_lens"].dtype, torch.int32)
        self.assertTrue(kwargs["seq_lens"].is_contiguous())
        self.assertEqual(kwargs["extend_seq_lens"].dtype, torch.int32)
        self.assertTrue(kwargs["extend_seq_lens"].is_contiguous())
        self.assertEqual(kwargs["full_to_swa_index_mapping"].dtype, torch.int64)

    def test_dsv4_fused_rope_preserves_inplace_contract(self):

        from sglang_kunlun.kernels import kernel_ops

        q_base = torch.zeros(2, 3, 8)
        q = q_base[..., 2:6]
        k_base = torch.zeros(2, 2, 8)
        k = k_base[..., 1:5]
        freqs_cis = torch.polar(torch.ones(16, 2), torch.zeros(16, 2))
        positions = torch.tensor([1, 3], dtype=torch.int64)
        q_rotated = torch.arange(q.numel(), dtype=q.dtype).reshape(2, 3, 4)
        k_rotated = torch.arange(k.numel(), dtype=k.dtype).reshape(2, 2, 4)
        op = mock.Mock(return_value=(q_rotated, k_rotated))

        with mock.patch.object(
            torch.ops.xspeedgate_ops, "flashinfer_rotary_embedding", op
        ):
            result = kernel_ops.dsv4_fused_rope_inplace_kunlun(
                q, k, freqs_cis, positions, inverse=True
            )

        self.assertIsNone(result)
        torch.testing.assert_close(q, q_rotated)
        torch.testing.assert_close(k, k_rotated)
        kwargs = op.call_args.kwargs
        self.assertTrue(kwargs["query"].is_contiguous())
        self.assertTrue(kwargs["key"].is_contiguous())
        self.assertEqual(kwargs["query"].shape, (2, 3, 4))
        self.assertEqual(kwargs["key"].shape, (2, 2, 4))
        self.assertFalse(kwargs["is_neox_style"])
        self.assertTrue(kwargs["inverse"])

    def test_dsv4_act_quant_matches_058_kunlun_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        hook_registry = types.ModuleType("sglang.srt.plugins.hook_registry")
        hook_registry.HookType = types.SimpleNamespace(REPLACE="replace")
        hook_registry.plugin_hook = lambda **_kwargs: lambda fn: fn
        old_hook_registry = sys.modules.get(hook_registry.__name__)
        sys.modules[hook_registry.__name__] = hook_registry
        module_name = "sglang_kunlun_test_act_quant"
        module_path = (
            kernel_ops.__file__.rsplit("/kernels/", 1)[0]
            + "/hooks/layers/attention/nsa/triton_kernel.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            if old_hook_registry is None:
                sys.modules.pop(hook_registry.__name__, None)
            else:
                sys.modules[hook_registry.__name__] = old_hook_registry
        act_quant_kunlun = module.act_quant_kunlun

        x = torch.tensor(
            [[-2.0, -1.0, 0.0, 2.0], [0.25, -0.5, 1.0, -1.0]],
            dtype=torch.bfloat16,
        ).contiguous()
        expected_q = torch.tensor(
            [[-127, -64, 0, 127], [32, -64, 127, -127]], dtype=torch.int8
        )
        expected_scale = torch.tensor([[2.0], [1.0]], dtype=torch.float32)

        def reference_quant2d(input_tensor, output, scale, force_sdnn=False):
            self.assertIs(input_tensor, x)
            self.assertTrue(force_sdnn)
            output.copy_(expected_q)
            scale.copy_(expected_scale)

        with mock.patch.object(
            module, "quant2d", side_effect=reference_quant2d
        ) as quant2d:
            actual_q, actual_scale = act_quant_kunlun(x, block_size=4)

        quant2d.assert_called_once()
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)

    def test_hadamard_matches_058_fp32_matmul_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        x = torch.arange(8, dtype=torch.float32).view(2, 4)
        expected = torch.full_like(x, 3.0)

        def reference_gen_hadamard_matrix(matrix, scale):
            self.assertEqual(matrix.dtype, torch.float32)
            self.assertEqual(matrix.shape, (4, 4))
            self.assertEqual(scale, 0.5)
            matrix.fill_(1.0)

        def reference_matmul(input_tensor, matrix, output, trans_a, trans_b, alpha, beta):
            self.assertIs(input_tensor, x)
            self.assertEqual(input_tensor.dtype, torch.float32)
            self.assertEqual(matrix.dtype, torch.float32)
            self.assertFalse(trans_a)
            self.assertTrue(trans_b)
            self.assertEqual(alpha, 1.0)
            self.assertEqual(beta, 0.0)
            output.copy_(expected)

        with mock.patch(
            "kunlun_ops.gen_hadamard_matrix",
            side_effect=reference_gen_hadamard_matrix,
        ) as gen_matrix, mock.patch(
            "kunlun_ops.matmul", side_effect=reference_matmul
        ) as matmul:
            actual = kernel_ops.hadamard_transform(x, 0.5)

        gen_matrix.assert_called_once()
        matmul.assert_called_once()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dsv4_hash_topk_uses_kunlun_operator_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        router_logits = torch.zeros(2, 4)
        input_ids = torch.tensor([3, 7], dtype=torch.int32)
        tid2eid = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.4, 0.6], [0.2, 0.8]])
        op = mock.Mock(return_value=(topk_ids, topk_weights))

        with mock.patch.object(
            torch.ops.xspeedgate_ops, "moe_hash_topk_fused", op
        ):
            weights, ids = kernel_ops.dsv4_hash_topk_kunlun(
                router_logits, input_ids, tid2eid, 0, 1.0
            )

        self.assertIs(weights, topk_weights)
        self.assertIs(ids, topk_ids)
        self.assertEqual(op.call_args.args[1].dtype, torch.int64)

    def test_dsv4_mqa_wo_a_einsum_matches_058_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        o = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).view(2, 3, 4)
        weight = torch.arange(3 * 5 * 4, dtype=torch.bfloat16).view(3, 5, 4)
        expected = torch.full((2, 3, 5), 7.0, dtype=torch.bfloat16)

        def reference_operator(input_tensor, weight_tensor):
            self.assertIs(input_tensor, o)
            self.assertTrue(input_tensor.is_contiguous())
            self.assertIs(weight_tensor, weight)
            return expected

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "einsum_tgd_grd_tgr",
            side_effect=reference_operator,
            create=True,
        ) as op:
            actual = kernel_ops.dsv4_mqa_wo_a_einsum_kunlun(o, weight)

        op.assert_called_once()
        self.assertIs(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dsv4_mqa_forward_uses_full_058_attention_sink(self):
        from sglang_kunlun.kernels import kernel_ops

        hook_registry = types.ModuleType("sglang.srt.plugins.hook_registry")
        hook_registry.HookType = types.SimpleNamespace(REPLACE="replace", AROUND="around")
        hook_registry.plugin_hook = lambda *_args, **_kwargs: lambda fn: fn
        forward_batch_info = types.ModuleType(
            "sglang.srt.model_executor.forward_batch_info"
        )
        forward_batch_info.ForwardBatch = object
        stubs = (hook_registry, forward_batch_info)
        old_modules = {stub.__name__: sys.modules.get(stub.__name__) for stub in stubs}
        for stub in stubs:
            sys.modules[stub.__name__] = stub

        module_path = (
            kernel_ops.__file__.rsplit("/kernels/", 1)[0] + "/models/deepseek_v4.py"
        )
        spec = importlib.util.spec_from_file_location(
            "sglang_kunlun_test_dsv4_model", module_path
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            for name, old_module in old_modules.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        original_torch = types.SimpleNamespace(einsum=lambda *_args: None)
        owner_module = types.ModuleType("sglang.fake_dsv4_model")
        owner_module.torch = original_torch
        sys.modules[owner_module.__name__] = owner_module
        self.addCleanup(sys.modules.pop, owner_module.__name__, None)

        local_sink = torch.zeros(64, dtype=torch.float32)
        full_sink = torch.arange(64, dtype=torch.float32)
        mqa = types.SimpleNamespace(
            tp_size=8,
            _attn_sink_local=local_sink,
            attn_sink=full_sink,
        )

        def original_forward(instance):
            self.assertIs(instance._attn_sink_local, full_sink)
            return "result"

        original_forward.__module__ = owner_module.__name__
        actual = module.mqa_forward_with_kunlun_wo_a(original_forward, mqa)

        self.assertEqual(actual, "result")
        self.assertIs(mqa._attn_sink_local, local_sink)
        self.assertIs(owner_module.torch, original_torch)

        gate = types.SimpleNamespace(
            weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        )
        hidden_states = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
        gate_output = module.moe_gate_forward_kunlun(gate, hidden_states)
        torch.testing.assert_close(
            gate_output, hidden_states @ gate.weight.T, rtol=0, atol=0
        )
        self.assertEqual(gate_output.dtype, torch.bfloat16)

        qkv_a = torch.arange(12, dtype=torch.bfloat16).view(2, 6)
        expected_kv = qkv_a[..., 2:] + 1
        fused_rope = mock.Mock()
        upstream_model = types.ModuleType("sglang.srt.models.deepseek_v4")
        upstream_model.fused_rope_inplace = fused_rope
        backend = types.SimpleNamespace(store_cache=mock.Mock())
        mqa_layer = types.SimpleNamespace(
            q_lora_rank=2,
            kv_norm=lambda value: value + 1,
            qk_rope_head_dim=2,
            freqs_cis=torch.empty(0),
            layer_id=7,
        )
        forward_batch = object()
        positions = torch.tensor([3, 4], dtype=torch.int64)
        with mock.patch.dict(
            sys.modules,
            {"sglang.srt.models.deepseek_v4": upstream_model},
        ):
            module.compute_kv_to_cache_kunlun(
                mqa_layer,
                torch.empty(2, 1),
                positions,
                forward_batch,
                backend,
                qkv_a=qkv_a,
            )

        fused_rope.assert_called_once()
        rope_args = fused_rope.call_args.args
        self.assertTrue(torch.equal(rope_args[0], expected_kv[..., -2:].unsqueeze(1)))
        self.assertIs(rope_args[2], mqa_layer.freqs_cis)
        self.assertIs(rope_args[3], positions)
        backend.store_cache.assert_called_once()
        store_kwargs = backend.store_cache.call_args.kwargs
        self.assertEqual(store_kwargs["layer_id"], 7)
        self.assertTrue(torch.equal(store_kwargs["swa_k"], expected_kv))
        self.assertIs(store_kwargs["forward_batch"], forward_batch)

        events = []
        fused_qk_rope = mock.Mock()

        def apply_fused_qk_rope(q, k, _freqs, _positions):
            events.append("rope")
            q.add_(4)
            k.add_(5)

        fused_qk_rope.side_effect = apply_fused_qk_rope
        upstream_model.fused_rope_inplace = fused_qk_rope
        indexer = mock.Mock(side_effect=lambda **_kwargs: events.append("indexer"))
        backend = types.SimpleNamespace(
            store_cache=mock.Mock(side_effect=lambda **_kwargs: events.append("store")),
            forward_core_compressor=mock.Mock(
                side_effect=lambda *_args: events.append("compressor")
            ),
        )
        q_lora = torch.arange(4, dtype=torch.float32).view(2, 2)
        kv_raw = torch.arange(8, dtype=torch.float32).view(2, 4)
        mqa_layer = types.SimpleNamespace(
            dsa_enable_prefill_cp=False,
            fuse_wqa_wkv=False,
            wq_a=lambda _value: (q_lora.clone(), None),
            wkv=lambda _value: (kv_raw.clone(), None),
            q_norm=lambda value: value + 1,
            wq_b=lambda value: (torch.cat((value, value), dim=-1), None),
            n_local_heads=1,
            head_dim=4,
            eps=1e-6,
            kv_norm=lambda value: value + 2,
            qk_rope_head_dim=2,
            freqs_cis=torch.empty(0),
            layer_id=2,
            indexer=indexer,
            compressor=object(),
        )
        forward_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_extend=lambda: True)
        )

        def rmsnorm(input_tensor, _weight, output, *_args):
            output.copy_(input_tensor + 3)

        env_gate = types.ModuleType(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate"
        )
        env_gate.is_unified_kv_triton = lambda: False
        with mock.patch.object(
            module.kunlun_ops, "rmsnorm", side_effect=rmsnorm
        ), mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.models.deepseek_v4": upstream_model,
                env_gate.__name__: env_gate,
            },
        ):
            actual_q, actual_kv = module.mqa_forward_prepare_058_kunlun(
                mock.Mock(),
                mqa_layer,
                torch.empty(2, 1),
                positions,
                forward_batch,
                backend,
            )

        expected_q = torch.cat((q_lora + 1, q_lora + 1), dim=-1).view(2, 1, 4)
        expected_q.add_(3)
        expected_q[..., -2:].add_(4)
        expected_kv = kv_raw + 2
        expected_kv[..., -2:].add_(5)
        self.assertEqual(events, ["rope", "store", "indexer", "compressor"])
        fused_qk_rope.assert_called_once()
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        self.assertIsNone(actual_kv)
        torch.testing.assert_close(
            backend.store_cache.call_args.kwargs["swa_k"],
            expected_kv,
            rtol=0,
            atol=0,
        )

        events.clear()
        fused_qk_rope.reset_mock()
        backend.store_cache.reset_mock()
        backend.forward_core_compressor.reset_mock()
        indexer.reset_mock()
        original_prepare = mock.Mock(
            side_effect=AssertionError("ordinary decode must use the 0.5.8 path")
        )
        q_padded = torch.full((2, 4, 4), -777.0)
        q_out = q_padded[:, 2:3, :]
        self.assertFalse(q_out.is_contiguous())
        decode_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_extend=lambda: False,
                is_decode_or_idle=lambda: True,
            )
        )

        with mock.patch.object(
            module.kunlun_ops, "rmsnorm", side_effect=rmsnorm
        ), mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.models.deepseek_v4": upstream_model,
                env_gate.__name__: env_gate,
            },
        ):
            decode_q, decode_kv = module.mqa_forward_prepare_058_kunlun(
                original_prepare,
                mqa_layer,
                torch.empty(2, 1),
                positions,
                decode_batch,
                backend,
                q_out,
            )

        original_prepare.assert_not_called()
        self.assertIs(decode_q, q_out)
        self.assertIsNone(decode_kv)
        self.assertEqual(events, ["rope", "store", "indexer", "compressor"])
        fused_qk_rope.assert_called_once()
        torch.testing.assert_close(q_out, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(
            q_padded[:, :2, :],
            torch.full((2, 2, 4), -777.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            q_padded[:, 3:, :],
            torch.full((2, 1, 4), -777.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            backend.store_cache.call_args.kwargs["swa_k"],
            expected_kv,
            rtol=0,
            atol=0,
        )

    def test_dsv4_linear_bf16_fp32_matches_058_torch_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        y = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.bfloat16)

        with mock.patch.object(torch, "mm") as mm:
            actual = kernel_ops.dsv4_linear_bf16_fp32_kunlun(x, y)

        mm.assert_not_called()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, x.float() @ y.float().t())

    def test_dsv4_silu_and_mul_clamp_matches_058_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        input = torch.tensor(
            [[12.0, -3.0, 20.0, -20.0], [-2.0, 4.0, 3.0, -5.0]]
        )
        output = torch.empty(2, 2)
        calls = []

        def swiglu(*, x, y):
            calls.append(x.clone())
            gate, up = x.chunk(2, dim=-1)
            y.copy_(torch.nn.functional.silu(gate) * up)

        kunlun_ops = types.ModuleType("kunlun_ops")
        kunlun_ops.swiglu = swiglu
        with mock.patch.dict(sys.modules, {"kunlun_ops": kunlun_ops}):
            kernel_ops.dsv4_silu_and_mul_clamp_kunlun(input, output, 10.0)

        gate, up = input.chunk(2, dim=-1)
        clamped = torch.cat(
            [gate.clamp(max=10.0), up.clamp(min=-10.0, max=10.0)], dim=-1
        )
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0], clamped, rtol=0, atol=0)
        expected_gate, expected_up = clamped.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(expected_gate) * expected_up
        torch.testing.assert_close(output, expected)

    def test_dsv4_metadata_adapts_legacy_eight_value_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        seq_lens = torch.tensor([0, 127, 128, 257], dtype=torch.int64)
        positions = torch.zeros(4, dtype=torch.int32)
        raw_out_loc = torch.arange(4, dtype=torch.int64)
        legacy_values = (
            raw_out_loc.to(torch.int32),
            positions,
            torch.zeros(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            raw_out_loc.to(torch.int32),
            positions,
            torch.ones(4, dtype=torch.int32),
            None,
        )
        op = mock.Mock(return_value=legacy_values)
        with mock.patch.object(
            torch.ops.xspeedgate_ops, "init_compressed_attn_metadata", op
        ):
            values = kernel_ops.dsv4_init_compression_metadata_kunlun(
                seq_lens,
                positions,
                raw_out_loc,
                compute_page_indices=False,
            )

        self.assertEqual(len(values), 9)
        self.assertEqual(values[0].dtype, torch.int32)
        self.assertEqual(values[4].dtype, torch.int32)
        torch.testing.assert_close(
            values[6], torch.tensor([0, 0, 1, 2], dtype=torch.int32)
        )

    def test_grouped_topk_preserves_058_statistic_padding(self):
        from sglang_kunlun.kernels import kernel_ops

        scores = mock.Mock(shape=(4, 256), device="xpu")
        bias = mock.Mock()
        statistic = object()
        values = object()
        indices = object()
        fake_ops = types.SimpleNamespace(moe_sigmoid_group_topk_norm=mock.Mock())

        with mock.patch.object(
            kernel_ops.torch,
            "empty",
            side_effect=[statistic, values, indices],
        ) as empty, mock.patch.dict(sys.modules, {"kunlun_ops": fake_ops}):
            result = kernel_ops.grouped_topk(scores, bias, 8, 4, 8, True, 2.5)

        self.assertEqual(empty.call_args_list[0].args, (12, 257))
        self.assertEqual(result, (values, indices))

    def test_register_all_installs_kernel_ops(self):
        registry = importlib.import_module("sglang_kunlun.hooks.registry")
        with mock.patch(
            "sglang_kunlun.kernels.kernel_ops.install"
        ) as install, mock.patch.object(registry.importlib, "import_module"):
            registry.register_all()
        install.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
