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

    def test_dsv4_compress_state_initializes_all_non_online_slots(self):
        from sglang.srt.mem_cache.deepseek_v4_compress_state import (
            CompressStatePool,
            KVAndScore,
        )
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        from sglang_kunlun.hooks import production_precision as runtime_precision

        target = (
            "sglang.srt.mem_cache.deepseek_v4_compress_state."
            "CompressStatePool.__init__"
        )
        self.assertTrue(
            any(
                hook_type == HookType.AFTER
                and hook is runtime_precision.initialize_non_online_compress_state_kunlun
                for hook_type, hook, _source in HookRegistry._hooks[target]
            )
        )

        def fake_alloc(pool, *, dtype, device, enable_memory_saver):
            pool.kv_score_buffer = KVAndScore(
                torch.full((pool._size, pool.last_dim), 7, dtype=dtype)
            )

        init_kwargs = dict(
            size=5,
            ring_size=128,
            overlap=False,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=128,
            swa_page_size=256,
        )
        with mock.patch.object(
            CompressStatePool, "_alloc_kv_score_buffer", fake_alloc
        ):
            pool = CompressStatePool(**init_kwargs)

        self.assertTrue((pool.kv_score_buffer.kv[:-1] == 7).all())
        self.assertTrue((pool.kv_score_buffer.score[:-1] == 7).all())
        runtime_precision.initialize_non_online_compress_state_kunlun(
            None,
            pool,
            **init_kwargs,
            online=False,
        )

        self.assertTrue(
            torch.equal(
                pool.kv_score_buffer.kv, torch.zeros_like(pool.kv_score_buffer.kv)
            )
        )
        self.assertTrue(torch.isneginf(pool.kv_score_buffer.score).all())

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
            (
                "sglang.kernels.ops.attention.dsv4.topk",
                "topk_transform_512",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.topk",
                "topk_transform_512_v2",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.elementwise",
                "fused_rope_inplace",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.elementwise",
                "fused_q_norm_rope",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.elementwise",
                "fused_q_indexer_rope_hadamard_quant",
            ),
            ("sglang.kernels.ops.moe.moe_fused_gate", "moe_fused_gate"),
            ("sglang.kernels.ops.attention.dsv4.moe", "hash_topk"),
            ("sglang.jit_kernel.dsv4.gemm", "linear_bf16_fp32"),
            ("sglang.kernels.ops.attention.dsv4.moe", "silu_and_mul_clamp"),
            (
                "sglang.kernels.ops.attention.dsv4.elementwise",
                "fused_k_norm_rope_flashmla",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.metadata_kernel",
                "init_compression_metadata",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.compress",
                "compress_forward",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.compress",
                "compress_norm_rope_store",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.quant_k_cache",
                "quant_to_nope_fp8_rope_bf16_pack_triton",
            ),
            (
                "sglang.kernels.ops.speculative.cache_locs",
                "assign_extend_cache_locs_uniform_func",
            ),
            (
                "sglang.kernels.ops.attention.dsv4.c128_cleanup",
                "clear_unaccepted_c128_draft_states",
            ),
        }
        self.assertTrue(expected_jit.issubset(kernel_ops.registered_jit_ops()))
        self.assertIn(
            (
                "sglang.kernels.ops.attention.dsv4.index_buf_accessor",
                "_set_k_and_s_triton",
            ),
            kernel_ops.registered_triton_ops(),
        )

    def test_spec_uniform_cache_locs_uses_torch_row_major_gather(self):
        from sglang_kunlun.kernels import kernel_ops

        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
            ],
            dtype=torch.int32,
        )

        output = kernel_ops.assign_extend_cache_locs_uniform_torch(
            req_pool_indices=torch.tensor([2, 0], dtype=torch.int32),
            req_to_token=req_to_token,
            start_offset=torch.tensor([1, 3], dtype=torch.int64),
            batch_size=2,
            draft_token_num=2,
            device=req_to_token.device,
        )

        self.assertEqual(output.dtype, torch.int64)
        self.assertEqual(output.tolist(), [21, 22, 3, 4])

    def test_dsv4_c128_cleanup_resets_only_rejected_ring_slots(self):
        from sglang_kunlun.kernels import kernel_ops

        state = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        original = state.clone()
        kernel_ops.clear_unaccepted_c128_draft_states_torch(
            state,
            req_pool_indices=torch.tensor([1, 0], dtype=torch.int32),
            seq_lens=torch.tensor([3, 1], dtype=torch.int64),
            accept_lens=torch.tensor([1, 2], dtype=torch.int32),
            ring_size=4,
            num_draft_tokens=3,
        )

        reset_row = torch.tensor(
            [0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf")]
        )
        self.assertTrue(torch.equal(state[4], reset_row))
        self.assertTrue(torch.equal(state[5], reset_row))
        self.assertTrue(torch.equal(state[3], reset_row))
        self.assertTrue(torch.equal(state[0], original[0]))
        self.assertTrue(torch.equal(state[1], original[1]))
        self.assertTrue(torch.equal(state[2], original[2]))
        self.assertTrue(torch.equal(state[6], original[6]))
        self.assertTrue(torch.equal(state[7], original[7]))

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

        self.assertEqual(pack.k_nope_fp8.shape[-1], 448 + 64)
        self.assertEqual(pack.k_nope_fp8.dtype, torch.float16)
        self.assertTrue(pack.k_nope_fp8.is_contiguous())
        self.assertIsNone(pack.k_rope_bf16)
        self.assertIsNone(pack.scale_k_nope_ue8m0)
        self.assertTrue(torch.equal(pack.k_nope_fp8, expected))
        self.assertFalse(torch.equal(pack.k_nope_fp8, values.to(torch.float16)))

    def test_dsv4_set_k_and_s_writes_torch_cache_slots(self):
        from types import SimpleNamespace

        from sglang_kunlun.kernels import kernel_ops

        buf = torch.zeros((2, 8 * 512), dtype=torch.float16)
        loc = torch.tensor([-1, 15], dtype=torch.int64)
        k_value = torch.randn((2, 512), dtype=torch.float16)
        pack = SimpleNamespace(k_nope_fp8=k_value)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "set_k_and_s_v4",
            side_effect=AssertionError("vendor SWA writer must not be called"),
        ) as set_k_and_s:
            kernel_ops.dsv4_set_k_and_s_kunlun(buf, loc, pack, page_size=8)

        set_k_and_s.assert_not_called()
        cache_rows = buf.view(-1, 512)
        torch.testing.assert_close(cache_rows[0], k_value[0], rtol=0, atol=0)
        torch.testing.assert_close(cache_rows[15], k_value[1], rtol=0, atol=0)

    def test_dsv4_mapping_writer_translates_raw_locations_in_torch(self):
        from types import SimpleNamespace

        from sglang_kunlun.kernels import kernel_ops

        buf = torch.zeros((2, 8 * 512), dtype=torch.float16)
        raw_loc_base = torch.tensor([3, 99, 7, 99], dtype=torch.int64)
        raw_loc = raw_loc_base[::2]
        self.assertFalse(raw_loc.is_contiguous())
        mapping = torch.arange(32, dtype=torch.int32)
        k_value = torch.randn((2, 512), dtype=torch.float16)
        pack = SimpleNamespace(k_nope_fp8=k_value)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "set_k_and_s_v4_with_mapping",
            side_effect=AssertionError("vendor mapped SWA writer must not be called"),
        ) as set_k_and_s:
            kernel_ops.dsv4_set_k_and_s_with_mapping_kunlun(
                buf, raw_loc, mapping, pack, page_size=8
            )

        set_k_and_s.assert_not_called()
        cache_rows = buf.view(-1, 512)
        torch.testing.assert_close(cache_rows[3], k_value[0], rtol=0, atol=0)
        torch.testing.assert_close(cache_rows[7], k_value[1], rtol=0, atol=0)

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
        q.copy_(torch.arange(q.numel(), dtype=q.dtype).reshape_as(q))
        k_base = torch.zeros(2, 2, 8)
        k = k_base[..., 1:5]
        k.copy_(torch.arange(k.numel(), dtype=k.dtype).reshape_as(k) + 1)
        angles = torch.outer(
            torch.arange(16, dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
        )
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        positions = torch.tensor([1, 3], dtype=torch.int64)

        def expected(value):
            value_complex = torch.view_as_complex(
                value.float().contiguous().reshape(*value.shape[:-1], -1, 2)
            )
            freqs = freqs_cis.index_select(0, positions).conj().unsqueeze(1)
            return torch.view_as_real(value_complex * freqs).flatten(-2)

        expected_q = expected(q)
        expected_k = expected(k)
        vendor_op = mock.Mock(
            side_effect=AssertionError("Torch reference RoPE must bypass vendor op")
        )
        with mock.patch.object(
            torch.ops.xspeedgate_ops, "flashinfer_rotary_embedding", vendor_op
        ):
            result = kernel_ops.dsv4_fused_rope_inplace_kunlun(
                q, k, freqs_cis, positions, inverse=True
            )

        self.assertIsNone(result)
        vendor_op.assert_not_called()
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(k, expected_k, rtol=0, atol=0)

    def test_dsv4_indexer_rope_uses_xspeedgate_fixed_layout_op(self):
        from sglang_kunlun.kernels import kernel_ops

        value = torch.randn(2, 64, 128)
        freqs_cis = torch.polar(
            torch.ones(16, 32),
            torch.randn(16, 32),
        )
        positions = torch.tensor([1, 3], dtype=torch.int64)
        rotated = value + 1
        vendor_op = mock.Mock(return_value=rotated)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "dsv4_rotate_gptj_tail",
            vendor_op,
        ):
            actual = kernel_ops._dsv4_rotate_gptj_tail(
                value, freqs_cis, positions
            )

        torch.testing.assert_close(actual, rotated, rtol=0, atol=0)
        kwargs = vendor_op.call_args.kwargs
        self.assertEqual(kwargs["value"].device, value.device)
        self.assertEqual(kwargs["value"].shape, (2, 64, 128))
        self.assertEqual(kwargs["freqs_cis"].dtype, torch.complex64)
        self.assertEqual(kwargs["positions"].dtype, torch.int32)
        self.assertTrue(kwargs["value"].is_contiguous())
        self.assertTrue(kwargs["freqs_cis"].is_contiguous())
        self.assertTrue(kwargs["positions"].is_contiguous())
        self.assertFalse(kwargs["inverse"])

    def test_dsv4_q_norm_rope_is_pure_torch(self):
        from sglang_kunlun.kernels import kernel_ops

        q_input = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6) - 5
        q_output = torch.empty_like(q_input)
        positions = torch.tensor([0, 2], dtype=torch.int64)
        angles = torch.outer(
            torch.arange(4, dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
        )
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        eps = 1e-6

        normalized = q_input * torch.rsqrt(
            q_input.square().mean(dim=-1, keepdim=True) + eps
        )
        tail = normalized[..., -4:]
        tail_complex = torch.view_as_complex(tail.contiguous().reshape(2, 2, 2, 2))
        rotated = torch.view_as_real(
            tail_complex * freqs_cis.index_select(0, positions).unsqueeze(1)
        ).flatten(-2)
        expected = normalized.clone()
        expected[..., -4:].copy_(rotated)

        vendor_rope = mock.Mock(
            side_effect=AssertionError("Torch reference RoPE must bypass vendor op")
        )
        vendor_norm = mock.Mock(
            side_effect=AssertionError("Torch reference RMSNorm must bypass vendor op")
        )
        with mock.patch.object(
            torch.ops.xspeedgate_ops, "flashinfer_rotary_embedding", vendor_rope
        ), mock.patch("kunlun_ops.rmsnorm", vendor_norm):
            kernel_ops.dsv4_fused_q_norm_rope_kunlun(
                q_input, q_output, eps, freqs_cis, positions
            )

        vendor_rope.assert_not_called()
        vendor_norm.assert_not_called()
        torch.testing.assert_close(q_output, expected, rtol=0, atol=0)

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

    def test_dsv4_q_indexer_reference_preserves_weight_output_rank(self):
        from sglang_kunlun.kernels import kernel_ops

        q_input = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
        weight = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        with mock.patch.object(
            kernel_ops, "_dsv4_rotate_gptj_tail", return_value=q_input
        ), mock.patch.object(
            kernel_ops, "_dsv4_hadamard_torch", return_value=q_input
        ):
            q_int8, weights = (
                kernel_ops.dsv4_fused_q_indexer_rope_hadamard_quant_kunlun(
                    q_input,
                    weight,
                    0.5,
                    torch.empty(0),
                    torch.tensor([0, 1]),
                )
            )

        q_scale = q_input.abs().amax(dim=-1, keepdim=True)
        expected_weights = weight.unsqueeze(-1) * 0.5 * q_scale
        self.assertEqual(q_int8.shape, q_input.shape)
        self.assertEqual(weights.shape, (2, 3, 1))
        torch.testing.assert_close(weights, expected_weights)

    def test_dsv4_moe_fused_gate_uses_torch_reference_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        router_logits = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        correction_bias = torch.tensor([10.0, 0.0, 0.0, 0.0])
        expected_ids = torch.tensor([[0, 3, 4], [0, 1, 4]], dtype=torch.int32)
        activated = torch.nn.functional.softplus(router_logits).sqrt()
        selected = torch.gather(activated, 1, expected_ids[:, :2].long())
        expected_weights = torch.cat(
            (
                selected / selected.sum(dim=-1, keepdim=True) * 1.5,
                torch.ones((2, 1)),
            ),
            dim=-1,
        )

        weights, ids = kernel_ops.dsv4_moe_fused_gate_kunlun(
            router_logits,
            correction_bias,
            topk=3,
            num_fused_shared_experts=1,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
        )

        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(ids, expected_ids)

        routed_weights, routed_ids = kernel_ops.dsv4_moe_fused_gate_kunlun(
            router_logits,
            correction_bias,
            topk=3,
            num_fused_shared_experts=0,
        )
        expected_routed_ids = torch.tensor(
            [[0, 3, 2], [0, 1, 2]], dtype=torch.int32
        )
        expected_routed_weights = torch.gather(
            activated, 1, expected_routed_ids.long()
        )
        expected_routed_weights /= expected_routed_weights.sum(
            dim=-1, keepdim=True
        )
        torch.testing.assert_close(routed_ids, expected_routed_ids)
        torch.testing.assert_close(routed_weights, expected_routed_weights)

    def test_dsv4_hash_topk_uses_torch_reference_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        router_logits = torch.tensor(
            [[-1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 0.0]]
        )
        input_ids = torch.tensor([1, 3], dtype=torch.int32)
        tid2eid = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [3, 0]], dtype=torch.int32
        )
        expected_ids = torch.tensor([[2, 3, 4], [3, 0, 4]], dtype=torch.int32)
        selected_logits = torch.tensor([[1.0, 2.0], [0.0, 3.0]])
        routed_weights = torch.nn.functional.softplus(selected_logits).sqrt()
        routed_weights /= routed_weights.sum(dim=-1, keepdim=True)
        expected_weights = torch.cat(
            (routed_weights, torch.full((2, 1), 0.5)), dim=-1
        )

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "moe_hash_topk_fused",
            side_effect=AssertionError("real HashTopK operator must not be called"),
        ) as op:
            weights, ids = kernel_ops.dsv4_hash_topk_kunlun(
                router_logits, input_ids, tid2eid, 1, 2.0
            )

        op.assert_not_called()
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(ids, expected_ids)

    def test_dsv4_mqa_wo_a_einsum_matches_058_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        o = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).view(2, 3, 4)
        weight = torch.arange(3 * 5 * 4, dtype=torch.bfloat16).view(3, 5, 4)
        expected = torch.einsum("tgd,grd->tgr", o, weight)

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "einsum_tgd_grd_tgr",
            side_effect=AssertionError("vendor MQA reduction must not be called"),
            create=True,
        ) as op:
            actual = kernel_ops.dsv4_mqa_wo_a_einsum_kunlun(o, weight)

        op.assert_not_called()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dsv4_model_hooks_cover_moe_gate_and_kv_store(self):
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
        forward_context = types.ModuleType(
            "sglang.srt.model_executor.forward_context"
        )
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
            {
                "sglang.srt.models.deepseek_v4": upstream_model,
                forward_context.__name__: forward_context,
            },
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
        store_call = backend.store_cache.call_args.kwargs
        self.assertEqual(store_call["layer_id"], 7)
        self.assertIs(store_call["forward_batch"], forward_batch)
        torch.testing.assert_close(store_call["swa_k"], expected_kv)

        events = []
        fused_qk_rope = mock.Mock()

        def apply_fused_qk_rope(q, k, _freqs, _positions):
            events.append("rope")
            q.add_(4)
            k.add_(5)

        fused_qk_rope.side_effect = apply_fused_qk_rope
        upstream_model.fused_rope_inplace = fused_qk_rope
        indexer = mock.Mock(side_effect=lambda **_kwargs: events.append("indexer"))

        def store_cache(**_kwargs):
            events.append("store")

        backend = types.SimpleNamespace(
            store_cache=mock.Mock(side_effect=store_cache),
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

        env_gate = types.ModuleType(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        )
        env_gate.is_unified_kv_triton = lambda: False
        with mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.models.deepseek_v4": upstream_model,
                "sglang.srt.model_executor.forward_context": forward_context,
                env_gate.__name__: env_gate,
            },
        ):
            actual_q, actual_kv = module.mqa_forward_prepare_kunlun(
                mock.Mock(),
                mqa_layer,
                torch.empty(2, 1),
                positions,
                forward_batch,
                backend,
            )

        expected_q = torch.cat((q_lora + 1, q_lora + 1), dim=-1).view(2, 1, 4)
        expected_q = expected_q * torch.rsqrt(
            expected_q.float().square().mean(dim=-1, keepdim=True) + mqa_layer.eps
        ).to(expected_q.dtype)
        expected_q[..., -2:].add_(4)
        expected_kv = kv_raw + 2
        expected_kv[..., -2:].add_(5)
        self.assertEqual(events, ["rope", "store", "indexer", "compressor"])
        fused_qk_rope.assert_called_once()
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        self.assertIsNone(actual_kv)
        backend.store_cache.assert_called_once()
        prepare_store_call = backend.store_cache.call_args.kwargs
        self.assertEqual(prepare_store_call["layer_id"], 2)
        self.assertIs(prepare_store_call["forward_batch"], forward_batch)
        torch.testing.assert_close(prepare_store_call["swa_k"], expected_kv)

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

        with mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.models.deepseek_v4": upstream_model,
                "sglang.srt.model_executor.forward_context": forward_context,
                env_gate.__name__: env_gate,
            },
        ):
            decode_q, decode_kv = module.mqa_forward_prepare_kunlun(
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
        backend.store_cache.assert_called_once()
        decode_store_call = backend.store_cache.call_args.kwargs
        self.assertEqual(decode_store_call["layer_id"], 2)
        self.assertIs(decode_store_call["forward_batch"], decode_batch)
        torch.testing.assert_close(decode_store_call["swa_k"], expected_kv)

    def test_dsv4_linear_bf16_fp32_preserves_shared_fp32_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
                y = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=dtype)

                actual = kernel_ops.dsv4_linear_bf16_fp32_kunlun(x, y)

                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(
                    actual,
                    torch.nn.functional.linear(x.float(), y.float()),
                    rtol=0,
                    atol=0,
                )

    def test_dsv4_silu_and_mul_clamp_matches_058_contract(self):
        from sglang_kunlun.kernels import kernel_ops

        input = torch.tensor(
            [[12.0, -3.0, 20.0, -20.0], [-2.0, 4.0, 3.0, -5.0]]
        )
        output = torch.empty(2, 2)

        gate, up = input.chunk(2, dim=-1)
        expected = (
            torch.nn.functional.silu(gate.clamp(max=10.0))
            * up.clamp(min=-10.0, max=10.0)
        )

        def forbidden_swiglu(*args, **kwargs):
            raise AssertionError("real Kunlun SwiGLU operator must not be called")

        kunlun_ops = types.ModuleType("kunlun_ops")
        kunlun_ops.swiglu = forbidden_swiglu
        with mock.patch.dict(sys.modules, {"kunlun_ops": kunlun_ops}):
            kernel_ops.dsv4_silu_and_mul_clamp_kunlun(input, output, 10.0)

        torch.testing.assert_close(output, expected)

    def test_dsv4_expand_prefill_causally_matches_padded_torch_contract(self):
        from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
            ExpandPrefillCausally,
        )
        from sglang_kunlun.kernels import kernel_ops

        result = kernel_ops.dsv4_expand_prefill_causally_torch(
            ExpandPrefillCausally,
            req_pool_indices=torch.tensor([5, 9], dtype=torch.int64),
            seq_lens=torch.tensor([6, 10], dtype=torch.int64),
            extend_seq_lens=torch.tensor([2, 3], dtype=torch.int64),
            extend_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens_cpu=None,
            extend_seq_lens_cpu=None,
            num_tokens=5,
            padded_num_tokens=7,
        )

        self.assertEqual(result.seq_lens_casual.tolist(), [5, 6, 8, 9, 10, 1, 1])
        self.assertEqual(
            result.req_pool_indices_repeated.tolist(), [5, 5, 9, 9, 9, 9, 9]
        )

    def test_dsv4_c4_decode_plan_matches_v2_byte_abi(self):
        from sglang_kunlun.kernels import kernel_ops

        req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
        req_to_token = torch.arange(3 * 16, dtype=torch.int32).view(3, 16)
        full_to_state = torch.arange(3 * 16, dtype=torch.int64)
        plan = kernel_ops.dsv4_compressor_decode_plan_torch(
            4,
            req_pool_indices,
            req_to_token,
            full_to_state,
            torch.tensor([5, 9], dtype=torch.int64),
            8,
            8,
        )

        self.assertEqual(plan.plan_d.dtype, torch.uint8)
        self.assertEqual(plan.plan_d.shape, (2, 16))
        self.assertEqual(
            plan.plan_d.view(torch.int32).tolist(),
            [[5, 20, 4, 5], [9, 40, 9, 10]],
        )

    def test_dsv4_prefill_plans_match_v2_c4_and_c128_contracts(self):
        from sglang_kunlun.kernels import kernel_ops

        req_to_token = torch.arange(3 * 256, dtype=torch.int32).view(3, 256)
        full_to_state = torch.arange(3 * 256, dtype=torch.int64)
        c4 = kernel_ops.dsv4_compressor_prefill_plan_torch(
            4,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([10], dtype=torch.int64),
            torch.tensor([10], dtype=torch.int64),
            req_to_token,
            full_to_state,
            8,
            8,
            10,
        )
        self.assertEqual(
            c4.plan_c.view(torch.int32).tolist(),
            [[4, (4 << 16) | 3, 0, 0], [8, 7, -1, 0]],
        )
        self.assertEqual(
            c4.plan_w.view(torch.int32).tolist(),
            [[ragged_id, ragged_id] for ragged_id in range(4, 10)],
        )

        c128 = kernel_ops.dsv4_compressor_prefill_plan_torch(
            128,
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([130], dtype=torch.int64),
            torch.tensor([130], dtype=torch.int64),
            req_to_token,
            full_to_state,
            256,
            256,
            130,
        )
        self.assertEqual(c128.plan_c.view(torch.int32).tolist(), [[128, 127, -1, 0]])
        self.assertEqual(
            c128.plan_w.view(torch.int32).tolist(),
            [[128, 640], [129, 641]],
        )

    def test_dsv4_c128_short_prefill_keeps_empty_compress_plan(self):
        from sglang_kunlun.kernels import kernel_ops

        plan = kernel_ops.dsv4_compressor_prefill_plan_torch(
            128,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            torch.arange(256, dtype=torch.int32).reshape(1, 256),
            torch.arange(256, dtype=torch.int64),
            256,
            256,
            8,
        )

        self.assertEqual(plan.plan_c.shape, (0, 16))
        self.assertEqual(
            plan.plan_w.view(torch.int32).tolist(),
            [[token_id, token_id] for token_id in range(8)],
        )

    def test_dsv4_prefill_graph_padding_uses_invalid_plan_sentinels(self):
        from sglang_kunlun.kernels import kernel_ops

        req_to_token = torch.arange(16, dtype=torch.int32).view(1, 16)
        full_to_state = torch.arange(16, dtype=torch.int64)
        plan = kernel_ops.dsv4_compressor_prefill_plan_torch(
            4,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64),
            req_to_token,
            full_to_state,
            8,
            8,
            8,
            use_cuda_graph=True,
        )
        plan_c = plan.plan_c.view(torch.int32)
        plan_w = plan.plan_w.view(torch.int32)

        self.assertEqual(plan_c.shape, (8, 4))
        self.assertEqual(plan_w.shape, (8, 2))
        self.assertEqual(plan_c[:2].tolist(), [[4, (4 << 16) | 3, 0, 0], [8, 7, -1, 0]])
        self.assertTrue((plan_c[2:, 0] == -1).all())
        self.assertEqual(plan_w[:4].tolist(), [[4, 4], [5, 5], [6, 6], [7, 7]])
        self.assertTrue((plan_w[4:] == -1).all())

    def test_dsv4_prefill_plan_supports_empty_batch(self):
        from sglang_kunlun.kernels import kernel_ops

        plan = kernel_ops.dsv4_compressor_prefill_plan_torch(
            128,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty((0, 0), dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            256,
            256,
            0,
            use_cuda_graph=True,
        )
        self.assertEqual(plan.plan_c.shape, (0, 16))
        self.assertEqual(plan.plan_w.shape, (0, 8))

    def test_dsv4_c4_paged_logits_use_physical_pages_scales_and_lengths(self):
        from sglang_kunlun.kernels import kernel_ops

        page_size = 64
        head_dim = 128
        page_bytes = page_size * (head_dim + 4)
        cache = torch.zeros((2, page_bytes), dtype=torch.uint8)
        values = cache[:, : page_size * head_dim].reshape(2, page_size, head_dim)
        scales = cache[:, page_size * head_dim :].view(torch.float32)
        values.view(torch.int8)[1, 0].fill_(1)
        values.view(torch.int8)[0, 0].fill_(2)
        scales[1, 0] = 0.25
        scales[0, 0] = 0.5

        q = torch.ones((1, 1, 2, head_dim), dtype=torch.int8)
        weight = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        actual = kernel_ops.dsv4_c4_paged_mqa_logits_torch(
            q_int8=q,
            kvcache_int8=cache.reshape(2, page_size, 1, head_dim + 4),
            weight=weight,
            seq_lens=torch.tensor([65], dtype=torch.int32),
            page_table=torch.tensor([[1, 0]], dtype=torch.int32),
            max_seq_len=66,
        )

        expected = torch.zeros((1, 66), dtype=torch.float32)
        expected[0, 0] = 48.0
        expected[0, 64] = 192.0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dsv4_compressed_attention_combines_lengths_sink_and_extra_keys(self):
        from sglang_kunlun.kernels import kernel_ops

        q = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]],
            dtype=torch.float16,
        )
        win_cache = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=torch.float16,
        )
        win_indices = torch.tensor([[0, 1, 3], [2, -1, 1]], dtype=torch.int32)
        win_lengths = torch.tensor([2, 1], dtype=torch.int32)
        extra_cache = torch.tensor([[1.0, 1.0], [3.0, 3.0]], dtype=torch.float16)
        extra_indices = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
        extra_lengths = torch.tensor([1, 0], dtype=torch.int32)
        sink = torch.tensor([0.0, -1.0], dtype=torch.float32)

        actual = kernel_ops.dsv4_compressed_attention_torch(
            q=q,
            win_cache=win_cache,
            win_indices=win_indices,
            win_lengths=win_lengths,
            softmax_scale=1.0,
            attn_sink=sink,
            extra_cache=extra_cache,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            query_block_size=1,
        )

        expected_rows = []
        for query, keys in (
            (q[0].float(), torch.cat((win_cache[:2], extra_cache[:1])).float()),
            (q[1].float(), win_cache[2:3].float()),
        ):
            scores = query @ keys.transpose(0, 1)
            probabilities = torch.softmax(
                torch.cat((scores, sink[:, None]), dim=1), dim=1
            )[:, :-1]
            expected_rows.append(probabilities @ keys)
        expected = torch.stack(expected_rows).to(torch.float16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dsv4_compress_store_updates_multi_page_int8_cache(self):
        from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
        from sglang_kunlun.kernels import kernel_ops

        page_size = 4
        page_bytes = page_size * (128 + 4)
        kv = torch.stack((torch.ones(128), torch.full((128,), -2.0)))
        plan_raw = torch.tensor([[4, 0, 0, 0], [8, 0, 0, 0]], dtype=torch.int32)
        plan = CompressorDecodePlan(4, plan_raw.view(torch.uint8))
        out_loc = torch.tensor([1, page_size + 2], dtype=torch.int64)
        kvcache = torch.zeros((2, page_bytes), dtype=torch.uint8)

        with mock.patch.object(
            kernel_ops, "_dsv4_norm_rope_torch", return_value=kv
        ), mock.patch.object(
            kernel_ops, "_dsv4_hadamard_torch", side_effect=lambda value: value
        ):
            kernel_ops.dsv4_compress_norm_rope_store_v2_torch(
                kv,
                plan,
                norm_weight=torch.ones(128),
                norm_eps=1e-6,
                freq_cis=torch.empty(0),
                out_loc=out_loc,
                kvcache=kvcache,
                page_size=page_size,
            )

        values = kvcache[:, : page_size * 128].reshape(2, page_size, 128)
        scales = kvcache[:, page_size * 128 :].view(torch.float32)
        torch.testing.assert_close(
            values.view(torch.int8)[0, 1], torch.full((128,), 127, dtype=torch.int8)
        )
        torch.testing.assert_close(
            values.view(torch.int8)[1, 2], torch.full((128,), -127, dtype=torch.int8)
        )
        torch.testing.assert_close(scales[0, 1], torch.tensor(1.0))
        torch.testing.assert_close(scales[1, 2], torch.tensor(2.0))

    def test_dsv4_c4_first_block_excludes_missing_overlap_history(self):
        from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
        from sglang_kunlun.kernels import kernel_ops

        head_dim = 2
        state = torch.zeros((2, 4, 4 * head_dim), dtype=torch.float32)
        normal_values = torch.arange(1, 5, dtype=torch.float32)[:, None].expand(-1, head_dim)
        state[0, :, head_dim : 2 * head_dim] = normal_values
        state[1, :, :head_dim] = 100.0
        kv_score_input = torch.tensor(
            [[100.0, 100.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        plan_raw = torch.tensor([[4, 3, 1, 0]], dtype=torch.int32)
        plan = CompressorDecodePlan(4, plan_raw.view(torch.uint8))

        actual = kernel_ops.dsv4_compress_forward_v2_torch(
            state,
            kv_score_input,
            torch.zeros((8, head_dim), dtype=torch.float32),
            plan,
            head_dim=head_dim,
            compress_ratio=4,
        )

        torch.testing.assert_close(
            actual,
            torch.full((1, head_dim), 2.5),
            rtol=0,
            atol=0,
        )

    def test_dsv4_c128_prefill_compresses_before_writing_current_state(self):
        from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan
        from sglang_kunlun.kernels import kernel_ops

        state = torch.zeros((1, 128, 2), dtype=torch.float32)
        state[..., 0] = 1.0
        plan_c = torch.tensor([[128, (127 << 16), 0, 0]], dtype=torch.int32)
        plan_w = torch.tensor([[0, 0]], dtype=torch.int32)
        plan = CompressorPrefillPlan(
            128,
            plan_c.view(torch.uint8),
            plan_w.view(torch.uint8),
            None,
        )
        kv_score_input = torch.tensor([[100.0, 0.0]], dtype=torch.float32)

        actual = kernel_ops.dsv4_compress_forward_v2_torch(
            state,
            kv_score_input,
            torch.zeros((128, 1), dtype=torch.float32),
            plan,
            head_dim=1,
            compress_ratio=128,
        )

        torch.testing.assert_close(
            actual,
            torch.tensor([[(127.0 + 100.0) / 128.0]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(state[0, 0], kv_score_input[0], rtol=0, atol=0)

    def test_dsv4_metadata_matches_v2_nine_value_contract_without_vendor_op(self):
        from sglang_kunlun.kernels import kernel_ops

        seq_lens = torch.tensor([0, 4, 128, 257], dtype=torch.int64)
        positions = seq_lens.to(torch.int32) - 1
        raw_out_loc = torch.tensor([0, 20, 256, 1028], dtype=torch.int64)
        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "init_compressed_attn_metadata",
            side_effect=AssertionError("vendor metadata op must not run"),
        ):
            values = kernel_ops.dsv4_init_compression_metadata_kunlun(
                seq_lens,
                positions,
                raw_out_loc,
                compute_page_indices=False,
            )

        self.assertEqual(len(values), 9)
        self.assertEqual(values[0].dtype, torch.int64)
        self.assertEqual(values[4].dtype, torch.int64)
        self.assertEqual(values[8], None)
        self.assertEqual(values[0].tolist(), [0, 5, 64, 0])
        self.assertEqual(values[4].tolist(), [0, 0, 2, 0])
        self.assertEqual(values[6].tolist(), [0, 0, 1, 2])

    def test_dsv4_metadata_supports_noncontiguous_inputs_and_page_mapping(self):
        from sglang_kunlun.kernels import kernel_ops

        seq_lens = torch.tensor([0, 9, 128, 1, 256, 1, 1, 1], dtype=torch.int64)[::2]
        positions = torch.tensor([-1, 0, 127, 0, 255, 0, 0, 0], dtype=torch.int32)[::2]
        raw_out_loc = torch.tensor([0, 0, 256, 0, 512, 0, 0, 0], dtype=torch.int64)[::2]
        page_table = torch.tensor(
            [[1, 2, 9, 9], [3, 4, 9, 9], [5, 6, 9, 9], [7, 8, 9, 9]],
            dtype=torch.int32,
        )[:, ::2]
        self.assertFalse(seq_lens.is_contiguous())
        self.assertFalse(positions.is_contiguous())
        self.assertFalse(raw_out_loc.is_contiguous())
        self.assertFalse(page_table.is_contiguous())

        with mock.patch.object(
            torch.ops.xspeedgate_ops,
            "init_compressed_attn_metadata",
            side_effect=AssertionError("vendor metadata op must not run"),
        ):
            values = kernel_ops.dsv4_init_compression_metadata_kunlun(
                seq_lens,
                positions,
                raw_out_loc,
                page_table,
                page_size=256,
            )

        self.assertEqual(values[8].tolist(), [
            [-1, -1, -1, -1],
            [6, -1, -1, -1],
            [10, 11, -1, -1],
            [-1, -1, -1, -1],
        ])

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
