import inspect
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_kunlun.hooks import ragged_draft_extend as ragged


class RaggedDraftMetadataContractTest(unittest.TestCase):
    def test_hook_registration_and_signature(self):
        target = (
            "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend."
            "KunlunDeepseekV4AttnBackend._build_forward_metadata"
        )
        registrations = HookRegistry._hooks[target]
        self.assertTrue(
            any(
                hook_type == HookType.AROUND
                and hook is ragged.build_ragged_draft_extend_metadata_kunlun
                for hook_type, hook, _source in registrations
            )
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    ragged.build_ragged_draft_extend_metadata_kunlun
                ).parameters
            ),
            (
                "original_fn",
                "self",
                "forward_batch",
                "max_seq_len_override",
                "use_prefill_cuda_graph",
            ),
        )

    def test_ragged_draft_uses_compact_prefill_metadata(self):
        req_to_token = torch.empty((8, 64), dtype=torch.int64)
        req_pool_indices = torch.tensor([3, 5], dtype=torch.int64)
        seq_lens = torch.tensor([11, 23], dtype=torch.int64)
        seq_lens_cpu = seq_lens.to(torch.int32)
        extend_seq_lens = torch.tensor([1, 3], dtype=torch.int32)
        out_cache_loc = torch.tensor([100, 104, 105, 106], dtype=torch.int64)
        marker = object()
        original = mock.Mock(return_value=object())
        prepare_forward = mock.Mock(return_value=0)
        init_prefill = mock.Mock(return_value=marker)
        mode = types.SimpleNamespace(is_draft_extend_v2=lambda: True)
        upstream = types.ModuleType(
            "sglang.srt.layers.attention.deepseek_v4_backend"
        )
        upstream.SWA_WINDOW = 4096
        upstream._get_logical_forward_mode = lambda _forward_batch: mode
        upstream._get_target_verify_bs = lambda batch: batch.batch_size
        owner = types.SimpleNamespace(
            req_to_token=req_to_token,
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
            swa_page_size=upstream.SWA_WINDOW,
            page_size=256,
            online_c128_mtp=types.SimpleNamespace(prepare_forward=prepare_forward),
            init_forward_metadata_prefill=init_prefill,
        )
        forward_batch = types.SimpleNamespace(
            _kunlun_ragged_draft_extend=True,
            forward_mode=mode,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=[1, 3],
            extend_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            out_cache_loc=out_cache_loc,
            batch_size=2,
            spec_info=None,
        )

        with mock.patch.dict(sys.modules, {upstream.__name__: upstream}):
            actual = ragged.build_ragged_draft_extend_metadata_kunlun(
                original,
                owner,
                forward_batch,
            )

        self.assertIs(actual, marker)
        original.assert_not_called()
        prepare_forward.assert_called_once()
        init_prefill.assert_called_once()
        call = init_prefill.call_args.kwargs
        self.assertEqual(call["max_seq_len"], 23)
        self.assertTrue(torch.equal(call["req_pool_indices"], req_pool_indices))
        self.assertTrue(torch.equal(call["seq_lens"], seq_lens.to(torch.int32)))
        self.assertEqual(call["seq_lens_cpu"], [11, 23])
        self.assertIs(call["out_cache_loc"], out_cache_loc)
        self.assertEqual(call["num_tokens"], 4)
        self.assertIs(call["extend_seq_lens"], extend_seq_lens)
        self.assertEqual(call["extend_seq_lens_cpu"], [1, 3])
        self.assertIs(call["extend_start_loc"], forward_batch.extend_start_loc)
        self.assertFalse(call["need_compress"])
        self.assertFalse(call["use_prefill_cuda_graph"])

    def test_non_ragged_draft_uses_upstream_metadata(self):
        marker = object()
        original = mock.Mock(return_value=marker)
        owner = object()
        forward_batch = types.SimpleNamespace(
            _kunlun_ragged_draft_extend=False,
        )

        actual = ragged.build_ragged_draft_extend_metadata_kunlun(
            original,
            owner,
            forward_batch,
        )

        self.assertIs(actual, marker)
        original.assert_called_once_with(
            owner,
            forward_batch,
            max_seq_len_override=None,
            use_prefill_cuda_graph=False,
        )


if __name__ == "__main__":
    unittest.main()
