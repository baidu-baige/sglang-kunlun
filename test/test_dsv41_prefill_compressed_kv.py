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
"""Exercise attention wiring on CPU without loading the Kunlun runtime."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sglang_kunlun/hooks/layers/attention/deepseek_v4_backend.py"


class Mode:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return False

    def is_decode(self):
        return self.name == "decode"

    def is_target_verify(self):
        return self.name == "verify"

    def is_extend_without_speculative(self):
        return self.name == "extend"


class Metadata:
    def __init__(self):
        self.swa_page_indices = torch.zeros((3, 128), dtype=torch.int32)
        self.swa_topk_lengths = torch.tensor([1, 2, 3], dtype=torch.int32)
        self.extra_indices = torch.arange(64, dtype=torch.int32).repeat(3, 1)
        self.extra_lengths = torch.ones(3, dtype=torch.int32)
        self.c4_sparse_page_indices = self.extra_indices
        self.c128_page_indices = self.extra_indices
        self.c4_sparse_topk_lengths = self.extra_lengths
        self.c128_topk_lengths_clamp1 = self.extra_lengths

    def sparse_page_indices(self, ratio):
        return self.extra_indices

    def sparse_topk_lengths(self, ratio):
        return self.extra_lengths


class Pool:
    swa_page_size = 128
    page_size = 256

    def __init__(self, ratio):
        self.swa_kv_pool = SimpleNamespace(kv_cache_total_dim=2)
        self.raw = torch.ones((2, 256))
        self.extra_page_size = 256 // ratio if ratio else 1
        self.extra = torch.full((2, self.extra_page_size * 2), 3.0)

    def get_swa_key_buffer_radix(self, layer_id):
        return self.raw

    def get_extra_key_buffer(self, layer_id):
        return self.extra

    def get_extra_key_page_size(self, layer_id):
        return self.extra_page_size

    def get_extra_key_bytes_per_token(self, layer_id):
        return 2


def load_forward():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "KunlunDeepseekV4AttnBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "forward")
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {
        "torch": torch, "os": os, "Optional": object, "Literal": object,
        "DeepSeekV4TokenToKVPool": Pool, "DSV4AttnMetadata": Metadata,
        "_LARGE_INDEXER_QUERY_THRESHOLD": 4096,
        "envs": SimpleNamespace(SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=
                                SimpleNamespace(get=lambda: False)),
    }
    # Postponed annotations avoid importing SGLang/vendor types for this wiring test.
    import __future__
    exec(compile(module, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["forward"]


class PrefillCompressedKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forward = staticmethod(load_forward())

    def run_forward(self, ratio, mode="extend", override=None):
        metadata = Metadata()
        backend = SimpleNamespace(
            mtp_enabled=False,
            forward_metadata=SimpleNamespace(core_attn_metadata=metadata),
            token_to_kv_pool=Pool(ratio),
            _kunlun_compressed_attention=lambda **kwargs: kwargs,
        )
        batch = SimpleNamespace(forward_mode=Mode(mode))
        q = torch.zeros((3, 8, 2))
        kv = torch.zeros((3, 2))
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_KUNLUN_DSA_RATIO_REPAIR", None)
            if override is not None:
                os.environ["SGLANG_KUNLUN_DSA_RATIO_REPAIR"] = override
            result = self.forward(
                backend, q, kv, kv, SimpleNamespace(layer_id=2), batch,
                compress_ratio=ratio, save_kv_cache=False, attn_sink=torch.zeros(8),
            )
        return result, metadata

    def test_default_preserves_compressed_kv_for_short_prefill(self):
        # These three queries all fit inside SWA128. Extra KV still belongs in
        # joint attention, rather than being omitted until the window overflows.
        for ratio in (1, 2, 4, 128):
            with self.subTest(ratio=ratio):
                result, metadata = self.run_forward(ratio)
                self.assertIsNotNone(result["extra_k_cache"])
                self.assertEqual(result["extra_k_cache"].shape, (2, 256 // ratio, 1, 2))
                self.assertEqual(result["extra_indices"].squeeze(1).tolist(),
                                 metadata.extra_indices.tolist())

    def test_uncompressed_layer_has_no_extra_branch(self):
        result, _ = self.run_forward(0)
        self.assertIsNone(result["extra_k_cache"])
        self.assertIsNone(result["extra_indices"])

    def test_decode_and_verify_still_consume_compressed_kv(self):
        for mode in ("decode", "verify"):
            result, _ = self.run_forward(2, mode=mode)
            self.assertIsNotNone(result["extra_k_cache"])

    def test_explicit_diagnostic_overrides_remain_available(self):
        for override in ("auto", "off"):
            result, _ = self.run_forward(2, override=override)
            self.assertIsNone(result["extra_k_cache"])
        result, _ = self.run_forward(2, mode="verify", override="auto")
        self.assertIsNotNone(result["extra_k_cache"])
        result, _ = self.run_forward(2, override="all")
        self.assertIsNotNone(result["extra_k_cache"])


if __name__ == "__main__":
    unittest.main()
