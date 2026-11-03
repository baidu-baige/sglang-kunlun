"""V4.1 output contracts against target upstream bodies; CPU tensors only."""

import ast
from contextvars import ContextVar
import os
from pathlib import Path
import types
import unittest
from unittest import mock

import msgspec
import torch

from test_dsv41_migration_cpu import BACKEND, ROOT, functions


UPSTREAM = Path(os.environ.get(
    "SGLANG_TEST_SOURCE",
    "/home/ljj/dsv4_1_v0522/code_sglang_0522/sglang/python/sglang",
))
ATTN = UPSTREAM / "srt/layers/attention"
INDEXER = ATTN / "dsv4/v41_indexer"
HOOK = ROOT / "sglang_kunlun/hooks/layers/attention/dsv41_torch_indexer.py"


def classes(path, *names, **extra):
    nodes = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    ns = {"torch": torch, "msgspec": msgspec, **extra}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return types.SimpleNamespace(**ns)


def metadata(ratios=(1, 2), rows=4, topk=512, prefill=True, buffers=None):
    core = types.SimpleNamespace(
        index_topk=topk, present_ratios=(0, *ratios),
        low_ratios=tuple(r for r in ratios if r in (1, 2)),
    )
    for ratio in ratios:
        setattr(core, f"c{ratio}_topk_lengths_clamp1", torch.arange(rows).int() + 1)
    functions(
        BACKEND, "init_flashmla_related",
        upstream=types.SimpleNamespace(_pad_last_dim=lambda x: x),
    ).init_flashmla_related(core, is_prefill=prefill, low_ratio_buffers=buffers)
    core.sparse_raw_indices = lambda r: getattr(core, f"c{r}_sparse_raw_indices")
    core.sparse_page_indices = lambda r: getattr(core, f"c{r}_sparse_page_indices")
    return core


class IndexerOutputsCPU(unittest.TestCase):
    def test_prefill_allocations_and_decode_identity(self):
        for ratios in ((1,), (2,), (1, 2)):
            for rows in (0, 4):
                for topk in (512, 1024):
                    core = metadata(ratios, rows, topk)
                    for ratio in ratios:
                        raw, pages = core.sparse_raw_indices(ratio), core.sparse_page_indices(ratio)
                        self.assertEqual(raw.shape, (rows, topk))
                        self.assertEqual(raw.dtype, torch.int32)
                        self.assertEqual(raw.device.type, "cpu")
                        self.assertIsNot(raw, pages)
                        self.assertTrue(torch.all(raw == -1))
                        if rows:
                            raw.fill_(9)
                            self.assertTrue(torch.all(pages == -1))
                    decode = metadata(ratios, rows, topk, prefill=False)
                    for ratio in ratios:
                        self.assertIsNone(decode.sparse_raw_indices(ratio))
        buffers = tuple(torch.tensor([i], dtype=torch.int32) for i in range(8))
        core = metadata(prefill=False, buffers=buffers)
        for ratio, offset in ((1, 2), (2, 6)):
            self.assertIs(getattr(core, f"c{ratio}_sparse_topk_lengths"), buffers[offset])
            self.assertIs(core.sparse_page_indices(ratio), buffers[offset + 1])
            self.assertIsNone(core.sparse_raw_indices(ratio))
        self.assertIsNone(metadata((4,)).c4_sparse_raw_indices)

    def test_full_topk_and_candidate_publish_with_both_hook_routes(self):
        structs = classes(INDEXER / "types.py", "CandidateMetadata", "PrefillInputs")
        scores = classes(INDEXER / "scoring.py", "RequestScores", "ChunkScores")
        scoring = functions(
            INDEXER / "scoring.py", "prefill_requests", "_score_chunks", "write_prefill",
            RequestScores=scores.RequestScores, ChunkScores=scores.ChunkScores,
            _TORCH_SCORE_BUDGET_BYTES=16,
        )
        blocks = classes(
            INDEXER / "dense_blocks.py", "BlockIds",
            CandidateMetadata=structs.CandidateMetadata,
        )
        for ratio in (1, 2):
            for cached in ("0", "1"):
                for publish in (False, True):
                    with self.subTest(ratio=ratio, cached=cached, publish=publish):
                        core = metadata(rows=4)
                        backend = types.SimpleNamespace(
                            forward_metadata=types.SimpleNamespace(core_metadata=core),
                            tail_forward_metadata=None,
                        )
                        context = ContextVar("test_indexer", default=None)
                        hooks = functions(
                            HOOK, "index_topk_context", "prefill_requests_kunlun", "_prep",
                            _ACTIVE_BACKEND=context, _CACHE_ATTR="_kunlun_lri_prep",
                        )
                        indexer = types.SimpleNamespace(
                            index_topk=512,
                            queries=lambda q, f: torch.ones(4, 1, 1),
                            head_weights=lambda x: torch.ones(4, 1),
                            scores=lambda q, k, w: k[:, 0].expand(len(q), -1),
                        )
                        raw, pages = core.sparse_raw_indices(ratio), core.sparse_page_indices(ratio)
                        inputs = structs.PrefillInputs(
                            indexer=indexer, layer_id=0, compress_ratio=ratio,
                            freqs_cis=torch.zeros(8), x=torch.zeros(4, 1), q_lora=None,
                            positions=torch.tensor([0, 1, 2, 3]), req_rows=torch.zeros(4).long(),
                            req_pool_indices=torch.tensor([0]), kv_page_table=None,
                            seq_lens_cpu=[4], rows_per_request=[4], rows_per_request_device=None,
                            out_raw_indices=raw, out_page_indices=pages,
                        )
                        table = (torch.arange(8) + 10 * ratio)[None]
                        worker = types.SimpleNamespace(
                            token_to_kv_pool=types.SimpleNamespace(
                                get_low_ratio_index_k_dequant=lambda layer, slots: slots.float()[:, None]
                            ),
                            req_to_token=table, topk_blocks=2, block_size=2,
                        )
                        def requests(**kwargs):
                            return hooks.prefill_requests_kunlun(scoring.prefill_requests, **kwargs)
                        # Stub only the block selection kernel; exercise real publication/writes.
                        select = lambda logits, lens, **kw: torch.zeros(
                            (len(logits), 1), dtype=torch.int32)
                        path = INDEXER / ("dense_blocks.py" if publish else "full_topk.py")
                        name = "_torch_publish_prefill" if publish else "_torch_prefill"
                        run = getattr(functions(
                            path, name, prefill_requests=requests, write_prefill=scoring.write_prefill,
                            BlockIds=blocks.BlockIds, select_candidate_block_ids=select,
                        ), name)
                        module = types.ModuleType("sglang.srt.layers.attention.dsv4.v41_indexer.scoring")
                        module.RequestScores = scores.RequestScores
                        module._score_chunks = scoring._score_chunks
                        with mock.patch.dict(os.environ, {"DSV4_LRI_TORCH_PREP_CACHE": cached}), mock.patch.dict(
                            "sys.modules", {module.__name__: module}
                        ):
                            for _ in range(2):
                                raw.fill_(99)
                                pages.fill_(99)
                                result = hooks.index_topk_context(lambda b: run(worker, inputs), backend)
                                self.assertIsNone(context.get())
                                for row in range(4):
                                    visible = (row + 1) // ratio
                                    self.assertEqual(raw[row, :visible].tolist(), list(range(visible)))
                                    self.assertEqual(pages[row, :visible].tolist(), list(range(10, 10 + visible)))
                                    self.assertTrue(torch.all(raw[row, visible:] == -1))
                                    self.assertTrue(torch.all(pages[row, visible:] == -1))
                                if publish:
                                    self.assertIsInstance(result, blocks.BlockIds)
                                    self.assertEqual(result.rows_per_request, [4])
                                    functions(
                                        ATTN / "deepseek_v4_backend.py", "_publish_candidate_metadata",
                                    )._publish_candidate_metadata(backend, result)
                                    self.assertIs(backend.forward_metadata.candidate_metadata, result)

    def test_context_restored_on_failure(self):
        context = ContextVar("test_failure", default=None)
        fn = functions(HOOK, "index_topk_context", _ACTIVE_BACKEND=context).index_topk_context
        context.set("outer")
        def fail(backend):
            self.assertEqual(context.get(), "inner")
            raise ValueError("expected")
        with self.assertRaisesRegex(ValueError, "expected"):
            fn(fail, "inner")
        self.assertEqual(context.get(), "outer")

    def test_cp_interleave_request_and_chunk_offsets(self):
        structs = classes(INDEXER / "types.py", "PrefillInputs")
        score_types = classes(INDEXER / "scoring.py", "RequestScores", "ChunkScores")
        scoring = functions(
            INDEXER / "scoring.py", "prefill_requests", "_score_chunks", "write_prefill",
            RequestScores=score_types.RequestScores, ChunkScores=score_types.ChunkScores,
            _TORCH_SCORE_BUDGET_BYTES=16,
        )
        hook = functions(
            HOOK, "prefill_requests_kunlun",
            _ACTIVE_BACKEND=ContextVar("cp_test", default=None),
        ).prefill_requests_kunlun
        interleave = functions(
            UPSTREAM / "srt/layers/cp/interleave.py", "interleave_rows_per_request",
        ).interleave_rows_per_request
        # Non-consecutive pool rows and reversed physical pages expose accidental
        # use of batch ordinals, local offsets, or one fabricated request.
        table = torch.arange(5 * 32).reshape(5, 32).flip(1) * 2
        for lengths, prefixes, pool_rows in (
            ([8], [0], [3]),
            ([3, 1, 9], [7, 0, 5], [3, 1, 4]),
            ([0, 1, 0], [4, 0, 8], [3, 1, 4]),
        ):
            global_req = torch.tensor([
                pool for pool, n in zip(pool_rows, lengths) for _ in range(n)
            ], dtype=torch.int64)
            global_pos = torch.tensor([
                prefix + j for prefix, n in zip(prefixes, lengths) for j in range(n)
            ])
            for size in (2, 4, 8):
                for rank in range(size):
                    local_pos = global_pos[rank::size]
                    local_req = global_req[rank::size]
                    n = len(local_pos)
                    for ratio in (1, 2):
                        for cached in ("0", "1"):
                            with self.subTest(lengths=lengths, size=size, rank=rank,
                                              ratio=ratio, cached=cached), torch.inference_mode():
                                raw = torch.full((n + 2, 3), 99, dtype=torch.int32)
                                pages = torch.full_like(raw, 99)
                                indexer = types.SimpleNamespace(
                                    index_topk=3,
                                    queries=lambda q, f: q[:, None, :],
                                    head_weights=lambda x: torch.ones(len(x), 1),
                                    scores=lambda q, k, w: k[:, 0].expand(len(q), -1),
                                )
                                seen = []
                                def topk(layer, x, q, req, pos, batch, *, rows_per_request):
                                    self.assertIsNone(req)
                                    inputs = structs.PrefillInputs(
                                        indexer=indexer, layer_id=0, compress_ratio=ratio,
                                        freqs_cis=torch.zeros(32), x=x, q_lora=q,
                                        positions=pos, req_rows=req,
                                        req_pool_indices=torch.tensor(pool_rows),
                                        kv_page_table=None, seq_lens_cpu=None,
                                        rows_per_request=rows_per_request,
                                        rows_per_request_device=None,
                                        out_raw_indices=raw, out_page_indices=pages,
                                    )
                                    def original(**kwargs):
                                        restored = kwargs["inputs"]
                                        torch.testing.assert_close(restored.req_rows, local_req)
                                        self.assertIs(restored.positions, pos)
                                        self.assertIs(restored.out_raw_indices, raw)
                                        yield from scoring.prefill_requests(**kwargs)
                                    pool = types.SimpleNamespace(
                                        get_low_ratio_index_k_dequant=lambda layer, slots: slots.float()[:, None]
                                    )
                                    with mock.patch.dict(os.environ, {"DSV4_LRI_TORCH_PREP_CACHE": cached}):
                                        for request, chunks in hook(
                                            original, inputs=inputs, token_to_kv_pool=pool, req_to_token=table,
                                        ):
                                            for chunk in chunks:
                                                torch.testing.assert_close(
                                                    chunk.tok, request.tok[chunk.rows])
                                                seen.extend(chunk.tok.tolist())
                                                scoring.write_prefill(
                                                    inputs, request, chunk,
                                                    chunk.scores.topk(request.k, dim=-1).indices,
                                                )
                                    self.assertIsNone(inputs.req_rows)
                                backend = types.SimpleNamespace(
                                    forward_metadata=types.SimpleNamespace(late_layer_tail=None),
                                    _low_ratio_index_topk=topk,
                                )
                                batch = types.SimpleNamespace(
                                    attn_cp_metadata=types.SimpleNamespace(total_seq_lens=sum(lengths)),
                                    extend_seq_lens_cpu=lengths, positions=global_pos,
                                )
                                cp = functions(
                                    ATTN / "deepseek_v4_backend.py", "_forward_low_ratio_sources_cp",
                                    interleave_rows_per_request=interleave, _as_int_list=list,
                                    get_parallel=lambda: types.SimpleNamespace(attn_cp_rank=rank, attn_cp_size=size),
                                    token_req_indices=lambda batch, num_tokens: global_req,
                                )._forward_low_ratio_sources_cp
                                cp(
                                    backend, layer=types.SimpleNamespace(indexer=indexer),
                                    x=torch.ones(n + 1, 1), q_lora=torch.ones(n + 1, 1),
                                    positions=torch.cat((local_pos, torch.tensor([999]))),
                                    forward_batch=batch, run_compressor=False, run_indexer=True,
                                )
                                self.assertEqual(len(seen), len(set(seen)))
                                for row, (pool, pos) in enumerate(zip(local_req.tolist(), local_pos.tolist())):
                                    visible = (pos + 1) // ratio
                                    slots = table[pool, torch.arange(visible) * ratio] // ratio
                                    k = min(3, visible)
                                    expected = slots.topk(k).indices.sort().values
                                    self.assertEqual(raw[row, :k].tolist(), expected.tolist())
                                    self.assertEqual(pages[row, :k].tolist(), slots[expected].tolist())
                                    self.assertTrue(torch.all(raw[row, k:] == -1))
                                    self.assertTrue(torch.all(pages[row, k:] == -1))
                                self.assertTrue(torch.all(raw[n:] == -1))
                                self.assertTrue(torch.all(pages[n:] == -1))

    def test_cp_rejects_inconsistent_request_metadata(self):
        hook = functions(HOOK, "prefill_requests_kunlun").prefill_requests_kunlun
        for counts in (None, [2], [1, -1], [1, 0], [2, 1]):
            inputs = types.SimpleNamespace(
                req_rows=None, rows_per_request=counts,
                req_pool_indices=torch.tensor([3, 1]), positions=torch.tensor([4, 5]),
            )
            original = mock.Mock()
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "per-request counts"):
                list(hook(original, inputs=inputs, token_to_kv_pool=None, req_to_token=None))
            original.assert_not_called()

    def test_input_builder_and_tail_publication(self):
        structs = classes(INDEXER / "types.py", "CandidateMetadata", "PrefillInputs")
        core = metadata()
        core.page_table = torch.zeros(4, 1, dtype=torch.int32)
        backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(core_metadata=core, late_layer_tail=None),
            _low_ratio_prefill_reads_page_indices=lambda batch: True,
        )
        batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[2, 2], extend_seq_lens=torch.tensor([2, 2]),
            req_pool_indices=torch.tensor([0, 1]), seq_lens_cpu=[2, 2],
        )
        upstream = functions(
            ATTN / "deepseek_v4_backend.py",
            "_make_low_ratio_prefill_indexer_inputs", "_publish_candidate_metadata",
            PrefillInputs=structs.PrefillInputs, _as_int_list=list,
            async_h2d=lambda values, **kw: torch.tensor(values, **kw),
        )
        for ratio in (1, 2):
            layer = types.SimpleNamespace(
                indexer=object(), layer_id=0, compress_ratio=ratio, freqs_cis=torch.zeros(4),
            )
            for cp_rows in (None, [2, 2]):
                inputs = upstream._make_low_ratio_prefill_indexer_inputs(
                    backend, layer, torch.zeros(4, 1), None,
                    None if cp_rows else torch.tensor([0, 0, 1, 1]),
                    torch.tensor([0, 1, 0, 1]), batch, cp_rows,
                )
                self.assertIs(inputs.out_raw_indices, core.sparse_raw_indices(ratio))
                self.assertIs(inputs.out_page_indices, core.sparse_page_indices(ratio))
                inputs.reset_outputs()
        tail_rows = functions(
            INDEXER / "types.py", "get_tail_row_indices",
            async_h2d=lambda values, **kw: torch.tensor(values, **kw),
        ).get_tail_row_indices
        block_type = classes(
            INDEXER / "dense_blocks.py", "BlockIds",
            CandidateMetadata=structs.CandidateMetadata, get_tail_row_indices=tail_rows,
        ).BlockIds
        published = block_type(
            blocks=torch.arange(8, dtype=torch.int32).reshape(4, 2),
            rows_per_request=[2, 2],
        )
        for cp in (None, object()):
            backend.tail_forward_metadata = types.SimpleNamespace(
                late_layer_tail=types.SimpleNamespace(
                    cp_metadata=cp, local_lens_cpu=[1, 1], extend_seq_lens_cpu=[1, 1],
                )
            )
            upstream._publish_candidate_metadata(backend, published)
            self.assertIs(backend.forward_metadata.candidate_metadata, published)
            tail = backend.tail_forward_metadata.candidate_metadata
            torch.testing.assert_close(tail.blocks, published.blocks[[1, 3]])
            self.assertEqual(tail.rows_per_request, [1, 1])


if __name__ == "__main__":
    unittest.main()
