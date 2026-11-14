"""Offline regression for NextN CP eligibility and the model-body recycle hook.

Execute source functions by AST to avoid importing the serving/plugin stack.
Only device collectives and model compute are mocked; CP row mapping, runner
dispatch, embedding preparation, correction and spec restoration are real code.
Run with python3 and optionally pass the sibling sglang checkout as argv[1].
"""

import ast
import contextlib
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch


def load_functions(path, names, namespace, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    selected = []
    for name in names:
        node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
        node.decorator_list = []
        selected.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def run(core):
    plugin = Path(__file__).resolve().parents[1]
    srt = core / "python/sglang/srt"
    rotation_path = plugin / "sglang_kunlun/hooks/models/glm53_mtp_rotation.py"
    cp_path = plugin / "sglang_kunlun/hooks/layers/cp/bcg.py"
    registrations = ast.parse(rotation_path.read_text())
    recycle = next(n for n in registrations.body if getattr(n, "name", None) == "nextn_forward_recycle")
    assert recycle.decorator_list[0].args[0].value == (
        "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.forward"
    ), "CP calls the body directly; a CausalLM hook is bypassed"

    class EagleDraftExtendInput(SimpleNamespace):
        pass

    class Mode:
        def __init__(self, name):
            self.name = name

        def is_context_parallel_extend(self):
            return self.name in ("extend", "mixed")

        def is_target_verify(self):
            return self.name == "verify"

        def is_draft_extend_v2(self):
            return self.name == "draft_v2"

        def is_extend(self):
            return self.name in ("extend", "mixed")

    eligible = load_functions(cp_path, ["is_cp_active_kunlun"], {})["is_cp_active_kunlun"]
    eagle_module = SimpleNamespace(EagleDraftExtendInput=EagleDraftExtendInput)
    with patch.dict(sys.modules, {"sglang.srt.speculative.eagle_info": eagle_module}):
        for mode in ("extend", "mixed", "verify", "draft_v2", "decode"):
            for spec in (None, EagleDraftExtendInput(), SimpleNamespace()):
                for ragged in (False, True):
                    batch = SimpleNamespace(
                        forward_mode=Mode(mode), spec_info=spec,
                        _kunlun_ragged_draft_extend=ragged,
                    )
                    expected = (
                        mode in ("extend", "mixed")
                        and not ragged
                        and (spec is None or isinstance(spec, EagleDraftExtendInput))
                    )
                    assert eligible(lambda fb: fb.forward_mode.is_context_parallel_extend(), batch) == expected
        batch = SimpleNamespace(forward_mode=Mode("extend"), spec_info=EagleDraftExtendInput())
        assert not eligible(lambda fb: False, batch), "upstream eligibility still decides"
    print("PASS: eligibility matrix (30 cases) and upstream veto")

    nextn = srt / "models/deepseek_nextn.py"
    embed_fn = load_functions(nextn, ["embed_input_ids"], {}, "DeepseekModelNextN")["embed_input_ids"]
    seam = load_functions(nextn, ["prepare_cp_inputs", "forward"], {}, "DeepseekV3ForCausalLMNextN")
    correction = load_functions(
        rotation_path, ["_apply_blocks", "nextn_forward_recycle"],
        {"torch": torch, "_RECYCLE": "recycle_correction"},
    )
    wrapped = correction["nextn_forward_recycle"]
    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda: None),
        is_tensor=torch.is_tensor, cat=torch.cat, arange=torch.arange,
    )
    parallel = SimpleNamespace(attn_cp_size=8, attn_cp_rank=0, attn_cp_group=None)
    padding = load_functions(
        srt / "layers/cp/padding.py",
        ["pad_local_rows", "pad_logical_token_to_physical"],
        {"torch": torch, "get_cp_padding_align_size": lambda: 8},
    )
    strategy_ns = load_functions(
        srt / "layers/cp/interleave.py",
        ["build_metadata", "_interleave_shard", "shard_hidden_states",
         "shard_position_ids", "gather_hidden_states", "_gather_interleaved_tensor",
         "materialize_full_mla_kv"],
        {"torch": torch, "pad_local_rows": padding["pad_local_rows"],
         "InterleaveContextParallelMetadata": SimpleNamespace,
         "get_parallel": lambda: parallel,
         "use_symmetric_memory": lambda *a, **k: contextlib.nullcontext(),
         "is_allocation_symmetric": lambda: False},
        "InterleaveCPStrategy",
    )
    strategy = SimpleNamespace()
    for name in ("build_metadata", "_interleave_shard", "shard_hidden_states",
                 "shard_position_ids", "gather_hidden_states",
                 "_gather_interleaved_tensor", "materialize_full_mla_kv"):
        setattr(strategy, name, MethodType(strategy_ns[name], strategy))
    strategy.cp_size = 8
    cp_ns = load_functions(
        srt / "layers/cp/utils.py",
        ["cp_shard_hidden_states", "cp_shard_position_ids", "cp_interleave_input_ids",
         "cp_shard_model_inputs", "cp_gather_after_forward"],
        {"contextmanager": contextlib.contextmanager,
         "is_cp_active": lambda fb: fb.attn_cp_metadata is not None,
         "get_cp_strategy": lambda: strategy,
         "get_moe_a2a_backend": lambda: SimpleNamespace(is_none=lambda: False)},
    )
    cp_ns["cp_shard_model_inputs"] = contextlib.contextmanager(cp_ns["cp_shard_model_inputs"])
    runner_ns = load_functions(
        srt / "model_executor/runner/eager_runner.py", ["_execute_extend_cp"],
        {"torch": cpu_torch, "cp_shard_model_inputs": cp_ns["cp_shard_model_inputs"],
         "cp_gather_after_forward": cp_ns["cp_gather_after_forward"]}, "EagerRunner",
    )
    target = torch.arange(38, dtype=torch.float32).reshape(19, 2)
    ids = torch.arange(19)
    ids[2] = 1000000  # MM sentinel must never reach the vocabulary embedding.
    blocks = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    expected_hidden = target @ blocks[0]
    mm_embeds = target + 100

    def embedding(token_ids):
        assert bool((token_ids < 100).all()), "MM sentinel reached embedding"
        return token_ids.float()[:, None].expand(-1, 2).clone()

    expected_embeds = mm_embeds.clone()
    expected_embeds[[6, 18]] = embedding(ids[[6, 18]])
    reference = expected_embeds + expected_hidden
    observations = []

    class Body:
        shared_head = SimpleNamespace(recycle_correction=blocks)
        embed_tokens = staticmethod(embedding)
        embed_input_ids = embed_fn

        def __call__(self, input_ids, positions, fb, input_embeds=None):
            return wrapped(self.compute, self, input_ids, positions, fb, input_embeds=input_embeds)

        @staticmethod
        def compute(self, input_ids, positions, fb, input_embeds=None):
            value = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids, fb)
            hidden = fb.spec_info.hidden_states
            observations.append((input_ids.clone(), positions.clone(), hidden.clone()))
            assert input_ids.shape[0] == positions.shape[0] == hidden.shape[0] == value.shape[0]
            return value + hidden

    body = Body()
    model = SimpleNamespace(
        model=body, pp_group=SimpleNamespace(is_last_rank=True),
        logits_processor=lambda ids, hidden, head, fb, *a, **k: hidden, lm_head=None,
    )
    model.prepare_cp_inputs = MethodType(seam["prepare_cp_inputs"], model)
    runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    fb = SimpleNamespace(
        input_ids=ids, positions=torch.arange(19),
        forward_mode=Mode("extend"), spec_info=EagleDraftExtendInput(hidden_states=target),
        mm_input_embeds=mm_embeds.clone(), contains_mm_inputs=lambda: True,
        extend_start_loc=torch.tensor([0, 7]), extend_seq_lens=torch.tensor([7, 12]),
        attn_cp_metadata=None,
    )
    noncp = seam["forward"](model, ids, fb.positions, fb)
    assert torch.equal(noncp, reference)
    assert fb.spec_info.hidden_states is target
    assert torch.equal(target, torch.arange(38).reshape(19, 2))
    assert torch.equal(observations[-1][2], expected_hidden)
    metadata = strategy.build_metadata(19, None, [7, 12])
    # Defaults normally supplied by the metadata dataclass.
    metadata.gather_index = None
    metadata.per_rank_logical_token = None
    padding["pad_logical_token_to_physical"](metadata)
    fb.attn_cp_metadata = metadata
    fb.input_ids_global = "original-global-ids"

    def collective(output, local):
        shards = []
        for rank in range(8):
            shard = reference[rank::8]
            padded = torch.zeros(8, 2)
            padded[:len(shard)] = shard
            shards.append(padded)
        assert torch.equal(local, shards[parallel.attn_cp_rank])
        output.copy_(torch.cat(shards))

    strategy_ns["attn_cp_all_gather_into_tensor"] = collective
    for rank in range(8):
        parallel.attn_cp_rank = strategy.cp_rank = rank
        cp = runner_ns["_execute_extend_cp"](runner, fb, {})
        assert torch.equal(cp, noncp), (rank, cp, noncp)
        local_ids, local_positions, corrected = observations[-1]
        valid = len(target[rank::8])
        assert torch.equal(corrected[:valid], expected_hidden[rank::8])
        assert torch.equal(local_ids[:valid], ids[rank::8])
        assert torch.equal(local_positions[:valid], fb.positions[rank::8])
        assert fb.spec_info.hidden_states is target
        assert fb.input_ids_global == "original-global-ids"
    print("PASS: 8 padded CP ranks match non-CP; MM sentinel and correction exactly once")

    # The native MLA handoff must replicate K while leaving Q on local rows.
    strategy_ns["torch"] = cpu_torch
    strategy.gather_kv_cache = strategy.gather_hidden_states
    prepare_kv = load_functions(
        srt / "layers/attention/dsa_backend.py", ["prepare_kv_for_attention"],
        {"dsa_use_prefill_cp": lambda fb: True, "is_cp_active": lambda fb: True,
         "get_cp_strategy": lambda: strategy},
    )["prepare_kv_for_attention"]
    local_kv = strategy.shard_hidden_states(reference, fb)
    k, rope = prepare_kv(
        SimpleNamespace(attn_mqa=None), fb,
        local_kv[:, :1].unsqueeze(1), local_kv[:, 1:].unsqueeze(1),
        defer_materialization=False,
    )
    cache_loc = torch.arange(19)
    assert k.shape[0] == rope.shape[0] == cache_loc.shape[0]
    assert torch.equal(torch.cat((k, rope), -1).squeeze(1), reference)
    print("PASS: native MLA materialization removes padding for full-cache writes")

    def fail(self, *args, **kwargs):
        raise RuntimeError("compute failure")

    fb.attn_cp_metadata = metadata
    del fb.input_ids_global
    try:
        with cp_ns["cp_shard_model_inputs"](expected_embeds, fb.positions, fb, ids):
            wrapped(fail, body, ids, fb.positions, fb)
    except RuntimeError as exc:
        assert str(exc) == "compute failure"
    else:
        raise AssertionError("missing compute failure")
    assert fb.spec_info.hidden_states is target
    assert not hasattr(fb, "input_ids_global")
    print("PASS: exception restores local correction, full spec state and absent global IDs")

    hash_fn = load_functions(
        plugin / "sglang_kunlun/hooks/layers/cp/input_ids.py",
        ["hash_topk_cp_shard_input_ids_kunlun"],
        {"os": os, "_shard_input_ids": lambda *a: (_ for _ in ()).throw(
            AssertionError("already aligned IDs must not be sharded twice"))},
    )["hash_topk_cp_shard_input_ids_kunlun"]
    aligned = ids[rank::8]
    result = hash_fn(lambda self, hidden, logits, input_ids: input_ids, None,
                     torch.zeros(len(aligned), 2), None, aligned)
    assert result is aligned
    print("PASS: direct input-ID hook preserves aligned CP rows")

    # Ordinary text inputs retain the vocabulary path, with no MM injection.
    fb.contains_mm_inputs = lambda: False
    fb.mm_input_embeds = None
    fb.input_ids = torch.arange(19)
    embeds, positions, kwargs = model.prepare_cp_inputs(fb)
    assert torch.equal(embeds, embedding(fb.input_ids))
    assert positions is fb.positions and kwargs == {}
    print("PASS: text-only CP preparation preserves the ordinary embedding path")


if __name__ == "__main__":
    checkout = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "sglang"
    run(checkout)
