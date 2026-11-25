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
"""Map bundled DSpark checkpoint quantization targets to runtime stage names."""
import os
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from . import deepseek_v4

_base = DeepseekV4ForCausalLM.hf_to_sglang_mapper


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.SoftmaxTemp.execute",
    type=HookType.REPLACE,
)
def softmax_temperature_kunlun(cls, *, logits, temperatures, rows_per_request):
    """Use device PyTorch operations instead of a CUDA-only FlashInfer kernel."""
    import torch

    rows, vocab = logits.shape
    if rows_per_request <= 0 or rows % rows_per_request:
        raise ValueError("DSpark softmax rows must divide evenly across requests")
    batch = rows // rows_per_request
    temperature = temperatures.reshape(batch, 1, 1).to(dtype=torch.float32)
    # Greedy requests carry temperature 0; one such row turns the whole scaled
    # tensor into inf/NaN. softmax(logits / 1) keeps the argmax, and the greedy
    # branch is selected downstream by the per-row mask, so substitute 1.
    temperature = torch.where(
        temperature > 0, temperature, torch.ones_like(temperature)
    )
    scaled = logits.float().reshape(batch, rows_per_request, vocab) / temperature
    return torch.softmax(scaled, dim=-1).reshape(rows, vocab)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.commit_swa.committed_swa_locations",
    type=HookType.REPLACE,
)
def committed_swa_locations_kunlun(cache_loc, full_to_swa_mapping, commit_lens, width):
    """Translate committed slots; capture masks invalid outputs with sentinel -1.

    Eager execution raises on invalid lengths or committed indices. During
    capture, invalid lengths mask the whole row and invalid indices mask the
    corresponding outputs, without reading a device predicate on the host.
    This prevents an unsafe gather but cannot repair missing committed KV.
    Like uncommitted slots, -1 must be skipped or routed to the reserved padding
    sink by the downstream writer. The Kunlun writer routes it to slot 0;
    the upstream paged and SWA allocators reserve that slot for dummy outputs.
    """
    import torch
    from sglang_kunlun.hooks.graph_state import is_graph_capturing

    integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    if any(value.dtype not in integer_types for value in (cache_loc, full_to_swa_mapping, commit_lens)):
        raise ValueError("DSpark SWA inputs must have integer dtype")
    if not (cache_loc.device == full_to_swa_mapping.device == commit_lens.device):
        raise ValueError("DSpark SWA inputs must share one device")
    if width <= 0 or cache_loc.numel() != commit_lens.numel() * width:
        raise ValueError("DSpark SWA commit shape does not match lengths and width")
    if cache_loc.numel() == 0:
        return torch.empty_like(cache_loc, dtype=torch.int32)
    lengths = commit_lens.detach().reshape(-1)
    capturing = is_graph_capturing()
    valid_lengths = (lengths >= 0) & (lengths <= width)
    if not capturing and not bool(valid_lengths.all()):
        raise ValueError(f"DSpark SWA commit lengths outside [0, {width}]: {lengths.tolist()}")
    locations = cache_loc.detach().reshape(-1, width).long()
    mapping = full_to_swa_mapping.detach().reshape(-1)
    committed = (
        torch.arange(width, device=cache_loc.device)[None, :] < lengths[:, None]
    ) & valid_lengths[:, None]
    in_bounds = (locations >= 0) & (locations < mapping.numel())
    invalid = committed & ~in_bounds
    if not capturing and bool(invalid.any()):
        raise ValueError("DSpark committed cache locations exceed SWA mapping bounds")
    if mapping.numel() == 0:
        return torch.full_like(cache_loc, -1, dtype=torch.int32)
    # Keep the gather shape fixed. Negative cache locations are sentinels, never
    # Python-style indices into the tail of the mapping.
    writable = committed & in_bounds
    indices = torch.where(writable, locations, torch.zeros_like(locations))
    mapped = mapping.gather(0, indices.reshape(-1)).reshape_as(locations)
    result = torch.where(writable, mapped.to(torch.int32), -1)
    return result.reshape(cache_loc.shape)


class DSparkWeightsMapper(WeightsMapper):
    """Map DSpark checkpoint names, adding the mtp->stages stage rename pass."""

    def _map_name(self, key):
        """Return the runtime name for ``key`` with the stage rename applied."""
        mapped = super()._map_name(key)
        if mapped is None:
            return None
        # WeightsMapper only applies ONE substring mapping. Stage and attention
        # renaming must both happen, so stage rename is a separate pass.
        return mapped.replace(r"mtp\.", r"stages\.", 1).replace("mtp.", "stages.", 1)


DeepseekV4ForCausalLMDSpark.hf_to_sglang_mapper = DSparkWeightsMapper(
    orig_to_new_substr=_base.orig_to_new_substr,
    orig_to_new_prefix=_base.orig_to_new_prefix,
    orig_to_new_suffix=_base.orig_to_new_suffix,
)
DeepseekV4ForCausalLMDSpark.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.load_weights",
    type=HookType.AROUND,
)
def load_dspark_weights(original_fn, self, weights):
    """Fold the draft ``wo_a`` INT8 scales before the stock loader runs."""
    # The draft wo_a also reads its unquantized weight directly. Fold checkpoint
    # INT8 scales just as the already-working target loader does.
    return original_fn(self, deepseek_v4._dequant_int8_wo_a(weights))


@plugin_hook(
    "sglang.srt.speculative.draft_worker_common.build_draft_tp_worker",
    type=HookType.AROUND,
)
def build_dspark_draft_worker(original_fn, **kwargs):
    """Route the DSpark draft to the Kunlun FP16-KV attention backend."""
    # The generic dsv4 backend assumes FP8-packed KV. Kunlun stores FP16 KV
    # and must use the same platform backend for the draft as for the target.
    if kwargs.get("algo_label") == "DSPARK" and kwargs.get("attention_backend_override") == "dsv4":
        kwargs["attention_backend_override"] = "kunlun_dsv4"
    return original_fn(**kwargs)


@plugin_hook(
    "sglang.srt.speculative.dflash_utils.compute_dflash_correct_drafts_and_bonus",
    type=HookType.REPLACE,
)
def correct_drafts_and_bonus_kunlun(*, candidates, target_predict):
    """Compute DSpark correct-draft counts and bonus tokens on their device."""
    import torch

    if candidates.ndim != 2 or candidates.shape != target_predict.shape:
        raise ValueError("DSpark candidates and target predictions must have the same 2D shape")
    if candidates.device != target_predict.device:
        raise ValueError("DSpark candidates and target predictions must share one device")
    width = candidates.shape[1]
    if width < 1:
        raise ValueError("DSpark acceptance requires at least one token per row")
    if width == 1:
        correct = torch.zeros(candidates.shape[0], device=candidates.device, dtype=torch.int64)
    else:
        matches = candidates[:, 1:] == target_predict[:, :-1]
        positions = torch.arange(width - 1, device=candidates.device, dtype=torch.int64)
        correct = torch.where(matches, width - 1, positions[None, :]).amin(dim=1)
    bonus_device = target_predict.gather(1, correct[:, None]).squeeze(1).to(torch.int64)
    correct_device = correct.to(torch.int32)
    return correct_device, bonus_device


@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_verify.TargetVerifyExecutor.accept_and_finalize",
    type=HookType.AROUND,
)
def trace_accept_finalize(original_fn, self, *args, **kwargs):
    """Keep finalized bonus and output tokens within vocabulary bounds."""
    from sglang_kunlun.hooks.graph_state import is_graph_capturing

    capturing = is_graph_capturing()
    result = original_fn(self, *args, **kwargs)
    if os.environ.get("KUNLUN_DSPARK_ACCEPT_GUARD", "1") == "1" and not capturing:
        vocab = int(
            getattr(
                self.target_worker.model_runner.model_config.hf_text_config,
                "vocab_size",
                0,
            )
        )
        if vocab > 0:
            bad_bonus = (result.bonus < 0) | (result.bonus >= vocab)
            bad_out = (result.out_tokens < 0) | (result.out_tokens >= vocab)
            if bool(bad_bonus.any()) or bool(bad_out.any()):
                result.bonus[bad_bonus] = 0
                result.out_tokens[bad_out] = 0
    return result


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept._row_argmax",
    type=HookType.REPLACE,
)
def reference_verify_argmax(logits, fused=False):
    """Return device-side argmax of verify logits, rejecting invalid rows."""
    import torch

    values = logits.detach()
    invalid = torch.isnan(values).any(dim=-1) | torch.isneginf(values).all(dim=-1)
    # Preserve the reference error contract. Only this scalar predicate is
    # synchronized; the full vocabulary logits remain on the device.
    if invalid.any():
        raise RuntimeError("Kunlun DSpark verify logits contain NaN or fully masked rows")
    result = values.argmax(dim=-1)
    return result


@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_verify."
    "TargetVerifyExecutor.run_non_compact",
    type=HookType.AROUND,
)
def guard_verify_ids(
    original_fn, self, *, batch, draft_input, verify_ids_2d, verify_window, sampling_info
):
    """Guard the upstream keyword-only run_non_compact input before forwarding.

    This Python entry runs before model graph replay, so replay inputs receive
    the eager error check too. Capture is a separate case: record a fixed-shape
    device mask, replacing invalid IDs with token 0 in place. Preserving storage
    also keeps proposal/acceptance aliases consistent. This is memory safety,
    not recovery of the intended token; corrupt capture inputs change semantics.
    """
    import torch
    from sglang_kunlun.hooks.graph_state import is_graph_capturing

    ids = verify_ids_2d
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("DSpark verify IDs must be a 2D int32/int64 tensor")
    vocab = int(getattr(
        self.target_worker.model_runner.model_config.hf_text_config, "vocab_size", 0
    ))
    if vocab <= 0:
        raise ValueError("DSpark verify ID guard requires a positive vocab_size")
    bad = (ids < 0) | (ids >= vocab)
    if is_graph_capturing():
        ids.masked_fill_(bad, 0)
    elif bool(bad.any()):
        raise ValueError("DSpark verify IDs outside vocabulary bounds")
    return original_fn(
        self, batch=batch, draft_input=draft_input, verify_ids_2d=ids,
        verify_window=verify_window, sampling_info=sampling_info,
    )
