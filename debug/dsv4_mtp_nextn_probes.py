"""MTP / NextN boundary probes for the DSV4 Kunlun backend.

These probes were previously installed by raw attribute assignment at the end of
``sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py``:

    DeepseekV4DecoderLayer.forward = _decoder_forward_with_mtp_probe
    DeepseekV4ModelNextN.forward = _nextn_forward_with_mtp_probe
    DeepseekV4ModelNextN.hc_head = _nextn_hc_head_with_mtp_probe
    DeepseekV4ForCausalLMNextN.forward = _nextn_causal_forward_with_mtp_probe

They now go through ``HookRegistry``, which also repairs stale
``from X import Y`` bindings that a bare ``setattr`` leaves pointing at the
unwrapped function.

Registration is **conditional**. Without ``DSV4_MTP_TENSOR_DUMP_DIR`` nothing is
registered at all, so ``DeepseekV4DecoderLayer.forward`` keeps its original
identity and no wrapper frame is added to the per-layer, per-token call path.
A decorator-based registration would have wrapped every forward unconditionally
and only skipped the payload, which is the wrong trade for a hot path.

Payload keys are unchanged: ``step{N}_layer0.nextn.<name>`` on the backend's
``_mtp_tensor_probe`` dict.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType

logger = logging.getLogger(__name__)

ENABLE_ENV_VAR = "DSV4_MTP_TENSOR_DUMP_DIR"


def probes_enabled() -> bool:
    return bool(os.environ.get(ENABLE_ENV_VAR))


def record_mtp_nextn_probe(name, tensor):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 1:
        return
    backend = get_attn_backend()
    step = getattr(backend, "speculative_step_id", 0)
    prefix = f"step{step}_layer0.nextn"
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe[f"{prefix}.{name}"] = tensor.clone()
    backend._mtp_tensor_probe = probe


def decoder_forward_with_mtp_probe(original_fn, self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and len(args) > 1:
        hidden_states = args[1]
    record_mtp_nextn_probe("decoder_input", hidden_states)
    output = original_fn(self, *args, **kwargs)
    output_hidden = output[0] if isinstance(output, tuple) else output
    record_mtp_nextn_probe("decoder_output", output_hidden)
    return output


def nextn_forward_with_mtp_probe(
    original_fn, self, input_ids, positions, forward_batch, input_embeds=None
):
    record_mtp_nextn_probe("spec_hidden", forward_batch.spec_info.hidden_states)
    output = original_fn(self, input_ids, positions, forward_batch, input_embeds)
    final_hidden = output[0] if isinstance(output, tuple) else output
    record_mtp_nextn_probe("final_hidden", final_hidden)
    return output


def nextn_hc_head_with_mtp_probe(original_fn, self, x, hc_fn, hc_scale, hc_base):
    record_mtp_nextn_probe("hc_head_input", x)
    output = original_fn(self, x, hc_fn, hc_scale, hc_base)
    record_mtp_nextn_probe("hc_head_output", output)
    return output


def nextn_causal_forward_with_mtp_probe(
    original_fn, self, input_ids, positions, forward_batch
):
    output = original_fn(self, input_ids, positions, forward_batch)
    logits = getattr(output, "next_token_logits", None)
    if logits is not None:
        record_mtp_nextn_probe("next_token_logits", logits)
    return output


_PROBE_HOOKS = (
    (
        "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.forward",
        decoder_forward_with_mtp_probe,
    ),
    (
        "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ModelNextN.forward",
        nextn_forward_with_mtp_probe,
    ),
    (
        "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ModelNextN.hc_head",
        nextn_hc_head_with_mtp_probe,
    ),
    (
        "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ForCausalLMNextN.forward",
        nextn_causal_forward_with_mtp_probe,
    ),
)


def install() -> int:
    """Register the probes only when the dump directory is configured."""
    if not probes_enabled():
        return 0
    # Imported lazily: the unit tests stub sglang.srt.plugins.hook_registry with
    # a module that only provides HookType and plugin_hook, so importing
    # HookRegistry at module scope would break collection.
    from sglang.srt.plugins.hook_registry import HookRegistry

    for target, hook in _PROBE_HOOKS:
        HookRegistry.register(target, hook, HookType.AROUND)
    logger.warning(
        "DSV4 MTP/NextN probes installed on %d targets because %s is set; "
        "this adds a wrapper frame to every decoder forward",
        len(_PROBE_HOOKS),
        ENABLE_ENV_VAR,
    )
    return len(_PROBE_HOOKS)


install()
