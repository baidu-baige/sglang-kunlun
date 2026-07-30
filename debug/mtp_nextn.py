"""Single-token MTP boundary probes for the DeepSeek-V4 NextN modules.

These wrappers were previously installed unconditionally at the bottom of
``kunlun_deepseek_v4_backend.py``.  They only record tensors when
``DSV4_MTP_TENSOR_DUMP_DIR`` is set, so the wrappers stay pass-through in
production, but they are monkey patches on upstream classes and therefore
belong to the debug package rather than to the production hooks.

``install()`` is idempotent and must be called after the DSV4 model modules are
importable (that is, from the attention backend hook, as before).
"""

from __future__ import annotations

import os

import torch

_installed = False


def _record(name: str, tensor) -> None:
    if (
        os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR") is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.shape[0] != 1
    ):
        return
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    step = getattr(backend, "speculative_step_id", 0)
    prefix = f"step{step}_layer0.nextn"
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe[f"{prefix}.{name}"] = tensor.clone()
    backend._mtp_tensor_probe = probe


def install() -> None:
    """Wrap the decoder/NextN forward boundaries with the MTP probes."""
    global _installed
    if _installed:
        return
    _installed = True

    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.models.deepseek_v4_nextn import (
        DeepseekV4ForCausalLMNextN,
        DeepseekV4ModelNextN,
    )

    original_decoder_forward = DeepseekV4DecoderLayer.forward
    original_nextn_forward = DeepseekV4ModelNextN.forward
    original_nextn_causal_forward = DeepseekV4ForCausalLMNextN.forward
    original_nextn_hc_head = DeepseekV4ModelNextN.hc_head

    def decoder_forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 1:
            hidden_states = args[1]
        _record("decoder_input", hidden_states)
        output = original_decoder_forward(self, *args, **kwargs)
        output_hidden = output[0] if isinstance(output, tuple) else output
        _record("decoder_output", output_hidden)
        return output

    def nextn_forward(self, input_ids, positions, forward_batch, input_embeds=None):
        _record("spec_hidden", forward_batch.spec_info.hidden_states)
        output = original_nextn_forward(
            self, input_ids, positions, forward_batch, input_embeds
        )
        final_hidden = output[0] if isinstance(output, tuple) else output
        _record("final_hidden", final_hidden)
        return output

    def nextn_hc_head(self, x, hc_fn, hc_scale, hc_base):
        _record("hc_head_input", x)
        output = original_nextn_hc_head(self, x, hc_fn, hc_scale, hc_base)
        _record("hc_head_output", output)
        return output

    def nextn_causal_forward(self, input_ids, positions, forward_batch):
        output = original_nextn_causal_forward(self, input_ids, positions, forward_batch)
        logits = getattr(output, "next_token_logits", None)
        if logits is not None:
            _record("next_token_logits", logits)
        return output

    DeepseekV4DecoderLayer.forward = decoder_forward
    DeepseekV4ModelNextN.forward = nextn_forward
    DeepseekV4ModelNextN.hc_head = nextn_hc_head
    DeepseekV4ForCausalLMNextN.forward = nextn_causal_forward
