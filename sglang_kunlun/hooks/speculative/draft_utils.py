"""Hook for ``sglang.srt.speculative.draft_utils.DraftBackendFactory``.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.speculative.draft_utils.DraftBackendFactory._create_backend",
    type=HookType.AROUND,
)
def _create_backend_kunlun(original_fn, self, backend_name, backend_map, error_template):
    """Route the ``kunlun`` draft attention backend through the platform.

    ``backend_name`` is the ServerArgs attribute consulted for the backend
    type: ``"decode_attention_backend"`` for the multi-step decode backend and
    ``"prefill_attention_backend"`` (or decode, per speculative_attention_mode)
    for the draft-extend backend. We replicate the upstream resolution order to
    decide whether the effective backend is ``"kunlun"``.
    """
    backend_type = (
        self.draft_attn_backend
        if self.draft_attn_backend
        else getattr(self.server_args, backend_name)
    )
    if backend_type is None:
        backend_type = self.server_args.attention_backend

    if backend_type == "kunlun_compressed":
        from sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend import (
            KunlunDeepseekV4AttnBackend,
            KunlunDeepseekV4MultiStepBackend,
        )

        if backend_name == "decode_attention_backend":
            return KunlunDeepseekV4MultiStepBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        return KunlunDeepseekV4AttnBackend(
            self.draft_model_runner, skip_prefill=False
        )

    if backend_type in ("dsa", "nsa"):
        from sglang_kunlun.hooks.layers.attention.kunlun_nsa_backend import (
            KunlunDSAAttnBackend,
            KunlunDSAMultiStepBackend,
        )

        if backend_name == "decode_attention_backend":
            return KunlunDSAMultiStepBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        return KunlunDSAAttnBackend(self.draft_model_runner, skip_prefill=False)

    if backend_type != "kunlun":
        return original_fn(self, backend_name, backend_map, error_template)

    from sglang.srt.platforms import current_platform

    # decode multi-step backend uses the "decode_attention_backend" slot;
    # draft-extend (prefill) uses the prefill/decode slot per attention mode.
    is_decode_multistep = backend_name == "decode_attention_backend" and (
        self.server_args.speculative_attention_mode != "decode"
    )

    if is_decode_multistep:
        cls = current_platform.get_draft_decode_attention_backend_cls()
        return cls(self.draft_model_runner, self.topk, self.speculative_num_steps)

    # draft-extend / prefill path: a single Kunlun attention backend that
    # also handles prefill (skip_prefill=False).
    cls = current_platform.get_draft_prefill_attention_backend_cls()
    return cls(self.draft_model_runner, skip_prefill=False)
