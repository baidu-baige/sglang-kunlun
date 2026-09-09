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
def _create_backend_kunlun(
    original_fn,
    self,
    backend_name,
    backend_map,
    error_template,
    stamps_children: bool = False,
):
    """Swap the Kunlun factories into ``backend_map`` and let upstream drive.

    ``backend_name`` is the slot consulted for the backend type:
    ``"decode_attention_backend"`` for the multi-step decode backend and
    ``"prefill_attention_backend"`` (or decode, per speculative_attention_mode)
    for the draft-extend backend.

    0.5.19 moved the stamping (``prefill/decode_attention_backend_str``) and the
    draft-decode sidecar wrapper *into* ``_create_backend``, and its map entries
    now return ``(stamp, backend)``. Rather than reimplementing that, we only
    replace the map entry for the resolved backend type and delegate, so every
    upstream post-step still runs.
    """
    backend_type = _resolve_backend_type(self, backend_name)
    is_decode_multistep = backend_name == "decode_attention_backend"

    def _make_compressed_backend():
        """Build the Kunlun DeepSeek-V4 (compressed) draft backend."""
        from sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend import (
            KunlunDeepseekV4AttnBackend,
            KunlunDeepseekV4MultiStepBackend,
        )

        if is_decode_multistep:
            return (
                backend_type,
                KunlunDeepseekV4MultiStepBackend(
                    self.draft_model_runner, self.topk, self.speculative_num_steps
                ),
            )
        return (
            backend_type,
            KunlunDeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False),
        )

    def _make_dsa_backend():
        """Build the Kunlun DeepSeek sparse attention (DSA/NSA) draft backend."""
        from sglang_kunlun.hooks.layers.attention.kunlun_nsa_backend import (
            KunlunDSAAttnBackend,
            KunlunDSAMultiStepBackend,
        )

        seed = getattr(self, "seed_dsa_topk_from_draft_extend", False)
        if is_decode_multistep:
            return (
                "dsa",
                KunlunDSAMultiStepBackend(
                    self.draft_model_runner,
                    self.topk,
                    self.speculative_num_steps,
                    seed_dsa_topk_from_draft_extend=seed,
                ),
            )
        return (
            "dsa",
            KunlunDSAAttnBackend(
                self.draft_model_runner,
                skip_prefill=False,
                seed_dsa_topk_from_draft_extend=seed,
            ),
        )

    def _make_platform_backend():
        """Build the generic Kunlun draft backend through the platform."""
        from sglang.srt.platforms import current_platform

        if is_decode_multistep and (
            self.server_args.speculative_attention_mode != "decode"
        ):
            cls = current_platform.get_draft_decode_attention_backend_cls()
            return (
                backend_type,
                cls(self.draft_model_runner, self.topk, self.speculative_num_steps),
            )
        cls = current_platform.get_draft_prefill_attention_backend_cls()
        return (
            backend_type,
            cls(self.draft_model_runner, skip_prefill=False),
        )

    factory = {
        "kunlun_compressed": _make_compressed_backend,
        "dsa": _make_dsa_backend,
        "nsa": _make_dsa_backend,
        "kunlun": _make_platform_backend,
    }.get(backend_type)
    if factory is not None:
        backend_map = dict(backend_map)
        backend_map[backend_type] = factory

    return original_fn(
        self,
        backend_name,
        backend_map,
        error_template,
        stamps_children=stamps_children,
    )


def _resolve_backend_type(self, backend_name):
    """Mirror upstream ``_create_backend`` backend-type resolution."""
    try:
        from sglang.srt.runtime_context import attention_backends

        prefill_backend, decode_backend = attention_backends()
        configured = (
            decode_backend
            if backend_name == "decode_attention_backend"
            else prefill_backend
        )
    except Exception:  # pragma: no cover - pre-0.5.19 layout
        configured = getattr(self.server_args, backend_name, None) or getattr(
            self.server_args, "attention_backend", None
        )
    return self.draft_attn_backend or configured

