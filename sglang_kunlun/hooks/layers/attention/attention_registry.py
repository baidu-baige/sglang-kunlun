"""Register the ``kunlun`` attention backend factory.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/attention_registry.py
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


try:
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
except Exception as e:  # pragma: no cover - upstream layout drift
    register_attention_backend = None
    logger.warning("Could not import register_attention_backend: %s", e)


if register_attention_backend is not None:
    @register_attention_backend("kunlun")
    def create_kunlun_attention_backend(runner):
        """Lazy-construct the generic Kunlun attention backend."""
        from .kunlun_backend import KunlunAttentionBackend

        return KunlunAttentionBackend(runner)

    @register_attention_backend("kunlun_compressed")
    def create_kunlun_compressed_attention_backend(runner):
        """Lazy-construct the Kunlun DeepSeek-V4 attention backend."""
        from .kunlun_deepseek_v4_backend import KunlunDeepseekV4AttnBackend

        return KunlunDeepseekV4AttnBackend(runner)

    @register_attention_backend("kunlun_nsa")
    def create_kunlun_nsa_attention_backend(runner):
        """Lazy-construct the Kunlun DeepSeek sparse attention (DSA/NSA) backend."""
        from .kunlun_nsa_backend import KunlunDSAAttnBackend

        return KunlunDSAAttnBackend(runner)

    # Upstream keys a lot of DSA-specific plumbing on the *name* "dsa"/"nsa"
    register_attention_backend("dsa")(create_kunlun_nsa_attention_backend)
    register_attention_backend("nsa")(create_kunlun_nsa_attention_backend)

    logger.info(
        "Registered 'kunlun', 'kunlun_compressed' and 'kunlun_nsa' attention "
        "backends; 'dsa'/'nsa' now resolve to KunlunDSAAttnBackend"
    )
