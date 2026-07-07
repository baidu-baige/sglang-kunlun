"""Register the ``kunlun`` attention backend factory.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/attention_registry.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
except Exception as e:  # pragma: no cover - upstream layout drift
    register_attention_backend = None
    logger.warning("Could not import register_attention_backend: %s", e)


if register_attention_backend is not None:
    @register_attention_backend("kunlun")
    def create_kunlun_attention_backend(runner):
        """Lazy-construct the Kunlun attention backend."""
        from .kunlun_backend import KunlunAttentionBackend

        return KunlunAttentionBackend(runner)

    logger.info("Registered 'kunlun' attention backend factory")