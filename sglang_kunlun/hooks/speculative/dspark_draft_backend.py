"""Route the DSpark DSV4 draft worker to the Kunlun compressed backend.

Upstream hardcodes ``DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"`` in
``dspark_config`` and passes it to ``build_draft_tp_worker`` as an
``attention_backend_override``. That override bypasses every Kunlun name
resolution path (the ``kunlun_compressed`` preservation wrapper in
``platform/srt.py`` and the ``DraftBackendFactory`` hook in ``draft_utils``),
so the draft ``ModelRunner`` builds the stock ``DeepseekV4AttnBackend`` while
the target runs ``KunlunDeepseekV4AttnBackend``. The stock backend's
triton/tilelang/JIT kernels are not available on P800, so the draft forward
returns garbage (constant argmax, NaN hidden states) and every draft token is
rejected.

Rewriting the constant is enough: the draft copy of ``ServerArgs`` sets both
``attention_backend`` and ``speculative_draft_attention_backend`` from it, and
``kunlun_compressed`` is a registered backend name.

Set ``SGLANG_KUNLUN_DSPARK_DRAFT_ATTENTION_BACKEND=dsv4`` to restore the
upstream value.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

_TARGET_BACKEND = os.environ.get(
    "SGLANG_KUNLUN_DSPARK_DRAFT_ATTENTION_BACKEND", "kunlun_compressed"
)


def _patch_dspark_draft_attention_backend() -> None:
    try:
        config_mod = importlib.import_module(
            "sglang.srt.speculative.dspark_components.dspark_config"
        )
    except Exception as e:  # pragma: no cover - upstream layout drift
        logger.warning("Could not import dspark_config to patch draft backend: %s", e)
        return

    previous = getattr(config_mod, "DSV4_DRAFT_ATTENTION_BACKEND", None)
    if previous == _TARGET_BACKEND:
        return

    config_mod.DSV4_DRAFT_ATTENTION_BACKEND = _TARGET_BACKEND

    # If the worker module already bound the constant into its own globals,
    # rebind there too (``from ... import DSV4_DRAFT_ATTENTION_BACKEND``).
    import sys

    worker_mod = sys.modules.get(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2"
    )
    if worker_mod is not None and hasattr(worker_mod, "DSV4_DRAFT_ATTENTION_BACKEND"):
        worker_mod.DSV4_DRAFT_ATTENTION_BACKEND = _TARGET_BACKEND

    logger.warning(
        "DSpark DSV4 draft attention backend: %r -> %r", previous, _TARGET_BACKEND
    )


_patch_dspark_draft_attention_backend()
