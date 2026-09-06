"""Declare Kunlun-only env fields that upstream SGLang does not define.

Some Kunlun hooks are gated by env flags that existed in the vendored 0.5.17
tree but are absent from upstream ``sglang.srt.environ.Envs`` (for example
``SGLANG_OPT_USE_FUSED_STORE_CACHE``). Reading them through ``envs.<NAME>``
would raise ``AttributeError`` at runtime, so the plugin registers the missing
fields itself instead of patching the main repo.

Fields are only added when absent, so a future upstream release that declares
one of them keeps its own default.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Attach missing Kunlun env fields to ``sglang.srt.environ.Envs``."""
    global _installed
    if _installed:
        return

    from sglang.srt.environ import EnvBool, Envs

    # name -> field factory, mirroring the 0.5.17 defaults.
    missing = {
        "SGLANG_OPT_USE_FUSED_STORE_CACHE": lambda: EnvBool(True),
        "SGLANG_TOPK_TRANSFORM_512_TORCH": lambda: EnvBool(False),
    }

    added = []
    for name, factory in missing.items():
        if hasattr(Envs, name):
            continue
        field = factory()
        # ``__set_name__`` only fires for class-body assignment, so set the
        # env var name that ``EnvField.get()`` reads explicitly.
        field.name = name
        setattr(Envs, name, field)
        added.append(name)

    if added:
        logger.info(
            "sglang-kunlun: registered %d compat env fields: %s",
            len(added),
            ", ".join(added),
        )

    _installed = True
