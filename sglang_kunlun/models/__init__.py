"""Kunlun model hooks and out-of-tree model registration."""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def configure_external_model_package() -> None:
    """Expose Kunlun model modules through SGLang's external model package env."""

    if envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
        logger.info(
            "sglang-kunlun: keep existing SGLANG_EXTERNAL_MODEL_PACKAGE=%s",
            envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get(),
        )
        return

    envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set(__name__)
    logger.info("sglang-kunlun: set SGLANG_EXTERNAL_MODEL_PACKAGE=%s", __name__)


configure_external_model_package()
