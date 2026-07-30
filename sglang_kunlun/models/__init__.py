"""Kunlun model hooks and out-of-tree model registration."""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

from . import deepseek_v4  # noqa: F401
from . import deepseek_v4_nextn  # noqa: F401
from . import deepseek_v4_precision  # noqa: F401

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
