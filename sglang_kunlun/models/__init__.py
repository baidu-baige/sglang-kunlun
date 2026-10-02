"""Kunlun model hooks and out-of-tree model registration.

hook 实现已全部迁移到 ``sglang_kunlun.hooks.models``；本包只保留 out-of-tree
model package 的设置（上游 ModelRegistry 据此解析外部模型模块）。为让 int4
name mapper 等注册在模型解析前就绪，这里从 hooks 包提前 import 一次。
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

# 提前 import：int4/量化 name mapper 的安装挂在 deepseek_v4 hook 模块的 import
# 副作用上，必须早于上游读取 ``hf_to_sglang_mapper``（模型加载）。
from sglang_kunlun.hooks.models import deepseek_v4 as _deepseek_v4  # noqa: F401

# Kimi-K3 W4A16 weight-loading hooks. SGLang imports the external model package
# from its model registry, which runs before any weight iterator, so
# registering them here keeps the packed-name protection and the shard trimming
# active independently of when the plugin's own hook bundle is installed.
from sglang_kunlun.hooks.models import (  # noqa: F401
    kimi_k3_w4a16_loader as _kimi_k3_w4a16_loader,
)

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
