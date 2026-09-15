# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
