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
"""Kunlun-local environment variables.

Kept out of the sglang main repo: these knobs only exist for the kunlun
plugin, so they are declared here and read through ``patch_envs``.
"""

from sglang.srt.environ import EnvBool, EnvField, EnvInt

# Temporarily enable _allow_set_name for class definition
original_allow_set_name = EnvField._allow_set_name
EnvField._allow_set_name = True


class Envs:
    """Kunlun-local environ."""

    # Clamp bound applied to the MoE combine output on fp16 models, which keeps
    # sink tokens from overflowing the fp16 range.
    SGLANG_FP16_LIMIT_IN_MOE = EnvInt(10)


envs = Envs()

# Restore the original value
EnvField._allow_set_name = original_allow_set_name
