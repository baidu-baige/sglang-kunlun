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

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt, EnvStr

# Temporarily enable _allow_set_name for class definition
original_allow_set_name = EnvField._allow_set_name
EnvField._allow_set_name = True


class Envs:
    """Kunlun-local environ."""

    # Clamp bound applied to the MoE combine output on fp16 models, which keeps
    # sink tokens from overflowing the fp16 range.
    SGLANG_FP16_LIMIT_IN_MOE = EnvInt(10)

    # ---------------------------------------------------------------------
    # AttentionStore (asradix) secondary KV cache.
    #
    # The upstream reference declares these as ``--enable-asradix-cache`` and
    # friends on the ``memory`` namespace, but a plugin cannot add fields to a
    # ``msgspec.Struct`` after the class is built, so the same knobs are env
    # vars here. Names and semantics track the CLI flags one-for-one; see
    # ``sglang_kunlun/mem_cache/as_config.py`` for the adapter that presents
    # them under the upstream field names.
    # ---------------------------------------------------------------------
    SGLANG_KUNLUN_ENABLE_ASRADIX_CACHE = EnvBool(False)
    # Number of XPUs per node, for AttentionStore Agent resource allocation.
    SGLANG_KUNLUN_PER_NODE_GPU_NUM = EnvInt(8)
    # Checkpoint spacing in tokens (>= page_size); 0 means "use page_size".
    SGLANG_KUNLUN_AS_SWA_STEP = EnvInt(0)
    # Use the tiered (position-dependent) spacing from the tiers spec below
    # instead of the single fixed step.
    SGLANG_KUNLUN_ENABLE_DYNAMIC_SWA_STEP = EnvBool(False)
    # Tiered spec 'bound:step,...' ending with a '0:<step>' (inf) tier.
    SGLANG_KUNLUN_SWA_STEP_TIERS = EnvStr("1024:256,4096:1024,0:4096")
    # Capacity multiplier for the SWA staging pool: how many concurrent
    # per-chunk checkpoints to reserve before the async write drains them.
    SGLANG_KUNLUN_AS_SWA_BUFFER_COEF = EnvFloat(3.0)


envs = Envs()

# Restore the original value
EnvField._allow_set_name = original_allow_set_name


# AttentionStore diagnostics that the upstream reference declares on sglang's own
# ``Envs``: content digest (D2H + hash per page) and the producer-event barrier
# check. The vendored asradix code reads them as ``envs.SGLANG_LOG_AS_*`` off
# sglang's environ, so inject them there rather than rewriting those reads --
# that also keeps them flippable at runtime the way upstream intends.
_SGLANG_ENV_INJECTIONS = {
    "SGLANG_LOG_AS_CACHE_DIGEST": False,
    "SGLANG_LOG_AS_BARRIER_DEBUG": False,
}


def patch_sglang_envs() -> None:
    """Declare the AttentionStore diagnostic envs on sglang's ``Envs``.

    ``EnvField`` is a plain class attribute (no ``__get__``), so ``envs.X``
    hands back the field object itself and injecting one is just a ``setattr``
    on the class plus the ``name`` that ``__set_name__`` would have assigned.
    """
    from sglang.srt.environ import envs as sglang_envs

    owner = type(sglang_envs)
    for name, default in _SGLANG_ENV_INJECTIONS.items():
        if getattr(owner, name, None) is not None:
            continue
        field = EnvBool(default)
        field.name = name
        setattr(owner, name, field)
