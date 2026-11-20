# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Kunlun layer hooks.

Importing this package registers all layer-level ``@plugin_hook`` targets.
"""

import importlib


def _target_exists(path: str) -> bool:
    """Whether the dotted hook target exists in the installed sglang.

    A hook whose target was renamed upstream must not be imported: the registry
    applies it at startup and a missing attribute aborts the launch with a
    ``Failed to apply hooks`` traceback (release rules treat that as fatal).
    0.5.21 renamed the dense-indexer tail to ``_low_ratio_index_topk`` and
    dropped the fp4/candidate helpers, so the two modules that replace them are
    native on that build and are skipped instead of registered.
    """
    module_path, _, attribute = path.rpartition(".")
    module_name, _, class_name = module_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    obj = getattr(module, class_name, None)
    return obj is not None and hasattr(obj, attribute)


_HAS_DENSE_INDEXER_TAIL = _target_exists(
    "sglang.srt.layers.attention.deepseek_v4_backend."
    "DeepseekV4AttnBackend._low_ratio_index_topk_dense"
)

from . import linear  # noqa: F401
from . import quantization  # noqa: F401
from . import moe  # noqa: F401
from . import rotary_embedding  # noqa: F401,E402
from . import vocab_parallel_embedding  # noqa: F401,E402
from . import mhc  # noqa: F401,E402
from . import mhc_head  # noqa: F401,E402
from . import engram  # noqa: F401,E402
from . import dsa_cp  # noqa: F401,E402
if _HAS_DENSE_INDEXER_TAIL:
    from . import dsv41_cp_indexer  # noqa: F401,E402
from . import dsv41_wo_a  # noqa: F401,E402
if _HAS_DENSE_INDEXER_TAIL:
    from . import dsv41_topk_sort  # noqa: F401,E402
from . import dsv41_pool_pairs  # noqa: F401,E402
from . import attn_tp_gather  # noqa: F401,E402

# Must be imported before upstream resolves attention backend modules.
from . import attention  # noqa: F401,E402

from . import zcp_gather_chunked  # noqa: F401,E402 round-10 chunked CP gather

# NOTE: ``dsv41_idx_mask`` (round-12/18 int32 + fp16 dense-indexer epilogue) is
# intentionally NOT imported in this build.  It monkey-patches
# ``dsv41_cp_indexer._dense_bf16_mqa_logits_torch`` with a full reimplementation
# that does not call ``_fused_mqa_epilogue``, so enabling it here would drop the
# native ``mqa_logits_sum_heads`` path (operator §7/§21, +25%).  Its int32 tail
# is moot once the epilogue runs inside that kernel; the fp16 score chain still
# needs to be ported into ``dsv41_cp_indexer`` and A/B'd on top of the native
# epilogue before it can ship.

# NOTE: ``dsv41_hc_mix_fast`` (round-13 pure-torch hc_mix_stats) is intentionally
# NOT imported.  ``mhc.py`` already REPLACEs the same target with the larger-M
# "vendor GEMM + stats-only inv-rms" split (bf16 matmul, no fp32 upcast); a later
# REPLACE registration here would silently override it and undo that split.
# Import it explicitly only on a build whose wheel lacks the ``mix == 0`` mode.

from . import dp_attention  # noqa: F401,E402
