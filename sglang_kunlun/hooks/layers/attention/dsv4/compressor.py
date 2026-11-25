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
"""Kunlun overrides for the v1 legacy DeepSeek-V4 compressor.

Corresponds to upstream ``sglang.srt.layers.attention.dsv4.compressor``
(``compress_old.py``-backed v1 path, selected when
``SGLANG_OPT_USE_COMPRESSOR_V2`` is False). The overrides here fix
call-signature / dispatch incompatibilities specific to running the v1 path on
Kunlun (mismatches introduced by the v2-oriented upstream callers).
"""

from __future__ import annotations

import functools
import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v1 legacy create_paged_compressor_data (compressor.py) signature patch.
# compressor_v2.py's create_paged_compressor_data accepts an
# ``online_state_slot_offset: int = 0`` kwarg (used to thread state-pool slot
# offsets through the online-compress path), but the v1 function does not
# declare this parameter at all. Callers written against the v2 signature
# (e.g. deepseek_v4_backend.py) pass this kwarg unconditionally, which raises
# ``TypeError: create_paged_compressor_data() got an unexpected keyword
# argument 'online_state_slot_offset'`` when running the v1 path. v1 has no
# online-compress branch, so the value is accepted but unused -- this patch
# only restores call-signature compatibility.
#
# ``deepseek_v4_backend.py`` binds the name into its own namespace with
# ``from ...compressor import create_paged_compressor_data``, but
# ``_propagate_patch`` rewrites every stale ``is original`` module attribute, so
# that reference is fixed automatically. That also gives the v1/v2 isolation for
# free: with ``SGLANG_OPT_USE_COMPRESSOR_V2`` the backend imported
# compressor_v2's function, a different object, which is left untouched.
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor.create_paged_compressor_data",
    type=HookType.AROUND,
)
def create_paged_compressor_data_kunlun(
    original_fn, *args, online_state_slot_offset: int = 0, **kwargs
):
    """Swallow the v2-only ``online_state_slot_offset`` kwarg (no-op on v1)."""
    return original_fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Compressor.forward_cuda alias (v1 legacy compressor.py). Kunlun masquerades
# as CUDA (torch.version.cuda is not None -> is_cuda()==True), but
# sglang-kunlun-standalone never registers itself as an OOT platform via
# current_platform.is_out_of_tree()/register_oot_forward(). As a result,
# MultiPlatformOp.dispatch_forward() falls through to the hardcoded
# "if _is_cuda: return self.forward_cuda" branch and lands on the
# unimplemented base-class forward_cuda, which raises NotImplementedError.
# Fix: alias Compressor.forward_cuda -> Compressor.forward_native so the
# native (pure-torch/Kunlun-kernel) path is used instead.
# ---------------------------------------------------------------------------

def patch_compressor_forward_cuda_alias():
    """Alias v1 Compressor.forward_cuda -> forward_native on Kunlun.

    Kunlun's torch build reports ``is_cuda()==True``, so
    ``MultiPlatformOp.dispatch_forward()`` picks ``forward_cuda`` unless the
    platform is registered as out-of-tree. ``Compressor`` only implements
    ``forward_native``, so without this alias any call raises
    ``NotImplementedError``.
    """
    try:
        from sglang.srt.layers.attention.dsv4.compressor import Compressor

        Compressor.forward_cuda = Compressor.forward_native
        logger.info(
            "sglang-kunlun: patched Compressor.forward_cuda -> forward_native"
        )
    except Exception as exc:  # noqa: BLE001 - v1 compressor.py may not be importable
        logger.warning(
            "sglang-kunlun: Compressor forward_cuda alias patch failed: %s", exc
        )


# ---------------------------------------------------------------------------
# Route the DSV4 backend's compress-plan builder to the v1 (compress_old) path.
# The stock ``deepseek_v4_backend`` hardcodes ``from ...compressor_v2 import
# create_paged_compressor_data`` (SGLANG_OPT_USE_COMPRESSOR_V2 is a no-op
# _DeprecatedEnv now). compressor_v2's ``CompressorPrefillPlan.generate``
# JIT-compiles a CUDA kernel (c_plan.cuh) via nvcc, which fails on this XPU
# image (host g++ lacks C++20 <concepts>). The v1 ``compressor.py`` path builds
# the plan through ``compress_old``, for which this plugin registers pure-torch
# ``CompressorPrefillPlan.generate`` / ``compress_forward`` hooks -- no CUDA JIT.
# So redirect the backend's module-level name to the v1 builder (which the
# ``create_paged_compressor_data_kunlun`` hook above already made accept the
# v2-only ``online_state_slot_offset`` kwarg).
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
    type=HookType.AROUND,
)
def create_paged_compressor_data_v1_route(original_fn, *args, **kwargs):
    """Send the backend's compress-plan call to the v1 compress_old builder."""
    from sglang.srt.layers.attention.dsv4.compressor import (
        create_paged_compressor_data as _v1_create,
    )

    return _v1_create(*args, **kwargs)
