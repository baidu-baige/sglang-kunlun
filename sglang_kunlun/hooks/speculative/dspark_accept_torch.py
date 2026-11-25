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
"""Force the DSpark accept kernels onto their torch implementations.

``SelectMixedAccept.execute`` picks triton whenever ``inputs_on_cuda`` is true.
On the Kunlun stack the device reports as cuda, so it selects
``select_mixed_accept_triton`` and the XPU Triton compiler fails while compiling
a new batch specialization:

    dspark_accept.py:525: error: operand #0 does not dominate this use
    RuntimeError: PassManager::run failed

``AcceptSampling.execute`` makes the same cpu/cuda choice, and its triton branch
computes the bonus token with ``gather_two_level_bonus_triton`` -- an int64
``tl.store`` gather of the same shape class that has been observed to leave rows
unwritten on XPU (see ``kernel_ops.sample_step_tokens_triton``). That silently
corrupts one token per sampling step, so temperature/top-p requests degrade
while greedy requests (``AcceptGreedy``, different bonus path) stay correct.

Both classes ship an equivalent pure-torch path, which is what CPU execution
already uses. Select it on Kunlun by default; set
``SGLANG_KUNLUN_DSPARK_ACCEPT_TRITON=1`` to opt back in for an A/B.

FinalizeAcceptLens and BuildOutTokens always use device torch operations.
The direct finalizer entry points used by the graph epilogue are routed too.
Acceptance caps and bonus gathers also use torch: their Triton launches are
not recorded by XMLIR, leaving capture-time values in the folded epilogue.
"""

from __future__ import annotations

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_FORCE_TORCH_ENV = "SGLANG_KUNLUN_DSPARK_ACCEPT_TRITON"
_announced = False
_sampling_announced = False


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.SelectMixedAccept.execute",
    type=HookType.AROUND,
)
def execute_kunlun(original_fn, *args, **kwargs):
    """Use the torch mixed-accept selector on XPU; triton cannot compile it."""
    global _announced
    if os.environ.get(_FORCE_TORCH_ENV) == "1":
        return original_fn(*args, **kwargs)

    from sglang.kernels.ops.speculative.dspark.dspark_accept import SelectMixedAccept

    if not _announced:
        _announced = True
        logger.warning(
            "sglang-kunlun: DSpark mixed-accept uses the torch selector "
            "(XPU triton compile fails); %s=1 restores triton",
            _FORCE_TORCH_ENV,
        )
    # ``execute`` is a classmethod and the AROUND wrapper forwards the bound
    # ``cls`` as args[0]; ``torch`` is a classmethod too, so passing it again
    # positionally would bind ``cls`` twice (TypeError on mixed batches).
    cls, *rest = args if args else (SelectMixedAccept,)
    return cls.torch(*rest, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.AcceptSampling.execute",
    type=HookType.AROUND,
)
def accept_sampling_kunlun(original_fn, *args, **kwargs):
    """Keep the sampling bonus token on torch; the XPU triton gather is lossy."""
    global _sampling_announced
    if os.environ.get(_FORCE_TORCH_ENV) == "1":
        return original_fn(*args, **kwargs)

    from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptSampling

    if not _sampling_announced:
        _sampling_announced = True
        logger.warning(
            "sglang-kunlun: DSpark sampling accept uses the torch bonus gather "
            "(XPU triton gather loses int64 rows); %s=1 restores triton",
            _FORCE_TORCH_ENV,
        )
    cls, *rest = args if args else (AcceptSampling,)
    return cls.torch(*rest, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.FinalizeAcceptLens.execute",
    type=HookType.AROUND,
)
def finalize_accept_lens_kunlun(original_fn, *args, **kwargs):
    """Keep acceptance lengths on torch, including CUDA-compatible devices."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import FinalizeAcceptLens

    cls, *rest = args if args else (FinalizeAcceptLens,)
    return cls.torch(*rest, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.finalize_accept_lens_triton",
    type=HookType.REPLACE,
)
@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_verify.finalize_accept_lens_triton",
    type=HookType.REPLACE,
)
def finalize_accept_lens_direct_kunlun(*, correct_len, cap_trim_lens, prefix_lens):
    """Route direct graph-epilogue calls through the same torch finalizer."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import finalize_accept_lens

    return finalize_accept_lens(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.CapCorrectLen.execute",
    type=HookType.AROUND,
)
def cap_correct_len_kunlun(original_fn, *args, **kwargs):
    """Record acceptance caps instead of executing Triton only at capture."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import CapCorrectLen

    cls, *rest = args if args else (CapCorrectLen,)
    return cls.torch(*rest, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.gather_row_bonus_triton",
    type=HookType.REPLACE,
)
def gather_row_bonus_kunlun(*, table, idx):
    """Read the replay's capped acceptance index and target token."""
    import torch

    # Inactive epilogue rows have verify_lens=0, hence a capped index of -1.
    # Match the upstream torch acceptance path's negative-index semantics.
    rows = torch.arange(table.shape[0], device=table.device)
    return table[rows, idx.long()].to(torch.int64)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_verify_window.BuildOutTokens.execute",
    type=HookType.AROUND,
)
def build_out_tokens_kunlun(original_fn, *args, **kwargs):
    """Build output tokens on the device using torch copy and scatter."""
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens

    cls, *rest = args if args else (BuildOutTokens,)
    return cls.torch(*rest, **kwargs)
