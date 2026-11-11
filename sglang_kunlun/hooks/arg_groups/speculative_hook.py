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
"""Hooks for ``sglang.srt.arg_groups.speculative_hook``.

The DSV4 phase task runs DSpark with decode DP attention while both roles keep
DeepEP. The stock ``_handle_dspark`` rejects that combination outright and only
permits the TP-MoE backends.

Swallowing the error is wrong: the rest of the function is what declares
``speculative_num_steps`` / ``speculative_eagle_topk`` /
``speculative_num_draft_tokens``. Returning early leaves them ``None`` and every
downstream consumer (SWA ring size, allocation sizing, draft backend setup)
fails later with a confusing ``NoneType`` arithmetic error. So on the opt-in we
retry the whole function with a view that reports an allowed MoE backend; all
declarations run, and only the one clause is bypassed.
"""

from __future__ import annotations

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_ALLOW_CP_DP_ENV = "SGLANG_KUNLUN_ALLOW_DSPARK_CP_DP"
logger = logging.getLogger(__name__)


class _AllowedMoeA2AView(object):
    """Proxy a resolved config view, reporting an allowed MoE backend."""

    def __init__(self, view):
        self._view = view

    @property
    def moe_a2a_backend(self):
        """Return the allowed backend label used during resolver retry.

        Returns:
            The string ``"none"`` so the resolver can declare DSpark defaults.
        """
        return "none"

    def __getattr__(self, name):
        return getattr(self._view, name)


@plugin_hook(
    "sglang.srt.arg_groups.speculative_hook._handle_dspark",
    type=HookType.AROUND,
)
def handle_dspark_kunlun(original_fn, server_args):
    """Retry the DSpark handler with the DeepEP clause opted out.

    Only active with ``SGLANG_KUNLUN_ALLOW_DSPARK_CP_DP=1`` and only for the
    ``moe_a2a_backend`` refusal. Without the env var the upstream error is
    re-raised unchanged.
    """
    try:
        return original_fn(server_args)
    except ValueError as exc:
        if (os.environ.get(_ALLOW_CP_DP_ENV) != "1"
                or "supports moe_a2a_backend" not in str(exc)):
            raise

    from sglang.srt.arg_groups import speculative_hook as hook

    logger.warning(
        "sglang-kunlun: %s=1; retrying the DSpark resolver with the DeepEP "
        "MoE-backend clause bypassed so its defaults are still declared",
        _ALLOW_CP_DP_ENV,
    )
    real_view = hook.resolving_view
    hook.resolving_view = lambda args: _AllowedMoeA2AView(real_view(args))
    try:
        return original_fn(server_args)
    finally:
        hook.resolving_view = real_view
