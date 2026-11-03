"""Relax a few combination checks for V4.1 + DSpark + PD (previously done by editing `sglang/` directly).

Background: The V4-0731 production setup used the `dp8 + deepep(low_latency) + DSpark`
combination. In V4.1, two newly added whitelists in `arg_groups` are overly conservative
and block the following config at startup:

    P: tp8 + attn_cp8 + DSpark + deepep(normal)      → attn_tp_size = 8//1//8 = 1
    D: tp8 + dp8(dp-attention) + DSpark + deepep(low_latency) → 8//8//1 = 1

The `attn_tp_size` on both sides are equal, so the equality requirement in
`disaggregation/common/conn.py` is already satisfied; the only thing being blocked is
that "the combination isn't recorded in the whitelist".

How it's relaxed: **without touching `sglang/`**, we wrap (AROUND) these two validate
functions and, while they run, temporarily push a declaration overlay onto
`server_args._resolved_overrides` (the stash at `arg_groups/overrides.py:131-135`, which
is what `resolution_result` reads), so that validation sees
`dp_size=1 / attn_dp_size=1 / enable_dp_attention=False / attn_cp_size=1`. The overlay is
removed immediately after the function returns — the actual resolution result is unaffected.

Two env switches, both defaulting to 0 = fully preserve upstream behavior:
  DSV4_ALLOW_DSPARK_PD_CP=1      relax the DP/CP conditions in `validate_deepseek_v41_features`
  DSV4_ALLOW_DSPARK_DP_DEEPEP=1  relax the dp-attention a2a whitelist in `_handle_dspark`

Known boundaries (checks that also get skipped when relaxed; all side-effect-free under the current config):
  * The `enable_decoder_swa_bounded_replay` branch in `validate_deepseek_v41_features`
    reads `enable_dp_attention`; on this line that switch is False, so the branch isn't entered.
    **If you want to enable `--enable-decoder-swa-bounded-replay`, remove this hook first, then re-evaluate.**
  * The dp-attention block in `_handle_dspark` is skipped entirely, which also skips the
    three checks: `--enable-dp-lm-head` being required, `SGLANG_RAGGED_VERIFY_MODE=static`
    (since 0.5.22 this one is also triggered by `SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION`,
    with defaults of `static` and `False` respectively), and `speculative_moe_a2a_backend`
    consistency. Among these, the `static` check also has a copy in
    `validate_deepseek_v41_features` (which is **not** relaxed), so it's still guaranteed;
    `enable_dp_lm_head` is ensured by the D-side startup script (always passed when dp>1).
"""

from __future__ import annotations

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_RELAX_FIELDS = {
    "dp_size": 1,
    "attn_dp_size": 1,
    "enable_dp_attention": False,
    "attn_cp_size": 1,
}
_MARKER_SOURCE = "sglang_kunlun.dsv41_relax(tmp)"
_LOGGED = set()


def _enabled(env_name: str) -> bool:
    return os.environ.get(env_name, "0") == "1"


def _push_overlay(server_args):
    """Temporarily push a declaration overlay, returning (stash, marker) for removal."""
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        server_args._resolved_overrides = stash
    marker = (_MARKER_SOURCE, dict(_RELAX_FIELDS))
    stash.append(marker)
    return stash, marker


def _pop_overlay(stash, marker) -> None:
    try:
        stash.remove(marker)
    except ValueError:  # should not happen in theory; even if it does, it must not mask the original exception
        logger.error(
            "sglang-kunlun: dsv41_relax overlay marker missing; resolution may "
            "carry a stale dp/cp override"
        )


def _run_relaxed(original_fn, server_args, env_name: str, what: str):
    if not _enabled(env_name):
        return original_fn(server_args)
    if what not in _LOGGED:
        _LOGGED.add(what)
        logger.warning(
            "sglang-kunlun: %s=1 ⇒ 临时按 dp=1/attn_dp=1/cp=1 放行 %s 的组合校验"
            "（真实解析结果不变）",
            env_name,
            what,
        )
    stash, marker = _push_overlay(server_args)
    try:
        return original_fn(server_args)
    finally:
        _pop_overlay(stash, marker)


@plugin_hook(
    "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
    type=HookType.AROUND,
)
def validate_deepseek_v41_features_kunlun(original_fn, server_args):
    """Run V4.1 validation with the opt-in temporary DP/CP overrides.

    Args:
        original_fn: Upstream feature validator.
        server_args: Server arguments whose resolution overlay is restored afterward.

    Returns:
        The upstream validator's result.
    """
    return _run_relaxed(
        original_fn,
        server_args,
        "DSV4_ALLOW_DSPARK_PD_CP",
        "validate_deepseek_v41_features",
    )


@plugin_hook(
    "sglang.srt.arg_groups.speculative_hook._handle_dspark",
    type=HookType.AROUND,
)
def handle_dspark_kunlun(original_fn, server_args):
    """Run DSpark setup with the opt-in DP-attention compatibility overlay.

    Args:
        original_fn: Upstream DSpark argument handler.
        server_args: Server arguments whose resolution overlay is restored afterward.

    Returns:
        The upstream handler's result.
    """
    return _run_relaxed(
        original_fn,
        server_args,
        "DSV4_ALLOW_DSPARK_DP_DEEPEP",
        "_handle_dspark",
    )
