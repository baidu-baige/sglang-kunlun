"""CPU regressions for the DSV4.1 P/D argument-relax overlay.

The decode role runs `--dp-size 8 --enable-dp-attention` with DeepEP, which the
designated upstream rejects unless the plugin's AROUND overlay hides the
attention-DP width. 0.5.22 renamed that width to `attn_dp_size`, so these checks
evaluate the real upstream gate expression instead of trusting field names.

Run directly with Python or through unittest discovery; no accelerator needed.
"""

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RELAX = ROOT / "sglang_kunlun/hooks/arg_groups/dsv41_relax.py"
UPSTREAM = Path("/home/ljj/dsv4_1_v0522/code_sglang_0522/sglang")
SPECULATIVE_HOOK = UPSTREAM / "python/sglang/srt/arg_groups/speculative_hook.py"

# The decode launcher's resolved layout: `--dp-size 8 --enable-dp-attention` is
# the deprecated spelling upstream resolves to attn_dp_size=8 / dp_size=1.
DECODE_RESOLVED = {
    "dp_size": 1,
    "attn_dp_size": 8,
    "enable_dp_attention": False,
    "attn_cp_size": 1,
    "moe_a2a_backend": "deepep",
    "enable_dp_lm_head": True,
    "speculative_moe_a2a_backend": None,
}


def _load_relax(names, extra=None):
    """Load the real relax helpers without importing the plugin registry.

    Args:
        names: Top-level function names required by a check.
        extra: Module globals the loaded functions read (for example `_LOGGED`).

    Returns:
        A namespace holding the production helpers and literal constants.
    """
    tree = ast.parse(RELAX.read_text())
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
    if found != set(names):
        raise AssertionError(f"Missing production definitions: {set(names) - found}")
    namespace = {"os": os, "logger": logging.getLogger(__name__)}
    namespace.update(extra or {})
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(RELAX), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _dspark_dp_gate():
    """The upstream `if` test that guards DSpark's attention-DP block.

    Returns:
        (expression source, referenced `cfg` attribute names) of the gate in the
        designated upstream `_handle_dspark`.
    """
    tree = ast.parse(SPECULATIVE_HOOK.read_text())
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_dspark"
    )
    for node in target.body:
        if not isinstance(node, ast.If):
            continue
        attrs = {
            child.attr
            for child in ast.walk(node.test)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "cfg"
        }
        # The guarded block is the one raising the DeepEP rejection.
        messages = [
            value.value
            for value in ast.walk(node)
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        if attrs and any("moe_a2a_backend 'none' " in text for text in messages):
            return ast.unparse(node.test), attrs
    raise AssertionError("upstream _handle_dspark has no attention-DP gate")


class PdRelaxCPU(unittest.TestCase):
    def test_overlay_values_cover_the_upstream_gate(self):
        """Every field the real gate reads must be in the relax overlay."""
        _, attrs = _dspark_dp_gate()
        fields = _load_relax({"_enabled"})._RELAX_FIELDS
        self.assertTrue(attrs)
        self.assertEqual(attrs - set(fields), set())

    def test_gate_fires_without_relax_and_is_suppressed_with_it(self):
        """The overlay must flip the actual upstream condition, not a renamed one."""
        expression, _ = _dspark_dp_gate()
        fields = _load_relax({"_enabled"})._RELAX_FIELDS
        plain = SimpleNamespace(**DECODE_RESOLVED)
        relaxed = SimpleNamespace(**{**DECODE_RESOLVED, **fields})
        self.assertTrue(eval(expression, {}, {"cfg": plain}))
        self.assertFalse(eval(expression, {}, {"cfg": relaxed}))

    def test_overlay_is_scoped_to_the_validated_call(self):
        """Relaxed values are visible inside the call and removed afterwards."""
        relax = _load_relax(
            {"_run_relaxed", "_push_overlay", "_pop_overlay", "_enabled"},
            extra={"_LOGGED": set()},
        )
        seen = []

        def original(server_args):
            seen.append([dict(fields) for _source, fields in server_args.stash])
            return "ok"

        server_args = SimpleNamespace(stash=[])
        server_args._resolved_overrides = server_args.stash
        with patch.dict(
            os.environ, {"DSV4_ALLOW_DSPARK_DP_DEEPEP": "1"}, clear=False
        ):
            self.assertEqual(
                relax._run_relaxed(
                    original, server_args, "DSV4_ALLOW_DSPARK_DP_DEEPEP", "_handle_dspark"
                ),
                "ok",
            )
        self.assertEqual(seen, [[relax._RELAX_FIELDS]])
        self.assertEqual(server_args.stash, [])

    def test_overlay_is_removed_when_validation_raises(self):
        """A rejected configuration must not leave a stale dp/cp override."""
        relax = _load_relax(
            {"_run_relaxed", "_push_overlay", "_pop_overlay", "_enabled"},
            extra={"_LOGGED": set()},
        )

        def original(server_args):
            raise ValueError("rejected")

        server_args = SimpleNamespace()
        server_args._resolved_overrides = []
        with patch.dict(
            os.environ, {"DSV4_ALLOW_DSPARK_DP_DEEPEP": "1"}, clear=False
        ):
            with self.assertRaises(ValueError):
                relax._run_relaxed(
                    original, server_args, "DSV4_ALLOW_DSPARK_DP_DEEPEP", "_handle_dspark"
                )
        self.assertEqual(server_args._resolved_overrides, [])

    def test_disabled_switch_keeps_upstream_behaviour(self):
        """With the env switch unset, no overlay is pushed at all."""
        relax = _load_relax(
            {"_run_relaxed", "_push_overlay", "_pop_overlay", "_enabled"},
            extra={"_LOGGED": set()},
        )
        observed = []

        def original(server_args):
            observed.append(getattr(server_args, "_resolved_overrides", None))
            return "untouched"

        server_args = SimpleNamespace()
        with patch.dict(
            os.environ, {"DSV4_ALLOW_DSPARK_DP_DEEPEP": "0"}, clear=False
        ):
            self.assertEqual(
                relax._run_relaxed(
                    original, server_args, "DSV4_ALLOW_DSPARK_DP_DEEPEP", "_handle_dspark"
                ),
                "untouched",
            )
        self.assertEqual(observed, [None])
        self.assertFalse(hasattr(server_args, "_resolved_overrides"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
