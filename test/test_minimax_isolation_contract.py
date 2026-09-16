"""Isolation contract: MiniMax-M3 support must not change other models' paths.

Every shared-file edit the MiniMax port relies on is additive or sits behind a
condition only the MiniMax contract can satisfy. These tests pin those
conditions so a later refactor cannot silently widen them (which is how the M3
line would start affecting DSV4 or other models again).

Checks that need the plugin's heavy import chain run in a subprocess: importing
it inside the pytest session perturbs module state for the other contract tests
(observed with ``test_ragged_draft_metadata_contract``).
"""

import ast
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "sglang_kunlun"
MINIMAX_DIR = PLUGIN_ROOT / "hooks" / "models" / "minimax"

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import os, torch
    os.environ.setdefault("SGLANG_IS_FLASHINFER_AVAILABLE", "False")

    # Only the two gate ops are stubbed (no device, no real kernel); the rest of
    # kunlun_ops must stay real because the plugin imports symbols from it.
    import kunlun_ops

    calls = []

    def moe_fused_gate(scores, bias, n_group, topk_group, topk, n_share, scale,
                       weights, indices, block_statistic):
        calls.append("minimax-kunlun")
        weights.zero_(); indices.zero_()

    def moe_fused_gate_dsv4(input, bias, weights, indices, topk, scoring_func,
                            n_share, renormalize, scale, apply_on_output):
        calls.append("dsv4")
        weights.zero_(); indices.zero_()

    kunlun_ops.moe_fused_gate = moe_fused_gate
    kunlun_ops.moe_fused_gate_dsv4 = moe_fused_gate_dsv4

    from sglang_kunlun.kernels import kernel_ops

    scores = torch.zeros(4, 8)
    bias = torch.zeros(8)

    # DSV4 (sqrtsoftplus) must never reach the MiniMax implementation.
    kernel_ops.moe_fused_gate_dsv4_kunlun(scores, bias, topk=2,
                                          scoring_func="sqrtsoftplus")
    assert calls[-1] == "dsv4", calls

    # sigmoid without the output-scaling flag is *not* the MiniMax contract and
    # must stay on the DSV4 op as well.
    kernel_ops.moe_fused_gate_dsv4_kunlun(scores, bias, topk=2,
                                          scoring_func="sigmoid")
    assert calls[-1] == "dsv4", calls

    # The MiniMax contract dispatches into the MiniMax package.
    kernel_ops.moe_fused_gate_dsv4_kunlun(
        scores, bias, topk=2, scoring_func="sigmoid",
        apply_routed_scaling_factor_on_output=True)
    assert calls[-1] == "minimax-kunlun", calls

    # The dense backend's paged KV helpers are inert for every other pool.
    from sglang_kunlun.hooks.layers.attention.kunlun_backend import (
        KunlunAttentionBackend,
    )

    class _NotMiniMaxPool:
        pass

    class _Stub:
        page_size = 128
        token_to_kv_pool = _NotMiniMaxPool()

    key = torch.zeros(128 * 4, 1, 128)
    value = torch.zeros(128 * 4, 1, 128)
    page_table = torch.zeros(1, 4, dtype=torch.int32)
    for helper in (KunlunAttentionBackend._paged_kv_view,
                   KunlunAttentionBackend._paged_kv_gather):
        got = helper(_Stub(), key, value, page_table)
        assert got[0] is key and got[1] is value and got[2] is page_table, helper

    print("ISOLATION_OK")
    """
)


def _hook_targets(tree_root: Path) -> set[str]:
    """All ``@plugin_hook("<target>", ...)`` targets under *tree_root*."""
    targets = set()
    for path in tree_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "plugin_hook" or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                targets.add(arg.value)
    return targets


class MiniMaxIsolationContractTest(unittest.TestCase):
    def test_minimax_hook_targets_are_exclusive_to_the_package(self):
        """No other module may register a hook on a MiniMax target."""
        minimax_targets = _hook_targets(MINIMAX_DIR)
        self.assertTrue(minimax_targets, "expected MiniMax hooks to be found")

        others = _hook_targets(PLUGIN_ROOT) - minimax_targets
        overlap = minimax_targets & others
        self.assertEqual(
            overlap,
            set(),
            f"MiniMax hook targets also registered elsewhere: {sorted(overlap)}",
        )

    def test_select_experts_branch_requires_the_minimax_contract(self):
        """The shared select_experts hook only forks on the M3 contract.

        Plain sigmoid (flag False) used to take ``fused_topk``; the source must
        keep that guarantee so no other sigmoid model changes behaviour.
        """
        source = (PLUGIN_ROOT / "hooks" / "layers" / "moe" / "topk.py").read_text()
        self.assertIn(
            'elif scoring_func == "sigmoid" and apply_routed_scaling_factor_on_output:',
            source,
            "the sigmoid fork must be gated on the MiniMax output-scaling flag",
        )

    def test_shared_paths_are_inert_and_dispatch_is_gated(self):
        """Behavioral checks, run in a subprocess to keep this session clean."""
        env = dict(os.environ)
        env.setdefault("SGLANG_IS_FLASHINFER_AVAILABLE", "False")
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertIn(
            "ISOLATION_OK",
            proc.stdout,
            f"isolation probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
