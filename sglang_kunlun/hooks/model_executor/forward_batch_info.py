"""Hooks for ``sglang.srt.model_executor.forward_batch_info``.

After enabling deepep (`--moe-a2a-backend deepep`), `require_attn_tp_gather()`
is true (`utils/common.py:3938`), so `require_mlp_sync()` is also true and every
forward goes through `ForwardBatch.prepare_mlp_sync_batch` to pad the token
count up to a multiple of attn_tp_size. This padding conflicts with
speculative's "fixed token width per request" when DP=1; see the description of
`_align_spec_mlp_sync_tokens` for details.
"""

from __future__ import annotations

import logging
import math

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_LOGGED_ALIGNMENT = False


def _align_global_num_tokens(self) -> None:
    """Pad the spec forward's mlp-sync token count to lcm(attn_tp_size, tokens per request).

    Upstream `prepare_mlp_sync_batch` first aligns each DP rank's token count
    with `ceil_align(x, attn_tp_size)` (`forward_batch_info.py:1455`, so that
    after MLP it can reduce-scatter along the attn_tp dimension), then derives
    the batch size back with
    `bs = num_tokens // spec_info.num_tokens_per_req` (same file 1551 / 1623).
    These two steps are only self-consistent when the aligned token count is
    still a multiple of the per-request token count:

      * With DP=8 (the 0731 production config), attn_tp_size=1 and ceil_align is
        a no-op, so it is not exposed;
      * With DP=1, attn_tp_size=8, which is coprime with DSPARK's 6 draft
        tokens; bs=1 yields num_tokens=8, bs=8//6=1, so every consumer that
        slices by "bs x tokens per request" fails its assertion (the engram
        hasher's `num_tokens == bs * block`, verify's `raw_out_loc` length check,
        etc.).

    Here we pad `global_num_tokens_cpu` to a multiple of lcm(attn_tp_size,
    tokens per request) before upstream reads it, making that upstream
    ceil_align a no-op. attn_tp_size and the per-request token count are both
    rank invariants, so every rank pads to the same value without extra
    communication. Production shapes are basically unaffected: with
    num_draft_tokens=6 and attn_tp=8, lcm=24; when bs is a multiple of 4
    (bs=32/64), 6xbs is already a multiple of 24, and only the small warmup bs
    pads a few extra rows.
    """
    global _LOGGED_ALIGNMENT

    spec_info = self.spec_info
    if spec_info is None or self.global_num_tokens_cpu is None:
        return
    mode = self.forward_mode
    # Only these modes take the upstream branch that derives bs from num_tokens (the condition at line 1524).
    if not (
        mode.is_decode()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or mode.is_idle()
    ):
        return
    per_req = getattr(spec_info, "num_tokens_per_req", -1)
    if per_req is None or per_req <= 1:
        return

    from sglang.srt.runtime_context import get_parallel

    attn_tp_size = get_parallel().attn_tp_size
    align = math.lcm(attn_tp_size, per_req)
    if align == per_req:
        # attn_tp_size already divides the per-request token count, so the upstream ceil_align does not
        # break the chunking.
        return

    aligned = [
        ((x + align - 1) // align) * align for x in self.global_num_tokens_cpu
    ]
    if aligned == list(self.global_num_tokens_cpu):
        return
    if not _LOGGED_ALIGNMENT:
        _LOGGED_ALIGNMENT = True
        logger.info(
            "kunlun: mlp-sync token 数按 lcm(attn_tp=%d, tokens/req=%d)=%d 对齐 "
            "(%s -> %s)",
            attn_tp_size,
            per_req,
            align,
            list(self.global_num_tokens_cpu),
            aligned,
        )
    self.global_num_tokens_cpu = aligned


_LOGGED_COMPUTE_POSITION_FALLBACK = False


@plugin_hook(
    "sglang.srt.model_executor.forward_batch_info.compute_position",
    type=HookType.REPLACE,
)
def _compute_position_kunlun(
    attn_backend: str,
    extend_prefix_lens,
    extend_seq_lens,
    extend_seq_lens_sum: int,
):
    """Build the extend positions with the vendor fused kernel.

    0.5.8 replaced the Triton body of ``compute_position_triton`` with
    ``torch.ops.xspeedgate_ops.compute_position_kernel``
    (``sgl_kernel/patch/model_executor/forward_batch_info.py:17-27``); the
    baseline prefill trace launches it as
    ``xinfer3::compute_position_kernel<int>`` from
    ``forward_batch_info.py:1007 compute_position``.

    The plugin already registers that same replacement for the Triton symbol
    (``kernels/kernel_ops.py:6715`` -> ``sglang.kernels.ops.attention.position``
    ``compute_position_kernel``), but it is unreachable: upstream only calls
    ``compute_position_triton`` when ``support_triton()`` is true
    (``forward_batch_info.py:2384``), and that reads
    ``current_platform.capabilities.supports_triton``
    (``utils/common.py:1484``), which this platform sets to False
    (``platform/srt.py:59``). So 0.5.23 silently takes
    ``compute_position_torch``, a per-request ``arange`` + ``cat``. Replacing the
    dispatcher instead of the Triton kernel is what makes the vendor op reachable
    without touching the capability flag, which other torch fallbacks depend on.
    """
    global _LOGGED_COMPUTE_POSITION_FALLBACK

    try:
        batch_size = extend_seq_lens.shape[0]
        # Upstream derives ``has_prefix`` the same way
        # (``kernels/ops/attention/position.py:11``); the vendor op takes an
        # empty prefix tensor to mean "no prefix", as the registered Triton
        # replacement already does (``kernel_ops.py:6728``).
        prefix_lens = extend_prefix_lens
        if prefix_lens is None or prefix_lens.shape[0] != batch_size:
            prefix_lens = torch.empty(
                0, device=extend_seq_lens.device, dtype=extend_seq_lens.dtype
            )
        positions, extend_start_loc = torch.ops.xspeedgate_ops.compute_position_kernel(
            prefix_lens, extend_seq_lens, extend_seq_lens_sum
        )
        # The op is templated on the index dtype, so it mirrors the input dtype;
        # upstream's contract is int64 positions and int32 start locs.
        if extend_start_loc.dtype != torch.int32:
            extend_start_loc = extend_start_loc.to(torch.int32)
        return positions.to(torch.int64), extend_start_loc
    except Exception as exc:  # noqa: BLE001 - never block a forward on this
        if not _LOGGED_COMPUTE_POSITION_FALLBACK:
            _LOGGED_COMPUTE_POSITION_FALLBACK = True
            logger.warning(
                "kunlun: xspeedgate_ops.compute_position_kernel unusable (%s); "
                "falling back to compute_position_torch. Positions are "
                "identical; the trace then lacks "
                "xspeedgate_ops::compute_position_kernel and "
                "xinfer3::compute_position_kernel.",
                exc,
            )
        from sglang.srt.model_executor.forward_batch_info import compute_position_torch

        return compute_position_torch(extend_prefix_lens, extend_seq_lens)


@plugin_hook(
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
    "prepare_mlp_sync_batch",
    type=HookType.AROUND,
)
def _align_spec_mlp_sync_tokens(original_fn, self, model_runner):
    _align_global_num_tokens(self)
    return original_fn(self, model_runner)
