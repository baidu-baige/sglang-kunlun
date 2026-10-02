"""Kunlun torch fallback for Kimi-K3 attention residual."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _aggregate_rows_kunlun(prefix_sum, bank, nvb, score_proj, score_norm):
    if nvb == 0:
        return prefix_sum
    rows = torch.cat([bank[:, :nvb, :], prefix_sum.unsqueeze(1)], dim=1)
    tokens, row_count, hidden = rows.shape
    normed = score_norm.forward_native(rows.reshape(tokens * row_count, hidden))
    scores = score_proj(normed)[0].reshape(tokens, row_count)
    probs = torch.softmax(scores.float(), dim=-1)
    return (probs.unsqueeze(-1) * rows.float()).sum(dim=1).to(prefix_sum.dtype)


@plugin_hook(
    "sglang.srt.layers.attn_residual.aggregate_stream",
    type=HookType.REPLACE,
)
def aggregate_stream_kunlun(prefix_sum, bank, nvb, score_proj, score_norm):
    """Aggregate attention-residual rows with the Torch fallback.

    Args:
        prefix_sum: Current residual prefix with shape ``[tokens, hidden]``.
        bank: Residual bank with shape ``[tokens, rows, hidden]``.
        nvb: Number of valid bank rows.
        score_proj: Projection used to produce aggregation scores.
        score_norm: Normalization module used before projection.

    Returns:
        The weighted residual aggregate in the prefix dtype.
    """
    return _aggregate_rows_kunlun(prefix_sum, bank, nvb, score_proj, score_norm)


@plugin_hook(
    "sglang.srt.layers.attn_residual._aggregate",
    type=HookType.REPLACE,
)
def aggregate_kunlun(
    prefix_sum,
    bank,
    nvb,
    score_proj,
    score_norm,
    out_norm,
    write_bank_row=False,
):
    """Replace the Triton score/combine path used by model forward.

    Args:
        prefix_sum: Current residual prefix with shape ``[tokens, hidden]``.
        bank: Residual bank with shape ``[tokens, rows, hidden]``.
        nvb: Bank row count or target row when writing a bank entry.
        score_proj: Projection used to produce aggregation scores.
        score_norm: Normalization module used before projection.
        out_norm: Final output normalization module.
        write_bank_row: Whether to store ``prefix_sum`` in bank row ``nvb``.

    Returns:
        The normalized residual result.

    Raises:
        IndexError: If ``write_bank_row`` selects an invalid bank row.
    """

    if write_bank_row:
        if nvb < 0 or nvb >= bank.shape[1]:
            raise IndexError(
                f"attention-residual bank row {nvb} is out of range "
                f"for bank shape {tuple(bank.shape)}"
            )
        bank[:, nvb, :].copy_(prefix_sum)
        return out_norm(prefix_sum)
    mixed = _aggregate_rows_kunlun(prefix_sum, bank, nvb, score_proj, score_norm)
    return out_norm(mixed)
