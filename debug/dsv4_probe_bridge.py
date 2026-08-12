"""Debug-only bridge for DeepSeek-V4 MoE tensor dumps."""

from __future__ import annotations

import os

import torch


def dump_selected_moe_rows(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> None:
    callback = getattr(layer, "_dsv4_moe_tensor_dump_callback", None)
    rows_text = os.getenv("DSV4_W8A8_MOE_DUMP_ROWS", "all")
    stages_text = os.getenv("DSV4_W8A8_MOE_DUMP_STAGES")
    if callback is None or not rows_text or not isinstance(value, torch.Tensor):
        return
    if stages_text and name not in stages_text.split():
        return

    if rows_text.strip().lower() == "all":
        rows = list(range(num_tokens))
    else:
        rows = [int(item) for item in rows_text.split(",") if item]
    rows = [row for row in rows if 0 <= row < num_tokens]
    if not rows or value.ndim == 0:
        return

    if value.shape[0] == num_tokens:
        indices = torch.tensor(rows, dtype=torch.long, device=value.device)
    elif value.shape[0] == num_tokens * top_k:
        indices = torch.tensor(
            [row * top_k + offset for row in rows for offset in range(top_k)],
            dtype=torch.long,
            device=value.device,
        )
    else:
        return
    callback(name, value.index_select(0, indices))
