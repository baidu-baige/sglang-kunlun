"""Kunlun ragged Draft Extend metadata hooks."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _ragged_extend_lengths(forward_batch, num_queries: int):
    if not forward_batch.forward_mode.is_draft_extend_v2():
        return None

    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = forward_batch.batch_size
    if lengths is None or len(lengths) != batch_size or batch_size <= 0:
        return None

    lengths = [int(length) for length in lengths]
    if sum(lengths) != num_queries:
        raise ValueError(
            "Kunlun ragged Draft Extend query count mismatch: "
            f"lengths={lengths}, num_queries={num_queries}"
        )
    if num_queries % batch_size == 0:
        uniform_length = num_queries // batch_size
        if all(length == uniform_length for length in lengths):
            return None
    return lengths


@plugin_hook(
    "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend."
    "KunlunDeepseekV4AttnBackend._make_lod",
    type=HookType.AROUND,
)
def make_ragged_draft_extend_lod_kunlun(
    original_fn, self, forward_batch, num_queries: int, device: torch.device
):
    """Build LoD metadata for non-uniform Draft Extend batches."""
    lengths = _ragged_extend_lengths(forward_batch, num_queries)
    if lengths is None:
        return original_fn(self, forward_batch, num_queries, device)

    batch_size = len(lengths)
    q_lod_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    if batch_size:
        torch.cumsum(
            torch.tensor(lengths, dtype=torch.int32),
            dim=0,
            out=q_lod_cpu[1:],
        )
    q_lod = q_lod_cpu.to(device, non_blocking=False)
    kv_lens = forward_batch.seq_lens[:batch_size].to(torch.int32)
    kv_lens_cpu = (
        forward_batch.seq_lens_cpu[:batch_size].to(torch.int32)
        if forward_batch.seq_lens_cpu is not None
        else kv_lens.to("cpu", non_blocking=False)
    )
    return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens
