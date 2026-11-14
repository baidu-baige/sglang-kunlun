"""Pad micro prefill batches so every CP rank carries at least one token.

A one-token prefill chunk split over ``attn_cp_size`` leaves the other ranks with
an empty shard. DeepEP's normal dispatch then reports a negative
``recv_num_tokens`` after its internal notify timeout and every prefill rank dies
with ``deep_ep.cpp:410 bkcl_notify_dispatch_standard failed`` -- measured
repeatedly, always after a ``#new-seq: N, #new-token: 1`` batch.

The fix is deliberately local: before dispatch, repeat rows until the local shard
has ``cp_size`` rows, and after combine drop the repeats. Repeating (rather than
zero-padding) keeps every routed expert id valid, which the zero-token path in
``dispatch_a`` cannot guarantee -- an all-zero topk sends the whole batch to
expert 0 on every rank, which is what makes the vendor's per-expert counts
disagree.
"""

from typing import Tuple

import torch


def pad_to_cp(hidden_states: torch.Tensor, cp_size: int) -> Tuple[torch.Tensor, int]:
    """Repeat rows until there are at least ``cp_size`` of them.

    Returns the padded tensor and the original row count, which the caller must
    hand back to :func:`unpad` after the collective so the padded rows never
    reach the caller.
    """
    rows = hidden_states.shape[0]
    if rows == 0 or rows >= cp_size:
        return hidden_states, rows
    reps = (cp_size + rows - 1) // rows
    padded = hidden_states.repeat((reps,) + (1,) * (hidden_states.dim() - 1))[:cp_size]
    return padded, rows


def pad_topk_to_cp(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, cp_size: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Same repetition for the router outputs, keeping them row-aligned."""
    rows = topk_ids.shape[0]
    if rows == 0 or rows >= cp_size:
        return topk_ids, topk_weights, rows
    reps = (cp_size + rows - 1) // rows
    ids = topk_ids.repeat((reps, 1))[:cp_size]
    weights = topk_weights.repeat((reps, 1))[:cp_size]
    return ids, weights, rows


def unpad(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Drop the repeated rows produced by :func:`pad_to_cp`."""
    return x if x.shape[0] == rows else x[:rows]
