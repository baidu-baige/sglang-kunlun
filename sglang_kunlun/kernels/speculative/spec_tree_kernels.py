"""Kunlun native replacements for the 0.5.23 JIT speculative-tree kernels.

0.5.23 moved ``build_tree_kernel_efficient`` / ``verify_tree_greedy`` out of the
prebuilt ``sgl_kernel`` into ``sglang.kernels.ops.speculative.tree``, which
JIT-compiles a CUDA module that will not build on Kunlun. Route both to the
plugin's native implementations (``sgl_kernel_kunlun.speculative``). The
positional contract matches 1:1 (selected_index/verified_seq_len/depth/
draft_token_num map onto the native top_scores_index/seq_lens/spec_steps/
num_verify_tokens).
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_TREE = "sglang.kernels.ops.speculative.tree"
_EAGLE = "sglang.kernels.ops.speculative.eagle"
_SAMPLING = "sglang.kernels.ops.speculative.sampling"


@plugin_hook(f"{_TREE}.build_tree_kernel_efficient", type=HookType.REPLACE)
def build_tree_kernel_efficient_kunlun(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode,
):
    """ """
    from sglang_kunlun.kernels.sgl_kernel_kunlun.speculative import (
        build_tree_kernel_efficient as _native,
    )

    _native(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


@plugin_hook(f"{_TREE}.verify_tree_greedy", type=HookType.REPLACE)
def verify_tree_greedy_kunlun(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    """ """
    from sglang_kunlun.kernels.sgl_kernel_kunlun.speculative import (
        verify_tree_greedy as _native,
    )

    _native(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


@plugin_hook(f"{_EAGLE}.prepare_verify_commit_outputs", type=HookType.REPLACE)
def prepare_verify_commit_outputs_kunlun(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    num_draft_tokens=0,
    mamba_track_interval=0,
    bonus_dtype=torch.int32,
):
    """Torch-native ``_verify_commit_outputs_kernel``.

    0.5.23's eagle commit op is a Triton kernel launched as ``kernel[grid](...)``;
    Triton is unsupported on Kunlun so the launch raises ``'function' object is
    not subscriptable``. Replicate it with tensor ops. Per row r:
    ``accepted = accept_lens[r]``, ``idx = accept_index[r, accepted-1]``,
    ``new_seq_lens = seq_lens + accepted``, ``bonus = predict[idx (wrapped)]``;
    with draft tokens also ``last_steps/correct/select`` and ``ids = predict``.
    """
    bs = accept_lens.numel()
    device = seq_lens.device
    new_seq_lens = torch.empty(
        (bs,),
        device=device,
        dtype=torch.promote_types(seq_lens.dtype, accept_lens.dtype),
    )
    bonus = torch.empty((bs,), device=predict.device, dtype=bonus_dtype)
    last_steps = track_steps = correct = select = ids = None
    if num_draft_tokens:
        last_steps = torch.empty((bs,), device=device, dtype=torch.int64)
        if mamba_track_interval:
            track_steps = torch.empty_like(last_steps)
        correct = torch.empty_like(accept_lens, memory_format=torch.contiguous_format)
        select = torch.empty_like(last_steps)
        ids = torch.empty((predict.numel(),), device=predict.device, dtype=torch.int64)
    if bs:
        predict_flat = predict.reshape(-1)
        predict_size = predict_flat.numel()
        rows = torch.arange(bs, device=device, dtype=torch.int64)
        accepted = accept_lens.to(torch.int64)
        length = seq_lens.to(torch.int64)
        idx = accept_index[rows, accepted - 1].to(torch.int64)
        new_seq_lens.copy_((length + accepted).to(new_seq_lens.dtype))
        idx_wrapped = torch.where(idx < 0, idx + predict_size, idx)
        bonus.copy_(predict_flat[idx_wrapped].to(bonus_dtype))
        if num_draft_tokens:
            last_steps.copy_(idx - rows * num_draft_tokens)
            correct.copy_((accepted - 1).to(correct.dtype))
            select.copy_(rows * num_draft_tokens + accepted - 1)
            # ids[row] = predict[row] for every row < predict_size (kernel copies
            # the whole predict buffer into ids, independent of bs).
            ids.copy_(predict_flat.to(torch.int64))
            if mamba_track_interval:
                post_length = length + accepted
                crossing = (length // mamba_track_interval) != (
                    post_length // mamba_track_interval
                )
                tracking_position = torch.clamp(
                    post_length // mamba_track_interval * mamba_track_interval
                    - length
                    - 1,
                    min=0,
                )
                track = (
                    accept_index[rows, tracking_position].to(torch.int64)
                    - rows * num_draft_tokens
                )
                track_steps.copy_(
                    torch.where(crossing, track, torch.full_like(track, -1))
                )
    if num_draft_tokens:
        return new_seq_lens, bonus, (last_steps, track_steps), (correct, select, ids)
    return new_seq_lens, bonus


@plugin_hook(f"{_SAMPLING}.tree_speculative_sampling_target_only", type=HookType.REPLACE)
def tree_speculative_sampling_target_only_kunlun(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
):
    """Route the non-greedy (temperature>0) EAGLE verify sampler to Kunlun.

    0.5.23 reaches this via ``sglang.kernels.ops.speculative.sampling`` which
    JIT-compiles a CUDA module (``<concepts>`` / nvcc) that will not build on
    Kunlun. The plugin already ships ``kunlun_ops.tree_speculative_sampling_
    target_only`` (wrapped in ``sgl_kernel_kunlun.sampling``); redirect to it.
    The stub binds this name only into the legacy ``sgl_kernel.sampling``
    namespace, which the 0.5.23 call path no longer imports.
    """
    from sglang_kunlun.kernels.sgl_kernel_kunlun.sampling import (
        tree_speculative_sampling_target_only as _native,
    )

    _native(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )
