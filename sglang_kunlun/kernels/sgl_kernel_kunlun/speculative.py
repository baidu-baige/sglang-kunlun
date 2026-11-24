"""Kunlun-backed implementations of speculative ``sgl_kernel`` APIs."""

from __future__ import annotations

from typing import Any

import torch


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    seq_lens: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: Any = None,
) -> None:
    """Build Eagle tree metadata using Kunlun's fused implementation."""

    from kunlun_ops import build_tree_efficient

    batch_size = seq_lens.numel()
    seq_lens_sum = -1
    parent_list_for_kernel = parent_list.to(torch.long)
    seq_lens_for_kernel = seq_lens.to(torch.int32)
    retrive_index_for_kernel = retrive_index.to(torch.long)
    retrive_next_token_for_kernel = retrive_next_token.to(torch.long)
    retrive_next_sibling_for_kernel = retrive_next_sibling.to(torch.long)

    build_tree_efficient(
        parent_list_for_kernel,
        top_scores_index,
        seq_lens_for_kernel,
        tree_mask,
        positions,
        retrive_index_for_kernel,
        retrive_next_token_for_kernel,
        retrive_next_sibling_for_kernel,
        topk,
        spec_steps,
        num_verify_tokens,
        batch_size,
        seq_lens_sum,
    )

    # kunlun_ops.build_tree_efficient fills positions/retrive arrays correctly
    # but leaves non-causal draft cells set in the FULL_MASK tree_mask (every
    # draft token would attend to future siblings -> corrupted target-verify
    # attention). Rebuild the causal closure host-side, mirroring bmu's
    # _fill_tree_mask_full (kernel_ops.py). FULL_MASK == 0 in TreeMaskMode;
    # None also means FULL_MASK on this backend (is_cpu is False on Kunlun).
    mode_val = (
        None if tree_mask_mode is None
        else int(getattr(tree_mask_mode, "value", tree_mask_mode))
    )
    if mode_val in (None, 0):
        _fill_tree_mask_full(
            tree_mask,
            seq_lens,
            retrive_next_token_for_kernel,
            retrive_next_sibling_for_kernel,
            num_verify_tokens,
            batch_size,
        )


def verify_tree_greedy(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Verify an Eagle tree greedily in-place.

    Prefer the fused on-device ``kunlun_ops.verify_tree_greedy`` kernel (one
    call, no host sync); fall back to the host traversal when the kernel is
    unavailable or the DSV4_MTP_FORCE_REJECT debug knob is set.
    """

    import os

    force_reject = os.getenv("DSV4_MTP_FORCE_REJECT", "0") == "1"
    if not force_reject:
        try:
            from kunlun_ops import verify_tree_greedy as _native_verify
        except (ImportError, AttributeError):
            _native_verify = None
        if _native_verify is not None:
            batch_size = candidates.size(0)
            num_draft_tokens = candidates.size(1)
            # Kernel reads target_predict as [bs, num_draft_tokens] int64 indexed
            # by the global retrive_index; keep it 2D (host path flattens it).
            tp = target_predict
            if tp.dim() == 1:
                tp = tp.reshape(batch_size, num_draft_tokens)
            # predicts / accept_index / accept_token_num are written in place;
            # read-only inputs are coerced to the int64 layout the kernel wants.
            _native_verify(
                predicts,
                accept_index,
                accept_token_num,
                candidates.to(torch.int64),
                retrive_index.to(torch.int64),
                retrive_next_token.to(torch.int64),
                retrive_next_sibling.to(torch.int64),
                tp.to(torch.int64),
            )
            return

    _verify_tree_greedy_host(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


def _verify_tree_greedy_host(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Host-side greedy tree verification (fallback)."""

    import os

    force_reject = os.getenv("DSV4_MTP_FORCE_REJECT", "0") == "1"
    batch_size = candidates.size(0)
    num_speculative_tokens = accept_index.size(1)

    if target_predict.dim() > 1:
        target_predict = target_predict.flatten()

    trace = os.getenv("DSV4_MTP_VERIFY_TRACE", "0") == "1"
    if trace:
        import logging

        logger = logging.getLogger(__name__)

    for batch_idx in range(batch_size):
        last_accepted_retrive_idx = retrive_index[batch_idx, 0].item()
        accept_index[batch_idx, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for _ in range(1, num_speculative_tokens):
            cur_index = retrive_next_token[batch_idx, cur_index].item()

            while cur_index != -1:
                draft_index = retrive_index[batch_idx, cur_index].item()
                draft_token_id = candidates[batch_idx, cur_index].item()
                target_token_id = target_predict[last_accepted_retrive_idx].item()

                if draft_token_id == target_token_id and not force_reject:
                    predicts[last_accepted_retrive_idx] = target_token_id
                    num_accepted += 1
                    accept_index[batch_idx, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                cur_index = retrive_next_sibling[batch_idx, cur_index].item()

            if cur_index == -1:
                break

        accept_token_num[batch_idx] = num_accepted
        predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx]
        if trace:
            base = retrive_index[batch_idx, 0].item()
            span = slice(base, base + candidates.shape[1])
            logger.warning(
                "[SPECDMP-VERIFY] row=%s accepted=%s candidates=%s "
                "target_argmax=%s bonus=%s",
                batch_idx,
                num_accepted,
                candidates[batch_idx].tolist(),
                target_predict[span].tolist(),
                int(predicts[last_accepted_retrive_idx]),
            )


def _fill_tree_mask_full(tree_mask, seq_lens, retrive_next_token, retrive_next_sibling, nv, bs):
    """Write the FULL_MASK causal tree mask (prefix + ancestor closure).

    ``kunlun_ops.build_tree_efficient`` fills positions/retrive arrays correctly
    but leaves non-causal draft cells set, which makes every draft token attend
    to future siblings and degenerates generation. Ported from bmu
    (eb5_adapt/kernels/kernel_ops.py:_fill_tree_mask_full).
    """
    flat = tree_mask.view(-1)
    off = 0
    nt = retrive_next_token.tolist()
    ns = retrive_next_sibling.tolist()
    seqs = seq_lens.tolist()
    for b in range(bs):
        sl = int(seqs[b])
        cols = sl + nv
        block = flat[off : off + nv * cols].view(nv, cols)
        block[:, :sl] = True
        block[:, sl:] = False
        parent = [-1] * nv
        for i in range(nv):
            c = nt[b][i]
            while c != -1:
                parent[c] = i
                c = ns[b][c]
        for q in range(nv):
            j = q
            while j != -1:
                block[q, sl + j] = True
                j = parent[j]
        off += nv * cols
