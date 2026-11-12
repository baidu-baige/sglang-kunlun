"""Kunlun-backed implementations of speculative ``sgl_kernel`` APIs."""

from __future__ import annotations

from typing import Any

import torch

_verify_fallback_warned = False


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

    The baseline 0.5.8 ran this as the fused vendor kernel: its
    ``sgl_kernel.verify_tree_greedy`` is a two-line wrapper around
    ``kunlun_ops.verify_tree_greedy`` (``sgl-kernel/python/sgl_kernel/
    speculative.py:47-70``), which is what shows up in the decode trace as
    ``xpuops3::infer_verify_tree_greedy``. The torch loop below therefore cannot
    be the primary path -- it is both a missing operator against the trace
    baseline and, with three ``.item()`` calls per tree node, a device sync per
    candidate. It is kept as the fallback (and as the reference for
    ``DSV4_MTP_FORCE_REJECT``), which is also how it was validated against the
    kernel in the first place.
    """

    import os

    force_reject = os.getenv("DSV4_MTP_FORCE_REJECT", "0") == "1"
    batch_size = candidates.size(0)
    num_speculative_tokens = accept_index.size(1)

    if target_predict.dim() > 1:
        target_predict = target_predict.flatten()

    if not force_reject:
        try:
            from kunlun_ops import verify_tree_greedy as _verify_tree_greedy

            # Same argument order and dtypes as 0.5.8: the index tensors are
            # already int64 (``eagle_info.py:72-79``) and the mutable outputs are
            # int32. The kernel wants ``target_predict`` 2-D --
            # ``_xflashinfer_sampling.py:1390`` raises "Tensor target_predict has
            # wrong dim! Expected 2" otherwise -- while the reference loop needs
            # it flat, so the two get different views of the same storage.
            _verify_tree_greedy(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                target_predict.reshape(batch_size, -1),
            )
            return
        except Exception as exc:  # noqa: BLE001 - fall back, never crash verify
            global _verify_fallback_warned
            if not _verify_fallback_warned:
                _verify_fallback_warned = True
                import logging

                logging.getLogger(__name__).warning(
                    "kunlun_ops.verify_tree_greedy unavailable (%s); using the "
                    "torch loop -- the decode trace will lack "
                    "infer_verify_tree_greedy and decode will be sync-bound",
                    exc,
                )

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
