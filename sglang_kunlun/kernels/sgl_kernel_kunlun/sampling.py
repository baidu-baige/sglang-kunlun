"""Kunlun-backed implementations of sgl_kernel sampling APIs.

Functions
---------
top_k_renorm_prob
    Wraps ``kunlun_ops.top_k_renorm_probs`` (note: plural 's' in the op name).
top_p_renorm_prob
    Wraps ``kunlun_ops.top_p_renorm_probs``.
tree_speculative_sampling_target_only
    Wraps ``kunlun_ops.tree_speculative_sampling_target_only``, adapting the
    keyword-argument call style used by ``eagle_info_v2.py:sample()``.
"""

import torch
import kunlun_ops


def top_k_renorm_prob(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """Renormalize probabilities by top-k thresholding (Kunlun).

    Mirrors the sgl_kernel CUDA API:
        top_k_renorm_prob(probs, top_ks) -> renorm_probs

    Parameters
    ----------
    probs:
        Float tensor of shape ``(batch_size, vocab_size)``.
    top_k:
        Int tensor of shape ``(batch_size,)`` with per-request top-k values.
        Values <= 0 are treated as "no top-k filtering" by the XPU kernel.

    Returns
    -------
    torch.Tensor
        Renormalized probabilities, same shape as ``probs``.
    """
    return kunlun_ops.top_k_renorm_probs(probs, top_k)


def top_p_renorm_prob(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Renormalize probabilities by top-p (nucleus) thresholding (Kunlun).

    Mirrors the sgl_kernel CUDA API:
        top_p_renorm_prob(probs, top_ps) -> renorm_probs

    Parameters
    ----------
    probs:
        Float tensor of shape ``(batch_size, vocab_size)``.
    top_p:
        Float tensor of shape ``(batch_size,)`` with per-request top-p values.

    Returns
    -------
    torch.Tensor
        Renormalized probabilities, same shape as ``probs``.
    """
    return kunlun_ops.top_p_renorm_probs(probs, top_p)


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 0.3,
    threshold_acc: float = 0.3,
    deterministic: bool = True,  # not used by XPU kernel
) -> None:
    """Tree speculative sampling (target-only) for Kunlun.

    Adapts the keyword-argument call style used by ``eagle_info_v2.py:sample()``
    to the positional ``kunlun_ops.tree_speculative_sampling_target_only`` API.

    Based on ``speculative_klx.py:tree_speculative_sampling_target_only_kunlun``
    from the mimo reference branch.
    """
    batch_size = candidates.size(0)
    num_spec_step = accept_index.size(1)
    num_draft_tokens = retrive_next_token.size(1)
    vocab_size = target_probs.size(-1)

    # NOTE: In the mimo reference (speculative_klx.py), predicts has shape
    # [bs * num_draft_tokens + 1] so [:-1] removes the bonus-token slot.
    # In aiak_sglang , predict is allocated as [bs * num_draft_tokens]
    # (no extra slot), so we view directly without slicing.
    kunlun_ops.tree_speculative_sampling_target_only(
        candidates.to(torch.int32),
        retrive_index.to(torch.int32),
        retrive_next_token.to(torch.int32),
        retrive_next_sibling.to(torch.int32),
        uniform_samples,
        target_probs,
        batch_size,
        num_spec_step,
        num_draft_tokens,
        vocab_size,
        threshold_single,
        threshold_acc,
        predicts.view(batch_size, num_draft_tokens),
        accept_index,
        accept_token_num,
        draft_probs,
    )
