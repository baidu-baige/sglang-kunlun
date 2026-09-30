"""Kunlun implementations of sgl_kernel top-k operations.

``fast_topk`` and ``fast_topk_v2`` are pure-torch fallbacks (no XPU-specific
kernel needed). ``fast_topk_transform_fused`` and
``fast_topk_transform_ragged_fused`` still rely on CUDA kernels
(``torch.ops.sgl_kernel.*``) that only exist in the CUDA sgl_kernel build; they
are left as stubs in ``bootstrap/sgl_kernel_stub.py`` and raise
``NotImplementedError`` if reached.
"""

import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    else:
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch fallback for ``sgl_kernel.fast_topk_v2``.

    Per row ``i``, select the ``topk`` highest scores inside the window
    ``[row_starts[i], row_starts[i] + lengths[i])`` (``row_starts=None`` means
    the window starts at 0) and return their **window-local** column indices.
    Rows with fewer than ``topk`` valid entries are padded with ``-1``.

    Matches the reference in ``sglang/kernels/aot/tests/test_topk.py``
    (``_ref_torch_impl``), which slices each row's window before calling
    ``torch.topk``. The CUDA kernel fills its index buffer in arrival order, so
    the output agrees as a set but not element-wise; the upstream test compares
    sorted indices for the same reason.

    ``lengths`` is per-row here (the reference test uses one scalar length for
    the whole batch), so validity is derived from the ``-inf`` fill rather than
    from the column rank -- ``sorted=False`` gives no usable rank.
    """

    assert score.dim() == 2, f"expected 2D score, got {tuple(score.shape)}"
    rows, cols = score.shape
    device = score.device
    if rows == 0:
        return torch.full((rows, topk), -1, dtype=torch.int32, device=device)

    lens = lengths.to(torch.int64)
    col = torch.arange(cols, device=device)
    if row_starts is None:
        base = None
        in_window = col[None, :] < lens[:, None]
    else:
        base = row_starts.to(torch.int64)[:, None]
        in_window = (col[None, :] >= base) & (col[None, :] < base + lens[:, None])

    masked = torch.where(in_window, score.float(), torch.full_like(score, float("-inf"), dtype=torch.float32))

    k = min(topk, cols)
    values, indices = torch.topk(masked, k, dim=1, sorted=False)
    if base is not None:
        indices = indices - base
    # Masked-out columns are exactly -inf, so a finite score means "real pick".
    out = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    out = out.to(torch.int32)
    if k < topk:
        out = torch.cat(
            [out, torch.full((rows, topk - k), -1, dtype=torch.int32, device=device)],
            dim=1,
        )
    return out


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    """Kunlun implementation of ``sgl_kernel.moe_fused_gate``."""

    import kunlun_ops

    if num_fused_shared_experts != 0:
        raise NotImplementedError("Kunlun moe_fused_gate does not support fused shared experts")

    num_tokens, num_experts = input_tensor.shape
    block_statistic = torch.empty(
        12, num_experts + 1, dtype=torch.int32, device=input_tensor.device
    )
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=input_tensor.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=input_tensor.device
    )
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=input_tensor,
        topk_index=topk_ids,
        norm_score=topk_weights,
        block_statistic=block_statistic,
        bias=bias.float(),
        scale=(routed_scaling_factor if apply_routed_scaling_factor_on_output else 1.0),
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_weights, topk_ids
