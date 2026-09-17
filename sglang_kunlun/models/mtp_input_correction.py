"""MTP input correction for the NextN/draft model."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Tuple

import torch

logger = logging.getLogger(__name__)

BLOCKS_SUFFIX = "mtp_input_correction.blocks"

#: Attribute a loader stashes the dense right-multiply matrix on.
MODEL_ATTR = "_kunlun_mtp_input_correction"


def mtp_input_correction_enabled() -> bool:
    """Whether the checkpoint's MTP input correction is applied (default on)."""
    return os.environ.get("SGLANG_KUNLUN_GLM_MTP_INPUT_CORRECTION", "1") not in (
        "0",
        "false",
        "False",
        "",
    )


def apply_block_correction(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """``x @ block_diag(blocks)`` (fp32 math, back to ``x``'s dtype).

    Kept in sync with the DSpark implementation; imported from there when
    available so both drafts share one definition.
    """
    return _dspark_apply(x, blocks)


def _dspark_apply(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    try:
        from sglang_kunlun.hooks.speculative.dspark_precision import (
            apply_block_correction as _impl,
        )

        return _impl(x, blocks)
    except Exception:
        n_blocks, block, _ = blocks.shape
        orig_dtype = x.dtype
        lead = x.shape[:-1]
        chunks = x.reshape(-1, n_blocks, block).to(torch.float32)
        out = torch.einsum("tnb,nbc->tnc", chunks, blocks.to(torch.float32))
        return out.reshape(*lead, n_blocks * block).to(orig_dtype)


def block_diag_matrix(blocks: torch.Tensor, dtype, device) -> torch.Tensor:
    """``[n_blocks, b, b]`` -> dense ``[n_blocks*b, n_blocks*b]`` right-multiply matrix."""
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError(
            f"{BLOCKS_SUFFIX} must be [n_blocks, b, b], got {tuple(blocks.shape)}"
        )
    n_blocks, block, _ = blocks.shape
    size = n_blocks * block
    full = torch.zeros((size, size), dtype=torch.float32, device=blocks.device)
    for index in range(n_blocks):
        start = index * block
        full[start : start + block, start : start + block] = blocks[index].to(
            torch.float32
        )
    return full.to(dtype=dtype, device=device)


def _forward_reads_rot_weight(module: torch.nn.Module) -> bool:
    """Whether the forward that would run actually consumes ``rot_weight``.

    The attribute existing is not enough: this fork's DSV4 NextN defines the
    slot but never reads it, so filling it would silently do nothing.
    """
    try:
        import inspect

        owner = type(module)
        forward = getattr(owner, "forward", None)
        if forward is not None and "rot_weight" in inspect.getsource(forward):
            return True
        inner = getattr(owner, "model", None)
        inner_forward = getattr(inner, "forward", None) if inner is not None else None
        return inner_forward is not None and "rot_weight" in inspect.getsource(
            inner_forward
        )
    except Exception:  # pragma: no cover - introspection only
        return False


def attach_correction(model: torch.nn.Module, blocks: torch.Tensor) -> None:
    """Stash the correction on the model; fill ``rot_weight`` when the slot exists."""
    inner = getattr(model, "model", None)
    reference = getattr(getattr(inner, "hnorm", None), "weight", None)
    if reference is None:
        reference = getattr(getattr(inner, "enorm", None), "weight", None)
    dtype = reference.dtype if reference is not None else torch.bfloat16
    device = reference.device if reference is not None else torch.device("cuda")

    size = int(blocks.shape[0]) * int(blocks.shape[1])
    if reference is not None and int(reference.shape[0]) != size:
        raise ValueError(
            "MTP input correction does not cover the hidden size: "
            f"{tuple(blocks.shape)} -> {size}, but the draft norm is "
            f"{tuple(reference.shape)}"
        )

    dense = block_diag_matrix(blocks, dtype=dtype, device=device)

    # Some drafts (DeepSeek-V3 style NextN) already consume a ``rot_weight`` slot
    # in their own forward and apply it themselves; others (this fork's DSV4
    # NextN) carry the attribute but never read it, in which case the worker
    # applies the correction at the draft-extend boundary. Decide from the
    # forward that would actually run, so it is never applied twice.
    holder = inner if inner is not None and hasattr(inner, "rot_weight") else None
    if holder is not None and _forward_reads_rot_weight(holder):
        holder.rot_weight = dense
        logger.info(
            "sglang-kunlun: MTP input correction %s -> rot_weight %s (%s)",
            tuple(blocks.shape),
            tuple(dense.shape),
            dtype,
        )
        return

    setattr(model, MODEL_ATTR, dense)
    logger.info(
        "sglang-kunlun: MTP input correction %s -> %s (%s), applied at the draft "
        "extend boundary",
        tuple(blocks.shape),
        tuple(dense.shape),
        dtype,
    )


def corrected_hidden(model: torch.nn.Module, hidden: torch.Tensor):
    """``hidden`` right-multiplied by the attached correction (no-op when absent).

    Pure: returns a new tensor (or ``hidden`` itself when there is nothing to
    do). Callers must call this exactly once per draft iteration — the target
    hidden it is applied to is per-batch, but applying it twice would square the
    correction.
    """
    if hidden is None or hidden.numel() == 0:
        return hidden
    correction = getattr(model, MODEL_ATTR, None)
    if correction is None:
        return hidden
    return torch.matmul(hidden.to(correction.dtype), correction).to(hidden.dtype)


def load_blocks_from_checkpoint(model_path) -> "torch.Tensor | None":
    """Read ``*.mtp_input_correction.blocks`` straight from a sharded checkpoint.

    Fallback for loaders that never surface the tensor (or when the loader hook
    is not in the active code path): the tensor is small (``3*2048*2048`` bf16 =
    25 MB), so pulling it out of the index by name is cheap and immune to weight
    stream plumbing.
    """
    if not model_path:
        return None
    import json
    import os

    index_path = os.path.join(str(model_path), "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        weight_map = json.load(open(index_path))["weight_map"]
        key = next((name for name in weight_map if name.endswith(BLOCKS_SUFFIX)), None)
        if key is None:
            return None
        from safetensors import safe_open

        with safe_open(
            os.path.join(str(model_path), weight_map[key]), framework="pt"
        ) as handle:
            return handle.get_tensor(key)
    except Exception as exc:  # pragma: no cover - checkpoint layout drift
        logger.warning("sglang-kunlun: could not read %s: %s", BLOCKS_SUFFIX, exc)
        return None


def strip_correction_blocks(
    model: torch.nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Pull the correction tensor out of the stream and attach it to ``model``."""
    if not mtp_input_correction_enabled():
        yield from weights
        return

    captured: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if BLOCKS_SUFFIX in name:
            captured.append((name, tensor))
            continue
        yield name, tensor

    if captured:
        for name, tensor in captured:
            logger.info("sglang-kunlun: captured %s for the MTP input correction", name)
        attach_correction(model, captured[-1][1])
