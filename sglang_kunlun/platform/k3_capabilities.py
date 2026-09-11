"""Disable NVIDIA-specific K3 fast paths on Kunlun."""

from __future__ import annotations

import os


UNSUPPORTED_K3_FEATURES = (
    "SGLANG_K3_AR_FUSION",
    "SGLANG_K3_SP_COLLECTIVE",
    "SGLANG_K3_SP_ATTN_RES",
    "SGLANG_K3_GEMM_AR",
    "SGLANG_K3_FUSED_FRONT",
    # New in 0.5.19 (``srt/environ.py:1576``, default False). Listed here so an
    # explicit opt-in from the environment cannot enable the radix-4 top-k fast
    # path, which has no Kunlun equivalent.
    "SGLANG_K3_RADIX4_TOPK",
)


def disable_unsupported_k3_features() -> None:
    """Force K3 onto separable torch/XCCL paths before SGLang reads flags."""
    for name in UNSUPPORTED_K3_FEATURES:
        os.environ[name] = "0"
