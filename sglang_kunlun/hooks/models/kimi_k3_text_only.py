"""Kimi-K3 tokenizer and multimodal policy hooks for Kunlun.
"""

from __future__ import annotations

import hashlib

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _kunlun_cpu_tensor_hash(tensor_list) -> int:
    """Match SGLang's CPU tensor hash without entering CUDA/Triton."""
    if isinstance(tensor_list, list):
        tensors = []
        for value in tensor_list:
            if isinstance(value, list):
                tensors.extend(value)
            else:
                tensors.append(value)
        hasher = hashlib.sha256()
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(tensor)!r}")
            tensor = tensor.detach().cpu().contiguous()
            hasher.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    else:
        tensor = tensor_list.detach().cpu().contiguous()
        hasher = hashlib.sha256()
        hasher.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)


@plugin_hook(
    "sglang.srt.managers.mm_utils.tensor_hash",
    type=HookType.REPLACE,
)
def kunlun_tensor_hash(tensor_list) -> int:
    """Hash Kunlun tensors on CPU; avoid CUDA IPC and NVIDIA Triton.

    Target unchanged in 0.5.19: ``srt/managers/mm_utils.py:824``.
    """
    return _kunlun_cpu_tensor_hash(tensor_list)


@plugin_hook(
    "sglang.srt.arg_groups.serving_hook.handle_multimodal_feature_transport",
    type=HookType.AROUND,
)
def handle_kunlun_multimodal_feature_transport(original_fn, server_args):
    """Resolve multimodal features to host transport before args are published."""
    from sglang.srt.arg_groups.overrides import declare_resolution

    declare_resolution(
        server_args,
        "kunlun_multimodal_feature_transport",
        mm_feature_transport="cpu",
        keep_mm_feature_on_device=False,
    )
    return original_fn(server_args)


@plugin_hook(
    "sglang.srt.multimodal.transport.determine_tensor_transport_mode",
    type=HookType.REPLACE,
)
def determine_kunlun_tensor_transport_mode(*args, **kwargs) -> str:
    """Use host-backed tensor transport because Kunlun has no CUDA storage IPC."""
    return "default"