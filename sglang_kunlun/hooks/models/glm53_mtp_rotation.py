"""Load QuaRot-R1 rotated checkpoints (vllm `vllm-glm53-w4a8-changes.patch`).

A rotated checkpoint folds ``diag(gamma_main) @ Q.T`` into the shared lm_head,
so the hidden states that reach that head live in the rotated basis. The MTP
branch (``model.layers.78``) has its own final-norm gain ``gamma_mtp``, so the
checkpoint also carries two block-diagonal corrections

    final_correction.blocks   = Q @ diag(gamma_mtp / gamma_main) @ Q.T   (logits)
    recycle_correction.blocks = same, recycled-hidden path only

Without them upstream loads the rotated lm_head but computes MTP logits from an
uncorrected norm gain -- measured 2026-11-14: accept rate collapsed to 2.2 % and
only 30/180 requests completed on `GLM-5.3-W4A8-INT8-Dynamic`.

sglang's `DeepseekModelNextN.shared_head` is a bare ``nn.Module`` with only
``norm`` and ``head``, so this module adds the two buffers, absorbs their
weights from ``load_weights``, and applies them at the two distinct sites:
    * logits path  -> wrap ``shared_head.head.forward`` (input side)
    * recycle path -> on the target hidden consumed by the MTP model body
"""

import logging

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_FINAL = "final_correction"
_RECYCLE = "recycle_correction"
_SUFFIXES = {
    "shared_head.final_correction.blocks": _FINAL,
    "shared_head.recycle_correction.blocks": _RECYCLE,
}


def _install_corrections(model, blocks: dict) -> None:
    """Attach the corrections and wrap the head's input side."""
    head_mod = model.shared_head
    for name, tensor in blocks.items():
        ref = head_mod.norm.weight
        if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
            raise ValueError(
                f"{name}: blocks must be [num_blocks, block, block], got "
                f"{tuple(tensor.shape)}"
            )
        covered = tensor.shape[0] * tensor.shape[1]
        if covered != ref.shape[0]:
            raise ValueError(
                f"{name}: covers {covered} channels but hidden_size is "
                f"{ref.shape[0]}"
            )
        head_mod.register_buffer(name, tensor.to(ref.device, ref.dtype), persistent=False)
        logger.info(
            "sglang-kunlun: %s loaded, %d blocks of %d",
            name,
            tensor.shape[0],
            tensor.shape[1],
        )

    final = getattr(head_mod, _FINAL, None)
    if final is None:
        return

    head = getattr(head_mod, "head", None)
    if head is None or getattr(head.forward, "_kunlun_rotated", False):
        return
    original = head.forward

    def forward(x):
        corr = getattr(head_mod, _FINAL, None)
        if corr is not None and x.shape[-1] == corr.shape[1] * corr.shape[0]:
            # blocks: [n, b, b] -> apply per block on the last axis
            y = x.reshape(*x.shape[:-1], corr.shape[0], corr.shape[1])
            x = torch.einsum("...ni,nij->...nj", y, corr).reshape(*x.shape)
        return original(x)

    forward._kunlun_rotated = True
    head.forward = forward


@plugin_hook(
    "sglang.srt.models.deepseek_nextn.DeepseekV3ForCausalLMNextN.load_weights",
    type=HookType.AROUND,
)
def nextn_load_weights_rotation(original, self, weights, *args, **kwargs):
    """Strip the QuaRot-R1 corrections out of the stream, then apply them."""
    pending = {}
    rest = []
    for name, tensor in weights:
        for suffix, key in _SUFFIXES.items():
            if name.endswith(suffix):
                pending[key] = tensor
                break
        else:
            rest.append((name, tensor))
    result = original(self, iter(rest), *args, **kwargs)
    if pending:
        _install_corrections(self.model, pending)
    return result


def _apply_blocks(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Apply a QuaRot-R1 block-diagonal factor on the last axis."""
    n, b = blocks.shape[0], blocks.shape[1]
    if x.shape[-1] != n * b:
        return x
    y = x.reshape(*x.shape[:-1], n, b)
    return torch.einsum("...ni,nij->...nj", y, blocks).reshape(*x.shape)


@plugin_hook(
    "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.forward",
    type=HookType.AROUND,
)
def nextn_forward_recycle(original, self, *args, **kwargs):
    """Restore the MTP gain on the recycled hidden before it enters hnorm.

    Consumer-side on purpose: `final_correction` already runs at the shared
    head's input, and the same tensor is handed to the next draft step, so
    correcting the producer would apply both factors to the logits path.
    """
    # The CP runner calls the model body directly, bypassing CausalLM.forward.
    head_mod = getattr(self, "shared_head", None)
    blocks = getattr(head_mod, _RECYCLE, None) if head_mod is not None else None
    if blocks is not None:
        fb = kwargs.get("forward_batch")
        if fb is None and len(args) >= 3:
            fb = args[2]
        spec = getattr(fb, "spec_info", None) if fb is not None else None
        hidden = getattr(spec, "hidden_states", None) if spec is not None else None
        if isinstance(hidden, torch.Tensor) and hidden.numel():
            spec.hidden_states = _apply_blocks(hidden, blocks)
            try:
                return original(self, *args, **kwargs)
            finally:
                # Do not accumulate the correction on a reused spec input.
                # Under CP this restores the local view; the runner then
                # restores the full-sequence target states.
                spec.hidden_states = hidden
    return original(self, *args, **kwargs)
