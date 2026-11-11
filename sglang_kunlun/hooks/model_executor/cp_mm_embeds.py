"""Restore two multimodal processing steps bypassed on the CP (prefill context parallel) path.

The current authoritative upstream baa9befa already provides ``prepare_language_model_inputs``;
this hook prefers to delegate validation, merging and remapping to it. The manual handling and
env switches below are only for compatibility with the old interface.

With prefill-CP enabled, the runner goes through `EagerRunner._execute_extend_cp`
(`model_executor/runner/eager_runner.py:382-399`), which **bypasses
`DeepseekV4ForCausalLM.forward` entirely** -- it shards input_embeds itself and directly calls
`model.model(...)` + `model.logits_processor(...)`. `ForCausalLM.forward` does two things that
image requests require before calling `self.model.forward(...)`, so both are lost:

1. **Merge image features** (`_prepare_mm_embeddings` in `models/deepseek_v4.py:5093-5098`).
   Without it each image placeholder token only gets the `image_token_id` embedding row, and the
   ViT output is never used. Definitive evidence: the same question with two completely different
   images yields a **byte-for-byte identical** answer.
2. **Rewrite the mm placeholder rows' hash ids to `image_token_id`**
   (`models/deepseek_v4.py:5099-5107`, upstream comment: \"remap prompt image hashes for
   Engram and routing\"). Without it, the layer-1 engram (config `engram_layer_ids=[1, 14]`)
   step that withdraws the image-token memory contribution via `input_ids == image_token_id`
   (`deepseek_v4.py:4465` `torch.where`) is always false, so image tokens get an extra engram
   memory looked up by hash; `vision_topk`'s VL routing bias also fails to select image tokens.

Measured (full MMMU 900 items, `max_tokens=16384`, P cp8+DSPARK / D dp8+deepep):
fixing only item 1 gives **0.6967**; fixing both gives **0.7689** (H20 reference 0.7656, no-CP
dp8 control 0.7611). Per-layer localization: layer-0 attention output has per-row cos 0.99999,
layer-1 attention input drops to 0.712 on the image rows and output to 0.370 -- layer 1 is
exactly the engram layer.
"""

from __future__ import annotations

import contextvars
import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# Whether the batch being executed can carry image tokens, published by the
# CP-extend hook below and read by `vision_topk_text_only_kunlun` at the bottom
# of this file. Defaults to True so that any forward path which does not go
# through that hook keeps the upstream behaviour: assume images are possible and
# pay for the VL bias. Failing "open" matters -- failing closed would silently
# route image tokens with the text bias.
_BATCH_HAS_MM = contextvars.ContextVar("kunlun_dsv41_batch_has_mm", default=True)


# managers/schedule_batch.py:160: the hash value filled into mm placeholder rows is always >= this threshold
_MM_PAD_SHIFT_VALUE = 1_000_000

_warned = False
_remap_warned = False


def _has_mm_items(forward_batch) -> bool:
    """Return whether the batch contains at least one multimodal input."""
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if not mm_inputs:
        return False
    return any(item is not None for item in mm_inputs)


@plugin_hook(
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_extend_cp",
    type=HookType.AROUND,
)
def cp_extend_with_mm_embeds(original_fn, self, forward_batch, kwargs):
    """Prepare MM inputs before the CP runner bypasses the model's forward.

    Args:
        original_fn: Wrapped EagerRunner CP execution method.
        self: EagerRunner instance.
        forward_batch: Batch whose scheduler IDs must survive execution.
        kwargs: Model keyword arguments, including optional input_embeds.

    Returns:
        The wrapped runner's output; incompatible MM inputs raise ValueError.
    """
    # Publishing the flag has to wrap the whole body, and the body has several
    # early returns, so it lives in `_cp_extend_with_mm_embeds` below and this
    # function is only the set/reset frame. ContextVar rather than a module
    # global on purpose: `reset` in `finally` is exception-safe, and a value set
    # here is invisible to other threads, so a forward that somehow runs off
    # this thread reads the safe default instead of a stale False.
    token = _BATCH_HAS_MM.set(_has_mm_items(forward_batch))
    try:
        return _cp_extend_with_mm_embeds(original_fn, self, forward_batch, kwargs)
    finally:
        _BATCH_HAS_MM.reset(token)


def _cp_extend_with_mm_embeds(original_fn, self, forward_batch, kwargs):
    """Body of the hook above; see its docstring for the contract."""
    global _warned

    model = self.model_runner.model
    # baa9befa exposes this boundary explicitly. Reuse its validation and ID
    # remapping, rather than bypassing validation when embeddings were supplied.
    prepare_inputs = getattr(model, "prepare_language_model_inputs", None)
    if prepare_inputs is not None:
        ids = forward_batch.input_ids
        model_ids, embeds = prepare_inputs(
            ids, forward_batch, kwargs.get("input_embeds")
        )
        kwargs = dict(kwargs, input_embeds=embeds)
        forward_batch.input_ids = model_ids
        try:
            return original_fn(self, forward_batch, kwargs)
        finally:
            forward_batch.input_ids = ids

    if os.environ.get("DSV4_CP_MM_EMBEDS", "1") != "1":
        return original_fn(self, forward_batch, kwargs)
    if kwargs.get("input_embeds") is not None or not _has_mm_items(forward_batch):
        return original_fn(self, forward_batch, kwargs)

    prepare = getattr(model, "_prepare_mm_embeddings", None)
    if prepare is None:
        if not _warned:
            _warned = True
            logger.warning(
                "[kunlun] CP extend 收到图片请求，但 %s 没有 _prepare_mm_embeddings，"
                "图片特征仍会被丢掉",
                type(model).__name__,
            )
        return original_fn(self, forward_batch, kwargs)

    # Same order as upstream ForCausalLM.forward: first merge image features using the
    # **original** ids (mm positions are hash values), and only then rewrite the ids --
    # `embed_mm_inputs` relies on the hash values to locate the mm ranges.
    kwargs["input_embeds"] = prepare(forward_batch.input_ids, forward_batch)
    if not _warned:
        _warned = True
        logger.info(
            "[kunlun] CP extend：已为图片请求补上 mm input_embeds"
            "（上游 _execute_extend_cp 只做 token 查表，图片特征会被丢掉）"
        )
    return _with_mm_ids_remapped(original_fn, self, forward_batch, kwargs, model)


def _with_mm_ids_remapped(original_fn, self, forward_batch, kwargs, model):
    """Rewrite the mm placeholder rows' hash ids back to image_token_id, then enter the CP model body.

    Equivalent to the masked_fill at upstream `ForCausalLM.forward:5099-5107`. Only replaces
    `forward_batch.input_ids` for the duration of the call (both CP sharding and the engram
    hasher read from it), restoring it before returning to avoid disturbing the scheduler's
    hash-keyed prefix matching.

    Args:
        original_fn: Wrapped CP execution method.
        self: EagerRunner instance.
        forward_batch: Batch containing scheduler hash IDs.
        kwargs: Prepared model arguments.
        model: Model providing vision configuration and image_token_id.

    Returns:
        The wrapped runner's output, with scheduler IDs restored afterwards.
    """
    global _remap_warned

    if os.environ.get("DSV4_CP_MM_IDS", "1") != "1":
        return original_fn(self, forward_batch, kwargs)

    ids = forward_batch.input_ids
    img_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if img_id is None or getattr(model, "vision", None) is None:
        return original_fn(self, forward_batch, kwargs)
    mask = ids >= _MM_PAD_SHIFT_VALUE
    if not bool(mask.any()):
        return original_fn(self, forward_batch, kwargs)

    if not _remap_warned:
        _remap_warned = True
        logger.info(
            "[kunlun] CP extend：把 %d/%d 行 mm 哈希 id 改写成 image_token_id=%s"
            "（补上上游 ForCausalLM.forward:5103 那一步，engram 与 VL 路由都依赖它）",
            int(mask.sum().item()), int(ids.shape[0]), img_id,
        )
    forward_batch.input_ids = ids.masked_fill(mask, img_id)
    try:
        return original_fn(self, forward_batch, kwargs)
    finally:
        forward_batch.input_ids = ids


@plugin_hook(
    "sglang.srt.multimodal.dsv41.vl_routing.vision_topk",
    type=HookType.AROUND,
)
def vision_topk_text_only_kunlun(original_fn, moe, logits, input_ids, *args, **kwargs):
    """Skip the VL bias machinery on batches that cannot contain image tokens.

    `deepseek_v2.py:1004` enables VL routing from a *weight* test
    (`e_score_correction_bias_vl is not None`), which this checkpoint satisfies
    on every layer, so text-only requests take it too. Upstream already narrows
    it with a *batch* test one line later -- but only on HIP
    (`_hip_moe.batch_has_images`). This is that narrowing for Kunlun, applied
    one level down because `vl_routing.vision_topk` is upstream code we do not
    edit.

    Passing `input_ids=None` is what the downstream contract already uses to
    mean "no alternative bias needed", and it is exact rather than an
    approximation: with no row matching `image_token_id`,
    `torch.where(all_false, bias_alt, bias) == bias`, so both implementations
    produce bit-identical results on the cheap branch.

    What it saves per full 16384-token prefill chunk (21 MoE layers see the full
    chunk):
      * native gate override: `use_alt` goes False, so
        `kunlun_ops.moe_fused_gate_dsv4` runs once per layer instead of once per
        bias vector. The second run's output is selected away in full by
        `torch.where(all_false, ...)`, i.e. ~10.5ms/step of provably discarded
        work (0.503ms per [16384,384] call).
      * torch reference path, if it is ever re-enabled: the per-row bias blend
        stops materialising a [16384,384] fp32 table (25.2MB) per layer,
        ~2.75ms/step, and the following add becomes a broadcast.

    The flag is only published on the CP-extend path, which is where prefill
    goes when `--enable-prefill-cp` is set. Anything else reads the default True
    and behaves exactly as before.
    """

    if input_ids is not None and not _BATCH_HAS_MM.get():
        input_ids = None
    return original_fn(moe, logits, input_ids, *args, **kwargs)
