# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal (image) prefill under prefill context parallelism (DSV4.1-Flash).

The CP extend path runs on ``EagerRunner._execute_extend_cp``, which calls
``model.model(...)`` (the decoder body) directly and therefore bypasses
``DeepseekV4ForCausalLM.forward`` -- the only place that

1. injects the image spans (``_prepare_mm_embeddings`` -> ``embed_mm_inputs`` ->
   ``get_image_feature`` -> ViT + Aligner), and
2. remaps the scheduler's image placeholders ``MM_PAD_SHIFT_VALUE + hash`` back
   to ``image_token_id``.

Without this hook a VL request under CP embeds every image row as a plain
``image_token_id`` lookup, hands ids ``>= 1_000_000`` to ``HashTopK`` (an
out-of-bounds gather) and disables ``vision_topk``'s image bias. The result is
silently wrong, not a crash.

The fix is the two steps in front of the existing CP boundary:

1. build the full-chunk embeddings with ``model._prepare_mm_embeddings`` (the
   ViT runs here; ``forward_batch.mm_input_embeds`` is set exactly as the
   non-CP path does, for the draft worker);
2. temporarily swap ``forward_batch.input_ids`` for the remapped copy, so the
   body's sharded ids, ``forward_batch.input_ids_global`` (``HashTopK`` /
   ``vision_topk``) and the logits processor all see ``image_token_id``.

The Engram hasher is fed the same tensor. Its kernel and torch fallback both
mask ``>= mm_pad_shift`` to ``image_token_id`` before hashing
(``srt/layers/engram.py``), so the already-remapped ids produce the identical
hash input; the raw ids are restored in ``finally`` before the runner returns.

Text-only batches never enter this hook's body (``_mm_active`` is false) and the
path stays byte-identical to the engine's.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_SHAPE_CHECKED = False
_CHECKED_ONCE: set[tuple] = set()


def _debug_enabled() -> bool:
    return os.environ.get("KL_MM_CP_DEBUG", "0") == "1"


def _check_mode() -> str:
    return os.environ.get("KL_MM_CP_CHECK", "once").lower()


def _rank_and_size() -> tuple[int, int]:
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        return int(parallel.attn_cp_rank), int(parallel.attn_cp_size)
    except Exception:  # pragma: no cover - diagnostics must never break a run
        return int(torch.cuda.current_device()), 1


def _mm_pad_shift() -> int:
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    return MM_PAD_SHIFT_VALUE


def _mm_active(model, forward_batch) -> bool:
    """True only when this forward carries image inputs we must inject first."""
    # An EPD language-only prefill owns no ViT but still receives image rows:
    # the encoder workers hand over precomputed embeddings, so the merge this
    # hook performs is exactly what such a prefill needs.
    if getattr(model, "vision", None) is None and not getattr(
        model, "mm_embedding_only", False
    ):
        return False
    mode = getattr(forward_batch, "forward_mode", None)
    is_decode_or_idle = getattr(mode, "is_decode_or_idle", None)
    if callable(is_decode_or_idle) and is_decode_or_idle():
        return False
    is_target_verify = getattr(mode, "is_target_verify", None)
    if callable(is_target_verify) and is_target_verify():
        return False
    contains_mm = getattr(forward_batch, "contains_mm_inputs", None)
    if callable(contains_mm):
        if not contains_mm():
            return False
    else:
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if not mm_inputs or not any(item is not None for item in mm_inputs):
            return False
    if getattr(forward_batch, "extend_seq_lens_cpu", None) is None:
        return False
    spec = getattr(forward_batch, "spec_algorithm", None)
    is_eagle = getattr(spec, "is_eagle", None)
    if callable(is_eagle) and is_eagle():
        # The EAGLE draft-extend runs outside CP and deepseek_v4_nextn ignores
        # forward_batch.mm_input_embeds, so image rows would be embedded from
        # placeholder ids. Refuse rather than answer wrongly.
        raise RuntimeError(
            "[mm-cp] multimodal prefill under CP is not supported with EAGLE; "
            "use DSpark or disable speculative decoding"
        )
    return True


def _require_engine_shape(original_fn) -> None:
    """Fail closed when the engine's CP boundary changed shape.

    The hook relies on the stock ``_execute_extend_cp`` honouring a supplied
    ``input_embeds`` and passing ``forward_batch.input_ids`` as
    ``complete_input_ids``. If a future image rewords/reorders that function the
    markers disappear; raising here is better than silently running the old
    assumption against a changed engine.
    """
    global _SHAPE_CHECKED
    if _SHAPE_CHECKED:
        return
    try:
        src = inspect.getsource(original_fn)
    except (OSError, TypeError):
        return
    for marker in (
        "cp_shard_model_inputs",
        "cp_gather_after_forward",
        "logits_processor(",
    ):
        if marker not in src:
            raise RuntimeError(
                f"[mm-cp] EagerRunner._execute_extend_cp no longer matches the "
                f"expected shape ({marker!r} missing); refusing to run "
                f"multimodal prefill under CP"
            )
    _SHAPE_CHECKED = True


def _record(stage: str, payload: dict) -> None:
    if not _debug_enabled():
        return
    directory = os.environ.get("KL_MM_CP_TRACE_DIR")
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        rank, _ = _rank_and_size()
        path = os.path.join(directory, f"mm_cp_rank{rank}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, **payload}) + "\n")
    except Exception:  # pragma: no cover - diagnostics must never break a run
        logger.exception("[mm-cp] failed to append shard evidence")


@plugin_hook(
    target=(
        "sglang.srt.model_executor.runner.eager_runner."
        "EagerRunner._execute_extend_cp"
    ),
    type=HookType.AROUND,
)
def execute_extend_cp_mm_kunlun(original_fn, self, forward_batch, kwargs):
    """Inject image spans + remap placeholder ids before the CP shard."""
    model = self.model_runner.model
    if not _mm_active(model, forward_batch):
        return original_fn(self, forward_batch, kwargs)

    if kwargs.get("input_embeds") is not None:
        raise RuntimeError(
            "[mm-cp] cannot combine input_embeds with image inputs"
        )
    _require_engine_shape(original_fn)

    raw = forward_batch.input_ids
    image_token_id = int(model.config.image_token_id)
    shift = _mm_pad_shift()

    embeds = model._prepare_mm_embeddings(raw, forward_batch)

    remapped = raw.masked_fill(raw >= shift, image_token_id)
    if _check_mode() != "off":
        key = ("full", torch.cuda.current_device())
        if _check_mode() == "always" or key not in _CHECKED_ONCE:
            _CHECKED_ONCE.add(key)
            if int(remapped.max()) >= int(model.config.vocab_size):
                raise RuntimeError(
                    "[mm-cp] image placeholder id leaked past the remap "
                    f"(max={int(remapped.max())}, vocab={int(model.config.vocab_size)})"
                )

    if _debug_enabled():
        rank, cp_size = _rank_and_size()
        total_image = int((remapped == image_token_id).sum())
        forward_batch.mm_cp_total_image = total_image
        forward_batch.mm_cp_image_token_id = image_token_id
        forward_batch.mm_cp_rows_batch = int(raw.numel())
        try:
            image_rows = raw >= shift
            image_mean = (float(embeds[image_rows].abs().mean())
                          if bool(image_rows.any()) else float("nan"))
            text_mean = float(embeds[~image_rows][:128].abs().mean())
            id_mean = float(
                model.get_input_embeddings()(
                    torch.tensor([image_token_id], device=embeds.device)
                ).abs().mean()
            )
            logger.info(
                "[mm-cp] embeds rank=%d img_rows=%d img_mean=%.5f text_mean=%.5f "
                "id_embed_mean=%.5f",
                rank, total_image, image_mean, text_mean, id_mean,
            )
        except Exception:  # noqa: BLE001 - diagnostics only
            logger.exception("[mm-cp] embed diagnostic failed")
        logger.info(
            "[mm-cp] rank=%d cp_size=%d rows_batch=%d total_image=%d",
            rank,
            cp_size,
            int(raw.numel()),
            total_image,
        )

    forward_batch.input_ids = remapped
    try:
        call_kwargs = dict(kwargs)
        call_kwargs["input_embeds"] = embeds
        return original_fn(self, forward_batch, call_kwargs)
    finally:
        forward_batch.input_ids = raw
        for attr in (
            "mm_cp_total_image",
            "mm_cp_image_token_id",
            "mm_cp_rows_batch",
        ):
            if hasattr(forward_batch, attr):
                delattr(forward_batch, attr)


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4Model.forward",
    type=HookType.BEFORE,
)
def record_body_mm_shard_kunlun(*args, **kwargs):
    """Record the image-token rows each CP rank's decoder body actually runs.

    This is the authoritative shard evidence: ``args[1]`` is the body's
    ``input_ids`` (the rank's interleave shard, zero padded), so summing the
    per-rank counts must equal the request's image span and every rank must own
    ``floor``/``ceil`` of it. Env gated; a no-op in production.
    """
    if not _debug_enabled() or len(args) < 2:
        return None
    model, input_ids = args[0], args[1]
    if not isinstance(input_ids, torch.Tensor):
        return None
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if image_token_id is None:
        return None
    forward_batch = args[3] if len(args) > 3 else kwargs.get("forward_batch")
    if forward_batch is None or not hasattr(forward_batch, "mm_cp_total_image"):
        return None

    if _check_mode() != "off":
        device = torch.cuda.current_device()
        key = ("body", device)
        if _check_mode() == "always" or key not in _CHECKED_ONCE:
            _CHECKED_ONCE.add(key)
            vocab = int(getattr(model.config, "vocab_size", 0) or 0)
            if vocab and int(input_ids.max()) >= vocab:
                raise RuntimeError(
                    "[mm-cp] image placeholder id reached the decoder body "
                    f"(max={int(input_ids.max())}, vocab={vocab})"
                )

    rank, cp_size = _rank_and_size()
    local_image = int((input_ids == int(image_token_id)).sum())
    total_image = int(forward_batch.mm_cp_total_image)
    embeds = args[4] if len(args) > 4 else kwargs.get("input_embeds")
    if isinstance(embeds, torch.Tensor) and embeds.shape[0] == input_ids.shape[0]:
        try:
            mask = input_ids == int(image_token_id)
            image_mean = (float(embeds[mask].abs().mean())
                          if bool(mask.any()) else float("nan"))
            text_mean = float(embeds[~mask][:64].abs().mean())
            logger.info(
                "[mm-cp] body embeds rank=%d local_image=%d img_mean=%.5f text_mean=%.5f",
                rank, local_image, image_mean, text_mean,
            )
        except Exception:  # noqa: BLE001 - diagnostics only
            logger.exception("[mm-cp] body embed diagnostic failed")
    logger.info(
        "[mm-cp] body rank=%d cp_size=%d local_image=%d total_image=%d rows_local=%d",
        rank,
        cp_size,
        local_image,
        total_image,
        int(input_ids.numel()),
    )
    _record(
        "body",
        {
            "ts": time.time(),
            "rank": rank,
            "cp_size": cp_size,
            "local_image": local_image,
            "total_image": total_image,
            "rows_local": int(input_ids.numel()),
            "rows_batch": int(forward_batch.mm_cp_rows_batch),
        },
    )
    return None
