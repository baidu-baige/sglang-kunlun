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
"""Out-of-graph precompute for the DeepSeek-V4.1 engram host table.

``SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`` keeps the ~198 GB n-gram table in
host memory (~23 GiB/GPU of HBM saved) but turns the lookup into a CPU gather
plus an H2D copy.  A ``full`` decode CUDA graph only replays device work, so
the gather runs once at capture and every replay reads the capture-time bytes:
the engram contribution is frozen.  DSPARK's block draft amplifies that into
the repetitive gibberish documented in the 2026-11-02 ku doc "decode 开 CUDA
Graph 输出乱码".

This module keeps the HBM saving *and* the ``full`` graph backend:

* right before every replay the decode graph runner recomputes ``hash_ids`` and
  the host-table gather for the captured static batch (eager, outside the
  graph) and writes ``emb`` into a persistent device buffer;
* the captured ``Engram`` graph only reads that buffer (a plain view), so each
  replay consumes fresh rows;
* the in-graph hasher is disabled in the same window, so history is committed
  exactly once, by the eager precompute.

The buffers are attached to the ``EngramEmbedding`` modules and must never be
reallocated after capture: the graphs hold their raw addresses.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ENV = "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"
# Capture-time contents are discarded, but an all-zero page feeds a per-token
# int8 quantiser a zero scale.  A small non-zero fill keeps the capture pass
# numerically boring without pretending to be a real lookup.
_CAPTURE_FILL = 0.02


def enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "y")


class _State:
    model = None
    hasher = None
    layers: tuple = ()
    max_tokens = 0
    # [max_tokens, n_engram_layers, n_hash_cols] int64.  Only its shape and
    # address are used: the captured graph slices it and the embed hook ignores
    # the values, so no hasher kernel ends up in the graph.
    dummy_ids: Optional[torch.Tensor] = None
    dummy_tokens: Optional[torch.Tensor] = None
    logged = False


_state = _State()
_tls = threading.local()


# ---------------------------------------------------------------------------
# Scope: true only inside graph capture / replay.  Eager prefill and the
# precompute itself must never see this set.
# ---------------------------------------------------------------------------
def active() -> bool:
    return getattr(_tls, "depth", 0) > 0


@contextmanager
def active_scope():
    _tls.depth = getattr(_tls, "depth", 0) + 1
    try:
        yield
    finally:
        _tls.depth -= 1


def is_bound() -> bool:
    return _state.hasher is not None and bool(_state.layers)


# ---------------------------------------------------------------------------
# Binding (called once per decode runner, before capture)
# ---------------------------------------------------------------------------
def bind(model, max_tokens: int) -> bool:
    """Find the engram hasher/layers and allocate the persistent buffers.

    Returns False when this model has no engram, in which case the hooks pass
    through untouched.
    """
    from sglang.srt.layers.engram import Engram, EngramHasher

    # ``engram_hasher`` lives on the inner ``DeepseekV4Model``, not on the
    # CausalLM wrapper passed in here, so discover both by module type.
    hasher = None
    layers = []
    for module in model.modules():
        if isinstance(module, Engram):
            layers.append(module)
        elif isinstance(module, EngramHasher):
            hasher = module
    layers = tuple(layers)
    if hasher is None:
        hasher = getattr(model, "engram_hasher", None)
    if not layers:
        # No engram in this model: the host-table flag is inert here.
        return False
    if hasher is None:
        raise RuntimeError(
            "engram modules were found but no EngramHasher; refusing to capture "
            "a full decode graph with the host table enabled, because the "
            "lookup would replay capture-time rows"
        )

    device = layers[0].wkv.weight.device
    head_dim = int(layers[0].embed.dim)
    n_layers = int(hasher.primes.shape[0])
    n_cols = int(hasher.offsets.shape[1])
    n_gram = int(hasher.max_ngram_size)

    if _state.dummy_ids is None or _state.dummy_ids.shape[0] < max_tokens:
        _state.dummy_ids = torch.zeros(
            (max_tokens, n_layers, n_cols), dtype=torch.int64, device=device
        )
        _state.dummy_tokens = torch.zeros(
            (max_tokens, n_gram), dtype=torch.int64, device=device
        )

    for engram in layers:
        embed = engram.embed
        buf = getattr(embed, "_kunlun_engram_precomp", None)
        if buf is not None:
            if buf.shape[0] < max_tokens:
                # Reallocating here would invalidate graphs already captured
                # against this address; max_num_token is constant, so this is
                # a hard error rather than a resize.
                raise RuntimeError(
                    "engram precompute buffer is smaller than the new capture "
                    f"size ({buf.shape[0]} < {max_tokens}); refusing to "
                    "reallocate after capture"
                )
            continue
        embed._kunlun_engram_precomp = torch.full(
            (max_tokens, n_cols, head_dim),
            _CAPTURE_FILL,
            dtype=torch.bfloat16,
            device=device,
        )

    _state.model = model
    _state.hasher = hasher
    _state.layers = layers
    _state.max_tokens = max_tokens

    if not _state.logged:
        logger.info(
            "sglang-kunlun: engram out-of-graph precompute enabled "
            "(layers=%d, max_tokens=%d, n_cols=%d, head_dim=%d)",
            len(layers),
            max_tokens,
            n_cols,
            head_dim,
        )
        _state.logged = True
    return True


# ---------------------------------------------------------------------------
# Hooks used from the model code
# ---------------------------------------------------------------------------
def dummy_hash_ids(num_tokens: int) -> torch.Tensor:
    if _state.dummy_ids is None or num_tokens > _state.dummy_ids.shape[0]:
        raise RuntimeError(
            "engram precompute dummy hash ids are not bound for this shape "
            f"({num_tokens} > {0 if _state.dummy_ids is None else _state.dummy_ids.shape[0]})"
        )
    return _state.dummy_ids[:num_tokens]


def dummy_verify_tokens(num_tokens: int) -> torch.Tensor:
    if _state.dummy_tokens is None or num_tokens > _state.dummy_tokens.shape[0]:
        raise RuntimeError(
            "engram precompute dummy verify tokens are not bound for this shape"
        )
    return _state.dummy_tokens[:num_tokens]


def take_precomputed(embed, indices: torch.Tensor) -> torch.Tensor:
    """Return the buffer the pre-replay pass just refreshed."""
    buf = getattr(embed, "_kunlun_engram_precomp", None)
    num_tokens = int(indices.shape[0])
    if buf is None:
        raise RuntimeError(
            "engram precompute buffer is missing; the capture hook must bind "
            "before the first active forward"
        )
    if num_tokens > buf.shape[0]:
        raise RuntimeError(
            f"engram precompute buffer too small: {num_tokens} > {buf.shape[0]}"
        )
    return buf[:num_tokens]


# ---------------------------------------------------------------------------
# Pre-replay pass
# ---------------------------------------------------------------------------
def precompute(forward_batch, num_tokens: int) -> None:
    """Recompute hash ids + the host gather for the captured static batch.

    Runs between ``load_batch`` and ``backend.replay``.  ``forward_batch`` is
    the capture-time ForwardBatch object: its tensor fields are views of the
    graph's static buffers (refreshed by ``load_batch``), and its fixed
    metadata (forward mode, padding mode, DP buffer len, spec info) is exactly
    what the captured Python saw.
    """
    if active():
        raise RuntimeError("engram precompute ran inside a captured region")

    from sglang.srt.layers.dp_attention import (
        set_dp_buffer_len,
        set_is_extend_in_batch,
    )

    set_dp_buffer_len(
        getattr(forward_batch, "global_dp_buffer_len", None),
        num_tokens,
        forward_batch.dp_padding_mode.is_max_len(),
        getattr(forward_batch, "global_num_tokens_cpu", None),
    )
    set_is_extend_in_batch(False)
    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None

    hash_ids = _state.hasher(forward_batch.input_ids, forward_batch)
    for engram in _state.layers:
        embed = engram.embed
        emb = embed(hash_ids[:, engram.layer_hash_index], forward_batch)
        buf = getattr(embed, "_kunlun_engram_precomp", None)
        if buf is None:
            raise RuntimeError("engram precompute buffer vanished after bind")
        if emb.shape[0] > buf.shape[0]:
            raise RuntimeError(
                f"engram precompute buffer too small: {emb.shape[0]} > {buf.shape[0]}"
            )
        if tuple(emb.shape[1:]) != tuple(buf.shape[1:]):
            raise RuntimeError(
                "engram precompute shape mismatch: "
                f"emb={tuple(emb.shape)} buf={tuple(buf.shape)}"
            )
        if emb.dtype != buf.dtype:
            raise RuntimeError(
                f"engram precompute dtype mismatch: {emb.dtype} != {buf.dtype}"
            )
        buf[: emb.shape[0]].copy_(emb)
