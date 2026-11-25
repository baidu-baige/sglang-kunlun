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
"""Page endpoint bindings for AttentionStore cache plans."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from .schema import pad_layer_buffers
from .transport import _BufferPageIO, _sdk_pod_config, _standard_namespace
from .types import (
    AttentionStoreBinding,
    AttentionStoreEndpointBinding,
    AttentionStoreEndpointRole,
    AttentionStoreInitContext,
)


def _key_buffer(pool, layer_id: int) -> torch.Tensor:
    getter = getattr(pool, "_get_key_buffer", None) or pool.get_key_buffer
    return getter(layer_id)


def _value_buffer(pool, layer_id: int) -> torch.Tensor:
    getter = getattr(pool, "_get_value_buffer", None) or pool.get_value_buffer
    return getter(layer_id)


def _uniform_page_bytes(
    buffers: Sequence[torch.Tensor],
    pools: Sequence[Any],
    page_size: int,
    name: str,
) -> int:
    if not buffers:
        raise ValueError(f"{name} requires at least one buffer")
    if len(buffers) != len(pools):
        raise ValueError(f"{name} buffer/pool metadata length mismatch")
    sizes = set()
    for buffer, pool in zip(buffers, pools):
        row_bytes = int(buffer[0].nbytes)
        pool_size = getattr(pool, "size", None)
        if pool_size is not None and buffer.shape[0] >= int(pool_size):
            sizes.add(row_bytes * page_size)
            continue
        local_page_size = int(getattr(pool, "page_size", page_size))
        if page_size % local_page_size:
            raise ValueError(
                f"{name} block size {page_size} is not divisible by "
                f"local page size {local_page_size}"
            )
        sizes.add(row_bytes * (page_size // local_page_size))
    if len(sizes) != 1:
        raise ValueError(f"{name} buffers have non-uniform block sizes: {sizes}")
    return sizes.pop()


def _page_layout_id(
    page_format: str,
    page_size: int,
    buffers: Mapping[str, Sequence[torch.Tensor]],
    schema_layer_ids: Sequence[int] = (),
) -> str:
    payload = {
        "schema_version": 2,
        "endpoint_kind": page_format,
        "page_size": page_size,
        "schema_layer_ids": tuple(map(int, schema_layer_ids)),
        "buffers": {
            name: sorted(
                {
                    json.dumps(
                        {
                            "dtype": str(buffer.dtype),
                            "page_item_shape": tuple(buffer.shape[1:]),
                            "stride": tuple(buffer.stride()),
                        },
                        sort_keys=True,
                    )
                    for buffer in tensors
                }
            )
            for name, tensors in buffers.items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _create_mha_page_endpoint(
    ctx: AttentionStoreInitContext,
    *,
    pool: Any,
    namespace: str,
    gpu_count: int,
    registration_endpoint_name: str | None = None,
) -> AttentionStoreEndpointBinding:
    """Bind an MHA K/V endpoint to a full-attention pool."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MHAConfig,
    )

    local_layer_ids = tuple(range(int(pool.layer_num)))
    if not local_layer_ids:
        raise ValueError("MHA page endpoint requires at least one attention layer")
    is_kda = registration_endpoint_name == "kda-page"
    layer_ids = tuple(ctx.page_layer_ids) if is_kda else local_layer_ids
    target_layer_num = len(layer_ids) if is_kda else ctx.page_layer_num
    key_buffers = [_key_buffer(pool, layer_id) for layer_id in local_layer_ids]
    value_buffers = [_value_buffer(pool, layer_id) for layer_id in local_layer_ids]
    if not is_kda:
        key_buffers = pad_layer_buffers(key_buffers, target_layer_num, "MHA K")
        value_buffers = pad_layer_buffers(value_buffers, target_layer_num, "MHA V")
    buffer_pools = [pool] * target_layer_num
    block_size = (
        ctx.block_size if registration_endpoint_name == "kda-page" else ctx.page_size
    )
    k_bytes = _uniform_page_bytes(key_buffers, buffer_pools, block_size, "MHA K")
    v_bytes = _uniform_page_bytes(value_buffers, buffer_pools, block_size, "MHA V")
    bindings = BufferBindings(
        k=[buffer.data_ptr() for buffer in key_buffers],
        v=[buffer.data_ptr() for buffer in value_buffers],
    )
    layout_id = _page_layout_id(
        "MHA",
        block_size,
        {"k": key_buffers, "v": value_buffers},
        layer_ids,
    )
    sdk = AttentionStoreSDK(
        _sdk_pod_config(
            ctx,
            namespace,
            endpoint_name=registration_endpoint_name,
            model_version=ctx.model_version,
            gpu_count=gpu_count,
        ),
        MHAConfig(
            # KDA clients share one namespace on the node.  Register the
            # complete node geometry; the request still contains only this
            # client's layer_ids and matching buffers.
            layer_num=ctx.page_layer_num if is_kda else target_layer_num,
            shard_id=ctx.page_shard_id,
            shard_num=ctx.page_shard_num,
            block_token_size=block_size,
            k_bytes=k_bytes,
            v_bytes=v_bytes,
        ),
        enable_as_kv_rw=True,
    )
    return AttentionStoreEndpointBinding(
        name="mha",
        registration_name=registration_endpoint_name,
        role=AttentionStoreEndpointRole.PAGE,
        store=_BufferPageIO(
            sdk,
            bindings,
            layer_ids,
            buffer_tensors={"k": key_buffers, "v": value_buffers},
            buffer_pools={"k": buffer_pools, "v": buffer_pools},
            local_page_size=ctx.page_size,
            block_size=block_size,
        ),
        namespace=namespace,
        layout_id=layout_id,
        bytes_per_block=(k_bytes + v_bytes)
        * (ctx.page_layer_num if is_kda else target_layer_num),
        payload_fields=("k", "v"),
    )


def _create_mla_page_endpoint(
    ctx: AttentionStoreInitContext,
    *,
    pool: Any,
    namespace: str,
    gpu_count: int,
    registration_endpoint_name: str | None = None,
) -> AttentionStoreEndpointBinding:
    """Bind an MLA latent-KV endpoint to a full-attention pool."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MLAConfig,
    )

    local_layer_ids = tuple(range(int(pool.layer_num)))
    if not local_layer_ids:
        raise ValueError("MLA page endpoint requires at least one attention layer")
    is_kda = registration_endpoint_name == "kda-page"
    layer_ids = tuple(ctx.page_layer_ids) if is_kda else local_layer_ids
    target_layer_num = len(layer_ids) if is_kda else ctx.page_layer_num
    kv_buffers = [_key_buffer(pool, layer_id) for layer_id in local_layer_ids]
    if not is_kda:
        kv_buffers = pad_layer_buffers(kv_buffers, target_layer_num, "MLA latent KV")
    buffer_pools = [pool] * target_layer_num
    kv_bytes = _uniform_page_bytes(
        kv_buffers, buffer_pools, ctx.block_size, "MLA latent KV"
    )
    bindings = BufferBindings(k=[buffer.data_ptr() for buffer in kv_buffers])
    layout_id = _page_layout_id(
        "MLA",
        ctx.block_size,
        {"kv_latent": kv_buffers},
        layer_ids,
    )
    sdk = AttentionStoreSDK(
        _sdk_pod_config(
            ctx,
            namespace,
            endpoint_name=registration_endpoint_name,
            model_version=ctx.model_version,
            gpu_count=gpu_count,
            pod_name=f"{ctx.pod_name}-tp{ctx.tp_rank}" if ctx.tp_size > 1 else None,
        ),
        MLAConfig(
            # For KDA shared-shard registration this is the complete
            # node-local layer geometry.  target_layer_num remains the local
            # pointer/buffer count so a PP client does not fabricate buffers
            # for layers owned by another client.
            layer_num=ctx.page_layer_num if is_kda else target_layer_num,
            shard_id=ctx.page_shard_id // ctx.tp_size,
            shard_num=ctx.page_shard_num // ctx.tp_size,
            block_token_size=ctx.block_size,
            kv_bytes=kv_bytes,
        ),
        enable_as_kv_rw=True,
    )
    return AttentionStoreEndpointBinding(
        name="mla",
        registration_name=registration_endpoint_name,
        role=AttentionStoreEndpointRole.PAGE,
        store=_BufferPageIO(
            sdk,
            bindings,
            layer_ids,
            buffer_tensors={"k": kv_buffers},
            buffer_pools={"k": buffer_pools},
            local_page_size=ctx.page_size,
            block_size=ctx.block_size,
        ),
        namespace=namespace,
        layout_id=layout_id,
        bytes_per_block=kv_bytes * (ctx.page_layer_num if is_kda else target_layer_num),
        payload_fields=("kv_latent",),
    )


def create_mha_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create an MHA binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MHAConfig,
    )

    if ctx.page_shard_id < 0:
        raise ValueError("MHA AttentionStore stage has no page shard")
    target_ids = tuple(ctx.target_layer_ids)
    key_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    value_buffers = [_value_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    buffer_pools = [ctx.kv_cache] * len(target_ids)
    if ctx.draft_kv_cache is not None:
        key_buffers.extend(
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        value_buffers.extend(
            _value_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        buffer_pools.extend([ctx.draft_kv_cache] * len(ctx.draft_layer_ids))
    if len(key_buffers) != len(ctx.page_layer_ids):
        raise ValueError("MHA page manifest and local buffer counts differ")
    key_buffers = pad_layer_buffers(key_buffers, ctx.page_layer_num, "MHA K")
    value_buffers = pad_layer_buffers(value_buffers, ctx.page_layer_num, "MHA V")
    buffer_pools = pad_layer_buffers(
        buffer_pools, ctx.page_layer_num, "MHA buffer pools"
    )
    k_bytes = _uniform_page_bytes(key_buffers, buffer_pools, ctx.page_size, "MHA K")
    v_bytes = _uniform_page_bytes(value_buffers, buffer_pools, ctx.page_size, "MHA V")
    namespace = _standard_namespace(ctx, "mha", ctx.page_shard_num)
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        MHAConfig(
            layer_num=ctx.page_layer_num,
            shard_id=ctx.page_shard_id,
            shard_num=ctx.page_shard_num,
            block_token_size=ctx.page_size,
            k_bytes=k_bytes,
            v_bytes=v_bytes,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(
        k=[buffer.data_ptr() for buffer in key_buffers],
        v=[buffer.data_ptr() for buffer in value_buffers],
    )
    page_store = _BufferPageIO(sdk, buffers, range(ctx.page_layer_num))
    endpoint = AttentionStoreEndpointBinding(
        name="mha",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id="mha",
        bytes_per_block=(k_bytes + v_bytes) * ctx.page_layer_num,
        payload_fields=("k", "v"),
    )
    return AttentionStoreBinding(
        plan_name="MHA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )


def create_mla_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create an MLA binding."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        MLAConfig,
    )

    if ctx.page_shard_id < 0:
        raise ValueError("MLA AttentionStore stage has no page shard")
    target_ids = tuple(ctx.target_layer_ids)
    kv_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    buffer_pools = [ctx.kv_cache] * len(target_ids)
    if ctx.draft_kv_cache is not None:
        draft_buffers = [
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        ]
        for phys_id, buffer in zip(ctx.draft_layer_ids, draft_buffers):
            # An MLA binding must never silently reinterpret an MHA draft layer.
            if (
                _value_buffer(ctx.draft_kv_cache, phys_id).data_ptr()
                != buffer.data_ptr()
            ):
                raise ValueError("MLA AttentionStore requires MLA-organized draft KV")
        kv_buffers.extend(draft_buffers)
        buffer_pools.extend([ctx.draft_kv_cache] * len(draft_buffers))
    if len(kv_buffers) != len(ctx.page_layer_ids):
        raise ValueError("MLA page manifest and local buffer counts differ")
    kv_buffers = pad_layer_buffers(kv_buffers, ctx.page_layer_num, "MLA latent KV")
    buffer_pools = pad_layer_buffers(
        buffer_pools, ctx.page_layer_num, "MLA buffer pools"
    )
    kv_bytes = _uniform_page_bytes(kv_buffers, buffer_pools, ctx.block_size, "MLA KV")
    namespace = _standard_namespace(ctx, "mla", ctx.page_shard_num)
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        MLAConfig(
            layer_num=ctx.page_layer_num,
            shard_id=ctx.page_shard_id,
            shard_num=ctx.page_shard_num,
            block_token_size=ctx.block_size,
            kv_bytes=kv_bytes,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(k=[buffer.data_ptr() for buffer in kv_buffers])
    page_store = _BufferPageIO(
        sdk,
        buffers,
        range(ctx.page_layer_num),
        buffer_tensors={"k": kv_buffers},
        buffer_pools={"k": buffer_pools},
        local_page_size=ctx.page_size,
        block_size=ctx.block_size,
    )
    endpoint = AttentionStoreEndpointBinding(
        name="mla",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id=_page_layout_id(
            "MLA",
            ctx.block_size,
            {"kv_latent": kv_buffers},
            range(ctx.page_layer_num),
        ),
        bytes_per_block=kv_bytes * ctx.page_layer_num,
        payload_fields=("kv_latent",),
    )
    return AttentionStoreBinding(
        plan_name="MLA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )


def _dsa_indexer_layer_ids(
    kv_cache, target_layer_ids: Sequence[int]
) -> tuple[int, ...]:
    mapping = getattr(kv_cache, "indexer_layer_mapping", None)
    if mapping is None:
        return tuple(target_layer_ids)
    target_set = set(target_layer_ids)
    layer_ids = tuple(
        layer_id
        for layer_id, _ in sorted(mapping.items(), key=lambda item: item[1])
        if layer_id in target_set
    )
    if not layer_ids:
        raise ValueError("DSA compact indexer mapping contains no physical layers")
    return layer_ids


def create_dsa_binding(ctx: AttentionStoreInitContext) -> AttentionStoreBinding:
    """Create a DSA binding and preserve compact physical indexer ordering."""
    from attentionstore_sdk.sdk import (
        AttentionStoreSDK,
        BufferBindings,
        DSAConfig,
    )

    if ctx.page_shard_id < 0:
        raise ValueError("DSA AttentionStore stage has no page shard")
    target_ids = tuple(ctx.target_layer_ids)
    latent_buffers = [_key_buffer(ctx.kv_cache, layer_id) for layer_id in target_ids]
    latent_pools = [ctx.kv_cache] * len(target_ids)
    physical_indexer_ids = _dsa_indexer_layer_ids(ctx.kv_cache, target_ids)
    indexer_buffers = [
        ctx.kv_cache.get_index_k_with_scale_buffer(layer_id)
        for layer_id in physical_indexer_ids
    ]
    indexer_pools = [ctx.kv_cache] * len(indexer_buffers)
    target_logical_ids = {layer_id: i for i, layer_id in enumerate(target_ids)}
    sdk_indexer_ids = [
        target_logical_ids[layer_id] for layer_id in physical_indexer_ids
    ]
    if ctx.draft_kv_cache is not None:
        draft_count = len(ctx.draft_layer_ids)
        latent_buffers.extend(
            _key_buffer(ctx.draft_kv_cache, layer_id)
            for layer_id in ctx.draft_layer_ids
        )
        latent_pools.extend([ctx.draft_kv_cache] * draft_count)
        for draft_offset, layer_id in enumerate(ctx.draft_layer_ids):
            if hasattr(ctx.draft_kv_cache, "get_index_k_with_scale_buffer"):
                buffer = ctx.draft_kv_cache.get_index_k_with_scale_buffer(layer_id)
            else:
                buffer = _value_buffer(ctx.draft_kv_cache, layer_id)
            indexer_buffers.append(buffer)
            indexer_pools.append(ctx.draft_kv_cache)
            sdk_indexer_ids.append(len(target_ids) + draft_offset)
    local_layer_num = len(latent_buffers)
    if local_layer_num != len(ctx.page_layer_ids):
        raise ValueError("DSA page manifest and local buffer counts differ")
    latent_buffers = pad_layer_buffers(
        latent_buffers, ctx.page_layer_num, "DSA latent KV"
    )
    latent_pools = pad_layer_buffers(
        latent_pools, ctx.page_layer_num, "DSA latent pools"
    )
    if ctx.page_layer_num > local_layer_num:
        padding = ctx.page_layer_num - local_layer_num
        indexer_buffers.extend([indexer_buffers[-1]] * padding)
        indexer_pools.extend([indexer_pools[-1]] * padding)
        sdk_indexer_ids.extend(range(local_layer_num, ctx.page_layer_num))
    latent_bytes = _uniform_page_bytes(
        latent_buffers, latent_pools, ctx.page_size, "DSA latent KV"
    )
    indexer_bytes = _uniform_page_bytes(
        indexer_buffers, indexer_pools, ctx.page_size, "DSA indexer"
    )
    namespace = _standard_namespace(ctx, "dsa", ctx.page_shard_num)
    sdk = AttentionStoreSDK(
        _sdk_pod_config(ctx, namespace),
        DSAConfig(
            layer_num=ctx.page_layer_num,
            shard_id=ctx.page_shard_id,
            shard_num=ctx.page_shard_num,
            block_token_size=ctx.page_size,
            latent_kv_bytes=latent_bytes,
            indexer_bytes=indexer_bytes,
            indexer_layer_ids=sdk_indexer_ids,
        ),
        enable_as_kv_rw=True,
    )
    buffers = BufferBindings(
        k=[buffer.data_ptr() for buffer in latent_buffers],
        v=[buffer.data_ptr() for buffer in indexer_buffers],
    )
    page_store = _BufferPageIO(sdk, buffers, range(ctx.page_layer_num))
    endpoint = AttentionStoreEndpointBinding(
        name="dsa",
        role=AttentionStoreEndpointRole.PAGE,
        store=page_store,
        namespace=namespace,
        layout_id="dsa",
        bytes_per_block=(latent_bytes + indexer_bytes) * ctx.page_layer_num,
        payload_fields=("kv_latent", "indexer"),
    )
    return AttentionStoreBinding(
        plan_name="DSA",
        cache_store=page_store,
        endpoints=(endpoint,),
    )
