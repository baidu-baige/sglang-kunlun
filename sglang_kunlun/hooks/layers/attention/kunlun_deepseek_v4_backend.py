"""Kunlun DeepSeek-V4 compressed-attention backend.

The upstream 0.5.14 backend owns metadata preparation and graph lifecycle.  This
module only replaces the FlashMLA-specific metadata/forward boundary with the
Kunlun compressed-attention operator.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import hashlib
import json
import logging
import os
import cocopod  # noqa: F401  # Registers DSV4 XSpeedGate attention operators.
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as upstream
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4AttnMetadata,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang_kunlun.kernels.kernel_ops import (
    dsv4_quant_k_cache_kunlun,
    dsv4_set_k_and_s_with_mapping_kunlun,
)


logger = logging.getLogger(__name__)
if os.environ.get("DSV4_C4_ATTN_METADATA_LOG") == "1":
    print(
        "[DSV4_C4_ATTN_BACKEND_IMPORT] "
        f"file={__file__} pid={os.getpid()} "
        f"TP_RANK={os.environ.get('TP_RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"flag={os.environ.get('DSV4_C4_ATTN_METADATA_LOG')}",
        flush=True,
    )

_ENABLE_DSV4_ACCURACY_DUMPS = False


def _dsv4_ifeval_diag_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _dsv4_ifeval_tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[dict]:
    if not isinstance(tensor, torch.Tensor):
        return None
    flat = tensor.detach().reshape(-1)
    cpu = flat.contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
        "numel": flat.numel(),
        "head": cpu[:16].tolist(),
        "tail": cpu[-16:].tolist() if cpu.numel() else [],
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }


def _append_dsv4_ifeval_backend_event(owner, event: dict) -> None:
    dump_dir = os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
    if not dump_dir or _dsv4_ifeval_diag_rank() != 0:
        return
    event_count = getattr(owner, "_dsv4_ifeval_diag_event_count", 0)
    max_events = int(os.environ.get("DSV4_IFEVAL_MTP_DIAG_MAX_EVENTS", "20000"))
    if event_count >= max_events:
        return
    os.makedirs(dump_dir, exist_ok=True)
    event["event_index"] = event_count
    event["pid"] = os.getpid()
    step_id = getattr(owner, "speculative_step_id", -1)
    event["speculative_step_id"] = int(step_id) if step_id is not None else -1
    with open(
        os.path.join(dump_dir, "backend_events_rank0.jsonl"),
        "a",
        encoding="utf-8",
    ) as output:
        output.write(json.dumps(event, separators=(",", ":")) + "\n")
    owner._dsv4_ifeval_diag_event_count = event_count + 1


def _summarize_c4_metadata_tensor(value):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return repr(value)
    flat = value.detach().cpu().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "head": flat[:8].tolist(),
        "tail": flat[-8:].tolist() if flat.numel() else [],
    }


def _clamp_c128_prefill_topk(
    compressed_topk: int,
    kv_lens_cpu: Optional[torch.Tensor],
    extend_seq_lens_cpu: Optional[List[int]],
    compress_ratio: int,
) -> int:
    if kv_lens_cpu is None or kv_lens_cpu.numel() == 0 or not extend_seq_lens_cpu:
        return compressed_topk
    chunk_start_pos = int(kv_lens_cpu[0]) - max(extend_seq_lens_cpu)
    if chunk_start_pos <= 0:
        return compressed_topk
    return min(compressed_topk, max(chunk_start_pos // compress_ratio, 1))


def _mtp_tensor_probe_enabled(layer_id: int, num_queries: int) -> bool:
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    if not dump_dir or os.environ.get("DSV4_MTP_PROBE_SEQ_LEN"):
        return False
    if (
        num_queries != 1
        and os.environ.get("DSV4_MTP_TENSOR_PROBE_ANY_BATCH") != "1"
    ):
        return False
    probe_layers = {
        int(value)
        for value in os.environ.get(
            "DSV4_MTP_TENSOR_PROBE_LAYERS", "0,21,42"
        ).split(",")
        if value
    }
    if layer_id not in probe_layers:
        return False
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    return rank == 0


def _mtp_writer_probe_enabled(layer_id: int) -> bool:
    return _mtp_tensor_probe_enabled(layer_id, 1) and (
        os.environ.get("DSV4_MTP_TENSOR_PROBE_WRITER") == "1"
    )


def _mtp_gather_cache_rows(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return cache.new_empty((*indices.shape, cache.shape[-1]))
    safe_indices = indices.reshape(-1).clamp(min=0, max=cache.shape[0] - 1).long()
    return cache.index_select(0, safe_indices).reshape(*indices.shape, cache.shape[-1])


def _mtp_tensor_stats(tensor: torch.Tensor) -> dict:
    flat = tensor.reshape(-1)
    stats = {
        "numel": flat.numel(),
        "zero_count": int((flat == 0).sum().item()) if flat.numel() else 0,
        "minus_one_count": int((flat == -1).sum().item()) if flat.numel() else 0,
        "negative_count": int((flat < 0).sum().item()) if flat.numel() else 0,
    }
    if flat.numel():
        stats.update(min=float(flat.min().item()), max=float(flat.max().item()))
    if tensor.is_floating_point():
        stats.update(
            nan_count=int(torch.isnan(flat).sum().item()),
            inf_count=int(torch.isinf(flat).sum().item()),
        )
    return stats


def _mtp_probe_matches_seq_lens(seq_lens_cpu, batch_size: int) -> bool:
    if seq_lens_cpu is None:
        return False
    values = torch.as_tensor(seq_lens_cpu)[:batch_size].reshape(-1)
    if not values.numel():
        return False
    expected = int(os.environ.get("DSV4_MTP_PROBE_SEQ_LEN", "-1"))
    if expected < 0:
        return int(values.max()) > 10000
    radius = int(os.environ.get("DSV4_MTP_PROBE_RADIUS", "8"))
    return bool((values - expected).abs().le(radius).any())


def _mtp_probe_layout(tensor: torch.Tensor) -> dict:
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def _dump_previous_mtp_tensor_probe(multistep_backend, forward_batch) -> None:
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    if (
        not dump_dir
        or rank != 0
        or not _mtp_probe_matches_seq_lens(
            forward_batch.seq_lens_cpu, forward_batch.batch_size
        )
    ):
        return

    cycle = getattr(multistep_backend, "_mtp_tensor_dump_cycle", 0)
    max_cycles = int(os.environ.get("DSV4_MTP_TENSOR_DUMP_CYCLES", "3"))
    if cycle >= max_cycles:
        return

    tensors = {}
    tensor_metadata = {}
    bs = forward_batch.batch_size

    def record(step: int, name: str, tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        key = f"step{step}.{name}"
        tensors[key] = tensor.detach().cpu()
        tensor_metadata[key] = _mtp_probe_layout(tensor)

    for step, backend in enumerate(multistep_backend.attn_backends):
        metadata = getattr(backend, "forward_metadata", None)
        core = getattr(metadata, "core_attn_metadata", None)
        if core is None:
            core = getattr(metadata, "core_metadata", None)
        indexer = getattr(metadata, "indexer_metadata", None)

        attention_aux = backend._attention_decode_aux.get(bs)
        if attention_aux is not None:
            _, q_lod, _, kv_lens = attention_aux
            record(step, "q_lod", q_lod)
            record(step, "kv_lens", kv_lens)

        page_table = getattr(core, "page_table", None)
        record(step, "page_table", page_table)
        if isinstance(page_table, torch.Tensor) and indexer is not None:
            max_seq_len = int(getattr(indexer, "max_c4_seq_len", 0))
            num_pages = min(
                (max_seq_len + 63) // 64,
                page_table.shape[1],
            )
            record(step, "block_table", page_table[:, :num_pages].to(torch.int32))
        record(step, "c4_seq_lens", getattr(indexer, "c4_seq_lens", None))
        record(
            step,
            "c4_sparse_page_indices",
            getattr(core, "c4_sparse_page_indices", None),
        )
        record(step, "c128_page_indices", getattr(core, "c128_page_indices", None))

    if not tensors:
        logger.warning("[DSV4_MTP_PROBE] no target replay metadata captured")
        return

    os.makedirs(dump_dir, exist_ok=True)
    version = os.environ.get("DSV4_MTP_TENSOR_DUMP_VERSION", "0514")
    path = os.path.join(dump_dir, f"mtp_tensor_{version}_cycle{cycle}_rank0.pt")
    torch.save(
        {
            "version": version,
            "cycle": cycle,
            "probe_seq_len": int(os.environ.get("DSV4_MTP_PROBE_SEQ_LEN", "-1")),
            "seq_lens": list(
                map(
                    int,
                    forward_batch.seq_lens_cpu[: forward_batch.batch_size],
                )
            ),
            "tensors": tensors,
            "tensor_metadata": tensor_metadata,
        },
        path,
    )
    logger.warning("[DSV4_MTP_PROBE] target tensor dump saved path=%s", path)
    multistep_backend._mtp_tensor_dump_cycle = cycle + 1


def _dump_eager_mtp_tensor_probe(backend) -> None:
    if getattr(backend, "speculative_num_steps", 0) > 1:
        return
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    probe = getattr(backend, "_mtp_tensor_probe", {})
    if not dump_dir or not probe:
        return
    cycle = getattr(backend, "_mtp_tensor_dump_cycle", 0)
    max_cycles = int(os.environ.get("DSV4_MTP_TENSOR_DUMP_CYCLES", "3"))
    if cycle >= max_cycles:
        return

    tensors = {}
    tensor_metadata = {}
    tensor_stats = {}
    for name, tensor in probe.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        cpu_tensor = tensor.detach().cpu()
        tensors[name] = cpu_tensor
        tensor_metadata[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": tuple(tensor.stride()),
            "device": str(tensor.device),
            "data_ptr": tensor.data_ptr(),
        }
        tensor_stats[name] = _mtp_tensor_stats(cpu_tensor)
    if not tensors:
        return

    os.makedirs(dump_dir, exist_ok=True)
    version = os.environ.get("DSV4_MTP_TENSOR_DUMP_VERSION", "0514")
    path = os.path.join(dump_dir, f"mtp_tensor_{version}_cycle{cycle}_rank0.pt")
    torch.save(
        {
            "version": version,
            "cycle": cycle,
            "tensors": tensors,
            "tensor_metadata": tensor_metadata,
            "tensor_stats": tensor_stats,
            "operator_contracts": getattr(backend, "_mtp_tensor_probe_contracts", {}),
        },
        path,
    )
    logger.warning("[DSV4_MTP_PROBE] target eager tensor dump saved path=%s", path)
    backend._mtp_tensor_dump_cycle = cycle + 1


_upstream_create_paged_compressor_data = upstream.create_paged_compressor_data


def _create_paged_compressor_data_kunlun(*args, **kwargs):
    """Adapt the 0.5.14 call site to the active compressor-v1 signature."""
    kwargs.pop("online_state_slot_offset", None)
    return _upstream_create_paged_compressor_data(*args, **kwargs)


upstream.create_paged_compressor_data = _create_paged_compressor_data_kunlun


def _generate_compressor_prefill_plan_kunlun(
    compress_ratio,
    num_q_tokens,
    seq_lens,
    extend_lens,
    device,
    use_cuda_graph=False,
):
    """Build compressor-v1 plans with the 0.5.8 XSpeedGate planner."""
    from sglang.jit_kernel.dsv4.compress_old import CompressorPrefillPlan

    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    if extend_lens.dtype != torch.int64:
        extend_lens = extend_lens.to(torch.int64)
    plan_tensor = torch.empty(
        (2, num_q_tokens, 16),
        dtype=torch.uint8,
        device=seq_lens.device,
        pin_memory=seq_lens.is_cpu,
    )
    plan_lens = torch.ops.xspeedgate_ops.plan_compress_prefill(
        extend_lens,
        seq_lens,
        plan_tensor[0],
        plan_tensor[1],
        compress_ratio,
        compress_ratio == 4,
        use_cuda_graph,
    )
    plan_device = plan_tensor.to(device, non_blocking=True)
    return CompressorPrefillPlan(
        compress_ratio,
        plan_device[0, : int(plan_lens[0])],
        plan_device[1, : int(plan_lens[1])],
    )


from sglang.jit_kernel.dsv4.compress_old import CompressorPrefillPlan

CompressorPrefillPlan.generate = staticmethod(_generate_compressor_prefill_plan_kunlun)


class KunlunDSV4AttnMetadata(DSV4AttnMetadata):
    """DSV4 metadata without CUDA FlashMLA scheduler objects."""

    def init_flashmla_related(self, is_prefill: bool = False) -> None:
        """Initialize Kunlun sparse metadata without FlashMLA schedulers."""
        if self.c4_sparse_topk not in (512, 1024):
            raise ValueError(f"unsupported Kunlun C4 top-k: {self.c4_sparse_topk}")
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = upstream._pad_last_dim(
            self.c4_sparse_page_indices
        )
        # Kunlun's custom prefill attention consumes physical page indices only.
        # Keep this unset to match the patched 0.5.8 xspeedgate top-k path; a raw
        # buffer would force the 0.5.14 JIT wrapper into its PyTorch fallback.
        self.c4_sparse_raw_indices = None
        self.c1_flashmla_metadata = None
        self.c4_flashmla_metadata = None
        self.c128_flashmla_metadata = None


def _copy_host_lengths_(destination, source, fill_value: int) -> None:
    count = min(source.numel(), destination.numel())
    destination[:count].copy_(source[:count])
    if count < destination.numel():
        destination[count:].fill_(fill_value)


def _is_graph_extend_mode(forward_mode) -> bool:
    return forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()


def _alloc_graph_extend_aux(
    batch_size: int, num_queries: int, device: torch.device
):
    num_queries_per_request = num_queries // batch_size if batch_size else 1
    q_lod_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32) * num_queries_per_request
    )
    q_lod = q_lod_cpu.to(device, non_blocking=False)
    kv_lens_cpu = torch.ones(batch_size, dtype=torch.int32)
    kv_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    query_lens = torch.full(
        (batch_size,),
        num_queries_per_request,
        dtype=torch.int32,
        device=device,
    )
    return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens, query_lens


def _get_graph_extend_aux(
    cache, batch_size: int, num_queries: int, device: torch.device
):
    if batch_size <= 0 or num_queries % batch_size:
        raise ValueError(
            "Kunlun DSV4 graph extend requires uniform queries per request: "
            f"batch_size={batch_size}, num_queries={num_queries}"
        )
    key = (batch_size, num_queries, device)
    aux = cache.get(key)
    if aux is None:
        aux = _alloc_graph_extend_aux(batch_size, num_queries, device)
        cache[key] = aux
    return aux


def _seq_lens_cpu_i32(forward_batch) -> torch.Tensor:
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.to("cpu", non_blocking=False)
    return seq_lens_cpu[: forward_batch.batch_size].to(torch.int32)


def _refresh_graph_host_lengths(
    forward_batch, attention_decode_aux, c4_decode_aux, graph_extend_aux
) -> None:
    """Refresh this backend's pointer-stable lengths before graph replay."""
    seq_lens = _seq_lens_cpu_i32(forward_batch)
    batch_size = forward_batch.batch_size

    if forward_batch.forward_mode.is_decode_or_idle():
        attention_aux = attention_decode_aux.get(batch_size)
        if attention_aux is not None:
            _copy_host_lengths_(attention_aux[2], seq_lens, fill_value=1)
            attention_aux[3].copy_(attention_aux[2])

        c4_context_lens = (seq_lens // 4) * 4
        for (cached_bs, _, _), aux in c4_decode_aux.items():
            if cached_bs == batch_size:
                _copy_host_lengths_(aux[2], c4_context_lens, fill_value=4)
                aux[3].copy_(aux[2])
        return

    if not _is_graph_extend_mode(forward_batch.forward_mode):
        return

    for (cached_bs, _, _), aux in graph_extend_aux.items():
        if cached_bs != batch_size:
            continue
        _, _, kv_lens_cpu, kv_lens, query_lens = aux
        _copy_host_lengths_(kv_lens_cpu, seq_lens, fill_value=1)
        kv_lens.copy_(kv_lens_cpu)
        torch.maximum(kv_lens, query_lens, out=kv_lens)


class KunlunDeepseekV4AttnBackend(DeepseekV4AttnBackend):
    """0.5.14 DSV4 control flow with the Kunlun attention boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attention_decode_aux = {}
        self._attention_graph_extend_aux = {}
        self._c4_decode_aux = {}

    def get_swa_page_indices(self, seq_lens_casual, req_pool_indices_repeated):
        """Translate request token offsets into Kunlun SWA page indices."""
        # Match the 0.5.8 contract: invalid history offsets are clamped to the
        # first physical row and remain valid indices; the length tensor masks
        # those rows. Upstream 0.5.14 writes -1 here, which changes graph replay
        # inputs and is not accepted by the Kunlun compressed-attention path.
        pos_causal = seq_lens_casual - 1
        offsets = torch.clamp(
            pos_causal.unsqueeze(1)
            - torch.arange(
                upstream.SWA_WINDOW, **self.cuda_int32_kwargs
            ).unsqueeze(0),
            min=0,
        )
        raw_indices = self.req_to_token[
            req_pool_indices_repeated[:, None], offsets
        ]
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(
            raw_indices
        ).to(torch.int32)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        """Initialize out-of-graph metadata and refresh replay lengths."""
        if in_capture and _is_graph_extend_mode(forward_batch.forward_mode):
            _get_graph_extend_aux(
                self._attention_graph_extend_aux,
                forward_batch.batch_size,
                forward_batch.positions.numel(),
                forward_batch.seq_lens.device,
            )
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        _refresh_graph_host_lengths(
            forward_batch,
            self._attention_decode_aux,
            self._c4_decode_aux,
            self._attention_graph_extend_aux,
        )

    def forward_c4_indexer(
        self,
        x,
        q_lora,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream=False,
        q_lora_ready=None,
        skip_compressor=False,
    ):
        """Run the version-pinned C4 flow with explicit Kunlun request state."""
        import torch.nn.functional as F

        from sglang.srt.layers.attention.dsv4 import indexer as upstream_indexer

        if forward_batch.forward_mode.is_idle():
            return

        token_to_kv_pool = self.token_to_kv_pool
        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata
        assert isinstance(indexer_metadata, upstream_indexer.PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        if enable_multi_stream:
            q_indexer, weights, c4_indexer_kv_cache = (
                self._forward_prepare_multi_stream(
                    x=x,
                    q_lora=q_lora,
                    c4_indexer=c4_indexer,
                    positions=positions,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    alt_streams=alt_streams,
                    q_lora_ready=q_lora_ready,
                )
            )
        else:
            assert q_lora_ready is None
            q_indexer, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                skip_compressor=skip_compressor,
            )

        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        use_fp4_indexer = c4_indexer.use_fp4_indexer
        head_dim_with_sf = 68 if use_fp4_indexer else 132

        if use_fp4_indexer:
            q_fp4, q_sf = q_indexer
            assert len(q_fp4.shape) == 3
            assert len(q_sf.shape) == 2
            q = (q_fp4.unsqueeze(1), q_sf.unsqueeze(1))
        else:
            assert len(q_indexer.shape) == 3
            q = q_indexer.unsqueeze(1)

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0],
            block_kv,
            num_heads_kv,
            head_dim_with_sf,
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if use_fp4_indexer:
            weights = weights.float()
            if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
                raise RuntimeError("DeepSeek V4 FP4 indexer requires DeepGEMM indexer.")
            from deep_gemm import fp8_fp4_paged_mqa_logits as fn
        elif envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_OPT_USE_AITER_INDEXER.get():
            fn = upstream_indexer._aiter_fp8_paged_mqa_logits
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            if upstream_indexer.is_sm120_supported():
                fn = upstream_indexer.fp8_paged_mqa_logits_torch_sm120
            else:
                fn = upstream_indexer.fp8_paged_mqa_logits_torch
        else:
            from deep_gemm import fp8_paged_mqa_logits as fn

        query_rows = (
            q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]
        )

        def match_num_queries(tensor, value):
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (
                0,
                query_rows - tensor.shape[0],
            )
            return F.pad(tensor, pad, value=value)

        c4_seq_lens = match_num_queries(indexer_metadata.c4_seq_lens, value=1)
        c4_seq_lens_for_logits = c4_seq_lens
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )
        use_tilelang = (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.get() and not use_fp4_indexer
        )
        use_aiter = envs.SGLANG_OPT_USE_AITER_INDEXER.get() and not use_fp4_indexer
        if (
            c4_seq_lens_for_logits.dim() == 1
            and not use_tilelang
            and not use_aiter
        ):
            c4_seq_lens_for_logits = c4_seq_lens_for_logits.unsqueeze(-1)

        logits = self._compute_c4_indexer_logits(
            fn=fn,
            q=q,
            c4_indexer_kv_cache=c4_indexer_kv_cache,
            weights=weights,
            c4_seq_lens=c4_seq_lens_for_logits,
            page_table=page_table,
            indexer_metadata=indexer_metadata,
            core_metadata=core_metadata,
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = upstream_indexer.get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None
        hisparse_coordinator = self.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            upstream_indexer.topk_transform_512_pytorch_vectorized(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            upstream_indexer.topk_transform_512_v2(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            upstream_indexer.topk_transform_512(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )

        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    ).to(torch.int32)
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)

    def _compute_c4_indexer_logits(
        self,
        *,
        fn,
        q,
        c4_indexer_kv_cache,
        weights,
        c4_seq_lens,
        page_table,
        indexer_metadata,
        core_metadata,
        forward_batch,
        c4_indexer,
    ):
        del fn, core_metadata
        return _compute_c4_logits_kunlun(
            backend=self,
            q_fp8=q,
            kvcache_fp8=c4_indexer_kv_cache,
            weight=weights,
            seq_lens=c4_seq_lens,
            page_table=page_table,
            max_seq_len=indexer_metadata.max_c4_seq_len,
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
        )

    def on_after_cuda_graph_warmup(self):
        """Skip FlashMLA metadata refresh, matching the 0.5.8 Kunlun backend."""
        metadata = self.forward_metadata
        if isinstance(metadata, upstream.DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, upstream.DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c1_flashmla_metadata = None
            core.c4_flashmla_metadata = None
            core.c128_flashmla_metadata = None

        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> KunlunDSV4AttnMetadata:
        """Build Kunlun core attention metadata for the current request."""
        assert self.swa_page_size == upstream.SWA_WINDOW
        seq_lens_casual = seq_lens_casual.to(torch.int32)
        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )
        # Match the 0.5.8 PyTorch metadata contract: mapping==0 denotes an
        # invalid/tombstoned SWA slot, and valid entries must be a leading
        # prefix. Preserve -1 sentinels for the Kunlun attention operator.
        effective_swa_len = (
            (swa_page_indices != 0)
            .int()
            .cumprod(dim=1)
            .sum(dim=1)
            .to(torch.int32)
        )
        swa_page_indices = upstream._pad_last_dim(
            swa_page_indices, multiples_of=upstream.PAGE_INDEX_ALIGNED_SIZE
        )
        swa_topk_lengths = torch.minimum(
            torch.clamp(seq_lens_casual, max=upstream.SWA_WINDOW),
            effective_swa_len,
        )
        page_table = req_to_token[
            req_pool_indices_repeated, :max_seq_len:self.page_size
        ]
        metadata = KunlunDSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=seq_lens_casual - 1,
            page_table=(page_table // self.page_size).to(torch.int32),
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=torch.clamp(seq_lens_casual, max=upstream.SWA_WINDOW),
            c4_sparse_topk=self.c4_topk,
        )
        # Apply the 0.5.8 effective-length contract after construction as well;
        # this keeps the value correct if upstream metadata initialization resets it.
        metadata.swa_topk_lengths = swa_topk_lengths
        if need_compress:
            metadata.init_compression_metadata()
            metadata.init_flashmla_related(is_prefill=is_prefill)
        else:
            metadata.c4_sparse_topk_lengths = None
            metadata.c4_sparse_page_indices = None
            metadata.c4_sparse_raw_indices = None
            metadata.c1_flashmla_metadata = None
            metadata.c4_flashmla_metadata = None
            metadata.c128_flashmla_metadata = None
        # Golden Prefill keeps -1 as the invalid-tail sentinel. Decode and
        # speculative verify/draft modes use the existing Kunlun page-0
        # normalization contract.
        if not is_prefill:
            for field_name in (
                "swa_page_indices",
                "c4_sparse_page_indices",
                "c128_page_indices",
            ):
                indices = getattr(metadata, field_name, None)
                if indices is not None:
                    indices.masked_fill_(indices == -1, 0)
        return metadata

    @staticmethod
    def _match_queries(tensor, size: int, value: int):
        if tensor is None or tensor.shape[0] == size:
            return tensor
        if tensor.shape[0] > size:
            return tensor[:size]
        return upstream._pad_tensor_to_size(tensor, size, value=value)

    def _make_lod(self, forward_batch, num_queries: int, device: torch.device):
        if forward_batch.forward_mode.is_decode_or_idle():
            batch_size = num_queries
            aux = self._attention_decode_aux.get(batch_size)
            if aux is None:
                q_lod_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
                q_lod = q_lod_cpu.to(device, non_blocking=False)
                kv_lens_cpu = torch.ones(batch_size, dtype=torch.int32)
                kv_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
                aux = q_lod_cpu, q_lod, kv_lens_cpu, kv_lens
                self._attention_decode_aux[batch_size] = aux
            q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = aux
            kv_lens.copy_(forward_batch.seq_lens[:batch_size].to(torch.int32))
            return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

        if _is_graph_extend_mode(forward_batch.forward_mode) and not getattr(
            forward_batch, "_kunlun_ragged_draft_extend", False
        ):
            batch_size = forward_batch.batch_size
            aux = _get_graph_extend_aux(
                self._attention_graph_extend_aux, batch_size, num_queries, device
            )
            q_lod_cpu, q_lod, kv_lens_cpu, kv_lens, query_lens = aux
            _copy_host_lengths_(
                kv_lens_cpu,
                _seq_lens_cpu_i32(forward_batch),
                fill_value=1,
            )
            kv_lens.copy_(forward_batch.seq_lens[:batch_size].to(torch.int32))
            torch.maximum(kv_lens, query_lens, out=kv_lens)
            return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

        lengths = forward_batch.extend_seq_lens_cpu
        if lengths is None:
            batch_size = forward_batch.seq_lens.shape[0]
            if batch_size == 0 or num_queries % batch_size:
                raise ValueError("cannot derive Kunlun DSV4 extend lengths")
            lengths = [num_queries // batch_size] * batch_size
        batch_size = len(lengths)
        q_lod_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        if batch_size:
            torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0, out=q_lod_cpu[1:])
        q_lod = q_lod_cpu.to(device, non_blocking=False)
        kv_lens = forward_batch.seq_lens[:batch_size].to(torch.int32)
        kv_lens_cpu = (
            forward_batch.seq_lens_cpu[:batch_size].to(torch.int32)
            if forward_batch.seq_lens_cpu is not None
            else kv_lens.to("cpu", non_blocking=False)
        )
        return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch
    ) -> None:
        """Use the 0.5.8 half-cache writer for the non-fused DSV4 path."""
        pool = self.token_to_kv_pool
        if (
            envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get()
            or pool.swa_kv_pool is None
            or pool.swa_kv_pool.store_dtype not in (torch.bfloat16, torch.float16)
        ):
            return super().store_cache(layer_id, swa_k, forward_batch)

        scheduler_raw_loc = forward_batch.out_cache_loc
        raw_loc = scheduler_raw_loc.to(dtype=torch.int32).contiguous()
        mapping = pool.full_to_swa_index_mapping
        assert mapping is not None
        swa_pool = pool.swa_kv_pool
        local_layer_id = pool._swa_local_layer_id(layer_id)
        cache = swa_pool.kv_buffer[local_layer_id].reshape(-1, 512)
        pack = dsv4_quant_k_cache_kunlun(swa_k)
        capture_probe = _mtp_writer_probe_enabled(layer_id)
        capture_ifeval_diag = bool(
            os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
            and _dsv4_ifeval_diag_rank() == 0
            and layer_id == 0
        )
        mapped_loc = None
        if capture_probe or capture_ifeval_diag:
            mapped_loc = mapping.index_select(0, raw_loc.long())
        if capture_probe:
            prefix = f"step{self.speculative_step_id}_layer{layer_id}"
            probe = getattr(self, "_mtp_tensor_probe", {})
            probe.update(
                {
                    f"{prefix}.store.scheduler_raw_loc": scheduler_raw_loc,
                    f"{prefix}.store.operator_raw_loc": raw_loc,
                    f"{prefix}.store.mapped_loc": mapped_loc.clone(),
                    f"{prefix}.store.k": pack.k_nope_fp8.clone(),
                }
            )
            self._mtp_tensor_probe = probe
            contracts = getattr(self, "_mtp_tensor_probe_contracts", {})
            contracts.setdefault(prefix, {}).update(
                {
                    "writer_page_size": swa_pool.page_size,
                    "writer_cache_shape": tuple(
                        swa_pool.kv_buffer[local_layer_id].shape
                    ),
                    "writer_cache_dtype": str(
                        swa_pool.kv_buffer[local_layer_id].dtype
                    ),
                }
            )
            self._mtp_tensor_probe_contracts = contracts
        dsv4_set_k_and_s_with_mapping_kunlun(
            swa_pool.kv_buffer[local_layer_id],
            raw_loc,
            mapping,
            pack,
            swa_pool.page_size,
        )
        if capture_probe or capture_ifeval_diag:
            cache_rows = _mtp_gather_cache_rows(cache, mapped_loc).clone()
        if capture_probe:
            probe[f"{prefix}.store.cache_rows"] = cache_rows
        if capture_ifeval_diag:
            packed = pack.k_nope_fp8
            write_matches_readback = bool(
                packed.numel() == cache_rows.numel()
                and torch.equal(packed.reshape(-1), cache_rows.reshape(-1))
            )
            _append_dsv4_ifeval_backend_event(
                self,
                {
                    "kind": "swa_store",
                    "layer_id": layer_id,
                    "write_matches_readback": write_matches_readback,
                    "scheduler_raw_loc": _dsv4_ifeval_tensor_summary(
                        scheduler_raw_loc
                    ),
                    "operator_raw_loc": _dsv4_ifeval_tensor_summary(raw_loc),
                    "mapped_loc": _dsv4_ifeval_tensor_summary(mapped_loc),
                    "packed_k": _dsv4_ifeval_tensor_summary(packed),
                    "cache_rows": _dsv4_ifeval_tensor_summary(cache_rows),
                },
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Run Kunlun attention for the requested compression ratio."""
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)
        assert k is v, "DeepseekV4 shares k and v"
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)

        core = self.forward_metadata.core_attn_metadata
        pool = self.token_to_kv_pool
        cache_dim = pool.swa_kv_pool.kv_cache_total_dim
        swa_size = pool.swa_window_size
        win_cache = pool.get_swa_key_buffer_radix(layer.layer_id)
        win_cache = win_cache.reshape(-1, cache_dim)

        win_indices = self._match_queries(core.swa_page_indices, q.shape[0], 0)
        win_indices = win_indices[:, :swa_size]
        if (
            os.environ.get("DSV4_MTP_PROBE") == "1"
            and os.environ.get("RANK", "0") == "0"
            and getattr(layer, "layer_id", -1) == 0
            and not getattr(self, "_dsv4_attention_probe_logged", False)
        ):
            logger.warning(
                "[DSV4_CALLSTACK] compressed_attention pre-normalization "
                "q_shape=%s q_dtype=%s win_cache_dtype=%s win_indices_shape=%s "
                "win_indices_min=%s win_indices_max=%s win_indices_head=%s",
                tuple(q.shape),
                q.dtype,
                pool.swa_kv_pool.kv_buffer[0].dtype,
                tuple(win_indices.shape),
                int(win_indices.min().item()) if win_indices.numel() else None,
                int(win_indices.max().item()) if win_indices.numel() else None,
                win_indices[0, :16].detach().cpu().tolist()
                if win_indices.numel()
                else [],
            )
            self._dsv4_attention_probe_logged = True
        # Keep the -1 sentinel exactly as in the 0.5.8 XSpeedGate path.
        win_indices = win_indices.contiguous()
        extra_cache = None
        extra_indices = None
        if compress_ratio == 4:
            extra_cache = pool.get_extra_key_buffer(layer.layer_id)
            extra_indices = core.c4_sparse_page_indices
        elif compress_ratio == 128:
            extra_cache = pool.get_extra_key_buffer(layer.layer_id)
            extra_indices = core.c128_page_indices
        extra_indices = self._match_queries(extra_indices, q.shape[0], -1)

        q_3d = q.squeeze(1) if q.ndim == 4 else q
        original_dtype = q_3d.dtype
        if q_3d.dtype != win_cache.dtype:
            q_3d = q_3d.to(win_cache.dtype)
        q_lod_cpu, q_lod, kv_lens_cpu, kv_lens = self._make_lod(
            forward_batch, q_3d.shape[0], q_3d.device
        )
        if (
            os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
            and _dsv4_ifeval_diag_rank() == 0
            and layer.layer_id == 0
        ):
            def sample_page_indices(indices):
                if not isinstance(indices, torch.Tensor) or indices.ndim < 2:
                    return indices
                if indices.shape[-1] <= 8:
                    return indices
                return torch.cat((indices[..., :4], indices[..., -4:]), dim=-1)

            sampled_win_indices = sample_page_indices(win_indices)
            sampled_win_cache_rows = _mtp_gather_cache_rows(
                win_cache, sampled_win_indices
            )
            sampled_extra_indices = sample_page_indices(extra_indices)
            sampled_extra_cache_rows = (
                _mtp_gather_cache_rows(extra_cache, sampled_extra_indices)
                if isinstance(extra_cache, torch.Tensor)
                and isinstance(sampled_extra_indices, torch.Tensor)
                else None
            )
            _append_dsv4_ifeval_backend_event(
                self,
                {
                    "kind": "attention_consume",
                    "layer_id": int(layer.layer_id),
                    "compress_ratio": int(compress_ratio),
                    "input_ids": _dsv4_ifeval_tensor_summary(
                        getattr(forward_batch, "input_ids", None)
                    ),
                    "positions": _dsv4_ifeval_tensor_summary(
                        getattr(forward_batch, "positions", None)
                    ),
                    "seq_lens": _dsv4_ifeval_tensor_summary(
                        getattr(forward_batch, "seq_lens", None)
                    ),
                    "req_pool_indices": _dsv4_ifeval_tensor_summary(
                        getattr(forward_batch, "req_pool_indices", None)
                    ),
                    "scheduler_out_cache_loc": _dsv4_ifeval_tensor_summary(
                        getattr(forward_batch, "out_cache_loc", None)
                    ),
                    "raw_out_loc": _dsv4_ifeval_tensor_summary(
                        getattr(core, "raw_out_loc", None)
                    ),
                    "page_table": _dsv4_ifeval_tensor_summary(
                        getattr(core, "page_table", None)
                    ),
                    "win_indices": _dsv4_ifeval_tensor_summary(win_indices),
                    "sampled_win_indices": _dsv4_ifeval_tensor_summary(
                        sampled_win_indices
                    ),
                    "sampled_win_cache_rows": _dsv4_ifeval_tensor_summary(
                        sampled_win_cache_rows
                    ),
                    "extra_indices": _dsv4_ifeval_tensor_summary(extra_indices),
                    "sampled_extra_indices": _dsv4_ifeval_tensor_summary(
                        sampled_extra_indices
                    ),
                    "sampled_extra_cache_rows": _dsv4_ifeval_tensor_summary(
                        sampled_extra_cache_rows
                    ),
                    "q_lod": _dsv4_ifeval_tensor_summary(q_lod),
                    "kv_lens": _dsv4_ifeval_tensor_summary(kv_lens),
                },
            )
        if (
            os.environ.get("DSV4_C4_ATTN_METADATA_LOG") == "1"
            and forward_batch.batch_size > 1
            and forward_batch.forward_mode.is_extend()
            and layer.layer_id == 0
            and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            )
        ):
            logger.warning(
                "[DSV4_C4_ATTN_METADATA] %s",
                json.dumps(
                    {
                        "backend_file": __file__,
                        "layer_id": layer.layer_id,
                        "forward_mode": str(forward_batch.forward_mode),
                        "batch_size": forward_batch.batch_size,
                        "q": _summarize_c4_metadata_tensor(q_3d),
                        "input_ids": _summarize_c4_metadata_tensor(
                            forward_batch.input_ids
                        ),
                        "positions": _summarize_c4_metadata_tensor(
                            forward_batch.positions
                        ),
                        "req_pool_indices": _summarize_c4_metadata_tensor(
                            forward_batch.req_pool_indices
                        ),
                        "seq_lens": _summarize_c4_metadata_tensor(
                            forward_batch.seq_lens
                        ),
                        "seq_lens_cpu": _summarize_c4_metadata_tensor(
                            forward_batch.seq_lens_cpu
                        ),
                        "extend_seq_lens_cpu": repr(
                            forward_batch.extend_seq_lens_cpu
                        ),
                        "out_cache_loc": _summarize_c4_metadata_tensor(
                            forward_batch.out_cache_loc
                        ),
                        "q_lod_cpu": _summarize_c4_metadata_tensor(q_lod_cpu),
                        "q_lod": _summarize_c4_metadata_tensor(q_lod),
                        "kv_lens_cpu": _summarize_c4_metadata_tensor(
                            kv_lens_cpu
                        ),
                        "kv_lens": _summarize_c4_metadata_tensor(kv_lens),
                        "raw_out_loc": _summarize_c4_metadata_tensor(
                            getattr(core, "raw_out_loc", None)
                        ),
                        "page_table": _summarize_c4_metadata_tensor(
                            getattr(core, "page_table", None)
                        ),
                        "swa_page_indices": _summarize_c4_metadata_tensor(
                            getattr(core, "swa_page_indices", None)
                        ),
                        "c4_sparse_page_indices": _summarize_c4_metadata_tensor(
                            getattr(core, "c4_sparse_page_indices", None)
                        ),
                    },
                    ensure_ascii=True,
                ),
            )
        if extra_cache is None:
            extra_cache = win_cache.new_empty((0, cache_dim))
            extra_indices = torch.empty(
                (q_3d.shape[0], 0), dtype=torch.int32, device=q_3d.device
            )
            compressed_topk = 0
            effective_ratio = 1
        else:
            page_width = pool.page_size // compress_ratio
            extra_cache = extra_cache[:, : page_width * cache_dim].reshape(-1, cache_dim)
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or _is_graph_extend_mode(forward_batch.forward_mode)
            ) and torch.cuda.is_current_stream_capturing():
                # During graph capture (decode, TARGET_VERIFY, DRAFT_EXTEND_V2),
                # kv_lens_cpu holds dummy placeholder values. Use the full static
                # stride so replay can consume the actual indices produced in-graph,
                # matching the 0.5.8 MAX_SEQ_LEN_FOR_CAPTURE behaviour.
                compressed_topk = extra_indices.shape[1]
            else:
                max_seq_len = int(kv_lens_cpu.max().item()) if kv_lens_cpu.numel() else 0
                compressed_topk = min(
                    max(max_seq_len // compress_ratio, 1), extra_indices.shape[1]
                )
                if (
                    compress_ratio == 128
                    and forward_batch.forward_mode.is_extend()
                    and not _is_graph_extend_mode(forward_batch.forward_mode)
                ):
                    compressed_topk = _clamp_c128_prefill_topk(
                        compressed_topk,
                        kv_lens_cpu,
                        forward_batch.extend_seq_lens_cpu,
                        compress_ratio,
                    )
            extra_indices = extra_indices[:, :compressed_topk].clamp(
                min=0, max=extra_cache.shape[0] - 1
            ).contiguous()
            effective_ratio = compress_ratio

        out = torch.zeros_like(q_3d)
        max_logits = torch.zeros(
            (q_3d.shape[0], q_3d.shape[1]),
            dtype=torch.float32,
            device=q_3d.device,
        )
        lse = torch.zeros_like(max_logits)
        if (
            _ENABLE_DSV4_ACCURACY_DUMPS
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1) == 0
            and q_3d.shape[0] == 8192
        ):
            snapshot = dict(getattr(torch, "_dsv4_accuracy_q_stages", {}))
            snapshot.update(
                {
                    "meta.pid": os.getpid(),
                    "meta.layer_id": getattr(layer, "layer_id", None),
                    "meta.backend_type": type(self).__name__,
                    "meta.layer_type": type(layer).__name__,
                    "attention.q_after_dtype_cast": q_3d.detach().cpu(),
                    "attention.win_kv": win_cache.detach().cpu(),
                    "attention.win_indices": win_indices.detach().cpu(),
                    "attention.com_kv": extra_cache.detach().cpu(),
                    "attention.com_indices": extra_indices.detach().cpu(),
                    "attention.qlod_cpu": q_lod_cpu.detach().cpu(),
                    "attention.qlod_xpu": q_lod.detach().cpu(),
                    "attention.kvseqlen_cpu": kv_lens_cpu.detach().cpu(),
                    "attention.kvseqlen_xpu": kv_lens.detach().cpu(),
                    "attention.sm_scale": self.softmax_scale,
                    "attention.max_window_size": win_indices.shape[1],
                    "attention.compress_ratio": effective_ratio,
                    "attention.com_topk": compressed_topk,
                }
            )
            torch.save(
                snapshot,
                f"/home/zx/debug_dumps/full_attention_0514_pid{os.getpid()}.pt",
            )

        if (
            _ENABLE_DSV4_ACCURACY_DUMPS
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1) == 0
            and q_3d.shape[0] == 8192
            and int(forward_batch.positions[0]) == 8192
        ):
            torch.save(
                {
                    "positions": forward_batch.positions.detach().cpu(),
                    "q": q_3d.detach().cpu(),
                    "win_kv": win_cache.detach().cpu(),
                    "win_indices": win_indices.detach().cpu(),
                    "com_kv": extra_cache.detach().cpu(),
                    "com_indices": extra_indices.detach().cpu(),
                    "qlod_cpu": q_lod_cpu.detach().cpu(),
                    "qlod_xpu": q_lod.detach().cpu(),
                    "kvseqlen_cpu": kv_lens_cpu.detach().cpu(),
                    "kvseqlen_xpu": kv_lens.detach().cpu(),
                    "max_logits_before": max_logits.detach().cpu(),
                    "lse_before": lse.detach().cpu(),
                    "softmax_scale": self.softmax_scale,
                    "causal": True,
                    "max_window_size": win_indices.shape[1],
                    "compress_ratio": effective_ratio,
                    "com_topk": compressed_topk,
                    "attn_sink": (
                        attn_sink.detach().cpu() if attn_sink is not None else None
                    ),
                },
                f"/home/zx/debug_dumps/matched_attention_0514_pid{os.getpid()}_obj{id(layer)}.pt",
            )

        dump_matched_output = (
            _ENABLE_DSV4_ACCURACY_DUMPS
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1) == 0
            and q_3d.shape[0] == 8192
            and int(forward_batch.positions[0]) == 8192
        )
        decode_probe_layers = {
            int(layer_id)
            for layer_id in os.environ.get("DSV4_DECODE_PROBE_LAYERS", "2").split(",")
            if layer_id
        }
        decode_probe_layer = getattr(layer, "layer_id", -1)
        capture_decode_probe = (
            os.environ.get("DSV4_DECODE_LAYER_DUMP") == "1"
            and decode_probe_layer in decode_probe_layers
            and (
                q_3d.shape[0] == 1
                or (
                    os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1"
                    and q_3d.shape[0] == 4
                )
            )
        )
        if capture_decode_probe:
            from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

            probe_prefix = f"layer{decode_probe_layer}"
            _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_attn_q"] = q_3d.clone()
            _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_win_indices"] = win_indices.clone()
            _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_com_indices"] = extra_indices.clone()
            _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_kv_lens"] = kv_lens.clone()
            for name, value in (
                ("input_ids", forward_batch.input_ids),
                ("positions", forward_batch.positions),
                ("seq_lens", forward_batch.seq_lens),
                ("req_pool_indices", forward_batch.req_pool_indices),
                ("scheduler_out_cache_loc", forward_batch.out_cache_loc),
            ):
                if isinstance(value, torch.Tensor):
                    _DSV4_DECODE_LAYER_OUTPUTS[
                        f"{probe_prefix}_cache_{name}"
                    ] = value.clone()
            for name in (
                "raw_out_loc",
                "swa_out_cache_loc",
                "c4_out_loc",
                "page_table",
                "swa_page_indices",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
            ):
                value = getattr(core, name, None)
                if isinstance(value, torch.Tensor):
                    _DSV4_DECODE_LAYER_OUTPUTS[
                        f"{probe_prefix}_cache_{name}"
                    ] = value.clone()
        debug_layer42 = (
            _ENABLE_DSV4_ACCURACY_DUMPS
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1) == 42
            and q_3d.shape[0] == 8192
            and int(forward_batch.positions[0]) == 24576
        )
        debug_layer42_summary = None
        if debug_layer42:
            debug_layer42_summary = {
                "q_nan": int(torch.isnan(q_3d).sum().item()),
                "win_cache_nan": int(torch.isnan(win_cache).sum().item()),
                "extra_cache_nan": int(torch.isnan(extra_cache).sum().item()),
                "win_indices_min": int(win_indices.min().item()),
                "win_indices_max": int(win_indices.max().item()),
                "extra_indices_min": int(extra_indices.min().item()),
                "extra_indices_max": int(extra_indices.max().item()),
                "kv_lens": kv_lens_cpu.tolist(),
            }

        q_op = q_3d.contiguous()
        win_cache_op = win_cache.contiguous()
        win_indices_op = win_indices.contiguous()
        extra_cache_op = extra_cache.contiguous()
        extra_indices_op = extra_indices.contiguous()
        out_op = out.contiguous()
        max_logits_op = max_logits.contiguous()
        lse_op = lse.contiguous()
        q_lod_cpu_op = q_lod_cpu.contiguous()
        q_lod_op = q_lod.contiguous()
        kv_lens_cpu_op = kv_lens_cpu.contiguous()
        kv_lens_op = kv_lens.contiguous()
        attn_sink_op = attn_sink.contiguous() if attn_sink is not None else None
        attention_aliases = getattr(
            forward_batch, "_dsv4_decode_attention_aliases", None
        )
        alias_layer = int(os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_LAYER", "2"))
        if attention_aliases is not None and layer.layer_id == alias_layer:
            operator_only_alias = (
                os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_OPERATOR_ONLY") == "1"
            )
            attention_aliases.update(
                {
                    "attention_q": q_op.clone() if operator_only_alias else q_op,
                    "attention_win_cache": win_cache_op,
                    "attention_win_indices": (
                        win_indices_op.clone() if operator_only_alias else win_indices_op
                    ),
                    "attention_extra_cache": extra_cache_op,
                    "attention_extra_indices": (
                        extra_indices_op.clone()
                        if operator_only_alias
                        else extra_indices_op
                    ),
                    "attention_q_lod_cpu": q_lod_cpu_op,
                    "attention_q_lod": (
                        q_lod_op.clone() if operator_only_alias else q_lod_op
                    ),
                    "attention_kv_lens_cpu": kv_lens_cpu_op,
                    "attention_kv_lens": (
                        kv_lens_op.clone() if operator_only_alias else kv_lens_op
                    ),
                    "attention_softmax_scale": torch.tensor(
                        self.softmax_scale, dtype=torch.float64
                    ),
                    "attention_causal": torch.tensor(True),
                    "attention_max_window_size": torch.tensor(
                        win_indices_op.shape[1], dtype=torch.int64
                    ),
                    "attention_compress_ratio": torch.tensor(
                        effective_ratio, dtype=torch.int64
                    ),
                    "attention_compressed_topk": torch.tensor(
                        compressed_topk, dtype=torch.int64
                    ),
                }
            )
            if attn_sink_op is not None:
                attention_aliases["attention_sink"] = attn_sink_op
            if not operator_only_alias:
                for name in (
                    "raw_out_loc",
                    "swa_out_cache_loc",
                    "c4_out_loc",
                    "page_table",
                    "swa_page_indices",
                    "swa_topk_lengths",
                    "c4_sparse_topk_lengths",
                ):
                    value = getattr(core, name, None)
                    if isinstance(value, torch.Tensor):
                        attention_aliases[f"metadata_{name}"] = value
                attention_aliases.update(
                    {
                        "metadata_req_pool_indices": forward_batch.req_pool_indices,
                        "metadata_req_to_token": self.req_to_token,
                    }
                )
        capture_mtp_probe = (
            forward_batch.forward_mode.is_extend()
            and not torch.cuda.is_current_stream_capturing()
            and _mtp_tensor_probe_enabled(
                getattr(layer, "layer_id", -1), q_op.shape[0]
            )
        )
        if capture_mtp_probe:
            prefix = f"step{self.speculative_step_id}_layer{layer.layer_id}"
            probe = getattr(self, "_mtp_tensor_probe", {})
            probe.update(
                {
                    f"{prefix}.metadata.input_ids": forward_batch.input_ids.clone(),
                    f"{prefix}.metadata.positions": forward_batch.positions.clone(),
                    f"{prefix}.metadata.seq_lens": forward_batch.seq_lens.clone(),
                    f"{prefix}.metadata.req_pool_indices": forward_batch.req_pool_indices.clone(),
                    f"{prefix}.metadata.scheduler_out_cache_loc": forward_batch.out_cache_loc.clone(),
                    f"{prefix}.metadata.raw_out_loc": core.raw_out_loc,
                    f"{prefix}.metadata.seq_lens_casual": core.seq_lens_casual,
                    f"{prefix}.metadata.positions_casual": core.positions_casual,
                    f"{prefix}.metadata.page_table": core.page_table,
                    f"{prefix}.metadata.swa_page_indices": core.swa_page_indices,
                    f"{prefix}.metadata.swa_topk_lengths": core.swa_topk_lengths,
                    f"{prefix}.metadata.effective_swa_len": (
                        (core.swa_page_indices != 0)
                        .to(torch.int32)
                        .cumprod(dim=-1)
                        .sum(dim=-1)
                        .clone()
                    ),
                    f"{prefix}.attention.q": q_op.clone(),
                    f"{prefix}.attention.win_indices": win_indices_op.clone(),
                    f"{prefix}.attention.win_cache_rows": _mtp_gather_cache_rows(
                        win_cache_op, win_indices_op
                    ).clone(),
                    f"{prefix}.attention.extra_indices": extra_indices_op.clone(),
                    f"{prefix}.attention.extra_cache_rows": _mtp_gather_cache_rows(
                        extra_cache_op, extra_indices_op
                    ).clone(),
                    f"{prefix}.attention.q_lod_cpu": q_lod_cpu_op,
                    f"{prefix}.attention.q_lod": q_lod_op.clone(),
                    f"{prefix}.attention.kv_lens_cpu": kv_lens_cpu_op,
                    f"{prefix}.attention.kv_lens": kv_lens_op.clone(),
                }
            )
            for name in (
                "swa_out_cache_loc",
                "c4_out_loc",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
                "c128_out_loc",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
            ):
                tensor = getattr(core, name, None)
                if isinstance(tensor, torch.Tensor):
                    probe[f"{prefix}.metadata.{name}"] = tensor
            indexer_metadata = getattr(self.forward_metadata, "indexer_metadata", None)
            if indexer_metadata is not None:
                for name in (
                    "c4_seq_lens",
                    "page_table",
                    "topk_metadata",
                    "deep_gemm_metadata",
                ):
                    tensor = getattr(indexer_metadata, name, None)
                    if isinstance(tensor, torch.Tensor):
                        probe[f"{prefix}.metadata.indexer_{name}"] = tensor
            if attn_sink_op is not None:
                probe[f"{prefix}.attention.attn_sink"] = attn_sink_op
            self._mtp_tensor_probe = probe
            contracts = getattr(self, "_mtp_tensor_probe_contracts", {})
            contracts.setdefault(prefix, {}).update(
                {
                    "softmax_scale": self.softmax_scale,
                    "causal": True,
                    "max_window_size": win_indices_op.shape[1],
                    "compress_ratio": effective_ratio,
                    "compressed_topk": compressed_topk,
                    "page_size": self.page_size,
                }
            )
            self._mtp_tensor_probe_contracts = contracts

        capture_prefill_backend_probe = (
            os.environ.get("DSV4_PREFILL_BACKEND_PROBE") == "1"
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1)
            == int(os.environ.get("DSV4_PREFILL_PROBE_LAYER", "0"))
            and q_op.shape[0] == 8192
            and int(forward_batch.positions[0].item()) == 8192
        )
        prefill_backend_probe = None
        if capture_prefill_backend_probe:
            tail = slice(7680, 8192)
            tail_win_indices = win_indices_op[tail]
            prefill_backend_probe = {
                "positions": forward_batch.positions.detach().cpu(),
                "q_tail": q_op[tail].detach().cpu(),
                "win_indices_tail": tail_win_indices.detach().cpu(),
                "win_cache_rows_tail": _mtp_gather_cache_rows(
                    win_cache_op, tail_win_indices
                ).detach().cpu(),
                "q_lod_cpu": q_lod_cpu_op.detach().cpu(),
                "q_lod": q_lod_op.detach().cpu(),
                "kv_lens_cpu": kv_lens_cpu_op.detach().cpu(),
                "kv_lens": kv_lens_op.detach().cpu(),
                "attn_sink": (
                    attn_sink_op.detach().cpu() if attn_sink_op is not None else None
                ),
                "swa_out_cache_loc": (
                    getattr(core, "swa_out_cache_loc").detach().cpu()
                    if isinstance(getattr(core, "swa_out_cache_loc", None), torch.Tensor)
                    else None
                ),
                "raw_out_loc": (
                    getattr(core, "raw_out_loc").detach().cpu()
                    if isinstance(getattr(core, "raw_out_loc", None), torch.Tensor)
                    else None
                ),
                "win_cache_shape": tuple(win_cache_op.shape),
                "win_cache_dtype": str(win_cache_op.dtype),
                "win_indices_shape": tuple(win_indices_op.shape),
                "softmax_scale": self.softmax_scale,
                "causal": True,
                "max_window_size": win_indices_op.shape[1],
                "compress_ratio": effective_ratio,
                "compressed_topk": compressed_topk,
                "page_size": self.page_size,
            }
            logger.warning(
                "DSV4_COMPRESSED_ATTN_STACK version=0514 backend=%s layer=%s "
                "q_shape=%s win_cache_shape=%s win_indices_shape=%s "
                "win_cache_ptr=%s side_stream=%s",
                f"{type(self).__module__}.{type(self).__qualname__}",
                getattr(layer, "layer_id", -1),
                tuple(q_op.shape),
                tuple(win_cache_op.shape),
                tuple(win_indices_op.shape),
                win_cache_op.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
                stack_info=True,
            )
        prefill_backend_probe_prefix_len = int(
            os.environ.get("DSV4_PREFILL_PROBE_PREFIX_LEN", "8192")
        )
        capture_prefill_backend_device_probe = (
            os.environ.get("DSV4_PREFILL_BACKEND_DEVICE_PROBE") == "1"
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and getattr(layer, "layer_id", -1)
            == int(os.environ.get("DSV4_PREFILL_PROBE_LAYER", "0"))
            and q_op.shape[0] == 8192
            and prefill_backend_probe_prefix_len
            in (forward_batch.extend_prefix_lens_cpu or [])
        )
        from debug.dsv4_probe_bridge import (
            dump_compressed_attention_inputs,
            dump_compressed_attention_outputs,
        )

        dump_compressed_attention_inputs(
            self,
            q_op=q_op,
            win_cache_op=win_cache_op,
            win_indices_op=win_indices_op,
            extra_cache_op=extra_cache_op,
            extra_indices_op=extra_indices_op,
            q_lod_cpu_op=q_lod_cpu_op,
            q_lod_op=q_lod_op,
            kv_lens_cpu_op=kv_lens_cpu_op,
            kv_lens_op=kv_lens_op,
            attn_sink_op=attn_sink_op,
            softmax_scale=self.softmax_scale,
            causal=True,
            effective_ratio=effective_ratio,
            compressed_topk=compressed_topk,
        )
        torch.ops.xspeedgate_ops.compressed_attention(
            q_op,
            win_cache_op,
            win_indices_op,
            extra_cache_op,
            extra_indices_op,
            out_op,
            max_logits_op,
            lse_op,
            q_lod_cpu_op,
            q_lod_op,
            kv_lens_cpu_op,
            kv_lens_op,
            self.softmax_scale,
            True,
            win_indices_op.shape[1],
            effective_ratio,
            compressed_topk,
            attn_sink_op,
            # Keep the C4 producer on the caller stream so PyTorch owns the
            # lifetime of the contiguous temporary inputs.
            side_stream=torch.cuda.current_stream().cuda_stream,
        )
        dump_compressed_attention_outputs(
            self,
            out_op=out_op,
            max_logits_op=max_logits_op,
            lse_op=lse_op,
        )
        if attention_aliases is not None and layer.layer_id == alias_layer:
            attention_aliases.update(
                {
                    "attention_output": (
                        out_op.clone() if operator_only_alias else out_op
                    ),
                    "attention_max_logits": (
                        max_logits_op.clone()
                        if operator_only_alias
                        else max_logits_op
                    ),
                    "attention_lse": (
                        lse_op.clone() if operator_only_alias else lse_op
                    ),
                }
            )
        if (
            capture_decode_probe
            and os.environ.get("DSV4_DECODE_PROBE_CACHE_ROWS") == "1"
            and forward_batch.seq_lens_cpu is not None
            and int(forward_batch.seq_lens_cpu[0])
            >= int(
                os.environ.get(
                    "DSV4_DECODE_PROBE_CACHE_ROWS_MIN_SEQ_LEN", "0"
                )
            )
        ):
            _DSV4_DECODE_LAYER_OUTPUTS[
                f"{probe_prefix}_win_cache_rows"
            ] = _mtp_gather_cache_rows(win_cache_op, win_indices_op)
            _DSV4_DECODE_LAYER_OUTPUTS[
                f"{probe_prefix}_com_cache_rows"
            ] = _mtp_gather_cache_rows(extra_cache_op, extra_indices_op)
        if capture_prefill_backend_device_probe:
            extra_unique_indices = torch.unique(extra_indices_op)
            win_unique_indices = torch.unique(win_indices_op)
            prefill_backend_device_probe = {
                "input_ids": forward_batch.input_ids.clone(),
                "positions": forward_batch.positions.clone(),
                "q_local": q_op[:, :8].clone(),
                "win_indices": win_indices_op.clone(),
                "win_unique_indices": win_unique_indices,
                "win_unique_cache_rows": win_cache_op.index_select(
                    0, win_unique_indices.to(torch.long)
                ).clone(),
                "extra_indices": extra_indices_op.clone(),
                "extra_unique_indices": extra_unique_indices,
                "extra_unique_cache_rows": extra_cache_op.index_select(
                    0, extra_unique_indices.to(torch.long)
                ).clone(),
                "output_local": out_op[:, :8].clone(),
                "max_logits_local": max_logits_op[:, :8].clone(),
                "lse_local": lse_op[:, :8].clone(),
                "q_lod_cpu": q_lod_cpu_op.clone(),
                "q_lod": q_lod_op.clone(),
                "kv_lens_cpu": kv_lens_cpu_op.clone(),
                "kv_lens": kv_lens_op.clone(),
                "attn_sink": (
                    attn_sink_op.clone() if attn_sink_op is not None else None
                ),
                "win_cache_shape": tuple(win_cache_op.shape),
                "extra_cache_shape": tuple(extra_cache_op.shape),
                "softmax_scale": self.softmax_scale,
                "causal": True,
                "max_window_size": win_indices_op.shape[1],
                "compress_ratio": effective_ratio,
                "compressed_topk": compressed_topk,
                "page_size": self.page_size,
                "side_stream": torch.cuda.current_stream().cuda_stream,
            }
            for name in (
                "raw_out_loc",
                "swa_out_cache_loc",
                "c128_out_loc",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
            ):
                tensor = getattr(core, name, None)
                if isinstance(tensor, torch.Tensor):
                    prefill_backend_device_probe[name] = tensor.clone()
            logger.warning(
                "DSV4_C128_PREFILL_STACK version=0514 backend=%s layer=%s "
                "q_shape=%s win_indices_shape=%s extra_indices_shape=%s "
                "win_cache_shape=%s extra_cache_shape=%s side_stream=%s",
                f"{type(self).__module__}.{type(self).__qualname__}",
                getattr(layer, "layer_id", -1),
                tuple(q_op.shape),
                tuple(win_indices_op.shape),
                tuple(extra_indices_op.shape),
                tuple(win_cache_op.shape),
                tuple(extra_cache_op.shape),
                torch.cuda.current_stream().cuda_stream,
                stack_info=True,
            )
            torch.save(
                {
                    key: value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in prefill_backend_device_probe.items()
                },
                (
                    "/home/zx/debug_dumps/prefill_backend_device_0514_layer"
                    f"{getattr(layer, 'layer_id', -1)}_prefix"
                    f"{prefill_backend_probe_prefix_len}_rank0.pt"
                ),
            )
        if capture_prefill_backend_probe:
            tail = slice(7680, 8192)
            prefill_backend_probe.update(
                {
                    "output_tail": out_op[tail].detach().cpu(),
                    "max_logits_tail": max_logits_op[tail].detach().cpu(),
                    "lse_tail": lse_op[tail].detach().cpu(),
                }
            )
            torch.save(
                prefill_backend_probe,
                "/home/zx/debug_dumps/prefill_backend_0514_repeat_layer0_chunk1_rank0.pt",
            )
        if capture_mtp_probe:
            probe[f"{prefix}.attention.output"] = out_op.clone()
            probe[f"{prefix}.attention.max_logits"] = max_logits_op.clone()
            probe[f"{prefix}.attention.lse"] = lse_op.clone()
            _dump_eager_mtp_tensor_probe(self)
        if capture_decode_probe:
            _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_attn_output"] = out.clone()
        if dump_matched_output:
            torch.save(
                out.detach().cpu(),
                f"/home/zx/debug_dumps/matched_attention_out_0514_pid{os.getpid()}_obj{id(layer)}.pt",
            )
        if debug_layer42:
            debug_layer42_summary["out_nan"] = int(torch.isnan(out).sum().item())
            torch.save(
                debug_layer42_summary,
                f"/home/zx/debug_dumps/attention_layer42_summary_0514_pid{os.getpid()}.pt",
            )

        return out.to(original_dtype) if out.dtype != original_dtype else out


def _compressor_forward_cuda_kunlun(self, x, forward_batch, attn_backend=None):
    """Restore the 0.5.8 Compressor.forward path for Kunlun's CUDA dispatch key."""

    return self.forward_native(x, forward_batch, attn_backend=attn_backend)


# Compressor inherits MultiPlatformOp but 0.5.14 does not define forward_cuda.
# Kunlun advertises the CUDA dispatch key, so bind the working native control
# flow explicitly before any model instance resolves its forward method.
from sglang.srt.layers.attention.dsv4.compressor import Compressor

Compressor.forward_cuda = _compressor_forward_cuda_kunlun


def _build_c4_prefill_contract(forward_batch, c4_seq_lens, page_table, device):
    """Build the request-level LoD contract used by the 0.5.8 extend path."""
    extend_lens = [int(value) for value in forward_batch.extend_seq_lens_cpu]
    qlod_cpu = torch.zeros(len(extend_lens) + 1, dtype=torch.int32)
    if extend_lens:
        qlod_cpu[1:] = torch.cumsum(torch.tensor(extend_lens, dtype=torch.int32), 0)
    last_rows = (qlod_cpu[1:] - 1).clamp(min=0, max=page_table.shape[0] - 1)
    c4_flat = c4_seq_lens.reshape(-1).to("cpu", dtype=torch.int32)
    per_req_k_lens = c4_flat.index_select(
        0, last_rows.clamp(max=c4_flat.numel() - 1).long()
    )
    klod_cpu = torch.zeros(len(extend_lens) + 1, dtype=torch.int32)
    if extend_lens:
        klod_cpu[1:] = torch.cumsum(per_req_k_lens * 4, 0)
    com_k_start_cpu = torch.zeros(len(extend_lens), dtype=torch.int32)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    return {
        "qlod_cpu": qlod_cpu,
        "qlod_xpu": qlod_cpu.to(device),
        "klod_cpu": klod_cpu,
        "klod_xpu": klod_cpu.to(device),
        "com_k_start_cpu": com_k_start_cpu,
        "com_k_start_xpu": com_k_start_cpu.to(device),
        "per_req_k_lens": per_req_k_lens,
        "last_rows": last_rows,
        "max_seq_k": int(per_req_k_lens.max().item()) * 4 if per_req_k_lens.numel() else 0,
        "max_seq_k_compressed": int(per_req_k_lens.max().item()) if per_req_k_lens.numel() else 0,
        "use_causal": prefix_lens is not None and all(int(value) < 3 for value in prefix_lens),
    }


def _gather_c4_prefill_kv(cache, page_table, contract, device):
    """Gather packed C4 pages exactly as the 0.5.8 extend path."""
    block_size = cache.shape[1]
    head_dim = cache.shape[-1] - 4
    cache_u8 = cache.contiguous().view(torch.uint8).reshape(
        cache.shape[0], block_size * (head_dim + 4)
    )
    k_pages = cache_u8[:, : block_size * head_dim].reshape(
        -1, block_size, head_dim
    )
    scale_pages = cache_u8[:, block_size * head_dim :].contiguous().view(
        torch.float32
    ).reshape(-1, block_size)
    lengths = contract["per_req_k_lens"]
    total_k = int(lengths.sum().item())
    k_contiguous = torch.empty((total_k, head_dim), dtype=torch.int8, device=device)
    k_scale_contiguous = torch.empty(total_k, dtype=torch.float32, device=device)
    offset = 0
    for request_index, k_len_value in enumerate(lengths.tolist()):
        k_len = int(k_len_value)
        if not k_len:
            continue
        num_pages = (k_len + block_size - 1) // block_size
        page_row = int(contract["last_rows"][request_index].item())
        pages = page_table[page_row, :num_pages].long()
        k_contiguous[offset : offset + k_len] = k_pages[pages].reshape(
            -1, head_dim
        )[:k_len].to(torch.int8)
        k_scale_contiguous[offset : offset + k_len] = scale_pages[pages].reshape(
            -1
        )[:k_len]
        offset += k_len
    return k_contiguous, k_scale_contiguous


def _select_c4_target_verify_rows(
    q_fp8, weight, seq_lens, page_table, num_requests
):
    """Match the 0.5.8 C4 TARGET_VERIFY representative-row contract."""
    num_queries = q_fp8.shape[0]
    if num_requests <= 0 or num_queries % num_requests:
        raise ValueError(
            "Kunlun C4 TARGET_VERIFY requires uniform queries per request: "
            f"num_requests={num_requests}, num_queries={num_queries}"
        )

    num_queries_per_request = num_queries // num_requests
    request_rows = torch.arange(
        num_requests, dtype=torch.long, device=q_fp8.device
    )
    page_table_is_per_request = page_table.shape[0] == num_requests
    if page_table_is_per_request:
        representative_rows = request_rows
    else:
        representative_rows = (
            request_rows * num_queries_per_request + num_queries_per_request - 1
        )

    q_fp8 = q_fp8.index_select(0, representative_rows)
    weight = weight.index_select(0, representative_rows)

    seq_lens = seq_lens.reshape(-1)
    if page_table_is_per_request:
        seq_lens = seq_lens[:num_requests]
    else:
        seq_lens = seq_lens.index_select(0, representative_rows)
        page_table = page_table.index_select(0, representative_rows)
    return q_fp8, weight, seq_lens, page_table, num_queries_per_request


def _compute_c4_logits_kunlun(
    *,
    backend,
    q_fp8,
    kvcache_fp8,
    weight,
    seq_lens,
    page_table,
    max_seq_len,
    forward_batch,
    c4_indexer,
):
    """Use the 0.5.8 Kunlun C4 operator with explicit batch ownership."""

    import kunlun_ops

    layer_id = c4_indexer.layer_id
    capture_mtp_probe = (
        forward_batch.forward_mode.is_extend()
        and not torch.cuda.is_current_stream_capturing()
        and _mtp_tensor_probe_enabled(layer_id, q_fp8.shape[0])
    )
    probe_prefix = f"step{backend.speculative_step_id}_layer{layer_id}"

    use_contiguous_prefill = (
        forward_batch is not None
        and forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.extend_seq_lens_cpu is not None
    )
    if use_contiguous_prefill:
        q = q_fp8.squeeze(1).contiguous()
        if q.dtype == torch.uint8:
            q = q.view(torch.int8)
        weights = weight.float().contiguous()
        contract = _build_c4_prefill_contract(
            forward_batch, seq_lens, page_table, q.device
        )
        k, k_scale = _gather_c4_prefill_kv(
            kvcache_fp8, page_table, contract, q.device
        )
        logits = torch.empty(
            (q.shape[0], contract["max_seq_k_compressed"]),
            dtype=torch.float32,
            device=q.device,
        )
        if k.shape[0]:
            kunlun_ops.c4a_mqa_logits(
                q=q,
                weight=weights,
                k=k,
                k_scale=k_scale,
                logits=logits,
                max_seq_q=max(int(value) for value in forward_batch.extend_seq_lens_cpu),
                max_seq_k=contract["max_seq_k"],
                qlod_cpu=contract["qlod_cpu"],
                qlod_xpu=contract["qlod_xpu"],
                klod_cpu=contract["klod_cpu"],
                klod_xpu=contract["klod_xpu"],
                com_k_start_cpu=contract["com_k_start_cpu"],
                com_k_start_xpu=contract["com_k_start_xpu"],
                is_causal=contract["use_causal"],
                compress_ratio=4,
                clean_logits=True,
            )
        else:
            logits.zero_()
        if logits.shape[1] < max_seq_len:
            padded = torch.full(
                (q.shape[0], max_seq_len),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            padded[:, : logits.shape[1]] = logits
            logits = padded
        dump_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
        dump_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
        if dump_dir and (not dump_tokens or q.shape[0] == dump_tokens):
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            if rank == int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
                sample_rows = torch.tensor(
                    [0, logits.shape[0] // 2, logits.shape[0] - 1],
                    dtype=torch.long,
                    device=logits.device,
                )
                os.makedirs(dump_dir, exist_ok=True)
                path = os.path.join(dump_dir, f"rank{rank}_c4_logits_{q.shape[0]}.pt")
                if not os.path.exists(path):
                    torch.save(
                        {
                            "sample_rows": sample_rows.detach().cpu(),
                            "logits": logits.index_select(0, sample_rows).detach().cpu(),
                            "max_seq_len": max_seq_len,
                        },
                        path,
                    )
        return logits

    num_queries_per_request = 1
    is_target_verify = (
        forward_batch is not None
        and forward_batch.forward_mode.is_target_verify()
    )
    if is_target_verify:
        q_fp8, weight, seq_lens, page_table, num_queries_per_request = (
            _select_c4_target_verify_rows(
                q_fp8,
                weight,
                seq_lens,
                page_table,
                forward_batch.batch_size,
            )
        )

    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    seq_lens = seq_lens.view(-1)[:batch_size]
    q = q_fp8.view(torch.int8) if q_fp8.dtype == torch.uint8 else q_fp8
    weights = weight.unsqueeze(1)

    cache_flat = kvcache_fp8.view(
        kvcache_fp8.shape[0], block_size * (head_dim + 4)
    )
    scale_offset = block_size * head_dim
    k_cache = (
        cache_flat[:, :scale_offset]
        .contiguous()
        .view(-1, block_size, 1, head_dim)
        .view(torch.int8)
    )
    num_pages = min(
        (max_seq_len + block_size - 1) // block_size,
        page_table.shape[1],
    )
    block_table = page_table[:, :num_pages].to(torch.int32)
    scale_pages = (
        cache_flat[:, scale_offset:]
        .contiguous()
        .view(torch.float32)
        .view(-1, block_size)
    )
    page_indices = block_table.to(torch.long)
    k_scale = scale_pages[page_indices].reshape(batch_size, -1)[
        :, :max_seq_len
    ].contiguous()

    key = (batch_size, q_fp8.device, max_seq_len)
    aux = backend._c4_decode_aux.get(key)
    if aux is None:
        qlod_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
        qlod_xpu = qlod_cpu.to(q_fp8.device)
        context_lens_cpu = torch.full(
            (batch_size,), 4, dtype=torch.int32
        )
        context_lens_xpu = torch.full(
            (batch_size,), 4, dtype=torch.int32, device=q_fp8.device
        )
        aux = qlod_cpu, qlod_xpu, context_lens_cpu, context_lens_xpu
        backend._c4_decode_aux[key] = aux
    qlod_cpu, qlod_xpu, context_lens_cpu, context_lens_xpu = aux
    raw_seq_lens_cpu = _seq_lens_cpu_i32(forward_batch)
    _copy_host_lengths_(
        context_lens_cpu,
        (raw_seq_lens_cpu // 4) * 4,
        fill_value=4,
    )
    context_lens_xpu.copy_(seq_lens.to(torch.int32) * 4)

    logits = torch.empty(
        (batch_size, 1, max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    layer_id = c4_indexer.layer_id
    attention_aliases = getattr(
        forward_batch, "_dsv4_decode_attention_aliases", None
    )
    alias_layer = int(
        os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_LAYER", "2")
    )
    if attention_aliases is not None and layer_id == alias_layer:
        attention_aliases.update(
            {
                "indexer_operator_q": q,
                "indexer_operator_weight": weights,
                "indexer_operator_seq_lens": seq_lens,
                "indexer_operator_block_table": block_table,
                "indexer_operator_k_cache": k_cache,
                "indexer_operator_k_scale": k_scale,
                "indexer_operator_q_lod_cpu": qlod_cpu,
                "indexer_operator_q_lod": qlod_xpu,
                "indexer_operator_context_lens_cpu": context_lens_cpu,
                "indexer_operator_context_lens": context_lens_xpu,
                "indexer_operator_logits": logits,
            }
        )
    capture_verify_probe = (
        os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1"
        and is_target_verify
    )
    if capture_verify_probe:
        from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

        probe_pages = min(140, block_table.shape[1])
        selected_k_cache = k_cache.index_select(
            0,
            block_table[:, :probe_pages]
            .reshape(-1)
            .long()
            .clamp(0, k_cache.shape[0] - 1),
        ).view(batch_size, probe_pages, block_size, 1, head_dim)
        prefix = "layer2_indexer_operator"
        _DSV4_DECODE_LAYER_OUTPUTS.update(
            {
                f"{prefix}_q": q.clone(),
                f"{prefix}_weight": weights.clone(),
                f"{prefix}_seq_lens": seq_lens.clone(),
                f"{prefix}_block_table": block_table.clone(),
                f"{prefix}_k_cache_pages": selected_k_cache.clone(),
                f"{prefix}_k_scale": k_scale[:, :9000].clone(),
                f"{prefix}_q_lod_cpu": qlod_cpu.clone(),
                f"{prefix}_q_lod": qlod_xpu.clone(),
                f"{prefix}_context_lens_cpu": context_lens_cpu.clone(),
                f"{prefix}_context_lens": context_lens_xpu.clone(),
            }
        )

    if capture_mtp_probe:
        probe = getattr(backend, "_mtp_tensor_probe", {})
        selected_k_cache = k_cache.index_select(
            0, block_table.reshape(-1).long().clamp(0, k_cache.shape[0] - 1)
        )
        probe.update(
            {
                f"{probe_prefix}.indexer.operator.q": q.clone(),
                f"{probe_prefix}.indexer.operator.weight": weights.clone(),
                f"{probe_prefix}.indexer.operator.seq_lens": seq_lens.clone(),
                f"{probe_prefix}.indexer.operator.page_table": page_table.clone(),
                f"{probe_prefix}.indexer.operator.block_table": block_table.clone(),
                f"{probe_prefix}.indexer.operator.k_cache_pages": selected_k_cache.clone(),
                f"{probe_prefix}.indexer.operator.k_scale": k_scale.clone(),
                f"{probe_prefix}.indexer.operator.q_lod_cpu": qlod_cpu,
                f"{probe_prefix}.indexer.operator.q_lod": qlod_xpu.clone(),
                f"{probe_prefix}.indexer.operator.context_lens_cpu": context_lens_cpu,
                f"{probe_prefix}.indexer.operator.context_lens": context_lens_xpu.clone(),
            }
        )
        backend._mtp_tensor_probe = probe
        contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
        contracts.setdefault(probe_prefix, {}).update(
            {
                "indexer_max_context_len": max_seq_len * 4,
                "indexer_compress_ratio": 4,
                "indexer_clean_logits": True,
                "indexer_use_xfa_boost": False,
            }
        )
        backend._mtp_tensor_probe_contracts = contracts
    kunlun_ops.c4a_paged_mqa_logits(
        q=q,
        weight=weights,
        k_cache=k_cache,
        k_scale=k_scale,
        logits=logits,
        max_context_len=max_seq_len * 4,
        qlod_cpu=qlod_cpu,
        qlod_xpu=qlod_xpu,
        context_lens_cpu=context_lens_cpu,
        context_lens_xpu=context_lens_xpu,
        block_table=block_table,
        compress_ratio=4,
        clean_logits=True,
        use_xfa_boost=False,
    )
    if capture_mtp_probe:
        probe[f"{probe_prefix}.indexer.operator.logits"] = logits.clone()
    logits = logits.squeeze(1)
    if is_target_verify:
        logits = (
            logits.unsqueeze(1)
            .expand(-1, num_queries_per_request, -1)
            .reshape(-1, max_seq_len)
            .contiguous()
        )
    return logits


class KunlunDeepseekV4MultiStepBackend(DeepseekV4MultiStepBackend):
    """0.5.14 multi-step control flow composed only of Kunlun backends."""

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        DeepseekV4AttnBackend.__init__(self, model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[KunlunDeepseekV4AttnBackend] = [
            KunlunDeepseekV4AttnBackend(
                model_runner,
                speculative_step_id=i,
                topk=topk,
                speculative_num_steps=speculative_num_steps,
            )
            for i in range(speculative_num_steps)
        ]
        self._mtp_probe_replay_calls = 0

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        # Seed replay from step 0, then copy its live metadata into every
        # later draft step that participates in this captured decode.
        from types import SimpleNamespace

        seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None and out_cache_loc.dtype != torch.int32:
            out_cache_loc = out_cache_loc.to(torch.int32)
        inner_fb = SimpleNamespace(
            batch_size=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            actual_forward_mode=getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            ),
            input_ids=getattr(forward_batch, "input_ids", None),
            positions=getattr(forward_batch, "positions", None),
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=seq_lens_i32,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=out_cache_loc,
            spec_info=forward_batch.spec_info,
        )
        if in_capture:
            for backend in self.attn_backends:
                backend.init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
        elif self.speculative_num_steps > 1:
            self.attn_backends[0].init_forward_metadata_out_graph(inner_fb)
            temp_metadata = self.attn_backends[0].forward_metadata
            for i in range(1, self.speculative_num_steps - 1):
                self.attn_backends[i].replay_cuda_graph_metadata_from(
                    bs=forward_batch.batch_size,
                    temp_metadata=temp_metadata,
                    bucket=upstream._GraphBucket.DECODE_OR_IDLE,
                )
        if not in_capture:
            _dump_previous_mtp_tensor_probe(self, forward_batch)
        if (
            in_capture
            or os.environ.get("DSV4_MTP_PROBE") != "1"
            or os.environ.get("RANK", "0") != "0"
            or forward_batch.seq_lens_cpu is None
            or int(forward_batch.seq_lens_cpu[: forward_batch.batch_size].max()) <= 10000
            or self._mtp_probe_replay_calls >= 3
        ):
            return

        self._mtp_probe_replay_calls += 1

        def _values(tensor, limit=8):
            if tensor is None:
                return None
            return tensor.detach().reshape(-1)[:limit].cpu().tolist()

        def _ptr(tensor):
            return tensor.data_ptr() if tensor is not None else None

        for step, backend in enumerate(self.attn_backends[: self.speculative_num_steps - 1]):
            metadata = backend.forward_metadata
            core = getattr(metadata, "core_attn_metadata", None)
            if core is None:
                logger.warning(
                    "[DSV4_MTP_PROBE] dsv4_metadata replay=%d step=%d metadata=%s",
                    self._mtp_probe_replay_calls,
                    step,
                    type(metadata).__name__,
                )
                continue
            logger.warning(
                "[DSV4_MTP_PROBE] dsv4_metadata replay=%d step=%d backend_step=%d "
                "seq_lens=%s seq_lens_cpu=%s positions=%s req_pool_indices=%s "
                "raw_out_loc=%s seq_lens_casual=%s positions_casual=%s "
                "page_table=%s swa_page_indices=%s swa_topk_lengths=%s "
                "c4_out_loc=%s c4_topk_raw=%s c4_topk_clamp1=%s "
                "ptrs=(raw:%s,page:%s,swa:%s,c4loc:%s)",
                self._mtp_probe_replay_calls,
                step,
                backend.speculative_step_id,
                _values(forward_batch.seq_lens),
                _values(forward_batch.seq_lens_cpu),
                _values(getattr(forward_batch, "positions", None)),
                _values(forward_batch.req_pool_indices),
                _values(getattr(core, "raw_out_loc", None), 12),
                _values(getattr(core, "seq_lens_casual", None)),
                _values(getattr(core, "positions_casual", None)),
                _values(getattr(core, "page_table", None), 4),
                _values(getattr(core, "swa_page_indices", None), 8),
                _values(getattr(core, "swa_topk_lengths", None)),
                _values(getattr(core, "c4_out_loc", None), 12),
                _values(getattr(core, "c4_topk_lengths_raw", None)),
                _values(getattr(core, "c4_topk_lengths_clamp1", None)),
                _ptr(getattr(core, "raw_out_loc", None)),
                _ptr(getattr(core, "page_table", None)),
                _ptr(getattr(core, "swa_page_indices", None)),
                _ptr(getattr(core, "c4_out_loc", None)),
            )


# Temporary single-token boundary probes used to localize MTP drift without
# cloning the large attention metadata for every layer.
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.srt.models.deepseek_v4_nextn import (
    DeepseekV4ForCausalLMNextN,
    DeepseekV4ModelNextN,
)


def _record_mtp_nextn_probe(name, tensor):
    if (
        os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR") is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.shape[0] != 1
    ):
        return
    backend = get_attn_backend()
    step = getattr(backend, "speculative_step_id", 0)
    prefix = f"step{step}_layer0.nextn"
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe[f"{prefix}.{name}"] = tensor.clone()
    backend._mtp_tensor_probe = probe


_original_decoder_forward_for_probe = DeepseekV4DecoderLayer.forward
_original_nextn_forward_for_probe = DeepseekV4ModelNextN.forward
_original_nextn_causal_forward_for_probe = DeepseekV4ForCausalLMNextN.forward
_original_nextn_hc_head_for_probe = DeepseekV4ModelNextN.hc_head


def _decoder_forward_with_mtp_probe(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and len(args) > 1:
        hidden_states = args[1]
    _record_mtp_nextn_probe("decoder_input", hidden_states)
    output = _original_decoder_forward_for_probe(self, *args, **kwargs)
    output_hidden = output[0] if isinstance(output, tuple) else output
    _record_mtp_nextn_probe("decoder_output", output_hidden)
    return output


def _nextn_forward_with_mtp_probe(
    self, input_ids, positions, forward_batch, input_embeds=None
):
    _record_mtp_nextn_probe("spec_hidden", forward_batch.spec_info.hidden_states)
    output = _original_nextn_forward_for_probe(
        self, input_ids, positions, forward_batch, input_embeds
    )
    final_hidden = output[0] if isinstance(output, tuple) else output
    _record_mtp_nextn_probe("final_hidden", final_hidden)
    return output


def _nextn_hc_head_with_mtp_probe(self, x, hc_fn, hc_scale, hc_base):
    _record_mtp_nextn_probe("hc_head_input", x)
    output = _original_nextn_hc_head_for_probe(self, x, hc_fn, hc_scale, hc_base)
    _record_mtp_nextn_probe("hc_head_output", output)
    return output


def _nextn_causal_forward_with_mtp_probe(self, input_ids, positions, forward_batch):
    output = _original_nextn_causal_forward_for_probe(
        self, input_ids, positions, forward_batch
    )
    logits = getattr(output, "next_token_logits", None)
    if logits is not None:
        _record_mtp_nextn_probe("next_token_logits", logits)
    return output


DeepseekV4DecoderLayer.forward = _decoder_forward_with_mtp_probe
DeepseekV4ModelNextN.forward = _nextn_forward_with_mtp_probe
DeepseekV4ModelNextN.hc_head = _nextn_hc_head_with_mtp_probe
DeepseekV4ForCausalLMNextN.forward = _nextn_causal_forward_with_mtp_probe
