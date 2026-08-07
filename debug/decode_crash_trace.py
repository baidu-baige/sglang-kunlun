"""Decode-stage crash/hang localization for the Kunlun DSV4 backend.

Every probe point here runs eagerly *outside* CUDA graph capture and replay, so
the last line printed before a device hang or kernel abort identifies which
graph launch never returned and which host-side metadata it was fed.

Gating:

* ``SGLANG_KUNLUN_DECODE_TRACE=1`` enables the trace.
* ``SGLANG_KUNLUN_DECODE_TRACE_SYNC=1`` additionally calls
  ``torch.cuda.synchronize()`` at the worker-level probe points, which turns an
  asynchronous device fault into a failure attributable to the preceding launch.
* ``SGLANG_KUNLUN_DECODE_TRACE_RANGES=1`` adds min/max for int32/int64 tensors.
  Off by default: Kunlun reports ``is_current_stream_capturing() == False`` under
  its own XCUDAGraph, so a reduction can land inside a capture, and uint8 plan
  buffers have no ``min`` implementation at all.
* ``SGLANG_KUNLUN_DECODE_TRACE_RANKS`` restricts output to a comma separated
  rank list. Default is every rank, because the 0.5.16 decode hang showed all
  eight ranks stuck in the same replay.

Output goes to stdout with ``flush=True``: the process is killed by the
scheduler watchdog, so buffered logging records are lost.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch

TRACE_ENV_VAR = "SGLANG_KUNLUN_DECODE_TRACE"
SYNC_ENV_VAR = "SGLANG_KUNLUN_DECODE_TRACE_SYNC"
RANGES_ENV_VAR = "SGLANG_KUNLUN_DECODE_TRACE_RANGES"
RANKS_ENV_VAR = "SGLANG_KUNLUN_DECODE_TRACE_RANKS"

_PREFIX = "[KUNLUN_DECODE_TRACE]"
_counters: dict[str, int] = {}
_last_event_ts: Optional[float] = None


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def enabled() -> bool:
    """Return whether decode tracing is active for this rank."""
    if os.environ.get(TRACE_ENV_VAR) != "1":
        return False
    ranks = os.environ.get(RANKS_ENV_VAR)
    if not ranks:
        return True
    allowed = {int(value) for value in ranks.split(",") if value.strip() != ""}
    return _rank() in allowed


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - absence of capture support means eager
        return False


def _tensor_summary(tensor: Any) -> str:
    if not isinstance(tensor, torch.Tensor):
        return repr(tensor)
    if tensor.numel() == 0:
        return f"shape={list(tensor.shape)} dtype={tensor.dtype} empty"
    summary = f"shape={list(tensor.shape)} dtype={tensor.dtype}"
    # Only int32/int64 reductions are implemented on Kunlun; uint8 plan buffers
    # raise NOT_IMPLEMENTED. Ranges stay opt-in because
    # ``is_current_stream_capturing`` reports False under the Kunlun XCUDAGraph,
    # so a reduction here can execute inside a capture.
    if (
        os.environ.get(RANGES_ENV_VAR) == "1"
        and tensor.dtype in (torch.int32, torch.int64)
        and tensor.numel() <= 4096
        and not _is_capturing()
    ):
        try:
            flat = tensor.detach().reshape(-1)
            summary += f" min={int(flat.min())} max={int(flat.max())}"
        except Exception as exc:  # noqa: BLE001 - tracing must never break the run
            summary += f" range_unavailable={type(exc).__name__}"
    return summary


def _batch_fields(forward_batch: Any) -> dict[str, Any]:
    if forward_batch is None:
        return {}
    fields: dict[str, Any] = {}
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None:
        fields["mode"] = str(mode)
    for name in ("batch_size", "input_ids", "seq_lens", "req_pool_indices", "out_cache_loc"):
        value = getattr(forward_batch, name, None)
        if value is None:
            continue
        if callable(value):
            # ScheduleBatch exposes batch_size() as a method, ForwardBatch as an int.
            try:
                value = value()
            except Exception as exc:  # noqa: BLE001 - tracing must never break the run
                value = f"<{type(exc).__name__}>"
        fields[name] = (
            _tensor_summary(value) if isinstance(value, torch.Tensor) else value
        )
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel():
        fields["seq_lens_cpu_max"] = int(seq_lens_cpu.max())
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is not None:
        fields["spec_info"] = type(spec_info).__name__
        for name in ("accept_length", "accept_length_cpu", "draft_token_num"):
            value = getattr(spec_info, name, None)
            if value is None:
                continue
            fields[f"spec_{name}"] = (
                _tensor_summary(value) if isinstance(value, torch.Tensor) else value
            )
    return fields


def maybe_sync(tag: str) -> None:
    """Synchronize the device so a hang is attributed to the previous launch."""
    if not enabled() or os.environ.get(SYNC_ENV_VAR) != "1":
        return
    if _is_capturing():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - tracing must never break the run
        trace(f"{tag}.sync_failed", error=repr(exc))


def trace(tag: str, forward_batch: Any = None, **fields: Any) -> None:
    """Emit one decode trace line for ``tag``."""
    global _last_event_ts

    if not enabled():
        return

    now = time.monotonic()
    step = _counters.get(tag, 0)
    _counters[tag] = step + 1
    since_previous = "-" if _last_event_ts is None else f"{(now - _last_event_ts) * 1e3:.1f}"
    _last_event_ts = now

    payload = dict(_batch_fields(forward_batch))
    payload.update(fields)
    rendered = " ".join(f"{key}={value}" for key, value in payload.items())
    print(
        f"{_PREFIX} rank={_rank()} pid={os.getpid()} tag={tag} count={step} "
        f"since_prev_ms={since_previous} {rendered}",
        flush=True,
    )


def trace_tensor_args(tag: str, **values: Any) -> None:
    """Emit shapes/dtypes for the arguments of one out-of-graph kernel call."""
    if not enabled():
        return
    trace(
        tag,
        capturing=_is_capturing(),
        **{name: _tensor_summary(value) for name, value in values.items()},
    )


def _host_range(tensor: Any) -> str:
    """Return min/max of a small length buffer, reduced on the host."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return "none"
    try:
        host = tensor.detach().reshape(-1).to("cpu")
        return f"{int(host.min())}..{int(host.max())}"
    except Exception as exc:  # noqa: BLE001 - tracing must never break the run
        return f"<{type(exc).__name__}>"


def _aux_summary(aux_map: Any, host_index: int, device_index: int) -> str:
    """Summarize the per-bucket length buffers a replayed graph reads."""
    if not aux_map:
        return "empty"
    parts = []
    for key, aux in list(aux_map.items())[:32]:
        parts.append(
            f"{key}[host={_host_range(aux[host_index])},"
            f"dev={_host_range(aux[device_index])}]"
        )
    return ";".join(parts)


def trace_replay_aux(
    tag: str,
    forward_batch: Any,
    *,
    attention_decode_aux: Any,
    c4_decode_aux: Any,
    graph_extend_aux: Any,
    **fields: Any,
) -> None:
    """Log the aux length buffers the next graph replay will consume.

    These buffers are the only per-step state a replayed graph sees, so a value
    that stops tracking ``seq_lens`` localizes both wrong attention spans and
    out-of-range page reads.
    """
    if not enabled():
        return
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    trace(
        tag,
        forward_batch=forward_batch,
        seq_lens_range=_host_range(
            seq_lens_cpu
            if isinstance(seq_lens_cpu, torch.Tensor)
            else getattr(forward_batch, "seq_lens", None)
        ),
        attention_decode_aux=_aux_summary(attention_decode_aux, 2, 3),
        c4_decode_aux=_aux_summary(c4_decode_aux, 2, 3),
        graph_extend_aux=_aux_summary(graph_extend_aux, 2, 3),
        **fields,
    )


def trace_verify_result(tag: str, result: Any) -> None:
    """Log EAGLE acceptance so wrong acceptance is separable from wrong logits.

    ``accept_lens`` pinned at ``num_draft_tokens`` on every request means the
    verify kernel accepts unconditionally, which produces looping output; varying
    lengths with wrong text points at the target logits instead.
    """
    if not enabled():
        return
    fields: dict[str, Any] = {}
    for name in ("accept_lens", "new_seq_lens", "next_token_ids"):
        value = getattr(result, name, None)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        try:
            host = value.detach().reshape(-1).to("cpu")
        except Exception as exc:  # noqa: BLE001 - tracing must never break the run
            fields[name] = f"<{type(exc).__name__}>"
            continue
        fields[name] = f"{list(value.shape)}:{host[:16].tolist()}"
    speculative = getattr(result, "speculative_num_draft_tokens", None)
    if speculative is not None:
        fields["num_draft_tokens"] = speculative
    trace(tag, **fields)


def _kernel_arg(tensor: Any) -> str:
    """Describe one kernel argument: shape, dtype, pointer and leading values.

    Plan buffers reach XSpeedGate as ``uint8`` views of packed int32 fields, so
    they are re-viewed as int32 before printing. The pointer is included because a
    buffer reallocated between capture and replay is invisible in the shape alone.
    """
    if not isinstance(tensor, torch.Tensor):
        return repr(tensor)
    parts = [f"shape={list(tensor.shape)}", f"dtype={tensor.dtype}"]
    parts.append(f"ptr=0x{tensor.data_ptr():x}")
    if tensor.numel() == 0:
        return " ".join(parts + ["empty"])
    if tensor.dtype in (torch.int32, torch.int64, torch.uint8) and tensor.numel() <= 65536:
        try:
            host = tensor.detach().to("cpu")
            if tensor.dtype == torch.uint8 and host.numel() % 4 == 0:
                host = host.reshape(-1).view(torch.int32)
            host = host.reshape(-1)
            parts.append(f"head={host[:8].tolist()}")
            parts.append(f"range={int(host.min())}..{int(host.max())}")
        except Exception as exc:  # noqa: BLE001 - tracing must never break the run
            parts.append(f"values=<{type(exc).__name__}>")
    return " ".join(parts)


def trace_kernel_args(tag: str, **values: Any) -> None:
    """Log pointer + leading values for the arguments of one kernel launch."""
    if not enabled():
        return
    trace(
        tag,
        capturing=_is_capturing(),
        **{name: _kernel_arg(value) for name, value in values.items()},
    )


def trace_call(tag: str, fn, *args, **kwargs):
    """Run ``fn`` bracketed by begin/end traces so a hang is localized.

    The traced batch is discovered from ``kwargs["forward_batch"]`` or the first
    positional argument exposing ``forward_mode``; nothing is consumed, so the
    wrapped call keeps its original arguments.
    """
    if not enabled():
        return fn(*args, **kwargs)

    forward_batch = kwargs.get("forward_batch")
    if forward_batch is None:
        for candidate in args:
            if hasattr(candidate, "forward_mode"):
                forward_batch = candidate
                break

    trace(f"{tag}.begin", forward_batch=forward_batch)
    maybe_sync(f"{tag}.begin")
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - re-raised after tracing
        trace(f"{tag}.raised", error=repr(exc))
        raise
    maybe_sync(f"{tag}.end")
    trace(f"{tag}.end")
    return result
