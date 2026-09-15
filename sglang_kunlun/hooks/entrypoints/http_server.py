# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""HTTP entrypoint hooks for Kunlun (P800).

The clean upstream sglang 0.5.14 ``http_server`` only registers ``/health`` and
``/health_generate``. The AIAK production image adds three deployment-facing
interfaces that the k8s gateway / PD orchestrator relies on:

* ``/ready``            – readiness probe that flips to 507 while the process is
                          being drained via an external stop signal.
* ``/health_forward``   – alias of the health-generate probe (kept for parity
                          with the AIAK image so existing probes keep working).
* ``/get_instance_info``– instance topology / live load snapshot consumed by the
                          gateway to route PD traffic.

We cannot edit the sglang tree. Routes are installed onto the module-level
FastAPI ``app`` by :func:`install_routes`, which is invoked from two places so
it works in every process topology:

* ``launch_server`` AROUND hook (below) — covers the single-process HTTP server.
* the ``sitecustomize`` meta-path finder — covers ``--tokenizer-worker-num N``
  where uvicorn spawns *fresh* worker processes that re-import ``http_server``
  without ever running ``launch_server``/``load_plugins``. Runtime mutation in
  the parent does not propagate to those workers, so the routes must be
  (re)attached at module-import time in each worker.

:func:`install_routes` is idempotent (it drops the managed paths before
re-adding), so running it from both entry points is safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid

from fastapi import Request
from fastapi.responses import Response

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# All health paths are served by the single shutdown-aware handler below,
# mirroring the AIAK image (``@app.get("/ready"|"/health"|"/health_forward"
# |"/health_generate")``).
_HEALTH_PATHS = ("/ready", "/health", "/health_forward", "/health_generate")
_INSTANCE_INFO_PATH = "/get_instance_info"


def _stopped_by_external() -> bool:
    """Return True when an external drainer asked this instance to stop.

    The orchestrator writes ``stop`` into the file pointed to by
    ``AIAK_STOP_SIGNAL_DIR`` (default ``/run/sglang/stop``) to gracefully take
    the instance out of rotation; ``/ready`` then reports 507 so the gateway
    stops routing new traffic to it.
    """
    signal_path = os.environ.get("AIAK_STOP_SIGNAL_DIR", "/run/sglang/stop")
    if not signal_path:
        return False
    try:
        if os.path.exists(signal_path):
            with open(signal_path, "r") as f:
                return f.read().strip().lower() == "stop"
    except Exception as e:  # noqa: BLE001 - probe must never raise
        logger.info("Error reading stop signal file: %s", e)
    return False


def install_routes(hs=None) -> None:
    """Add the AIAK health/ready/instance-info routes onto sglang's ``app``.

    ``hs`` is the ``sglang.srt.entrypoints.http_server`` module. When called from
    the ``sitecustomize`` import hook the freshly-executed module is passed in
    directly (it is not yet published in ``sys.modules`` for a plain import);
    when called from the ``launch_server`` hook it is imported here.

    ``Request``/``Response`` are imported at module scope (not locally) so that
    FastAPI can resolve the ``request: Request`` annotation of the handler
    closures — under ``from __future__ import annotations`` the annotation is a
    string resolved against the *module* globals, not the local namespace.
    """
    if hs is None:
        from sglang.srt.entrypoints import http_server as hs

    app = getattr(hs, "app", None)
    if app is None:
        return

    async def health_generate(request: Request) -> Response:
        """Health / readiness probe (shutdown-aware, AIAK-compatible)."""
        tm = hs._global_state.tokenizer_manager

        # Readiness: report 507 while being drained so the gateway de-routes us.
        if request.url.path.rstrip("/") == "/ready" and _stopped_by_external():
            logger.warning("Server is shutting down")
            return Response(content="Server is shutting down", status_code=507)

        if tm.gracefully_exit:
            logger.info("Health check request received during shutdown. Returning 503.")
            return Response(status_code=503)

        if tm.server_status == hs.ServerStatus.Starting:
            return Response(status_code=503)

        # Cheap liveness for /health unless generation-based probing is enabled.
        if (
            not hs.envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
            and request.url.path == "/health"
        ):
            return Response(status_code=200)

        sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
        rid = f"{hs.HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

        if getattr(tm, "is_image_gen", False):
            gri = tm.get_image_gen_health_check_request(rid, sampling_params)
        elif tm.is_generation:
            gri = hs.GenerateReqInput(
                rid=rid,
                input_ids=[0],
                sampling_params=sampling_params,
                log_metrics=False,
            )
            if tm.server_args.disaggregation_mode != hs.DisaggregationMode.NULL.value:
                gri.bootstrap_host = hs.FAKE_BOOTSTRAP_HOST
                gri.bootstrap_room = 0
        else:
            gri = hs.EmbeddingReqInput(
                rid=rid,
                input_ids=[0],
                sampling_params=sampling_params,
                log_metrics=False,
            )

        async def gen():
            async for _ in tm.generate_request(gri, request):
                break

        task = asyncio.create_task(gen())

        # Any response from the detokenizer/scheduler => server is healthy.
        tic = time.time()
        while time.time() < tic + hs.HEALTH_CHECK_TIMEOUT:
            await asyncio.sleep(1)
            if tm.last_receive_tstamp > tic:
                task.cancel()
                tm.rid_to_state.pop(rid, None)
                tm.server_status = hs.ServerStatus.Up
                return Response(status_code=200)

        task.cancel()
        tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
        last_receive_time = time.strftime(
            "%H:%M:%S", time.localtime(tm.last_receive_tstamp)
        )
        logger.error(
            "Health check failed. Server couldn't get a response from detokenizer "
            "for last %s seconds. tic start time: %s. last_heartbeat time: %s",
            hs.HEALTH_CHECK_TIMEOUT,
            tic_time,
            last_receive_time,
        )
        tm.rid_to_state.pop(rid, None)
        tm.server_status = hs.ServerStatus.UnHealthy
        return Response(status_code=503)

    async def get_instance_info():
        """Return instance topology + live load, as consumed by the gateway."""
        tm = hs._global_state.tokenizer_manager
        server_args = tm.server_args

        chunked_prefill_enabled = server_args.chunked_prefill_size != -1

        current_input_tokens = 0
        for state in tm.rid_to_state.values():
            obj = getattr(state, "obj", None)
            input_ids = getattr(obj, "input_ids", None)
            if isinstance(input_ids, list):
                current_input_tokens += len(input_ids)

        max_running_requests = None
        if server_args.max_running_requests:
            max_running_requests = (
                server_args.max_running_requests * server_args.dp_size
            )

        return {
            "kv_role": server_args.disaggregation_mode,
            "dp_ranks": list(range(server_args.dp_size)),
            "dp_size": server_args.dp_size,
            "tp_size": server_args.tp_size,
            "nnodes": server_args.nnodes,
            "chunked_prefill_enabled": chunked_prefill_enabled,
            "chunked_prefill_size": server_args.chunked_prefill_size,
            "active_request_count": len(tm.rid_to_state),
            "max_running_requests": max_running_requests,
            "current_input_tokens": current_input_tokens,
        }

    # Drop any pre-existing routes we are about to (re)define so the
    # shutdown-aware handler is authoritative and we don't leave duplicates.
    managed = set(_HEALTH_PATHS) | {_INSTANCE_INFO_PATH}
    app.router.routes = [
        r for r in app.router.routes if getattr(r, "path", None) not in managed
    ]

    for path in _HEALTH_PATHS:
        app.add_api_route(path, health_generate, methods=["GET"])
    app.add_api_route(_INSTANCE_INFO_PATH, get_instance_info, methods=["GET"])

    logger.info(
        "sglang-kunlun: installed HTTP routes %s (pid=%s)",
        list(_HEALTH_PATHS) + [_INSTANCE_INFO_PATH],
        os.getpid(),
    )


def _ephemeral_port_floor() -> int:
    """Lowest port the kernel may hand out for an ephemeral bind."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 32768


def pin_nccl_port(server_args) -> None:
    """Pin ``nccl_port`` below the ephemeral range for dp-attention servers.

    sglang picks ``nccl_port`` with ``get_free_port()`` (bind port 0, then close)
    and the ``DataParallelController`` allocates its per-worker ZMQ ports the
    same way a moment later. Linux hands out the *same* just-released ephemeral
    port again, so the controller keeps binding exactly the port the rank-0
    scheduler is about to use for its TCPStore, and startup dies with
    ``EADDRINUSE`` on that port. Choosing a free port below
    ``ip_local_port_range`` makes the collision impossible: no ephemeral bind
    can ever land there.
    """
    if server_args.nccl_port is not None or not server_args.enable_dp_attention:
        return
    reserved = {server_args.port, server_args.disaggregation_bootstrap_port}
    for port in range(_ephemeral_port_floor() - 1, 1024, -1):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
        server_args.nccl_port = port
        logger.info("sglang-kunlun: pinned nccl_port to %d", port)
        return
    logger.warning("sglang-kunlun: found no free port below the ephemeral range")


_WARMUP_TOKENS_ENV = "SGLANG_KUNLUN_WARMUP_TOKENS"
# Off by default: the prompt length that exercises the long-context path is
# deployment-specific, and a prompt larger than the SWA pool can hold wedges the
# scheduler (the request can never finish, so the pool never frees). Set
# ``SGLANG_KUNLUN_WARMUP_TOKENS`` to a length the pool can actually hold to
# enable it.
_DEFAULT_WARMUP_TOKENS = 0


def long_context_prefill_warmup(server_args) -> None:
    """Run one long synthetic prefill before the server accepts traffic.

    The stock disaggregation warm-up only pushes four tokens through the model,
    which never touches the long-context prefill path (NSA context parallelism,
    the paged compress/indexer caches, the MoE dispatch workspaces). Those
    buffers are allocated lazily and are read past the region the first request
    fills, so on a fresh process the first *long* request reads uninitialised
    device memory (NaN) and degenerates into an empty / BOS-only answer, while
    every later request is fine. Pushing one long prompt through here makes the
    first real request behave like the second one.
    """
    if server_args.disaggregation_mode != "prefill":
        return
    num_tokens = int(os.environ.get(_WARMUP_TOKENS_ENV, _DEFAULT_WARMUP_TOKENS))
    # ``--max-prefill-tokens -1`` means "no limit"; only clamp against a real cap.
    if server_args.max_prefill_tokens and server_args.max_prefill_tokens > 0:
        num_tokens = min(num_tokens, server_args.max_prefill_tokens)
    if num_tokens <= 0:
        return

    import requests
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

    dp_size = max(server_args.dp_size, 1)
    # Deterministic spread of token ids so the MoE routing touches many experts.
    input_ids = [(i * 7919) % 60000 + 16 for i in range(num_tokens)]
    payload = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * dp_size,
        "bootstrap_room": [
            i * (2**63 // dp_size) + (i % max(server_args.tp_size, 1)) + 1
            for i in range(dp_size)
        ],
        "input_ids": [input_ids] * dp_size,
    }
    headers = {}
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"
    t0 = time.time()
    res = requests.post(
        server_args.url() + "/generate",
        json=payload,
        headers=headers,
        timeout=1800,
        verify=server_args.ssl_verify(),
    )
    logger.info(
        "sglang-kunlun: long-context warmup (%d tokens) -> %s in %.1fs",
        num_tokens,
        res.status_code,
        time.time() - t0,
    )


def start_long_context_warmup(server_args) -> None:
    """Fire :func:`long_context_prefill_warmup` once the server is up.

    Runs in a background thread so it cannot block ``launch_server``; it waits
    for the health endpoint (which only answers after sglang's own warm-up) and
    then pushes one long prompt through the model.
    """
    import threading

    if server_args.disaggregation_mode != "prefill":
        return
    if int(os.environ.get(_WARMUP_TOKENS_ENV, _DEFAULT_WARMUP_TOKENS)) <= 0:
        # Disabled: don't leave a thread polling the health endpoint for 30 min.
        return

    def runner():
        import requests

        url = server_args.url()
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                res = requests.get(
                    url + "/health_generate", timeout=60, verify=server_args.ssl_verify()
                )
                if res.status_code == 200:
                    break
            except Exception:  # noqa: BLE001 - server not up yet
                pass
            time.sleep(5)
        else:
            logger.warning(
                "sglang-kunlun: long-context warmup skipped (server not ready)"
            )
            return
        try:
            long_context_prefill_warmup(server_args)
        except Exception:  # noqa: BLE001 - warm-up must never break serving
            logger.exception("sglang-kunlun: long-context warmup failed")

    threading.Thread(target=runner, name="kunlun-long-warmup", daemon=True).start()
    logger.info("sglang-kunlun: long-context warmup thread started")


@plugin_hook(
    "sglang.srt.entrypoints.http_server.launch_server",
    type=HookType.AROUND,
)
def launch_server_kunlun(original_fn, *args, **kwargs):
    """Install the AIAK deployment routes onto ``app`` before uvicorn starts.

    Covers the single-process HTTP server. For ``--tokenizer-worker-num N`` the
    worker processes are handled by the ``sitecustomize`` import hook instead.
    """
    try:
        install_routes()
    except Exception:  # noqa: BLE001 - never block server launch
        logger.exception("sglang-kunlun: failed to install health/instance routes")
    server_args = kwargs.get("server_args") or (args[0] if args else None)
    if server_args is not None:
        pin_nccl_port(server_args)
        try:
            start_long_context_warmup(server_args)
        except Exception:  # noqa: BLE001 - never block server launch
            logger.exception("sglang-kunlun: failed to start long-context warmup")
    return original_fn(*args, **kwargs)

