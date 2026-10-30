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
"""Extra scheduler/tokenizer metrics that upstream sglang does not expose.

This module re-creates, from hooks and without touching upstream
``SchedulerStats``, metrics the production Grafana dashboards query:

Iteration timing (``mostrecent`` gauges)
    ``generation_time`` decode-step wall time; ``run_batch_time`` its forward-only
    part; ``iter_token_process_time`` the remainder (scheduling / overhead).

Static / DP-balance gauges
    ``num_max_batchs`` (``--max-running-requests``); ``dp_balance`` (1 - Gini of
    per-DP token counts); ``idle_batch_ratio``; ``decode_bs_util``;
    ``prefill_chunk_util``.

Per-request latency histograms
    ``request_waiting_time``, ``request_first_token_forward_time``,
    ``request_zmq_time_seconds``. Histogram quantiles are not additive, so these
    aggregate spans cannot be reconstructed from the per-phase
    ``per_stage_req_latency_seconds{stage=...}``.

Token-length aliases
    ``per_prompt_len`` / ``per_prompt_gen_len``: aliases of
    ``prompt_tokens_histogram`` / ``generation_tokens_histogram``. They ride the
    tokenizer's ``observe_one_finished_request``, which under PD only fires on
    DECODE, so they are re-emitted scheduler-side on PREFILL too -- alongside the
    per-request counters ``num_requests_total`` / ``num_aborted_requests_total``,
    the token counters ``prompt_tokens_total`` / ``cached_tokens_total`` and the
    prefill histograms ``kvcache_transfer_time`` / ``prompt_calculation_time``
    (see the "prefill-side per-request metrics" section).

Preheated counters (visible from startup)
    ``num_bootstrap_failed_reqs_total`` / ``num_transfer_failed_reqs_total`` /
    ``num_aborted_requests_total`` / ``evicted_tokens_total`` /
    ``num_mm_requests_total``. A labeled counter is invisible until first
    observed, so each is pre-registered with ``inc(0)``. Two need more than a
    nudge: ``evicted_tokens_total`` lives on ``RadixCacheMetricsCollector``, and
    ``num_mm_requests_total`` does not exist upstream at all.

Notes: everything uses ``perf_counter()``; the iteration timings are recomputed
every decode step; ``run_batch_time`` is the scheduler's blocked-on-GPU span (as
in the fork) and ``generation_time`` the commit-to-commit cadence. See the
per-hook comments.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


class _MetricState:
    """Per-process holder for the values the hooks below produce.

    A plain module-level singleton rather than extra ``SchedulerStats`` fields:
    there is exactly one scheduler per process, the upstream stats container is
    copied/serialised in places we do not control, and this keeps the port to a
    single file.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.dp_rank: Optional[int] = None
        self.max_running_requests: Optional[int] = None
        self.chunked_prefill_size: Optional[int] = None

        # Iteration timing.
        self.forward_start_time = 0.0
        self.forward_finish_time = 0.0
        self.last_commit_time = 0.0  # previous decode-step commit, for step cadence
        self.generation_time = 0.0
        self.run_batch_time = 0.0
        self.iter_token_process_time = 0.0

        # DP balance. Defaults are the "no data yet" values.
        self.dp_balance = 0.0
        self.idle_batch_ratio = 1.0
        self.decode_bs_util = 0.0
        self.prefill_chunk_util = 0.0


STATE = _MetricState()


# Prompt-length histogram buckets, copied verbatim from upstream
# (observability/metrics_collector.py ``default_bucket_prompt_tokens``). Kept in
# sync so the prefill-side ``per_prompt_len`` emission (``_observe_prefill_prompt_len``)
# uses the exact same buckets as the tokenizer-side alias, letting prometheus
# multiprocess merge the two definitions without bucket drift.
_DEFAULT_PROMPT_TOKENS_BUCKETS = [
    100, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500,
    25000, 27500, 30000, 35000, 40000, 60000, 80000, 100000, 200000, 300000,
    400000, 600000, 800000, 1000000, 1100000,
]

# Latency buckets (seconds), copied verbatim from the fork (metrics/collector.py:750-796).
# Shared by the request_first_token_forward_time / request_zmq_time histograms and
# the prefill-side kvcache_transfer_time / prompt_calculation_time histograms.
_LATENCY_BUCKETS = [
    0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2,
    3, 5, 10, 20, 40, 60, 80, 120,
]


def _dp_balance(tokens: List[int]) -> float:
    """``1 - Gini`` of the per-DP token counts; 1.0 means perfectly balanced."""
    total_dp = len(tokens)
    total_tokens = sum(tokens)
    if total_dp <= 1:
        return 1.0 if total_dp == 1 else 0.0
    if total_tokens <= 0:
        return 0.0
    acc = 0
    for i, val in enumerate(sorted(tokens), start=1):
        acc += (2 * i - total_dp - 1) * val
    return 1.0 - acc / (total_dp * total_tokens)


def _is_representative_rank(scheduler) -> bool:
    """True on the one rank per DP group that should emit per-request metrics.

    A prefill request is drained on every scheduler process holding a shard of it
    (attn-TP ranks, PP stages and attn-CP ranks all see the SAME request; only DP
    groups see distinct ones), so per-request metrics must be emitted exactly once
    per DP group -- otherwise a ``sum by(...)`` across ranks reads N x the value.
    """
    ps = getattr(scheduler, "ps", None)
    return ps is not None and (
        getattr(ps, "attn_tp_rank", 0) == 0
        and getattr(ps, "pp_rank", 0) == 0
        and getattr(ps, "attn_cp_rank", 0) == 0
    )


# ─────────────────────────── metric definition / emit ───────────────────────────


@plugin_hook(
    "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.__init__",
    type=HookType.AFTER,
)
def _define_aiak_metrics(_result, self, *args, **kwargs) -> None:
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Gauge as _PromGauge
    from prometheus_client import Histogram as _PromHistogram

    Gauge = self._gauge_cls or _PromGauge
    Histogram = self._histogram_cls or _PromHistogram
    Counter = self._counter_cls or _PromCounter
    labelnames = self.labels.keys()

    def gauge(name: str, documentation: str, mostrecent: bool = False):
        extra = {"multiprocess_mode": "mostrecent"} if mostrecent else {}
        try:
            return Gauge(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
                **extra,
            )
        except ValueError:
            # Another layer already owns the name in this process (e.g. an image
            # that ships the internal platform stack). Skipping keeps the
            # collector constructible -- raising here would kill the scheduler.
            logger.info("sglang-kunlun: %s already defined; skipping", name)
            return None

    def histogram(name: str, documentation: str, buckets: List[float]):
        try:
            return Histogram(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
                buckets=buckets,
            )
        except ValueError:
            logger.info("sglang-kunlun: %s already defined; skipping", name)
            return None

    def counter(name: str, documentation: str, extra_labels: List[str] = ()):
        try:
            return Counter(
                name=name,
                documentation=documentation,
                labelnames=list(labelnames) + list(extra_labels),
            )
        except ValueError:
            logger.info("sglang-kunlun: %s already defined; skipping", name)
            return None

    # ``multiprocess_mode`` matches the fork's definitions exactly: the four
    # scalars below are "last known value" samples, the DP gauges are not.
    self.aiak_num_max_batchs = gauge(
        "sglang:num_max_batchs", "The number of max running requests", mostrecent=True
    )
    self.aiak_generation_time = gauge(
        "sglang:generation_time", "The generation time", mostrecent=True
    )
    self.aiak_run_batch_time = gauge(
        "sglang:run_batch_time", "The run batch time", mostrecent=True
    )
    self.aiak_iter_token_process_time = gauge(
        "sglang:iter_token_process_time", "The time between token", mostrecent=True
    )
    self.aiak_dp_balance = gauge(
        "sglang:dp_balance",
        "Variance of token counts across all DPs, indicating load balance",
    )
    self.aiak_idle_batch_ratio = gauge(
        "sglang:idle_batch_ratio", "Idle batch ratio across all DPs"
    )
    self.aiak_decode_bs_util = gauge(
        "sglang:decode_bs_util", "Active decode requests ratio per DP worker."
    )
    self.aiak_prefill_chunk_util = gauge(
        "sglang:prefill_chunk_util", "Used prefill tokens ratio per DP worker."
    )

    # Buckets copied verbatim from the fork's definitions -- changing them would
    # make cross-version quantile comparison meaningless.
    self.aiak_request_waiting_time = histogram(
        "sglang:request_waiting_time",
        "Histogram of request waiting time in seconds",
        [
            0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2,
            5, 10, 20, 40, 60, 80, 120, 160,
        ],
    )
    self.aiak_request_first_token_forward_time = histogram(
        "sglang:request_first_token_forward_time",
        "Histogram of request first token forward time in seconds",
        _LATENCY_BUCKETS,
    )
    self.aiak_request_zmq_time = histogram(
        "sglang:request_zmq_time_seconds",
        "Histogram of zmq time in seconds.",
        _LATENCY_BUCKETS,
    )

    # ── Renamed metrics: keep the old names as drop-in aliases ────────────────
    # These were renamed upstream; a dashboard still querying the old names would
    # read "no data". Same observations, same label sets. Guarded: if another
    # layer already owns the name, skip it instead of crashing the collector.
    self.aiak_num_prefill_prealloc_queue_reqs = gauge(
        "sglang:num_prefill_prealloc_queue_reqs",
        "Deprecated alias for num_prefill_bootstrap_queue_reqs.",
        mostrecent=True,
    )
    self.aiak_gpu_execution_seconds_total = None
    self.aiak_dp_cooperation_gpu_execution_seconds_total = None
    try:
        self.aiak_gpu_execution_seconds_total = Counter(
            name="sglang:gpu_execution_seconds_total",
            documentation="Deprecated alias for forward_execution_seconds_total.",
            labelnames=list(labelnames) + ["category"],
        )
    except ValueError:
        logger.info(
            "sglang-kunlun: gpu_execution_seconds_total already defined; skipping"
        )
    try:
        self.aiak_dp_cooperation_gpu_execution_seconds_total = Counter(
            name="sglang:dp_cooperation_gpu_execution_seconds_total",
            documentation=(
                "Deprecated alias for dp_cooperation_forward_execution_seconds_total."
            ),
            labelnames=list(labelnames) + ["category", "num_prefill_ranks"],
        )
    except ValueError:
        logger.info(
            "sglang-kunlun: dp_cooperation_gpu_execution_seconds_total already "
            "defined; skipping"
        )

    # Prefill-side metrics that upstream only emits from the tokenizer -- which
    # under PD only fires on DECODE, so a PREFILL node reads "no data". Defined
    # ONLY on prefill schedulers; the histograms reuse the tokenizer buckets so
    # prometheus multiprocess merges the definitions cleanly. Emitted from the
    # inflight-queue drain (see ``_observe_prefill_prompt_len``).
    self.aiak_per_prompt_len = None
    self.aiak_prompt_tokens_histogram = None
    self.aiak_kvcache_transfer_time = None
    self.aiak_prompt_calculation_time = None
    self.aiak_num_requests_total = None
    self.aiak_num_aborted_requests_total = None
    self.aiak_prompt_tokens_total = None
    self.aiak_cached_tokens_total = None
    _server_args = kwargs.get("server_args")
    if _server_args is None and len(args) >= 5:
        _server_args = args[4]
    if getattr(_server_args, "disaggregation_mode", None) == "prefill":
        from sglang.srt.observability.utils import generate_buckets

        _prompt_len_buckets = generate_buckets(
            getattr(_server_args, "prompt_tokens_buckets", None),
            _DEFAULT_PROMPT_TOKENS_BUCKETS,
        )
        for _attr, _name in (
            ("aiak_per_prompt_len", "sglang:per_prompt_len"),
            ("aiak_prompt_tokens_histogram", "sglang:prompt_tokens_histogram"),
        ):
            setattr(
                self,
                _attr,
                histogram(
                    _name,
                    "Prompt token length, emitted on the prefill node.",
                    _prompt_len_buckets,
                ),
            )

        # Emitted from the prefill scheduler (prefill.py:545/678); values come
        # from req.time_stats in the same drain hook:
        # prompt_calculation_time = transfer_queue_entry - forward_entry,
        # kvcache_transfer_time = completion - transfer_queue_entry.
        self.aiak_kvcache_transfer_time = histogram(
            "sglang:kvcache_transfer_time",
            "Histogram of kvcache transfer time measured in prefill",
            _LATENCY_BUCKETS,
        )
        self.aiak_prompt_calculation_time = histogram(
            "sglang:prompt_calculation_time",
            "Histogram of prompt calculation time measured in prefill",
            _LATENCY_BUCKETS,
        )

        self.aiak_num_requests_total = counter(
            "sglang:num_requests_total",
            "Number of finished requests, emitted on the prefill node.",
        )
        self.aiak_num_aborted_requests_total = counter(
            "sglang:num_aborted_requests_total",
            "Number of aborted requests, emitted on the prefill node.",
        )
        # Token counters. ``cached_tokens_total`` carries the same extra
        # ``cache_source`` label as upstream (metrics_collector.py:1507-1511) so the
        # P-side series merge with the D-side ones instead of conflicting.
        self.aiak_prompt_tokens_total = counter(
            "sglang:prompt_tokens_total",
            "Number of prefill (prompt) tokens processed, emitted on the prefill node.",
        )
        self.aiak_cached_tokens_total = counter(
            "sglang:cached_tokens_total",
            "Number of cached prompt tokens by source, emitted on the prefill node.",
            ["cache_source"],
        )

        # Pre-register the prefill-side counters at 0 so they are visible from
        # startup. A labeled counter stays invisible until its first increment;
        # aborts are rare (healthy load has none), so without this the prefill
        # panel reads "no data" instead of 0. Mirrors the tokenizer-side preheat
        # in ``_prepublish_tokenizer_counters``.
        for _attr in (
            "aiak_num_requests_total",
            "aiak_num_aborted_requests_total",
            "aiak_prompt_tokens_total",
        ):
            _counter = getattr(self, _attr, None)
            if _counter is not None:
                _counter.labels(**self.labels).inc(0)
        if self.aiak_cached_tokens_total is not None:
            # device/host are the two sources upstream always reports; storage_*
            # only exists with an L3 backend, so it is left to appear on first hit.
            for _source in ("device", "host"):
                self.aiak_cached_tokens_total.labels(
                    **self.labels, cache_source=_source
                ).inc(0)

    # Pre-register the default label set so a healthy instance publishes 0 instead
    # of nothing: a labeled Prometheus counter stays invisible until its first
    # observation, which the dashboard reads as "no data". Mirrors the internal
    # platform stack (internal-platform-stack-dsv41.diff:3469-3473).
    self.num_bootstrap_failed_reqs.labels(**self.labels).inc(0)
    self.num_transfer_failed_reqs.labels(**self.labels).inc(0)


@plugin_hook(
    "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.log_stats",
    type=HookType.AFTER,
)
def _emit_aiak_gauges(_result, self, _stats, *args, **kwargs) -> None:
    # Every gauge may be None when another layer already owns the metric name
    # (see the ``gauge`` helper); ``_log_gauge`` does not tolerate None.
    for _gauge, _value in (
        (self.aiak_num_max_batchs, STATE.max_running_requests or 0),
        (self.aiak_generation_time, STATE.generation_time),
        (self.aiak_run_batch_time, STATE.run_batch_time),
        (self.aiak_iter_token_process_time, STATE.iter_token_process_time),
        (self.aiak_dp_balance, STATE.dp_balance),
        (self.aiak_idle_batch_ratio, STATE.idle_batch_ratio),
        (self.aiak_decode_bs_util, STATE.decode_bs_util),
        (self.aiak_prefill_chunk_util, STATE.prefill_chunk_util),
        # Alias: same value as num_prefill_bootstrap_queue_reqs.
        (
            self.aiak_num_prefill_prealloc_queue_reqs,
            getattr(_stats.num_prefill_bootstrap_queue_reqs, "total", 0),
        ),
    ):
        if _gauge is not None:
            self._log_gauge(_gauge, _value)


@plugin_hook(
    "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector."
    "increment_forward_execution_seconds",
    type=HookType.AROUND,
)
def _alias_gpu_execution_seconds(
    original_fn, self, category, t, dp_cooperation_info=None, *args, **kwargs
):
    # Mirror gpu_execution_seconds_total -> forward_execution_seconds_total (and
    # the dp_cooperation variant) onto the old names so dashboards still querying
    # them keep working.
    result = original_fn(self, category, t, dp_cooperation_info, *args, **kwargs)
    alias = getattr(self, "aiak_gpu_execution_seconds_total", None)
    if alias is not None:
        alias.labels(**self.labels, category=category).inc(t)
    if dp_cooperation_info is not None:
        d_alias = getattr(
            self, "aiak_dp_cooperation_gpu_execution_seconds_total", None
        )
        if d_alias is not None:
            d_alias.labels(
                **self.labels, category=category, **dp_cooperation_info.to_labels()
            ).inc(t)
    return result


# ───────────────── tokenizer-side token-length aliases ────────────────────────
#
# The internal dashboard queries ``sglang:per_prompt_len`` / ``per_prompt_gen_len``,
# which the internal platform stack keeps as deprecated aliases of the newer
# ``prompt_tokens_histogram`` / ``generation_tokens_histogram`` families -- same
# observations, buckets and labels, different name. The aliases were dropped, so
# any panel/query that names them reads "no data". Re-created here without
# renaming the upstream metrics.
#
# These live on ``TokenizerMetricsCollector`` (the tokenizer process), not on the
# scheduler collector: the token-length histograms are emitted from
# ``observe_one_finished_request``, which only the tokenizer calls.


@plugin_hook(
    "sglang.srt.observability.metrics_collector.TokenizerMetricsCollector.__init__",
    type=HookType.AFTER,
)
def _define_tokenizer_alias_metrics(_result, self, *args, **kwargs) -> None:
    from prometheus_client import Histogram as _PromHistogram

    Histogram = self._histogram_cls or _PromHistogram
    labelnames = self.labels.keys()

    # ``_upper_bounds`` is the sorted bucket list prometheus_client stores on the
    # metric; reusing it makes the alias a true drop-in (same buckets as upstream,
    # including any ``--prompt-tokens-buckets`` override).
    try:
        self.aiak_per_prompt_len = Histogram(
            name="sglang:per_prompt_len",
            documentation="Deprecated alias for prompt token length.",
            labelnames=labelnames,
            buckets=self.prompt_tokens_histogram._upper_bounds,
        )
        self.aiak_per_prompt_gen_len = Histogram(
            name="sglang:per_prompt_gen_len",
            documentation="Deprecated alias for generation token length.",
            labelnames=labelnames,
            buckets=self.generation_tokens_histogram._upper_bounds,
        )
    except ValueError:
        # The name is owned by another layer (e.g. an image that already ships the
        # internal platform stack) -- it is already defined and observed, so skip.
        self.aiak_per_prompt_len = None
        self.aiak_per_prompt_gen_len = None
        logger.info(
            "sglang-kunlun: per_prompt_len alias already defined upstream; skipping"
        )


@plugin_hook(
    "sglang.srt.observability.metrics_collector.TokenizerMetricsCollector."
    "observe_one_finished_request",
    type=HookType.AROUND,
)
def _observe_token_alias_histograms(
    original_fn, self, labels, prompt_tokens, generation_tokens, *args, **kwargs
):
    result = original_fn(self, labels, prompt_tokens, generation_tokens, *args, **kwargs)
    # Same observations as the upstream families (metrics_collector.py:1679/1683).
    if getattr(self, "aiak_per_prompt_len", None) is not None:
        self.aiak_per_prompt_len.labels(**labels).observe(float(prompt_tokens))
        self.aiak_per_prompt_gen_len.labels(**labels).observe(float(generation_tokens))
    return result


@plugin_hook(
    "sglang.srt.observability.metrics_collector.TokenizerMetricsCollector.__init__",
    type=HookType.AFTER,
)
def _prepublish_tokenizer_counters(_result, self, *args, **kwargs) -> None:
    # ``num_aborted_requests_total`` only exists once a request is aborted, so a
    # healthy instance reports nothing and the panel reads "no data". Publish 0 up
    # front, as the internal platform stack does
    # (internal-platform-stack-dsv41.diff:3736-3738).
    self.num_aborted_requests_total.labels(**self.labels).inc(0)

    # ``num_mm_requests_total`` does not exist upstream at all; the internal
    # platform stack adds it next to ``num_requests_total`` and pre-registers it
    # the same way (diff:3723-3728). Defined under an ``aiak_`` attribute so an
    # upstream copy is never clobbered, and skipped if another layer already owns
    # the name (``ValueError`` from the registry, as with the token-length aliases).
    from prometheus_client import Counter as _PromCounter

    Counter = self._counter_cls or _PromCounter
    try:
        self.aiak_num_mm_requests_total = Counter(
            name="sglang:num_mm_requests_total",
            documentation="Number of multimodal requests processed.",
            labelnames=self.labels.keys(),
        )
        self.aiak_num_mm_requests_total.labels(**self.labels).inc(0)
    except ValueError:
        self.aiak_num_mm_requests_total = None
        logger.info(
            "sglang-kunlun: num_mm_requests_total already defined upstream; skipping"
        )


@plugin_hook(
    "sglang.srt.managers.tokenizer_manager.TokenizerManager.collect_metrics",
    type=HookType.AROUND,
)
def _count_mm_requests(original_fn, self, state, recv_obj, i, *args, **kwargs):
    # Upstream's ``observe_one_finished_request`` has no ``is_multimodal`` argument,
    # so the observation the internal platform stack makes there (diff:3787-3789)
    # has to be reproduced from the call site: ``collect_metrics`` runs once per
    # finished request and carries ``state.obj``, whose ``contains_mm_input()`` is
    # exactly what K3 passes down.
    result = original_fn(self, state, recv_obj, i, *args, **kwargs)

    collector = getattr(self, "metrics_collector", None)
    mm = getattr(collector, "aiak_num_mm_requests_total", None)
    if mm is None or not getattr(state, "finished", False):
        return result

    obj = getattr(state, "obj", None)
    contains_mm_input = getattr(obj, "contains_mm_input", None)
    if not callable(contains_mm_input) or not contains_mm_input():
        return result

    # Rebuild the same per-request label set ``collect_metrics`` feeds to
    # ``observe_one_finished_request`` (metrics_collector.py:2370-2377).
    labels = dict(collector.labels)
    custom_labels = getattr(obj, "custom_labels", None)
    if custom_labels:
        labels.update(custom_labels)
    if getattr(self, "enable_priority_scheduling", False):
        priority = getattr(obj, "priority", None)
        if priority is not None:
            labels["priority"] = str(priority)
    mm.labels(**labels).inc(1)
    return result


# ──────────────────── radix-cache counter preheat ─────────────────────
#
# ``sglang:evicted_tokens_total`` is the fourth counter in the monitoring
# contract's ``metric.preheat`` group (``monitoring-contract.md`` class E). An
# instance under no cache pressure never evicts, so the labeled counter is never
# observed and the eviction panel reads "no data" instead of zero. The internal
# platform stack pre-registers it in ``RadixCacheMetricsCollector.__init__``
# (internal-platform-stack-dsv41.diff:3861-3864); mirrored here. The collector is
# only built when a hierarchical cache is configured, which is precisely when the
# panel applies.


@plugin_hook(
    "sglang.srt.observability.metrics_collector.RadixCacheMetricsCollector.__init__",
    type=HookType.AFTER,
)
def _preheat_radix_cache_counters(_result, self, *args, **kwargs) -> None:
    self.eviction_num_tokens.labels(**self.labels).inc(0)


# ──────────────────────────── one-time scheduler limits ──────────────────────────


@plugin_hook(
    "sglang.srt.managers.scheduler_components.metrics_reporter."
    "SchedulerMetricsReporter._init_metrics",
    type=HookType.AFTER,
)
def _capture_scheduler_limits(_result, self, *args, **kwargs) -> None:
    # ``__post_init__`` sets ``enable_metrics`` before calling ``_init_metrics``,
    # and ``self.scheduler`` is a constructor field, so both are readable here.
    STATE.enabled = bool(self.enable_metrics)
    STATE.dp_rank = self.dp_rank
    STATE.max_running_requests = getattr(self.scheduler, "max_running_requests", None)
    STATE.chunked_prefill_size = getattr(self.scheduler, "chunked_prefill_size", None)


# ─────────────────────────────── DP balance ─────────────────────────────────────


@plugin_hook(
    "sglang.srt.managers.scheduler_components.dp_attn._update_gather_batch",
    type=HookType.AFTER,
)
def _compute_dp_balance(
    _result, batch, mlp_sync_info, require_mlp_tp_gather, *args, **kwargs
) -> None:
    # Only DP-0 computes these (the all-gathered list is identical on every rank,
    # so one writer is enough). ``None`` means DP attention is off, in which case
    # this rank is the only writer.
    if not STATE.enabled or STATE.dp_rank not in (0, None):
        return

    tokens = mlp_sync_info.global_num_tokens
    if not tokens:
        return

    forward_mode = batch.forward_mode
    if forward_mode.is_idle():
        return

    total_dp = len(tokens)
    STATE.idle_batch_ratio = tokens.count(0) / total_dp
    STATE.dp_balance = _dp_balance(tokens)

    mean_tokens = sum(tokens) / total_dp
    if forward_mode.is_extend():
        if STATE.chunked_prefill_size:
            STATE.prefill_chunk_util = mean_tokens / STATE.chunked_prefill_size
    elif STATE.max_running_requests:
        STATE.decode_bs_util = mean_tokens / STATE.max_running_requests


# ─────────────────────────── request ZMQ time ────────────────────────────────────
#
# ``request_zmq_time_seconds`` = scheduler-recv minus tokenizer-dispatch (both
# ``perf_counter()``, comparable across processes). The stamp cannot ride the
# tokenized object's ``time_stats`` across ZMQ (``APIServerReqTimeStats`` drops
# every timestamp in ``__getstate__``), so it is carried on the object itself:
# tokenizer stamps it -> scheduler copies it onto ``recv_req.time_stats`` after
# recv -> ``new_from_obj`` carries it onto the scheduler-side stats (hence the
# class attribute installed below) -> the arrive stamp derives ``kunlun_zmq_time``.

_DISPATCH_ATTR = "kunlun_dispatch_to_scheduler_time"


def _install_time_stats_field() -> None:
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

    if not hasattr(SchedulerReqTimeStats, _DISPATCH_ATTR):
        setattr(SchedulerReqTimeStats, _DISPATCH_ATTR, 0.0)


_install_time_stats_field()


@plugin_hook(
    "sglang.srt.managers.tokenizer_manager.TokenizerManager._send_one_request",
    type=HookType.AROUND,
)
def _stamp_dispatch_one(original_fn, self, tokenized_obj, *args, **kwargs):
    tokenized_obj.kunlun_dispatch_to_scheduler_time = time.perf_counter()
    return original_fn(self, tokenized_obj, *args, **kwargs)


@plugin_hook(
    "sglang.srt.managers.tokenizer_manager.TokenizerManager._send_batch_request",
    type=HookType.AROUND,
)
def _stamp_dispatch_batch(original_fn, self, tokenized_objs, *args, **kwargs):
    ts = time.perf_counter()
    for obj in tokenized_objs:
        obj.kunlun_dispatch_to_scheduler_time = ts
    return original_fn(self, tokenized_objs, *args, **kwargs)


def _bridge_dispatch_to_time_stats(req) -> None:
    if hasattr(req, "batch"):
        for sub in req.batch:
            _bridge_dispatch_to_time_stats(sub)
        return
    dispatch = getattr(req, _DISPATCH_ATTR, None)
    if dispatch is None:
        return
    time_stats = getattr(req, "time_stats", None)
    if time_stats is not None:
        setattr(time_stats, _DISPATCH_ATTR, dispatch)


@plugin_hook(
    "sglang.srt.managers.scheduler_components.request_receiver."
    "SchedulerRequestReceiver.recv_requests",
    type=HookType.AFTER,
)
def _bridge_dispatch_time(_result, *args, **kwargs) -> None:
    if not _result:
        return
    for req in _result:
        _bridge_dispatch_to_time_stats(req)


@plugin_hook(
    "sglang.srt.observability.req_time_stats."
    "SchedulerReqTimeStats.set_scheduler_recv_time",
    type=HookType.AROUND,
)
def _stamp_arrive_scheduler_time(original_fn, self, ts=None):
    # ``set_scheduler_recv_time`` is called from ``Req.__init__`` -- the same
    # point the fork stamped ``arrive_scheduler_time``. ``scheduler_recv_time`` is
    # set by the original.
    result = original_fn(self, ts)
    dispatch = getattr(self, _DISPATCH_ATTR, 0.0)
    if dispatch > 0.0 and self.scheduler_recv_time > 0.0:
        self.kunlun_zmq_time = max(0.0, self.scheduler_recv_time - dispatch)
    return result


# ────────────────────── per-request latency histograms ───────────────────────


@plugin_hook(
    "sglang.srt.observability.req_time_stats."
    "SchedulerReqTimeStats.set_forward_entry_time",
    type=HookType.AROUND,
)
def _observe_request_waiting_time(original_fn, self, ts=None):
    # AROUND (pre-check) so a repeat call (or a post-retract reset) cannot emit a
    # bogus sample computed from a stale ``forward_entry_time``.
    first_entry = self.forward_entry_time == 0.0
    result = original_fn(self, ts)
    if not (first_entry and self.enable_metrics):
        return result

    zmq_time = getattr(self, "kunlun_zmq_time", 0.0)
    if zmq_time > 0.0 and self.metrics_collector.aiak_request_zmq_time is not None:
        self.metrics_collector._log_histogram(
            self.metrics_collector.aiak_request_zmq_time, zmq_time
        )

    from sglang.srt.disaggregation.utils import DisaggregationMode

    if self.disagg_mode == DisaggregationMode.PREFILL:
        start = self.prefill_bootstrap_queue_entry_time
    elif self.disagg_mode == DisaggregationMode.DECODE:
        start = self.decode_transfer_queue_entry_time
    else:
        start = self.wait_queue_entry_time
    if start > 0.0 and self.metrics_collector.aiak_request_waiting_time is not None:
        self.metrics_collector._log_histogram(
            self.metrics_collector.aiak_request_waiting_time,
            self.forward_entry_time - start,
        )
    return result


@plugin_hook(
    "sglang.srt.observability.req_time_stats."
    "SchedulerReqTimeStats.set_completion_time",
    type=HookType.AROUND,
)
def _observe_request_first_token_forward_time(original_fn, self, ts=None):
    result = original_fn(self, ts)
    if not self.enable_metrics:
        return result

    from sglang.srt.disaggregation.utils import DisaggregationMode

    if self.disagg_mode == DisaggregationMode.PREFILL:
        # The P node finishes a request when its KV transfer completes, so
        # ``completion_time`` *is* the first-token-out moment.
        start = self.prefill_bootstrap_queue_entry_time
    elif self.disagg_mode == DisaggregationMode.DECODE:
        return result
    elif self.last_decode_finish_time == 0.0:
        # Non-PD: only emit for requests that finished on their first token.
        start = self.wait_queue_entry_time
    else:
        return result

    hist = self.metrics_collector.aiak_request_first_token_forward_time
    if start > 0.0 and hist is not None:
        self.metrics_collector._log_histogram(hist, self.completion_time - start)
    return result


# ─────────────────────────────── iteration timing ────────────────────────────────
#
# Under the overlap scheduler one loop iteration launches ``batch_N`` and then
# consumes ``batch_{N-1}`` (scheduler.py: ``run_batch`` -> ``result_queue`` ->
# ``pop_and_process``). ``run_batch_time`` is therefore read from a single global
# start scalar, exactly as the fork does: the span is "launch of N .. N-1's device
# copy landed", i.e. the time the scheduler sits blocked on the GPU, which on a
# saturated engine is one step's forward time.
#
# Pairing the span to its own batch instead (start_{N-1} .. finish_{N-1}) looks
# more correct but also counts N-1's queueing wait behind N-2, which roughly
# doubles the value and collapses ``iter_token_process_time`` to 0.
#
# ``generation_time`` is the commit-to-commit cadence rather than one Python-loop
# wall time: under overlap the loop is not the step, so a per-loop span can be
# shorter than the measured forward and the subtraction would underflow.


def _run_batch_time() -> float:
    """Latest forward span: ``run_batch`` entry -> the device copy landing."""
    return max(STATE.forward_finish_time - STATE.forward_start_time, 0.0)


@plugin_hook(
    "sglang.srt.managers.scheduler.Scheduler.run_batch",
    type=HookType.AROUND,
)
def _stamp_forward_start(original_fn, self, batch, *args, **kwargs):
    # Same position as the fork's ``iter_forward_start_time`` (top of ``run_batch``).
    if STATE.enabled:
        STATE.forward_start_time = time.perf_counter()
    return original_fn(self, batch, *args, **kwargs)


def _stamp_forward_finish(result) -> None:
    # Stamp *after* the device copy has landed so ``run_batch_time`` covers the
    # real forward, matching the fork (which stamped just past the ``.tolist()``
    # D2H in the output processor). ``synchronize()`` on an already-completed
    # event is a no-op, so doing it before the original runs does not add a
    # stall.
    copy_done = getattr(result, "copy_done", None)
    if copy_done is not None:
        copy_done.synchronize()
    STATE.forward_finish_time = time.perf_counter()


@plugin_hook(
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_decode",
    type=HookType.AROUND,
)
def _stamp_forward_finish_decode(original_fn, self, batch, result, *args, **kwargs):
    # Decode stamps the forward finish and commits ``run_batch_time`` here;
    # ``generation_time`` / ``iter_token_process_time`` are committed just below in
    # ``report_decode_stats`` (same ordering as the fork).
    if STATE.enabled:
        _stamp_forward_finish(result)
        STATE.run_batch_time = _run_batch_time()
    return original_fn(self, batch, result, *args, **kwargs)


@plugin_hook(
    "sglang.srt.managers.scheduler_components.metrics_reporter."
    "SchedulerMetricsReporter.report_decode_stats",
    type=HookType.BEFORE,
)
def _commit_iter_timing(*args, **kwargs) -> None:
    # Runs once per decode step (process_batch_result_decode -> report_decode_stats).
    # ``generation_time`` is the step cadence (time between consecutive commits):
    # under the overlap scheduler that is the real per-step wall time, whereas a
    # single Python-loop span is shorter than the async forward and made
    # ``iter_token_process_time`` negative. ``run_batch_time`` was already paired
    # to this step's batch in ``_stamp_forward_finish_decode``.
    if not STATE.enabled:
        return
    now = time.perf_counter()
    if STATE.last_commit_time > 0.0:
        STATE.generation_time = now - STATE.last_commit_time
        STATE.iter_token_process_time = max(
            STATE.generation_time - STATE.run_batch_time, 0.0
        )
    STATE.last_commit_time = now


@plugin_hook(
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_prefill",
    type=HookType.AROUND,
)
def _stamp_forward_finish_prefill(original_fn, self, batch, result, *args, **kwargs):
    # Prefill commits ``run_batch_time`` (and only that) from its output path;
    # ``generation_time`` / ``iter_token_process_time`` keep the last decode value
    # so a prefill iteration cannot spike them.
    if STATE.enabled:
        _stamp_forward_finish(result)
        STATE.run_batch_time = _run_batch_time()
    return original_fn(self, batch, result, *args, **kwargs)


@plugin_hook(
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
    "process_batch_result_disagg_prefill",
    type=HookType.AROUND,
)
def _stamp_forward_finish_disagg_prefill(
    original_fn, self, batch, result, *args, **kwargs
):
    # PD prefill dispatches to process_batch_result_disagg_prefill, not
    # process_batch_result_prefill (scheduler.py:3379-3382), so without this hook
    # the one above never fires and run_batch_time stays 0 on a prefill node.
    if STATE.enabled:
        _stamp_forward_finish(result)
        STATE.run_batch_time = _run_batch_time()
    return original_fn(self, batch, result, *args, **kwargs)


@plugin_hook(
    "sglang.srt.managers.scheduler_components.metrics_reporter."
    "SchedulerMetricsReporter.report_prefill_stats",
    type=HookType.AROUND,
)
def _fix_prefill_num_running_reqs(
    original_fn, self, batch, prefill_stats, *args, **kwargs
):
    # num_running_reqs on a PREFILL node is the current prefill batch size
    # (scheduler_metrics_mixin.py:186-189: `self.running_batch` is decode-only, so
    # use the prefill batch size -> prefill_stats.num_new_seqs). Upstream instead
    # feeds self.running_batch.reqs into PrefillStats.from_adder
    # (scheduler.py:2932), which is empty on a prefill node, so it is always 0.
    # Re-log it from prefill_stats.num_new_seqs.
    result = original_fn(self, batch, prefill_stats, *args, **kwargs)
    scheduler = getattr(self, "scheduler", None)
    mc = getattr(self, "metrics_collector", None)
    if scheduler is None or mc is None:
        return result
    from sglang.srt.disaggregation.utils import DisaggregationMode

    if getattr(scheduler, "disaggregation_mode", None) != DisaggregationMode.PREFILL:
        return result
    # Same representative-rank gate as the other prefill-side metrics.
    if not _is_representative_rank(scheduler):
        return result
    mc._log_gauge(mc.num_running_reqs, getattr(prefill_stats, "num_new_seqs", 0))
    return result


# ──── prefill-side per-request metrics (prompt length + request counts) ───────
#
# ``per_prompt_len`` / ``prompt_tokens_histogram``, ``num_requests_total`` /
# ``num_aborted_requests_total``, ``prompt_tokens_total`` / ``cached_tokens_total``
# and ``kvcache_transfer_time`` / ``prompt_calculation_time`` all normally ride the
# tokenizer, which under PD only fires on DECODE. Re-emit them here from
# the P scheduler: the inflight-queue drain returns the reqs whose KV transfer
# just completed (prefill.py:799-835). All are defined in ``_define_aiak_metrics``.


@plugin_hook(
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
    "process_disagg_prefill_inflight_queue",
    type=HookType.AFTER,
)
def _observe_prefill_prompt_len(_result, self, *args, **kwargs) -> None:
    if not STATE.enabled or not _result:
        return
    # Emit on the single representative rank per DP group (see
    # ``_is_representative_rank``); otherwise a ``sum by(...)`` across TP/PP/CP
    # ranks reads N x the true value.
    if not _is_representative_rank(self):
        return
    collector = getattr(self, "metrics_collector", None)
    if collector is None:
        return
    hists = [
        h
        for h in (
            getattr(collector, "aiak_per_prompt_len", None),
            getattr(collector, "aiak_prompt_tokens_histogram", None),
        )
        if h is not None
    ]
    hist_kv = getattr(collector, "aiak_kvcache_transfer_time", None)
    hist_calc = getattr(collector, "aiak_prompt_calculation_time", None)
    num_requests = getattr(collector, "aiak_num_requests_total", None)
    num_aborted = getattr(collector, "aiak_num_aborted_requests_total", None)
    prompt_tokens = getattr(collector, "aiak_prompt_tokens_total", None)
    cached_tokens = getattr(collector, "aiak_cached_tokens_total", None)
    if (
        not hists
        and hist_kv is None
        and hist_calc is None
        and num_requests is None
        and num_aborted is None
        and prompt_tokens is None
        and cached_tokens is None
    ):
        return

    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
    except Exception:
        FAKE_BOOTSTRAP_HOST = None

    for req in _result:
        # Skip fake-bootstrap (single-node profiling) requests entirely: they are
        # neither a real completion nor a real abort.
        if (
            FAKE_BOOTSTRAP_HOST is not None
            and getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST
        ):
            continue

        # Aborted prefills (bootstrap/transfer failure -> FINISH_ABORT) count
        # toward num_aborted_requests_total, and are NOT counted as completions
        # nor sampled into the prompt-length histograms.
        reason = getattr(req, "finished_reason", None)
        if reason is not None and type(reason).__name__ == "FINISH_ABORT":
            if num_aborted is not None:
                num_aborted.labels(**collector.labels).inc(1)
            continue

        # Successful prefill completion: request count + prompt length + the
        # prefill-side latency histograms.
        if num_requests is not None:
            num_requests.labels(**collector.labels).inc(1)

        origin_ids = getattr(req, "origin_input_ids", None)
        if origin_ids is not None:
            prompt_len = float(len(origin_ids))
            for hist in hists:
                collector._log_histogram(hist, prompt_len)
            if prompt_tokens is not None:
                prompt_tokens.labels(**collector.labels).inc(prompt_len)

        # Cache breakdown. Reuses the output streamer's own helper
        # (output_streamer.py:58-80) so the ``cache_source`` values are byte-for-byte
        # the ones the tokenizer side would publish, including the storage backend
        # suffix.
        if cached_tokens is not None:
            streamer = getattr(self, "output_streamer", None)
            details = (
                streamer.get_cached_tokens_details(req)
                if streamer is not None
                else None
            )
            if details:
                _backend = details.get("storage_backend") or "unknown"
                for _source, _value in (
                    ("device", details.get("device", 0)),
                    ("host", details.get("host", 0)),
                    (f"storage_{_backend}", details.get("storage", 0)),
                ):
                    if _value > 0:
                        cached_tokens.labels(
                            **collector.labels, cache_source=_source
                        ).inc(_value)
            elif getattr(req, "cached_tokens", 0) > 0:
                # No breakdown available: same fallback as the tokenizer side
                # (metrics_collector.py:1670-1673).
                cached_tokens.labels(**collector.labels, cache_source="total").inc(
                    req.cached_tokens
                )

        ts = getattr(req, "time_stats", None)
        if ts is not None and (hist_kv is not None or hist_calc is not None):
            forward_entry = getattr(ts, "forward_entry_time", 0.0) or 0.0
            transfer_in = getattr(ts, "prefill_transfer_queue_entry_time", 0.0) or 0.0
            completion = getattr(ts, "completion_time", 0.0) or 0.0
            # prompt_calculation_time = prefill forward span (prefill.py:545).
            if hist_calc is not None and forward_entry > 0.0 and transfer_in > 0.0:
                collector._log_histogram(hist_calc, transfer_in - forward_entry)
            # kvcache_transfer_time = KV transfer span (prefill.py:678).
            if hist_kv is not None and transfer_in > 0.0 and completion > 0.0:
                collector._log_histogram(hist_kv, completion - transfer_in)
