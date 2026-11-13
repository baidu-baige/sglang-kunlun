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
"""Kunlun plugin hooks for ``sglang.srt.layers.moe.token_dispatcher.deepep``."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
    is_tbo_enabled,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_blackwell,
    load_json_config,
)


if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    from deep_ep import Config
    from deep_ep.buffer_v2 import BufferV2 as Buffer
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


_DEEPEP_COMBINE_SYNC = None


def _deepep_combine_sync_enabled() -> bool:
    """Whether to force the combine completion wait (default: yes)."""
    global _DEEPEP_COMBINE_SYNC
    if _DEEPEP_COMBINE_SYNC is None:
        import os

        _DEEPEP_COMBINE_SYNC = (
            os.environ.get("SGLANG_KUNLUN_DEEPEP_COMBINE_SYNC", "1") != "0"
        )
        if not _DEEPEP_COMBINE_SYNC:
            logger.warning(
                "[kunlun] SGLANG_KUNLUN_DEEPEP_COMBINE_SYNC=0: skipping the deepep "
                "combine completion wait, which is known to corrupt MoE output on "
                "this backend"
            )
    return _DEEPEP_COMBINE_SYNC


_comm_stream_state = None
# stream id -> torch.cuda.ExternalStream.  ``buffer.get_comm_stream()`` builds a
# fresh ``ExternalStream`` on every call and that constructor re-resolves the
# device index through ``torch/cuda/_utils.py`` + ``torch/_utils.py`` each time
# (~30 us per call in a profiled prefill; 2 calls per MoE layer).  The stream id
# is fixed for a buffer's lifetime, so the object is cached per id.
_comm_streams: dict = {}


def _cached_comm_stream(buffer):
    """``buffer.get_comm_stream()`` without rebuilding the wrapper every call."""
    stream_id = int(buffer.runtime.get_cuda_stream_id())
    stream = _comm_streams.get(stream_id)
    if stream is None:
        stream = torch.cuda.ExternalStream(stream_id)
        _comm_streams[stream_id] = stream
    return stream


def _order_after_comm_stream(buffer) -> bool:
    """Make the caller's stream wait for deep_ep's comm stream (no host stop).

    ``libbkcl``'s all-to-all runs on deep_ep's own stream and neither the
    ``async_finish`` event nor ``EventOverlap`` orders it against the caller's
    stream, which is why a host-wide sync was used.  Ordering the caller stream
    against the *whole* comm stream with an event is the same dependency without
    blocking the host.  Returns False (caller falls back to the device sync) if
    the stream is not reachable on this build.
    """
    global _comm_stream_state
    try:
        comm = _cached_comm_stream(buffer)
        if comm is None:
            raise RuntimeError("deep_ep returned no comm stream")
        torch.cuda.current_stream().wait_stream(comm)
        if _comm_stream_state is None:
            _comm_stream_state = True
            logger.warning(
                "[kunlun] deep_ep comm-stream ordering active: caller stream "
                "waits the comm stream device-side (no host sync per layer)"
            )
        return True
    except Exception as exc:
        if _comm_stream_state is None:
            _comm_stream_state = False
            logger.warning(
                "[kunlun] deep_ep comm-stream ordering unavailable (%s); "
                "falling back to torch.cuda.synchronize()",
                exc,
            )
        return False


def _deepep_completion_wait(buffer) -> None:
    """Order the caller's stream after a dispatch/combine all-to-all.

    Device-side event ordering (no host stop); the host-wide sync remains only
    as the fallback for builds where the comm stream is unreachable.  Measured
    8x60k EP-fake prefill on 8xP800: 23813.9 -> 24079.7 tok/s, +1.12%.
    """
    if not _deepep_combine_sync_enabled() or not _not_capturing_cuda_graph():
        return
    if _order_after_comm_stream(buffer):
        return
    torch.cuda.synchronize()


_capture_mode_probe = None
_capture_mode_probed = False


def _not_capturing_cuda_graph() -> bool:
    """Whether we are outside CUDA-graph capture (a host sync inside capture fails).

    The probe used to re-run ``from ... import`` on every call (2x per MoE
    layer); the module reference is resolved once and reused.
    """
    global _capture_mode_probe, _capture_mode_probed
    if not _capture_mode_probed:
        _capture_mode_probed = True
        try:
            from sglang.srt.model_executor.runner_utils.capture_mode import (
                get_is_capture_mode,
            )

            _capture_mode_probe = get_is_capture_mode
        except Exception:
            _capture_mode_probe = None
    if _capture_mode_probe is None:
        return True
    try:
        return not _capture_mode_probe()
    except Exception:
        return True


def _kunlun_wait_event(event) -> None:
    """Stream-ordered wait on an async deep_ep dispatch/combine (capturable).

    Upstream waits with ``hook() if self.return_recv_hook else
    event.current_stream_wait()``; the kunlun port left that else-branch
    commented out and relied on the ``torch.cuda.synchronize()`` in
    dispatch_b/combine_b instead. That sync is (correctly) skipped under
    cuda-graph capture (``_not_capturing_cuda_graph``), so inside a captured
    decode graph there was *no* completion wait at all and the next layer could
    read a half-landed all-to-all. The event deep_ep returns is stream-ordered,
    so waiting on it is both correct and capturable.
    """
    if event is None:
        return
    wait = getattr(event, "current_stream_wait", None)
    if wait is not None:
        wait()


def _kunlun_release_event(owner, attr: str) -> None:
    """Drop a deep_ep event once its completion wait has been issued.

    ``buffer_v2.low_latency_{dispatch,combine}`` return
    ``EventOverlap(event, tensors_to_record)``, and deep_ep keeps the input, the
    packed recv buffer and the layout tensors alive through that tuple because
    ``Tensor.record_stream`` cannot be used under cuda-graph capture. That
    reference is only needed for the async window between launching the kernel
    and waiting on it -- a window ``dispatch_b``/``combine_b`` have just closed
    with ``_kunlun_wait_event``. The wait is stream-ordered, so everything the
    caller issues afterwards, including the allocator handing the same block to
    another tensor, is ordered after the all-to-all.

    Keeping the event instead pins, per MoE layer, one packed recv buffer
    (``[num_local_experts, tokens_per_rank * ranks, hidden]``) plus the combine
    input: ~360 MiB per layer at tokens/rank=64, ~14 GiB over the 40 layers of
    DSV4.1-Flash. Eager decode carried that as a permanent floor, and during
    cuda-graph capture it also forced the graph's private pool to size for every
    layer at once instead of reusing one layer's scratch. Measured on an 8-rank
    tp8/ep8/dp8 DSV4.1-Flash decode (tokens/rank=64, graph bs up to 64): capture
    peak 34.1 -> 2.3 GiB, resident graph pool 20.7 -> 2.5 GiB.
    """
    event = getattr(owner, attr, None)
    if event is None:
        return
    if hasattr(event, "extra_tensors"):
        event.extra_tensors = None
    setattr(owner, attr, None)


class DeepEPPDispatchHooks(DispatcherBaseHooks):
    """DeepEPPDispatchHooks"""

    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        """format"""
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        """format"""
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP normal combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        """format"""
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        """format"""
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPDispatchMode(IntEnum):
    """DeepEPDispatchMode"""
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    """DeepEPBuffer"""
    _buffers = {}
    _dispatch_modes = {}
    _active_key = None
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
    ):
        """get_deepep_buffer"""
        # Keep a strong reference to the communication group in the key. Equal
        # world sizes do not imply equal communication domains or layouts.
        key = (
            group, hidden_size, param_bytes, deepep_mode,
            num_max_dispatch_tokens_per_rank, num_experts,
        )
        cls._active_key = key
        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts
        cls._dispatch_mode = cls._dispatch_modes.get(key)
        cls._buffer = cls._buffers.get(key)
        if cls._buffer is not None:
            return cls._buffer

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
        if deepep_mode == DeepEPMode.NORMAL:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms
        elif deepep_mode == DeepEPMode.LOW_LATENCY:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        elif deepep_mode == DeepEPMode.AUTO:
            # low-latency and normal mode all need run
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = max(
                DeepEPConfig.get_instance().num_sms, num_experts // group.size()
            )
        else:
            raise NotImplementedError

        # if not _is_npu:
        total_num_sms = torch.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count
        if (
            (deepep_mode != DeepEPMode.LOW_LATENCY)
            and not is_tbo_enabled()
            and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)
        ):
            logger.warning(
                f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                f"This may result in highly suboptimal performance. "
                f"Consider using --deepep-config to change the behavior."
            )
        
        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            # TODO can be false when unneeded
            allow_mnnvl=True,
            num_experts=num_experts,
        )
        cls._buffers[key] = cls._buffer
        
        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        """clean_buffer"""
        if cls._buffer is None or not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        """set_dispatch_mode_as_normal"""
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL
        cls._dispatch_modes[cls._active_key] = cls._dispatch_mode

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        """set_dispatch_mode_as_low_latency"""
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY
        cls._dispatch_modes[cls._active_key] = cls._dispatch_mode

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):
        """set_dispatch_mode"""
        if mode.is_low_latency():
            cls.set_dispatch_mode_as_low_latency()
        elif mode.is_normal():
            cls.set_dispatch_mode_as_normal()
        else:
            raise Exception("unsupported mode")


@plugin_hook(
    "sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPConfig",
    type=HookType.REPLACE,
)
class DeepEPConfig(BaseDispatcherConfig):
    """Kunlun DeepEP dispatcher configuration."""
    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        """get_instance"""
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    """_DeepEPDispatcherImplBase"""
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        # A large value will lead to large memory occupation, thus users should change it accordingly
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch_a"""
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        """dispatch_b"""
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine_a"""
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        """combine_b"""
        raise NotImplementedError

    def _get_buffer(self):
        """_get_buffer"""
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        """set_quant_config"""
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        """set_overlap_args"""
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        """clear_overlap_args"""
        self.overlap_args = None
        self.meta_overlap_args = None


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    """_DeepEPDispatcherImplNormal"""
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch_a"""
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        
        import kunlun_ops
        topk_ids = topk_ids.to(torch.int32)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states_int8 = torch.empty(
                num_tokens,
                hidden_size,
                device='cuda',
                dtype=torch.int8
        )
        hidden_states_scale = torch.empty(
                num_tokens,
                1,
                device='cuda',
                dtype=torch.float32,
        ) # can't be col major
        if num_tokens > 0:
            kunlun_ops.quant2d(hidden_states, hidden_states_int8, hidden_states_scale)
        x_to_dispatch = (hidden_states_int8, hidden_states_scale)
        hidden_states = x_to_dispatch
        
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        """dispatch_b"""
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        # Symmetric with combine_b: libbkcl's dispatch is likewise not
        # ordered against the caller's stream.
        _deepep_completion_wait(self._get_buffer())

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        """_dispatch_core"""
        buffer = self._get_buffer()

        num_tokens_per_rank = None
        num_tokens_per_rdma_rank = None
        num_tokens_per_expert = None
        is_token_in_rank = None
        
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.
        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine_a"""
        output = hidden_states

        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        """combine_b"""
        hidden_states, event = self._combine_core(output, previous_event)
        # Kunlun: ``buffer.combine()`` is NOT ordered against the caller's stream
        # even when ``async_finish=False`` (the deepep backend here is libbkcl, not
        # the CUDA reference implementation), so the consumer of ``hidden_states``
        # can observe a half-landed all-to-all. Upstream's
        # ``... if self.async_finish else ()`` therefore skips the only completion
        # wait and the next layer runs on a partially written tensor: garbage
        # hidden states -> dead logits -> the sampler emits token id 0
        # (``<|begin_of_sentence|>``) forever, and on longer sequences the same
        # window widens into ``error code= 700, an illegal memory access``.
        #
        # Measured on DeepSeek-V4-Flash, TP8/EP8, 781-token bare prompt, 20 trials,
        # temperature=0: 0/20 usable without this wait (and the server dies), 19/20
        # with it; syncing after ``dispatch_b`` instead only reaches 2/20, so the
        # missing wait is specifically on the combine side.
        #
        # Set SGLANG_KUNLUN_DEEPEP_COMBINE_SYNC=0 to restore the upstream
        # (defective on kunlun) behaviour -- used for the negative half of the
        # A/B proof.
        event.current_stream_wait() if self.async_finish else ()
        # Unconditional: the event wait above is not sufficient on this
        # backend (that is the defect -- libbkcl's combine is not ordered
        # against the caller's stream), and with async_finish=False there is
        # no wait at all. Measured 0/20 -> 19/20 on a 781-token prompt.
        # ``_deepep_completion_wait`` keeps that dependency (wait on the whole
        # comm stream) but does it device-side when the mode allows, so the
        # host is not stopped on every MoE layer.
        _deepep_completion_wait(self._get_buffer())
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        """_combine_core"""
        buffer = self._get_buffer()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        """_get_buffer"""
        buffer = DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )
        DeepEPBuffer.set_dispatch_mode_as_normal()
        return buffer


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    """_DeepEPDispatcherImplLowLatency"""
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()
        self.quant_config = {}
        buffer = self._get_buffer()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch_a"""
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        # KUNLUN_LL_IDLE_PAD: the native low_latency dispatch/combine faults -714 when a
        # dp-attention DP rank is idle (0 tokens). Pad to a single dummy token so this rank
        # still joins the collective with a valid shape; combine_b drops the dummy row.
        self._ll_orig_num_tokens = hidden_states.shape[0]
        if self._ll_orig_num_tokens == 0:
            hidden_states = hidden_states.new_zeros((1, hidden_states.shape[1]))
            topk_ids = topk_ids.new_zeros((1, topk_ids.shape[1]))
            topk_weights = topk_weights.new_zeros((1, topk_weights.shape[1]))

        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        hook,
    ):
        """dispatch_b"""
        if self.return_recv_hook:
            hook()
        else:
            _kunlun_wait_event(getattr(self, "_ll_dispatch_event", None))
        _kunlun_release_event(self, "_ll_dispatch_event")

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        deepep_output = DeepEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """_dispatch_core"""
        use_nvfp4 = use_fp8 = False
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            use_nvfp4 = True
        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
            use_fp8 = True

        buffer = self._get_buffer()
        # Kunlun deep_ep low_latency_dispatch asserts a 2-D contiguous bfloat16
        # input; the float16 model hidden_states must be cast to bf16 first.
        if isinstance(hidden_states, torch.Tensor):
            if not hidden_states.is_cuda:
                hidden_states = hidden_states.to(torch.cuda.current_device())
            if hidden_states.dtype != torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
            hidden_states = hidden_states.contiguous()
        if isinstance(topk_ids, torch.Tensor) and not topk_ids.is_cuda:
            topk_ids = topk_ids.to(torch.cuda.current_device())
            if not hidden_states.is_cuda:
                hidden_states = hidden_states.to(torch.cuda.current_device())
        if isinstance(topk_ids, torch.Tensor) and not topk_ids.is_cuda:
            topk_ids = topk_ids.to(torch.cuda.current_device())
        packed_recv_hidden, self.packed_recv_count, self.handle, _ll_event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_ids,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        self._ll_dispatch_event = _ll_event
        return packed_recv_hidden, self.packed_recv_count, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine_a"""
        hidden_states, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, hook

    def combine_b(self, hidden_states, hook):
        """combine_b"""
        overlap_args = self.overlap_args
        if overlap_args is not None:
            overlap_args.stream.wait_stream(self.device_module.current_stream())

        if self.return_recv_hook:
            hook()
        else:
            _kunlun_wait_event(getattr(self, "_ll_combine_event", None))

        # Same defect as the normal-mode combine_b above: when
        # ``return_recv_hook`` is False there is no completion wait at all, and on
        # this backend ``low_latency_combine`` is not ordered against the caller's
        # stream, so the next layer can read a half-landed all-to-all.
        if _deepep_combine_sync_enabled() and _not_capturing_cuda_graph():
            torch.cuda.synchronize()

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        # The combine result is written asynchronously. Converting it in
        # _combine_core would enqueue a read before this completion wait.
        if (
            self.params_dtype is not None
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.dtype != self.params_dtype
        ):
            hidden_states = hidden_states.to(self.params_dtype)

        _kunlun_release_event(self, "_ll_combine_event")

        # KUNLUN_LL_IDLE_PAD: drop the dummy row added for an idle rank in dispatch_a.
        if getattr(self, "_ll_orig_num_tokens", None) == 0:
            hidden_states = hidden_states[:0]
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """_combine_core"""
        buffer = self._get_buffer()
        overlap_args = self.overlap_args
        meta_overlap_args = self.meta_overlap_args

        ctx = nullcontext()
        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            ctx = torch.cuda.stream(overlap_args.stream)

            if is_blackwell():
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    src_signals=overlap_args.signal,
                    src_signal_expect_value=overlap_args.threshold,
                )
            else:
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    packed_recv_count=self.packed_recv_count,
                    comp_signal=overlap_args.signal,
                    block_m=meta_overlap_args["block_m"],
                    threshold=meta_overlap_args["threshold"],
                    num_sms=overlap_args.num_sms,
                )
        else:
            overlap_args_dict = {}

        with ctx:
            if isinstance(hidden_states, torch.Tensor):
                if hidden_states.dtype != torch.bfloat16:
                    hidden_states = hidden_states.to(torch.bfloat16)
                hidden_states = hidden_states.contiguous()
            combined_hidden_states, _ll_event, hook = buffer.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                handle=self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **overlap_args_dict,
            )

        self._ll_combine_event = _ll_event
        self.packed_recv_count = self.handle = None
        return combined_hidden_states, hook

    def _get_buffer(self):
        """_get_buffer"""
        buffer = DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return buffer


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


@plugin_hook(
    "sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPDispatcher",
    type=HookType.REPLACE,
)
class DeepEPDispatcher(BaseDispatcher):
    """Kunlun DeepEP dispatcher implementation."""
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        # KUNLUN_LL_NORMAL_FALLBACK: stash kwargs so we can lazily build the NORMAL dispatcher even when
        # --deepep-mode low_latency was requested (Kunlun native low_latency_combine faults
        # -714 on dp-attention idle ranks; we fall back to idle-tolerant NORMAL).
        self._normal_common_kwargs = common_kwargs
        self._normal_async_finish = async_finish
        self._normal_dispatcher = None
        self._low_latency_dispatcher = None
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        """dispatch"""
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch_a"""
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        """dispatch_b"""
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        """combine"""
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        """combine_a"""
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        """combine_b"""
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _ensure_normal_dispatcher(self) -> _DeepEPDispatcherImplBase:
        """KUNLUN_LL_NORMAL_FALLBACK: lazily build the NORMAL dispatcher (not created when mode==low_latency)."""
        if self._normal_dispatcher is None:
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=self._normal_async_finish,
                **self._normal_common_kwargs,
            )
        return self._normal_dispatcher

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        """_get_impl"""
        # KUNLUN_LL_IDLE_PAD: use the NORMAL dispatcher only for AUTO/NORMAL modes. An explicit
        # --deepep-mode low_latency decode MUST use the capturable LL path: NORMAL dispatch calls
        # bkcl_notify_dispatch_standard (dynamic, blocking) which deadlocks inside the decode
        # CUDA-graph capture -> notify_dispatch timeout. Idle DP ranks (the reason the old code
        # force-fell-back to NORMAL under dp-attention) are handled by the 1-token padding in
        # _DeepEPDispatcherImplLowLatency.dispatch_a/combine_b above, so LL is now idle-tolerant.
        if self.deepep_mode.enable_normal():
            return self._ensure_normal_dispatcher()
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        """_update_stage"""
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        """set_quant_config"""
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        """set_overlap_args"""
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        """clear_overlap_args"""
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        """register_deepep_dispatch_hook"""
        return self._deepep_dispatch_hooks.register_hook(hook)
