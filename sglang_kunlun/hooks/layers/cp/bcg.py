"""Kunlun hook for ``sglang.srt.layers.cp.bcg`` (CP + breakable CUDA graph).

Upstream gates context-parallel prefill capture on ``supports_prefill_cp_bcg``,
which only whitelists ``cp_strategy=zigzag`` with the ``trtllm_mha`` prefill
backend. Kunlun runs DSV4 prefill CP as the *interleave* (round-robin) strategy
with the ``kunlun_compressed`` backend. Everything else the CP-BCG path needs is
already strategy-agnostic and available here:

* ``init_cp_strategy`` selects ``InterleaveCPStrategy`` for ``--cp-strategy
  interleave``, and that strategy implements ``can_apply`` /
  ``shard_hidden_states`` / ``shard_position_ids`` / ``gather_hidden_states``
  for token ``i -> rank i % cp_size``.
* ``SGLANG_ENABLE_CP_V2`` is already auto-enabled for DSA + interleave in
  ``ServerArgs._handle_context_parallelism``, so ``is_cp_v2_active`` and the
  CP-local BCG input (``PrefillCPBCGInput``) work without further plumbing.
* ``DeepEPMoE.forward`` (Kunlun plugin) carries the BCG eager-break contract
  that keeps the rank-coupled DeepEP NORMAL a2a out of the captured graph.

So the only missing piece is the whitelist entry. Keep the upstream predicate
as the fallback: a configuration upstream already accepts keeps its result, and
every non-Kunlun configuration keeps its upstream verdict.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_KUNLUN_PREFILL_CP_STRATEGY = "interleave"
_KUNLUN_PREFILL_ATTENTION_BACKEND = "kunlun_compressed"

# 0.5.19 的 core 已经原生实现了 0.5.17 时代由本插件承担的 CP-v2 + BCG 几何
# （init_forward_metadata_prefill 自己 pad + reindex、PrefillCPBCGInput.prepare 自己
# 分片、模型侧有自己的 CP-v2 分支），所以 0.5.17 的那组「padded-global 元数据 +
# 手动 reindex」钩子已整体删除，这里只保留 Kunlun 特有的部分：
#   * supports_prefill_cp_bcg 放宽到 interleave + kunlun_compressed，
#   * CP-v2 强制（server_args 钩子，见 hooks/server_args.py），
#   * 压缩器 eager 断点 + 紧凑 plan 重建（build/replay 两个钩子），
#   * 回放尾部的 DSV4 元组契约、interleave 回放桶选择，
#   * ragged draft 保持 pre-CP-v2 布局。


@plugin_hook(
    "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
    type=HookType.REPLACE,
)
def is_dsa_enable_prefill_cp_kunlun():
    """Keep an explicit DSA prefill CP request authoritative under CP-v2.

    Upstream's CP-v2 branch answers this from ``is_deepseek_dsa(config)``, whose
    architecture allowlist predates ``DeepseekV4ForCausalLM``. With CP-v2
    enabled for DSV4 that would silently switch off the DSA CP row split, the
    communicator's CP moves, the compressor's CP layout and the Kunlun
    attention's CP branch -- while CP-v2 itself shards the embeddings, leaving
    the two layouts inconsistent. The explicit flag (set by the DSV4 CP
    validation at argument-resolution time) wins; everything else keeps the
    upstream verdict.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.enable_dsa_prefill_context_parallel:
        return True
    if not envs.SGLANG_ENABLE_CP_V2.get() or parallel.attn_cp_size <= 1:
        return False

    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.arg_groups.overrides import model_config_of
    from sglang.srt.runtime_context import get_server_args

    # 0.5.19 dropped ServerArgs.get_model_config(); the pipeline's resolver-safe
    # helper is the way in.
    return is_deepseek_dsa(model_config_of(get_server_args()).hf_config)


# NOTE: 0.5.17 的 ``cp_split_and_rebuild_data/position`` REPLACE 钩子在 0.5.19 下线：
# core 的 CP-v2 分片（``PrefillCPBCGInput.prepare`` + 模型侧 CP-v2 分支）已经把行
# 分片与 CP-v2 embedding 对齐，插件再替换一次会与原生几何冲突。


def _rebuild_compact_compress_plans_kunlun(self, forward_batch, metadata):
    """Rebuild the c4/c128 plans the way the eager path builds them.

    The compress kernels process one row of ``plan_c`` per row of work, so the
    plan's row count *is* the kernel's cost.  The vendor planner returns the
    compact plan (one row per compression group: ``num_q_tokens / ratio``) only
    when it is handed **CPU** ``seq_lens``/``extend_lens`` *and*
    ``use_cuda_graph=False``; with device tensors -- which the capture-shaped
    metadata path passes, because it withholds the CPU lists -- it pads the plan
    to ``num_q_tokens`` rows instead.  Measured on an 8192-token prefill step:
    c128 ``infer_compress_fast_c128_compress`` 19.36ms per call with the padded
    plan vs 0.26ms with the compact one, 20 calls per step.

    The padded shape only exists to keep the plan's shape stable for capture.
    With the compressor running as an eager break (``forward_core_compressor``
    above) nothing captured reads these tensors, so the eager build is both safe
    and much cheaper; ``refresh_for_breakable_cuda_graph_replay_kunlun`` keeps
    the per-replay objects instead of copying them into the capture-time ones.
    """

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        create_paged_compressor_data,
    )

    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if seq_lens_cpu is None or extend_lens_cpu is None:
        return

    def _as_list(value):
        return list(value.tolist()) if hasattr(value, "tolist") else list(value)

    for ratio in (4, 128):
        attr_name = f"c{ratio}_compress_metadata"
        if getattr(metadata, attr_name, None) is None:
            # ``need_compress=False`` builds dummy (None) plans; leave those.
            continue
        if ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            # The online planner already returns its own compact plan.
            continue
        setattr(
            metadata,
            attr_name,
            create_paged_compressor_data(
                compress_ratio=ratio,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=_as_list(seq_lens_cpu),
                extend_lens=forward_batch.extend_seq_lens,
                extend_lens_cpu=_as_list(extend_lens_cpu),
                use_prefill_cuda_graph=False,
            ),
        )


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._build_forward_metadata",
    type=HookType.AROUND,
)
def rebuild_compact_compress_plans_kunlun(
    original, self, forward_batch, *args, **kwargs
):
    """0.5.19：CP-v2 几何交给 core，这里只把压缩 plan 重建成紧凑形状。

    core 的 ``init_forward_metadata_prefill`` 在 CP-v2 下按
    ``sum(per_rank_actual_token)`` pad 行数并自己 ``apply_cp_reindex``，所以
    0.5.17 那套「padded-global 元数据 + 手动 reindex」不再需要（见模块头注释）。
    但 core 依然把压缩 plan 交给定形（``num_q_tokens`` 行）的 vendor planner——
    压缩算子的开销 ∝ plan 行数（c128 下 128 倍），而压缩器在 BCG 下是 eager 断点，
    没有任何被捕获的东西读 plan，所以这里在 core 建完之后把两份 plan 按 eager 的
    方式（CPU 的 seq/extend 长度 + ``use_cuda_graph=False``）重建一遍。
    ``_rebuild_compact_compress_plans_kunlun`` 里的条件（``_compressor_runs_eager``）
    保证只在 BCG 作用域内生效；BCG=off 的 eager 路径本来就拿 CPU 长度，天然紧凑。
    """
    metadata = original(self, forward_batch, *args, **kwargs)
    if _compressor_runs_eager() and metadata is not None:
        _rebuild_compact_compress_plans_kunlun(self, forward_batch, metadata)
    return metadata


def _compressor_runs_eager() -> bool:
    """Whether the c4/c128 compressor goes through the eager break below.

    The break wrapper only fires inside the breakable CUDA graph's
    capture/replay scope, which is exactly when the runner sets
    ``is_in_breakable_cuda_graph()``.  ``cp/bcg.py`` and the plan builder in
    ``kernels/kernel_ops.py`` have to agree on this condition, so they share it.
    """

    return is_in_breakable_cuda_graph()


def _forward_core_compressor_impl(
    original_fn, self, x, forward_batch, layer_id, compressor
):
    return original_fn(self, x, forward_batch, layer_id, compressor)


# The eager-break form of the compressor call (see the hook below).
_eager_forward_core_compressor = eager_on_graph(True)(_forward_core_compressor_impl)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor_v2.CompressorBackendMixin.forward_core_compressor",
    type=HookType.AROUND,
)
def forward_core_compressor_kunlun(
    original_fn, self, x, forward_batch, layer_id, compressor
):
    """Run the c4/c128 compressor between graph segments instead of inside one.

    DSV4 resolves ``forward_core_compressor`` to ``compressor_v2.forward_unified``
    (the v2 mixin aliases it), so the hook has to target that class -- the v1
    ``dsv4.compressor.CompressorBackendMixin`` is not on this backend's MRO.

    The compress-v2 pipeline is what made the early BCG profile look like a
    regression: on an 8192-token prefill step
    ``infer_compress_fast_c128_compress`` costs 19.36ms per call x 20 calls
    (0.26ms eager) plus ~50ms of ``infer_dsv4_compress_norm_rope_store_128``,
    which is more than the graph saves (launch gaps, the CP all-gather).

    That cost is *not* what capture freezes, though: with this break in place the
    compressor's kernels are launched eagerly at replay (host-correlated
    ``cudaLaunchKernel``) and still took 19.36ms, because the plan handed to the
    kernel is the graph-shaped one (``num_q_tokens`` rows instead of the compact
    ``num_q_tokens/compress_ratio``).  The break is what makes restoring the
    compact plan possible (``kernel_ops.dsv4_compressor_prefill_plan_kunlun``
    reads ``_compressor_runs_eager``): with the compressor eager, nothing
    captured reads the plan any more.

    Ordering still holds: the break completes before the next segment, so the
    compressed cache is written before anything reads it.  The wrapper is
    transparent when no capture is active, so the eager and tc_piecewise paths
    keep calling the original method.
    """
    return _eager_forward_core_compressor(
        original_fn, self, x, forward_batch, layer_id, compressor
    )


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DSV4Metadata"
    ".refresh_for_breakable_cuda_graph_replay_",
    type=HookType.AROUND,
)
def refresh_for_breakable_cuda_graph_replay_kunlun(
    original, self, static_metadata, *args, **kwargs
):
    """Adopt the live compressor plans instead of copying them into the captured ones.

    The original copies the freshly built per-replay plans into the plan objects
    from capture time (``maybe_copy_inplace``), which needs both shapes to match.
    That is what forces the planner into graph mode; with the compressor running
    as an eager break the planner instead returns the compact plan, whose row
    count follows the live token count and therefore differs between a partial
    bucket and its capture.

    Assigning the fresh objects first makes that copy a self-copy, i.e. a no-op
    with no shape constraint.  Upstream already takes this route for the online
    c128 plan ("may carry Python-side planner state, so assign the freshly built
    per-replay object"); it is equally safe for the paged plan here because
    nothing captured reads it -- the c4/c128 plans are consumed only by
    ``forward_core_compressor``, which is an eager break, and by the indexer /
    attention metadata that carry their own tensors.
    """

    if _compressor_runs_eager():
        if static_metadata.c4_compress_metadata is not None:
            self.c4_compress_metadata = static_metadata.c4_compress_metadata
        if static_metadata.c128_compress_metadata is not None:
            self.c128_compress_metadata = static_metadata.c128_compress_metadata
    return original(self, static_metadata, *args, **kwargs)


# NOTE: 0.5.17 的 ``cp_round_robin_input_ids`` REPLACE 钩子在 0.5.19 下线：core 的
# CP-v2 行分片已按 ``max_rank_len`` pad，MoE gating 的形状断言由原生路径满足。


@plugin_hook(
    "sglang.srt.layers.cp.utils.is_cp_v2_active",
    type=HookType.REPLACE,
)
def is_cp_v2_active_kunlun(forward_batch) -> bool:
    """Keep the ragged draft extend on its pre-CP-v2 layout.

    The DSPARK ragged draft extend builds its own row layout
    (``_kunlun_ragged_draft_extend``): sharding only one of its sides through
    CP-v2 leaves the attention with query rows that do not match its position
    rows. Those batches keep the model-side DSA split instead, exactly as
    before CP-v2 was enabled.
    """
    if getattr(forward_batch, "_kunlun_ragged_draft_extend", False):
        return False
    # Speculative batches (target verify / draft extend) build their own row
    # layout: the plugin's LoD paths special-case them, and CP-v2's eager shard
    # only mirrors plain prefill extends -- sharding one side leaves the
    # attention with 5 query rows against 8 position rows. Keep them on the
    # model-side DSA split, as before CP-v2 was enabled.
    if getattr(forward_batch, "spec_info", None) is not None:
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is not None and (
        forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
    ):
        return False

    from sglang.srt.layers.cp.utils import enable_cp_v2, get_cp_strategy

    if not enable_cp_v2():
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False
    strategy = get_cp_strategy()
    if strategy is None:
        return False
    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False
    return strategy.can_apply(len(input_ids), forward_batch)


@plugin_hook(
    "sglang.srt.layers.cp.bcg.execute_prefill_cp_bcg",
    type=HookType.REPLACE,
)
def execute_prefill_cp_bcg_kunlun(
    runner,
    forward_batch,
    static_forward_batch,
    static_num_tokens,
    raw_num_tokens,
    **kwargs,
):
    """0.5.19 的回放尾部丢了 DSV4 的 (hidden_states, pre_hc_head) 元组契约。

    DSV4 的 body 返回 ``(hidden_states, hidden_states_before_norm)``；普通
    ``model.forward`` 会先拆开这对再进 LogitsProcessor，而 CP-BCG 直接 replay
    body，必须保留同样的契约（0.5.17 的 core 是这么做的，0.5.19 迁移时丢了，
    于是 ``logits_processor`` 拿到元组去 ``hidden_states[last_index]`` 直接
    ``IndexError: tuple index out of range``）。这里按 0.5.17 的语义重写：
    拆元组、gather 两个流、并把 ``hidden_states_before_norm`` 传给
    LogitsProcessor（0.5.19 的签名仍接受该参数）。
    """
    from sglang.srt.layers.cp.bcg import _slice_output_rows
    from sglang.srt.layers.cp.utils import cp_gather_after_forward
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    cp_input = runner.prefill_cp_bcg_input
    assert cp_input is not None
    model = runner.model_runner.model
    with runner._prefill_forward_context(
        static_forward_batch,
        num_tokens=static_num_tokens,
        raw_num_tokens=raw_num_tokens,
    ):
        local_output = runner.backend.replay(
            ShapeKey(size=static_num_tokens),
            static_forward_batch,
            **kwargs,
        )
        local_output = _slice_output_rows(local_output, cp_input.live_local_tokens)

        capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
        aux_hidden_states = None
        if capture_aux_hidden_states:
            local_output, aux_hidden_states = local_output

        hidden_states_before_norm = None
        if isinstance(local_output, tuple):
            hidden_states, hidden_states_before_norm = local_output
        else:
            hidden_states = local_output

        if not model.pp_group.is_last_rank:
            return (
                (hidden_states, aux_hidden_states)
                if capture_aux_hidden_states
                else hidden_states
            )

        hidden_states = cp_gather_after_forward(
            hidden_states, static_forward_batch, torch.cuda.current_stream()
        )
        if aux_hidden_states is not None:
            if torch.is_tensor(aux_hidden_states):
                aux_hidden_states = cp_gather_after_forward(
                    aux_hidden_states,
                    static_forward_batch,
                    torch.cuda.current_stream(),
                )
            else:
                aux_hidden_states = [
                    cp_gather_after_forward(
                        hidden, static_forward_batch, torch.cuda.current_stream()
                    )
                    for hidden in aux_hidden_states
                ]
            hidden_states_before_norm = None
        elif hidden_states_before_norm is not None:
            hidden_states_before_norm = cp_gather_after_forward(
                hidden_states_before_norm,
                static_forward_batch,
                torch.cuda.current_stream(),
            )
        return model.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            model.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=hidden_states_before_norm,
        )


@plugin_hook(
    "sglang.srt.layers.cp.bcg.PrefillCPBCGInput.select_replay_bucket_for_batch",
    type=HookType.AROUND,
)
def select_replay_bucket_for_batch_kunlun(
    original, self, *, num_tokens, extend_seq_lens, capture_num_tokens, max_padding_factor
):
    """把 CP 回放桶选择扩展到 interleave（round-robin）策略。

    0.5.19 core 的 ``required_local_tokens`` 只认 ``ZigzagCPStrategy``，对
    interleave 返回 ``None`` ⇒ ``can_run_graph`` 恒 False，所有 prefill 都退
    eager。这里在原逻辑返回 None 时，用策略自己的 ``build_metadata`` 算出
    interleave 的 per-rank 行数上界（对齐后）再走同一个桶选择。
    """
    result = original(
        self,
        num_tokens=num_tokens,
        extend_seq_lens=extend_seq_lens,
        capture_num_tokens=capture_num_tokens,
        max_padding_factor=max_padding_factor,
    )
    if result is not None or extend_seq_lens is None:
        return result

    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size

    strategy = get_cp_strategy()
    if strategy is None:
        return None
    extend_lens = [int(x) for x in extend_seq_lens]
    metadata = strategy.build_metadata(
        num_tokens=sum(extend_lens), seqs_len=None, extend_seqs_len=extend_lens
    )
    align_size = get_cp_padding_align_size()
    required = (
        max(metadata.per_rank_actual_token) + align_size - 1
    ) // align_size * align_size
    return self.select_replay_bucket(
        num_tokens=num_tokens,
        required_local_tokens=required,
        capture_num_tokens=capture_num_tokens,
        max_padding_factor=max_padding_factor,
    )


@plugin_hook(
    "sglang.srt.layers.cp.bcg.supports_prefill_cp_bcg",
    type=HookType.AFTER,
)
def supports_prefill_cp_bcg_kunlun(result, server_args, *args, **kwargs):
    """Accept CP-BCG for the Kunlun interleave + kunlun_compressed combination."""
    if result:
        return result

    resolved = server_args._resolved() if hasattr(server_args, "_resolved") else None
    if resolved is None:
        # 0.5.19: resolved fields live in the resolution stash, projected by
        # resolved_view; the attention backends come from the same helper the
        # core's own supports_prefill_cp_bcg uses.
        from sglang.srt.arg_groups.overrides import (
            attention_backends_of,
            resolved_view,
        )

        resolved = resolved_view(server_args)
        prefill_attention_backend, _ = attention_backends_of(resolved)
    else:
        prefill_attention_backend, _ = server_args._resolved_attention_backends()
    # 0.5.19 keeps the resolved CP decisions in the stash, so the raw
    # ServerAttrs fields can still read False after `--enable-dsa-prefill-...`
    # was resolved into enable_prefill_cp=True; read everything from the view.
    supported = (
        bool(getattr(resolved, "enable_prefill_cp", False))
        and resolved.attn_cp_size == getattr(resolved, "tp_size", server_args.tp_size)
        and getattr(resolved, "cp_strategy", None) == _KUNLUN_PREFILL_CP_STRATEGY
        and prefill_attention_backend == _KUNLUN_PREFILL_ATTENTION_BACKEND
    )
    if supported:
        logger.warning(
            "Enabling breakable prefill CUDA graph for Kunlun CP: "
            "cp_strategy=%s, attn_cp_size=%s, prefill_attention_backend=%s; "
            "capture runs the CP-local body and gathers before logits.",
            getattr(resolved, "cp_strategy", None),
            resolved.attn_cp_size,
            prefill_attention_backend,
        )
    return supported
