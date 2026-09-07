"""DSpark 精度修复：verify 后清理被拒草稿状态 + draft 权重量化规则对齐。

原先这些改动直接落在 sglang/ 里，现全部收敛到插件：

1. ``clear_unaccepted_c4_states``：上游只为 ratio-128 提供了 ring 清理
   （``deepseek_v4_memory_pool.clear_unaccepted_c128_draft_states``），ratio-4
   连方法都没有。这里给 pool 补上，行号按 c4 的寻址方式算：
   ``row = swa_loc // swa_page_size * ring_size + swa_loc % ring_size``
   （一行是一个 slot，不是 ratio 个 slot 的组）。这一条上游没有对应实现。
2. verify 之后调用 c128 + c4 两个清理：上游那个 c128 清理只在
   ``eagle_worker_common`` 的 verify 里被调用，而 ``DSparkWorkerV2`` 继承
   ``BaseSpecWorker``、不走那条路径，自己也只提交 mamba state，所以 DSpark 下
   两个 ring 都没人清。被拒草稿会留在压缩历史里被后续步骤当作已提交上下文
   读到，表现为输出重复退化、accept len 虚高（钉在 gamma+1）。
3. draft 权重前缀 ``stages.N`` -> ``mtp.N``，并把 checkpoint 里假设已融合的
   ``wq_a`` / ``wkv`` 补进量化 ignore：否则 draft 的 bf16 权重会被当成 int8
   量化目标而截断成全 0，main_proj 输出 0 -> RMSNorm(0) -> NaN，草稿全被拒。
"""

from __future__ import annotations

import logging
import re

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_CLEAR_LOGGED = False

# checkpoint 只把融合后的 wqkv_a 写进 ignore，而 DSparkAttention 是拆开的两个 linear
_DSPARK_EXTRA_QUANT_IGNORE = (
    r"re:.*mtp\.\d+\.self_attn\.wq_a$",
    r"re:.*mtp\.\d+\.self_attn\.wkv$",
)


def _clear_unaccepted_c4_states(self, rejected_locs: torch.Tensor) -> None:
    """把被拒草稿 token 对应的 ratio-4 compress-state 行置为无效。

    c4 不像 c128 那样按 request 寻址，行号来自 token 的 SWA slot。置无效的方式是
    kv 半边写 0、score 半边写 -inf，即把该 slot 从组内 pooling softmax 里剔除。

    行号的 ground truth 是 ``c_plan.cuh`` 的 ``compute_loc``：
    ``swa_page * ring_size + swa_loc % ring_size``，**不除** ``compress_ratio``。
    那个 ``/ compress_ratio`` 只属于 ``plan_c.read_page``（压缩页索引），不属于
    raw state ring。多除一次会把 4 个 slot 折叠到同一行、且只覆盖 buffer 前 1/4。
    """
    if rejected_locs is None or rejected_locs.numel() == 0:
        return
    pools = [
        p
        for p in list(self.compress_state_pools)
        + list(getattr(self, "indexer_compress_state_pools", []) or [])
        if p is not None and p.ratio == 4
    ]
    if not pools:
        return
    swa_loc = self.translate_loc_from_full_to_swa(rejected_locs).to(torch.int64)
    swa_loc = swa_loc.reshape(-1)
    for pool in pools:
        kv_score = getattr(getattr(pool, "kv_score_buffer", None), "kv_score", None)
        if kv_score is None:
            continue
        ring_size = pool.ring_size
        rows = swa_loc // self.swa_page_size * ring_size + swa_loc % ring_size
        # 未映射的 slot（swa_loc < 0）折到 row 0 会破坏合法状态，直接丢掉。
        keep = (swa_loc >= 0) & (rows >= 0) & (rows < kv_score.shape[0])
        rows = torch.unique(rows[keep])
        if rows.numel() == 0:
            continue
        half = kv_score.shape[-1] // 2
        flat = (
            kv_score.reshape(kv_score.shape[0], -1)
            if kv_score.dim() > 2
            else kv_score
        )
        flat[rows, :half] = 0
        flat[rows, half:] = float("-inf")


def _install_c4_state_clear() -> None:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    DeepSeekV4TokenToKVPool.clear_unaccepted_c4_states = _clear_unaccepted_c4_states


_install_c4_state_clear()


def _rejected_locs(batch, num_draft: int):
    """verify 的 out_cache_loc 里 commit_lens 之后的那些 slot。"""
    loc = getattr(batch, "out_cache_loc", None)
    bs = batch.req_pool_indices.numel()
    if loc is None or num_draft <= 1 or bs <= 0 or loc.numel() != bs * num_draft:
        return None
    return loc.view(bs, num_draft)


@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2."
    "_commit_target_mamba_states_after_verify",
    type=HookType.AFTER,
)
def clear_unaccepted_compress_states_kunlun(result, self, *args, **kwargs):
    """verify 之后清掉被拒草稿写下的 c128 / c4 compress-state。

    挂在 mamba 提交之后，是因为这里能同时拿到 batch、verify 前的 seq_lens 和
    commit_lens，且 batch.seq_lens 还没被 accept 结果覆盖。EAGLE/MTP 在
    ``eagle_worker_common.verify_target_output`` 里做的是同一件事。

    上游的 ``clear_unaccepted_c128_draft_states`` 只在 ``eagle_worker_common`` 的
    verify 里被调用，而 ``DSparkWorkerV2`` 直接继承 ``BaseSpecWorker``、不走那条
    路径，所以 DSpark 下 c128 和 c4 两个 ring 都没有人清。
    """
    batch = kwargs.get("batch")
    seq_lens_pre_verify = kwargs.get("seq_lens_pre_verify")
    commit_lens = kwargs.get("commit_lens")
    if batch is None or commit_lens is None:
        return result
    if batch.forward_mode.is_idle() or batch.req_pool_indices.numel() == 0:
        return result
    allocator = self.target_worker.model_runner.token_to_kv_pool_allocator
    if allocator is None:
        return result
    kvcache = allocator.get_kvcache()
    num_draft = int(self.verify_num_draft_tokens)

    clear_c128 = getattr(kvcache, "clear_unaccepted_c128_draft_states", None)
    clear_c4 = getattr(kvcache, "clear_unaccepted_c4_states", None)
    # 接线自检，每个 rank 只打一次：这个 hook 的各条提前 return 都是静默的，
    # 尤其 clear_c4 取不到（说明 _install_c4_state_clear 没生效）会让清理退回空转，
    # 而空转正是当初那个 bug 的形态。所以要把两个能力的解析结果都打出来。
    global _CLEAR_LOGGED
    if not _CLEAR_LOGGED:
        _CLEAR_LOGGED = True
        logger.info(
            "[DSPARK_CLEAR_STATES] verify 后 compress-state 清理已挂上 "
            "(c128=%s, c4=%s, num_draft=%s)",
            clear_c128 is not None,
            clear_c4 is not None,
            num_draft,
        )

    if clear_c128 is not None:
        clear_c128(
            batch.req_pool_indices, seq_lens_pre_verify, commit_lens, num_draft
        )

    loc_2d = _rejected_locs(batch, num_draft)
    if clear_c4 is not None and loc_2d is not None:
        keep = commit_lens.to(loc_2d.device).reshape(-1, 1)
        offsets = torch.arange(num_draft, device=loc_2d.device).reshape(1, num_draft)
        clear_c4(loc_2d[offsets >= keep])
    return result


def _quant_config_targets_draft_attn(quant_config) -> bool:
    """checkpoint 是否真的量化了 draft 的 wq_a/wkv。

    W8A8 那份 checkpoint 里 ``mtp.N.attn.wq_a`` 只有 ``.weight``、没有
    ``weight_scale``，必须补进 ignore 才不会被截断成 0（就是下面那个 hook 的由来）。
    但 W4A8 那份是真的按 int8 量化了这两层、带 weight_scale，此时补 ignore 会反过来
    把它们当未量化权重加载。所以先看 target_scheme_map 里有没有命中它们。
    """
    targets = getattr(quant_config, "target_scheme_map", None)
    if not targets:
        return False
    probes = ("model.mtp.0.self_attn.wq_a", "model.mtp.0.self_attn.wkv")
    for target in targets:
        if not isinstance(target, str) or not target.startswith("re:"):
            continue
        try:
            pattern = re.compile(target[3:])
        except re.error:
            continue
        if any(pattern.fullmatch(name) for name in probes):
            return True
    return False


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.__init__",
    type=HookType.AROUND,
)
def align_dspark_quant_ignore_kunlun(original_fn, self, *args, **kwargs):
    """把 checkpoint 的量化 ignore 规则补齐后再建 draft，避免 bf16 权重被量化成 0。"""
    quant_config = kwargs.get("quant_config")
    if quant_config is None:
        for arg in args:
            if hasattr(arg, "ignore"):
                quant_config = arg
                break
    ignore = getattr(quant_config, "ignore", None)
    if isinstance(ignore, list):
        if _quant_config_targets_draft_attn(quant_config):
            logger.info(
                "[DSPARK_QUANT_IGNORE] checkpoint 已量化 draft 的 wq_a/wkv，跳过补 ignore"
            )
        else:
            for pattern in _DSPARK_EXTRA_QUANT_IGNORE:
                if pattern not in ignore:
                    ignore.append(pattern)
    return original_fn(self, *args, **kwargs)



@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DSparkV4Stage.__init__",
    type=HookType.AROUND,
)
def align_dspark_stage_prefix_kunlun(original_fn, self, *args, **kwargs):
    """draft 权重前缀改成 checkpoint 用的 ``mtp.N``，ignore 规则才能匹配上。"""
    prefix = kwargs.get("prefix")
    if isinstance(prefix, str) and "stages." in prefix:
        kwargs["prefix"] = prefix.replace("stages.", "mtp.")
    return original_fn(self, *args, **kwargs)
