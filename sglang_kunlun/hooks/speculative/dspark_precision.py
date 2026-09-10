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


def _clear_unaccepted_c4_states(self, rejected_locs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """把被拒草稿 token 对应的 ratio-4 compress-state 行置为无效。

    c4 不像 c128 那样按 request 寻址，行号来自 token 的 SWA slot。置无效的方式是
    kv 半边写 0、score 半边写 -inf，即把该 slot 从组内 pooling softmax 里剔除。

    行号的 ground truth 是 ``c_plan.cuh`` 的 ``compute_loc``：
    ``swa_page * ring_size + swa_loc % ring_size``，**不除** ``compress_ratio``。
    那个 ``/ compress_ratio`` 只属于 ``plan_c.read_page``（压缩页索引），不属于
    raw state ring。多除一次会把 4 个 slot 折叠到同一行、且只覆盖 buffer 前 1/4。

    ``mask`` 让调用方传完整的 ``[bs, num_draft]`` loc 加一个布尔掩码，而不是先在
    host 上做布尔选择——布尔选择的输出形状依赖数据，会强制一次 D2H 同步。
    行号只取决于 ``(ring_size, kv_score.shape[0])``，而 43 层的 c4 pool 这两个值
    是一样的，所以按这个 key 缓存：原实现在每个 pool 里重算一遍 rows，还各带一次
    ``rows[keep]`` 布尔选择和一次 ``torch.unique``（两者都同步），~80 个 pool 就是
    ~160 次同步、~1400 个 kernel。
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
    base_keep = swa_loc >= 0
    if mask is not None:
        base_keep = base_keep & mask.reshape(-1).to(swa_loc.device)
    row_cache: dict[tuple[int, int], torch.Tensor] = {}
    for pool in pools:
        kv_score = getattr(getattr(pool, "kv_score_buffer", None), "kv_score", None)
        if kv_score is None:
            continue
        ring_size = pool.ring_size
        num_rows = kv_score.shape[0]
        cache_key = (int(ring_size), int(num_rows))
        rows = row_cache.get(cache_key)
        if rows is None:
            rows = swa_loc // self.swa_page_size * ring_size + swa_loc % ring_size
            # 未映射的 slot（swa_loc < 0）折到 row 0 会破坏合法状态，直接丢掉。
            keep = base_keep & (rows >= 0) & (rows < num_rows)
            rows = rows[keep]
            row_cache[cache_key] = rows
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
        rejected = offsets >= keep
        clear_c4(loc_2d, mask=rejected)
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


# ---------------------------------------------------------------------------
# R1（QuaRot）旋转补偿
#
# 量化时对权重做了旋转，旋转后 draft 私有的 RMSNorm gamma 和折进共享（target）权重里的
# 旋转对不上，checkpoint 里因此额外带了 4 个补偿矩阵：
#   mtp.{0,1,2}.context_correction.blocks   [4, 1024, 1024] bf16
#   mtp.2.final_correction.blocks           [4, 1024, 1024] bf16
# 内容是 C = Q @ diag(g) @ Q.T 的块对角残差（hc_mult=4、hidden=4096 => 4 块 1024×1024）。
# 上游 sglang 0.5.17 没有这两个模块，权重被 load_weights 里的
# "DSpark V4 draft: unexpected weight" 直接丢弃 => draft 质量偏低、accept rate 偏低。
# 实测补上后 300 条评测里 accept len 3.06 -> 3.63、accept rate 0.412 -> 0.527，三个
# 测试集分数基本持平。
#
# 参考实现来自 vLLM 侧的同一改动（scripts/vllm-dsv4-w4a8-changes.patch）。
# 非旋转 checkpoint（如 fp16 那份 W8A8）不带这些张量，所有 hook 自动退化成 identity。
# ---------------------------------------------------------------------------
_CORRECTION_RE = re.compile(r"^mtp\.(\d+)\.(context_correction|final_correction)\.blocks$")


def apply_block_correction(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """``x @ block_diag(blocks)``，按块右乘。

    ``blocks`` 是 ``[n_blocks, b, b]``，要求 ``x`` 最后一维 == n_blocks * b。
    用 fp32 运算再 cast 回 ``x`` 的 dtype（gamma 解析上会相消，量级 ~O(1)）。
    """
    n_blocks, b, _ = blocks.shape
    orig_dtype = x.dtype
    lead = x.shape[:-1]
    xr = x.reshape(-1, n_blocks, b).to(torch.float32)
    out = torch.einsum("tnb,nbc->tnc", xr, blocks.to(torch.float32))
    return out.reshape(*lead, n_blocks * b).to(orig_dtype)


class _BlockCorrection(torch.nn.Module):
    """只用来持有一份 ``blocks`` 缓冲，按需挂到 stage 上。"""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("blocks", weight.detach().clone(), persistent=False)


class _NormWithBlockCorrection(torch.nn.Module):
    """``norm`` 之后紧接一次块对角补偿。

    final_correction 的作用点是 ``_logits_from_x_post_hc`` 里
    ``x = last.norm(x_post_hc)`` 之后、乘 lm_head 之前。``last.norm`` 全文件只有那一处
    调用点，所以把 norm 包一层等价于在中间插一步，比 REPLACE 整个 logits 函数更稳
    （不用把 fp32 lm_head / markov TP 分片那些分支抄一遍、跟着上游漂移）。
    注意不能把补偿折进 lm_head 权重：lm_head 是和 target 共享的。
    """

    def __init__(self, norm: torch.nn.Module, weight: torch.Tensor) -> None:
        super().__init__()
        self.norm = norm
        self.register_buffer("blocks", weight.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        return apply_block_correction(self.norm(x), self.blocks)


def _attach_correction(model, stage_id: int, kind: str, weight: torch.Tensor) -> None:
    """把补偿矩阵挂到对应位置。找不到目标就直接抛，不静默跳过。"""
    stage = model.stages[stage_id]
    # 只有最后一个 stage 有 norm，所以设备要从任意一个已存在的参数上取，
    # 否则前面的 stage 会留在 CPU 上，einsum 时报 "mat2 is on cpu"。
    device = next(stage.parameters()).device
    weight = weight.to(device)
    if kind == "context_correction":
        stage.context_correction = _BlockCorrection(weight)
    else:
        norm = stage.norm
        if isinstance(norm, _NormWithBlockCorrection):
            raise RuntimeError("final_correction 被加载了两次")
        stage.norm = _NormWithBlockCorrection(norm, weight)


def _strip_correction_blocks(model, weights):
    """从权重流里摘出补偿矩阵并就地挂上，其余原样透传。"""
    found = []
    for name, tensor in weights:
        m = _CORRECTION_RE.match(name)
        if m is None:
            yield name, tensor
            continue
        _attach_correction(model, int(m.group(1)), m.group(2), tensor)
        found.append(name)
    if found:
        logger.info(
            "[DSPARK_R1] 已挂上 %d 个旋转补偿矩阵: %s", len(found), ", ".join(sorted(found))
        )


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.load_weights",
    type=HookType.AROUND,
)
def load_dspark_correction_blocks_kunlun(original_fn, self, weights, *args, **kwargs):
    """补偿矩阵不是注册参数，先从权重流里摘出来挂好，剩下的交给上游 loader。"""
    return original_fn(self, _strip_correction_blocks(self, weights), *args, **kwargs)


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark."
    "write_target_hidden_kv",
    type=HookType.REPLACE,
)
def write_target_hidden_kv_with_correction_kunlun(
    self, *, main_hidden, swa_loc, positions, pool
):
    """写 target hidden 的 KV 时按 stage 施加 context_correction。

    这条路径的输入是 main_x（走 main_proj + main_norm，**没有 attn_norm**），所以要撤掉
    折进旋转后 wkv 的 attn_norm gamma、重新施加 main_norm gamma，即
    C_k = Q @ diag(gamma_main_norm / gamma_attn_norm_k) @ Q.T。每个 stage 的 C_k 不同，
    而上游 ``CommitKvProj.execute`` 是三个 stage 共享一个 main_x 的融合实现，用不了；
    有补偿时退回逐 stage 投影（复用已有的 ``kv_proj_only``），没有补偿时保持融合路径。
    """
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj

    main_x = self.project_target_hidden(main_hidden)
    swa_loc = swa_loc.to(torch.int32)
    corrections = [getattr(stage, "context_correction", None) for stage in self.stages]
    if any(corr is not None for corr in corrections):
        kvs = [
            stage.self_attn.kv_proj_only(
                main_x
                if corr is None
                else apply_block_correction(main_x, corr.blocks)
            )
            for stage, corr in zip(self.stages, corrections)
        ]
    else:
        kvs = CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[stage.self_attn.wkv for stage in self.stages],
        )
    for stage, kv in zip(self.stages, kvs):
        attn = stage.self_attn
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=attn.layer_id,
            swa_loc=swa_loc,
            kv=kv,
            kv_weight=attn.kv_norm.weight.data,
            eps=attn.eps,
            freqs_cis=attn.freqs_cis,
            positions=positions,
        )


