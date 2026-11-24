"""SWA 滑窗回收：补上 dflash / DSPARK decode 路径缺失的那次 evict。

上游 ``ScheduleBatch.prepare_for_decode`` 在开投机时把 decode 准备全交给
``spec_utils.spec_prepare_for_decode``，那里按算法分流：

- eagle / ngram → ``eagle_utils.eagle_prepare_for_decode``，第一件事就是
  ``batch.maybe_evict_swa()``，循环里 ``r.decode_batch_idx += 1``
- 不开投机 → ``ScheduleBatch.prepare_for_decode`` 走 ``alloc_for_decode``，
  里面同样调 ``batch.maybe_evict_swa()``
- dflash 家族（含 DSPARK）→ ``DFlashDraftInputV2.prepare_for_decode``，**两件都没做**

于是 SWA 池只进不出：日志里每一行 decode 的 ``#swa token`` 和 ``#full token`` 数值完全
相等，而 SWA 池只有 full 池的 1/10（``swa_full_tokens_ratio=0.1``），所以 full 用到 0.10
的时候 SWA 已经 1.00。接着 ``SWATokenToKVPoolAllocator.available_size()`` 取
``min(full, swa)`` ≈ 0 → ``check_decode_mem`` 失败 → ``retract_decode`` 退掉一条 →
这条立刻被重新 prefill 灌回来 → 几十秒后再次撑满，服务卡在 retract/重灌的循环里，
请求永远结束不了。

修法就是把 eagle 那两件事补回来，顺序也保持一致（先 evict 再自增：evict 的 decode 分支
门槛是 ``decode_batch_idx >= 1``，第一个 decode step 不 evict 是上游对 overlap 调度的
要求）。``decode_batch_idx`` 那条不能省，它在这条路径上一直是 0，光加 evict 不会生效。

放在插件里而不是直接改 sglang：这是上游的缺口，但我们只在昆仑这条链上验证过。

Fix 后实测：``swa/full`` 从 1.000 掉到 0.111，单请求 full=4608 时 swa=512
（= 2 个 page = sliding_window 128 + page 256 向上取整），峰值 swa usage 0.03、
retract 0 次，且输出与修复前逐字节相同。
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

@plugin_hook(
    "sglang.srt.speculative.dflash_info_v2.DFlashDraftInputV2.prepare_for_decode",
    type=HookType.BEFORE,
)
def evict_swa_before_dflash_decode(self, batch, *args, **kwargs):
    """在 dflash/DSPARK 分配下一步 KV 之前回收滑窗外的 SWA 槽位。

    必须早于 ``alloc_for_spec_decode``（本函数体后半段）执行，腾出来的槽位才对这一步的
    分配可见 —— BEFORE hook 正好在这个位置。返回 None 表示不改原始入参。
    """
    if batch.batch_size() == 0:
        return None

    batch.maybe_evict_swa()
    for req in batch.reqs:
        req.decode_batch_idx += 1

    return None


# `create_idle_input()` builds zero-*width* tensors (`topk_p`, `topk_index` and
# `hidden_states` are all `(0, 0)`), but `merge_batch` cats them with a live
# batch's `[bs, topk]` / `[bs, hidden]` along dim 0, where every other dimension
# has to agree. It does not, so merging an idle side kills the scheduler with
#   RuntimeError: Sizes of tensors must match except in dimension 0.
#                 Expected size 0 but got size 16 for tensor number 1 in the list.
# Reached on the PD decode role with DSPARK when a newly bootstrapped request
# group is merged into a running batch, which depends on how max_running and the
# live concurrency relate: on P800 8-card fake-KV decode, max_running 16 with 12
# concurrent requests crashes all eight schedulers, while 16/8 and 12/12 are fine.
# This keeps upstream's logic and adds the empty-side short-circuit.
@plugin_hook(
    "sglang.srt.speculative.dflash_info_v2.DFlashDraftInputV2.merge_batch",
    type=HookType.REPLACE,
)
def merge_batch_kunlun(self, spec_info) -> None:
    """Merge two draft inputs, tolerating an idle (zero-width) side."""
    import torch

    if self.nxt_kv_lens_cpu is not None:
        assert spec_info.nxt_kv_lens_cpu is not None
        self.nxt_kv_lens_cpu = torch.cat(
            [self.nxt_kv_lens_cpu, spec_info.nxt_kv_lens_cpu]
        )
        self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())
    elif spec_info.nxt_kv_lens_cpu is not None:
        self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu
        self.nxt_kv_lens_sum = spec_info.nxt_kv_lens_sum

    if self.future_indices is not None:
        assert spec_info.future_indices is not None
        self.future_indices = torch.cat(
            [self.future_indices, spec_info.future_indices]
        )
        return

    # Two shapes of "no draft state yet" reach this call and neither can go into
    # torch.cat: an idle side with no rows at all (create_idle_input), and a side
    # whose rows exist but whose per-row width is still 0, which is how a request
    # that has not run a draft step yet is carried. The first is dropped, the
    # second is widened with zeros -- a zero draft is simply rejected by verify,
    # so at most one step's acceptance is lost, whereas a crash takes the node.
    if spec_info.topk_p.shape[0] == 0:
        return
    if self.topk_p.shape[0] == 0:
        self.topk_p = spec_info.topk_p
        self.topk_index = spec_info.topk_index
        self.bonus_tokens = spec_info.bonus_tokens
        self.new_seq_lens = spec_info.new_seq_lens
        self.hidden_states = spec_info.hidden_states
        return

    def _widen(t, like):
        """Give a zero-width `[n, 0]` tensor the width of `like`."""
        if t.dim() == 2 and like.dim() == 2 and t.shape[1] == 0 < like.shape[1]:
            return torch.zeros(
                (t.shape[0], like.shape[1]), dtype=t.dtype, device=t.device
            )
        return t

    self.topk_p = _widen(self.topk_p, spec_info.topk_p)
    self.topk_index = _widen(self.topk_index, spec_info.topk_index)
    self.hidden_states = _widen(self.hidden_states, spec_info.hidden_states)
    spec_info.topk_p = _widen(spec_info.topk_p, self.topk_p)
    spec_info.topk_index = _widen(spec_info.topk_index, self.topk_index)
    spec_info.hidden_states = _widen(spec_info.hidden_states, self.hidden_states)

    self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
    self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
    self.bonus_tokens = torch.cat([self.bonus_tokens, spec_info.bonus_tokens], dim=0)
    self.new_seq_lens = torch.cat([self.new_seq_lens, spec_info.new_seq_lens], dim=0)
    self.hidden_states = torch.cat(
        [self.hidden_states, spec_info.hidden_states], dim=0
    )
