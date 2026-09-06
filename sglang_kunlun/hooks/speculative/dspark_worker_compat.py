"""0.5.19 上游签名对齐：``DSparkWorkerV2.forward_batch_generation``。

``scheduler.run_batch`` 的非 overlap 投机分支统一按
``forward_batch_generation(batch, pp_proxy_tensors=...)`` 调用 worker（见
``sglang/srt/managers/scheduler.py``），0.5.19 的 ``EAGLEWorkerV2`` /
``DFlashWorkerV2`` / ``NgramWorker`` 都带了 ``pp_proxy_tensors=None`` 形参，只有
``DSparkWorkerV2`` 漏了，于是 ``--speculative-algorithm DSPARK
--disable-overlap-schedule`` 一进 run_batch 就 ``TypeError``。

昆仑不跑 pipeline parallel，所以这里只补形参并要求它必须为 None：真出现非 None
说明配置越界，宁可显式失败也不静默丢参数。
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.speculative.dspark_components.dspark_worker_v2."
    "DSparkWorkerV2.forward_batch_generation",
    type=HookType.AROUND,
)
def _dspark_forward_batch_generation_accepts_pp_proxy(
    original_fn,
    self,
    batch,
    on_publish=None,
    grammar_barrier=None,
    pp_proxy_tensors=None,
):
    if pp_proxy_tensors is not None:
        raise NotImplementedError(
            "DSPARK on Kunlun does not support pipeline parallel: "
            "forward_batch_generation got non-None pp_proxy_tensors."
        )
    return original_fn(
        self, batch, on_publish=on_publish, grammar_barrier=grammar_barrier
    )
