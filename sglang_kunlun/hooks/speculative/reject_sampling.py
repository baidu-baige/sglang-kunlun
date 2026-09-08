"""昆仑温度采样路径：chain rejection sampling 算子 + draft softmax 回落。

两处修复都只在 ``temperature > 0`` 时才起作用；``temperature=0`` 走贪心接受路径，
既不算 draft_probs 也不做 rejection sampling，两者都碰不到。
换句话说，**温度 >0 在昆仑上原本完全不可用**，缺任一处都跑不通。

1. ``chain_speculative_sampling_triton``：上游是纯 Triton 实现（内部 kernel 就叫
   ``speculative_sampling_classic_kernel``），在昆仑上跑不了。XSpeedGate 提供了同名的
   XPU 算子（AICapX-1302），入参与上游 wrapper 逐个对齐，可以直接顶替。
   算法是 chain rejection sampling：按 ``u * draft_p < target_p`` 判接受，拒绝处从残差
   分布 ``normalize(relu(target - draft))`` 采 bonus token，输出分布无偏。
   算子不存在时（旧版 xspeedgate_ops）自动回落到上游 Triton，不影响启动。

   注意它有两个调用方，**DSpark 走的是第二个、无条件调用**：

   - ``srt/speculative/eagle_utils.py`` —— 受 ``--speculative-use-rejection-sampling``
     控制，关掉时走 ``tree_speculative_sampling_target_only``（插件已在
     ``sgl_kernel_stub`` 里绑成昆仑实现）。只影响 EAGLE / dflash。
   - ``kernels/ops/speculative/dspark/dspark_accept.py`` —— DSpark 自己的路径：
     ``dspark_verify.accept_draft_tokens`` -> ``AcceptSampling.execute`` -> ``cls.triton``
     -> 这里，**没有任何开关判断**。所以 DSpark 下不需要加那个 flag，加了也没作用。

2. ``SoftmaxTemp.execute``：算 draft_probs 用的带温度 softmax。上游分派是"在 cuda 上且
   flashinfer 能 import 就用 flashinfer"，而昆仑伪装成 cuda、flashinfer 包也装着，于是
   选到 flashinfer 的预编译 CUDA kernel，运行期报
   ``OnlineSoftmax failed with error code invalid device function``，直接打挂 scheduler
   （表现是请求 hang / 连接被掐断）。
   注意 ``SGLANG_IS_FLASHINFER_AVAILABLE=False`` 挡不住它——那里是模块顶部直接
   ``from flashinfer.sampling import softmax``，绕过了这个开关。
"""


from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_OP_NAME = "speculative_sampling_classic_kernel"
_LOGGED = False


def _resolve_op():
    """拿到 XPU 算子；旧版 xspeedgate_ops 没有这个算子时返回 None。"""
    try:
        import xspeedgate_ops  # noqa: F401
    except ImportError:
        return None
    return getattr(torch.ops.xspeedgate_ops, _OP_NAME, None)


@plugin_hook(
    "sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton",
    type=HookType.AROUND,
)
def chain_speculative_sampling_kunlun(
    original_fn,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,
):
    """转调 XSpeedGate 的 XPU 算子，结果原地写回三个输出张量。

    ``threshold_single`` / ``threshold_acc`` / ``deterministic`` 以及
    ``retrive_next_token`` / ``retrive_next_sibling``（tree 接口的遗留入参）
    算子侧不使用，但仍按上游签名传下去，避免以后算子启用它们时又要改调用点。
    """
    op = _resolve_op()
    global _LOGGED
    if op is None:
        if not _LOGGED:
            _LOGGED = True
            logger.warning(
                "[KUNLUN_SPEC_SAMPLING] xspeedgate_ops 缺少 %s，回落到上游 Triton 实现",
                _OP_NAME,
            )
        return original_fn(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            threshold_single,
            threshold_acc,
            deterministic,
        )
    if not _LOGGED:
        _LOGGED = True
        logger.info("[KUNLUN_SPEC_SAMPLING] 使用 xspeedgate_ops.%s", _OP_NAME)
    op(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    )


_SOFTMAX_LOGGED = False


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_accept.SoftmaxTemp.execute",
    type=HookType.REPLACE,
)
def softmax_temp_execute_kunlun(cls, *args, **kwargs) -> torch.Tensor:
    """带温度 softmax 走纯 torch 实现。

    上游分派顺序是 flashinfer -> triton -> torch，昆仑上前两者都不可用：flashinfer 是
    预编译的 CUDA kernel（invalid device function），Triton 这套环境也不走。
    torch 版本就是 ``logits / temp`` 再 softmax，5 行，直接可用。
    """
    global _SOFTMAX_LOGGED
    if not _SOFTMAX_LOGGED:
        _SOFTMAX_LOGGED = True
        logger.info(
            "[KUNLUN_SPEC_SAMPLING] SoftmaxTemp 走 torch 实现（绕开 flashinfer）"
        )
    return cls.torch(*args, **kwargs)
