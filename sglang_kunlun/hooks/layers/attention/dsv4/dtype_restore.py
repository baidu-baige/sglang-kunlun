"""fp16 服务下未量化 Linear 的 dtype 恢复（dsv4 组件共用工具）。

上游在部分共享基类/组件里把投影的 ``params_dtype`` 固定成 bf16；当服务以
fp16 拉起时，这里在组件 ``__init__`` 之后把未量化 Linear 的参数与权重恢复到
请求的 dtype。供 ``indexer`` / ``compressor`` 的 hook 以及 models 层的
``MqaAttentionBase`` hook 共用。
"""

from __future__ import annotations

import torch

_FP16_DTYPE_NAMES = frozenset(("fp16", "float16", "half"))


def restore_requested_fp16_parameter_dtype(*linears):
    """把未量化 Linear 恢复到服务请求的 fp16 dtype（非 fp16 请求时不动）。"""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.server_args import get_global_server_args

    if get_global_server_args().dtype not in _FP16_DTYPE_NAMES:
        return

    for linear in linears:
        if not isinstance(
            getattr(linear, "quant_method", None), UnquantizedLinearMethod
        ):
            continue
        linear.params_dtype = torch.float16
        weight = getattr(linear, "weight", None)
        if weight is not None and weight.dtype != torch.float16:
            weight.data = weight.data.to(dtype=torch.float16)
