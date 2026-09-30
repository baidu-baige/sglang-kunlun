"""Register K3 causal-convolution kernel overrides."""

from __future__ import annotations

from sglang_kunlun.kernels.causal_conv1d import (
    causal_conv1d_update_kunlun,
    kda_causal_conv1d_prefill_kunlun,
    kda_causal_conv1d_update_kunlun,
)
from sglang_kunlun.kernels.k3_recurrent import (
    track_mamba_states_all_layers_kunlun,
    track_mamba_states_if_needed_kunlun,
)
from sglang_kunlun.kernels.kernel_ops import register_jit_op
from sglang_kunlun.kernels.kernel_registry import register_kernel_override


register_jit_op(
    "sglang.kernels.ops.mamba.causal_conv1d_triton",
    "causal_conv1d_fn",
)(kda_causal_conv1d_prefill_kunlun)
register_jit_op(
    "sglang.kernels.ops.mamba.causal_conv1d_triton",
    "causal_conv1d_update",
)(kda_causal_conv1d_update_kunlun)
register_jit_op(
    "sglang.kernels.ops.mamba.mamba_state_scatter_triton",
    "track_mamba_states_if_needed",
)(track_mamba_states_if_needed_kunlun)
register_jit_op(
    "sglang.kernels.ops.mamba.mamba_state_scatter_triton",
    "track_mamba_states_all_layers",
)(track_mamba_states_all_layers_kunlun)


@register_kernel_override(
    "mamba.causal_conv1d_update", note="Kunlun causal-conv decode"
)
def causal_conv1d_update(*args, **kwargs):
    """Forward the registry decode call to the Kunlun implementation."""
    return causal_conv1d_update_kunlun(*args, **kwargs)
