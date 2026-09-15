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
"""Kunlun compressed-tensors W4A8 (packed INT4 weights, INT8 activations)."""

from __future__ import annotations

from .compressed_tensors_wna16_moe import KunlunCompressedTensorsWNA16MoE

__all__ = ["KunlunCompressedTensorsW4A8Int8MoE"]


class KunlunCompressedTensorsW4A8Int8MoE(KunlunCompressedTensorsWNA16MoE):
    """Packed INT4 weights with dynamic per-token INT8 activations.

    Reuses the GPTQ packed-INT4 layout of ``KunlunCompressedTensorsWNA16MoE``.
    This is the Kunlun equivalent of compressed-tensors W4A8: ``moe_fc_v3``
    with ``tgemm_type="int4_wo_int8"``, ``use_pack_int4=True``, and per-token
    activation quant. It is not the NPU nibble-packed W4A8 scheme
    (offset/clip/scale_bias).
    """

    quantize_activations = True

    def __init__(self, scheme_dict, quant_format: str | None = None) -> None:
        """Validate W4A8 activations, then reuse the packed-INT4 weight path."""
        super().__init__(scheme_dict, quant_format=quant_format)
        input_quant = scheme_dict.get("input_activations")
        if (
            input_quant is None
            or input_quant.num_bits != 8
            or not input_quant.dynamic
        ):
            raise ValueError(
                "Kunlun W4A8 fused MoE requires dynamic per-token INT8 "
                f"activations, got {input_quant}"
            )
