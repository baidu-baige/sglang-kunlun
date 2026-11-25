# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Capture-state helper that is reliable on the Kunlun backend.

``torch.cuda.is_current_stream_capturing()`` did not report the capture
region here, so a metadata synchronize ran inside draft graph capture and
failed with ``wait for noc idle timeout``. SGLang's own capture-mode flag is
set around the capture region; prefer it and keep the CUDA check as fallback.
"""
import torch


def is_graph_capturing() -> bool:
    """Check framework capture mode before falling back to the CUDA query.

    Returns:
        True when either available capture-state query reports capture;
        False when neither does or both queries are unavailable.
    """
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False
