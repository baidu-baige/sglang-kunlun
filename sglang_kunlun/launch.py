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
"""Explicit launcher for SGLang on Kunlun XPU."""

from __future__ import annotations

import os
import runpy


def main() -> None:
    """Run SGLang launch_server after installing Kunlun bootstrap hooks."""
    # torch_xmlir must initialize before SGLANG_PLATFORM is exposed.  When
    # torch sees the Kunlun platform during its first import, the runtime
    # loads XCCL through the sitecustomize path and can corrupt the allocator.
    import torch  # noqa: F401

    os.environ["SGLANG_KUNLUN_EXPLICIT_LAUNCH"] = "1"
    os.environ.setdefault("SGLANG_PLATFORM", "kunlun")

    from sglang_kunlun.bootstrap import _kunlun_pre_shim

    _kunlun_pre_shim()
    try:
        import sglang.srt.server_args as server_args

        choices = getattr(server_args, "ATTENTION_BACKEND_CHOICES", None)
        if choices is not None:
            for backend in ("kunlun", "kunlun_dsv4"):
                if backend not in choices:
                    choices.append(backend)
    except Exception:
        pass

    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
