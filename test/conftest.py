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
"""Let the hook modules import without SGLang's third-party dependency tree.

``unit-test.yml`` installs ``sglang --no-deps`` deliberately, so ``import
sglang`` stops at the first missing dependency here (numpy, then orjson, then
psutil, ...). Installing them would mean adopting SGLang's whole tree, pinned
``torch`` included, which is what ``--no-deps`` is avoiding.

The hook modules need exactly two names from SGLang at import time, so supply a
stub registry when the real one cannot be imported. A full install -- a Kunlun
runner, a developer box -- keeps using the real registry, so the decorator
contract is still exercised where SGLang is actually present.
"""

import enum
import sys
import types


def _real_registry_importable() -> bool:
    try:
        import sglang.srt.plugins.hook_registry  # noqa: F401
    except Exception:
        return False
    return True


class _StubHookType(enum.Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


def _stub_plugin_hook(target: str, type: _StubHookType = _StubHookType.AFTER):
    """Accept the registration and return the function unchanged."""

    def register(fn):
        return fn

    return register


def _install_stub_registry() -> None:
    registry = types.ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = _StubHookType
    registry.plugin_hook = _stub_plugin_hook

    # A __path__ keeps importlib.util.find_spec able to walk these, which is how
    # the hooks probe for the kernel module they patch.
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

    sys.modules["sglang.srt.plugins.hook_registry"] = registry
    sys.modules["sglang.srt.plugins"].hook_registry = registry
    sys.modules["sglang.srt"].plugins = sys.modules["sglang.srt.plugins"]
    sys.modules["sglang"].srt = sys.modules["sglang.srt"]


if not _real_registry_importable():
    _install_stub_registry()
