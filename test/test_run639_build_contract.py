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
"""CPU-only checks for the reviewed Pod engine and XSpeedGate build inputs."""
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "dockerfile/install.sh").read_text()
DOCKER = (ROOT / "dockerfile/Dockerfile").read_text()
VERSION = "0.5.21.dev263+gc610c4039"
WHEEL = ROOT / f"dockerfile/vendor/sglang-{VERSION}-py3-none-any.whl"


def function(name):
    return re.search(r"^function " + name + r"\(\) \{\n.*?^\}", INSTALL, re.M | re.S)[0]


class BuildContract(unittest.TestCase):
    def shell(self, body, **variables):
        definitions = "\n".join(function(n) for n in (
            "vendored_sglang_wheel", "validate_sglang_wheel", "clone_sglang"
        ))
        assignments = "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in variables.items())
        # Use this test's interpreter even on CPU hosts without a python3 alias.
        script = f"set -euo pipefail\npython3() {{ {shlex.quote(sys.executable)} \"$@\"; }}\n{definitions}\n{assignments}\n{body}"
        return subprocess.run(["bash", "-c", script], text=True, capture_output=True)

    def test_vendored_wheel_hash_and_metadata(self):
        self.assertEqual(hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
                         "55d4e9b8354ecc8987c4782d1765e12ede1da8e63cd3a8ee0acb1e0735b36a92")
        result = self.shell('w=$(vendored_sglang_wheel); validate_sglang_wheel "$w"',
                            plugin_dir=ROOT, SGLANG_EXPECTED_VERSION=VERSION)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_corrupt_or_missing_vendor_wheel_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            vendor = Path(directory) / "dockerfile/vendor"
            vendor.mkdir(parents=True)
            for payload in (None, b"corrupted wheel"):
                if payload is not None:
                    (vendor / WHEEL.name).write_bytes(payload)
                result = self.shell('vendored_sglang_wheel', plugin_dir=directory,
                                    SGLANG_EXPECTED_VERSION=VERSION)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_wrong_source_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "wrong.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("sglang.dist-info/METADATA", "Name: sglang\nVersion: 0.5.19\n")
            result = self.shell(f"validate_sglang_wheel {shlex.quote(str(wheel))}",
                                SGLANG_EXPECTED_VERSION=VERSION)
            self.assertNotEqual(result.returncode, 0)

    def test_explicit_source_requires_commit_before_network(self):
        for commit in ("", "NA"):
            result = self.shell('clone_sglang https://invalid.example main "$commit"', commit=commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require an explicit SGLANG_COMMIT", result.stderr)

    def test_default_build_uses_only_local_engine(self):
        self.assertIn('ARG SGLANG_ENGINE_SOURCE="wheel"', DOCKER)
        self.assertIn('ARG SGLANG_COMMIT=""', DOCKER)
        self.assertIn('--SGLANG_ENGINE_SOURCE="${SGLANG_ENGINE_SOURCE}"', DOCKER)
        self.assertIn('SGLANG_ENGINE_SOURCE="wheel"', INSTALL)
        dispatch = INSTALL[INSTALL.index('    case "${SGLANG_ENGINE_SOURCE}" in'):]
        self.assertIn('wheel) vendored_sglang_wheel >/dev/null ;;', dispatch)
        self.assertNotIn('SGLANG_ENGINE_SOURCE="source"', dispatch)
        engine = function("install_aiak_sglang")
        self.assertIn('validate_sglang_wheel "${built_wheels[0]}" || return 1', engine)
        self.assertIn('pip3 install --no-deps "${vendored_whl}"', engine)
        self.assertIn('echo "triton==${triton_ver}"', engine)

    def test_delivered_xspeedgate_is_verified_and_installed_whole(self):
        # The build ships the delivered XSpeedGate wheel (1.7.0+cbe2814,
        # torch29 build; the torch25 build of the same version is ABI
        # incompatible with this runtime), not the stock 242f72b wheel:
        # cast_permute01 / infer_hc_combine_g4 / fp4 index-k quantize+store /
        # mqa round-2 live only in the delivered one.
        block = INSTALL[INSTALL.index('        # Install the delivered XSpeedGate wheel'):]
        block = block[:block.index('        # Older delivered wheels shipped xspeedgate_ops only')]
        self.assertIn('https://vllm-ai-models.bj.bcebos.com/aiak_share/20261109/torch29/xspeedgate_ops-1.7.0%2Bcbe2814.torch29-cp310-cp310-linux_x86_64.whl', block)
        self.assertIn('0ae115bf811c9698b7682c52b8cebca3d7e1b409b11b2c3a9d49a36d55399ad3', block)
        self.assertLess(block.index('sha256sum -c -'), block.index('pip3 install'))
        self.assertIn('pip3 install "$whl_file" --force-reinstall --no-deps', block)
        self.assertNotIn('xsgops20261021r', INSTALL)
        # cocopod/ is re-extracted from the stock wheel (directory only).
        tail = INSTALL[INSTALL.index('        # Older delivered wheels shipped xspeedgate_ops only'):]
        self.assertIn('z.extractall(dst, members)', tail)


if __name__ == "__main__":
    unittest.main()
