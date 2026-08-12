#!/bin/bash
set -euo pipefail
ROOT=/home/zx/code/v0.5.17/sglang-kunlun
export PYTHONPATH="$ROOT:$ROOT/../sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_OPT_USE_COMPRESSOR_V2=1
export DSV4_KUNLUN_REFERENCE_ONLY=1
UPSTREAM="$ROOT/../sglang"
python_status="$(git -C "$UPSTREAM" status --porcelain -- python)"
if [[ -n "$python_status" ]]; then
    printf '%s\n' "Restored upstream python is dirty:" "$python_status" >&2
    exit 1
fi
printf '%s\n' "Restored upstream python status: clean"
cd "$ROOT"
python -m compileall -q sglang_kunlun test
python -m unittest test.test_production_precision_contract -v
python -m unittest discover -s test -p 'test_*.py' -v
