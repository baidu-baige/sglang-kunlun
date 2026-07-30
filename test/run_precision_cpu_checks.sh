#!/bin/bash
set -euo pipefail
ROOT=/home/zx/code/open_src/sglang/v0.5.14_precision_migration_20260730/baidu/sglang-kunlun
export PYTHONPATH="$ROOT:$ROOT/../sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_PLATFORM=kunlun
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
