#!/usr/bin/env bash
# 验证 EB5 服务：等待 ready → health → 4 个不同 prompt 的并发请求。
#
# 用法：
#   bash 01_eb/tool/verify_eb5.sh [PORT]
#   默认 PORT=30082；日志取 $BINGOO/logs 下最新的 eb5_kunlun_*.log。
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_BINGOO_DIR="$HERE"
while [ "$_BINGOO_DIR" != / ] && [ ! -x "$_BINGOO_DIR/env/bin/python" ]; do
  _BINGOO_DIR=$(dirname "$_BINGOO_DIR")
done
BINGOO=${BINGOO:-$_BINGOO_DIR}
unset _BINGOO_DIR

PORT=${1:-30082}
READY_TIMEOUT=${READY_TIMEOUT:-600}
# 环境里可能已有 LOG 变量（本机默认为 "log"），用专用变量名覆盖。
LOG=${EB5_LOG:-}
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  LOG=$(ls -t "$BINGOO"/logs/eb5_kunlun_*.log 2>/dev/null | head -1)
fi

[ -n "${LOG:-}" ] && [ -f "$LOG" ] || { echo "❌ 找不到日志，请先起服务"; exit 1; }
echo "[verify] log=$LOG port=$PORT"

# ---- 1) 等待 ready ----
deadline=$((SECONDS + READY_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if grep -aq "fired up and ready to roll" "$LOG"; then
    echo "[verify] ✅ ready ($((SECONDS))s)"
    break
  fi
  if grep -aqE "OutOfMemory|CUDA error|died" "$LOG"; then
    echo "[verify] ❌ 日志出现致命错误"; tail -30 "$LOG"; exit 1
  fi
  sleep 3
done

if ! grep -aq "fired up and ready to roll" "$LOG"; then
  echo "[verify] ❌ ${READY_TIMEOUT}s 内未 ready"; tail -30 "$LOG"; exit 1
fi

# ---- 2) health（ready 瞬间可能 503，重试）----
code=000
for _ in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
    "http://127.0.0.1:$PORT/health_generate" || true)
  [ "$code" = "200" ] && break
  sleep 3
done
echo "[verify] health=$code"
[ "$code" = "200" ] || exit 1

# ---- 3) 4 个不同 prompt（相同 prompt 会路由到同一 DP rank）----
TMP=$(mktemp -d /tmp/eb5_verify.XXXXXX)
prompts=(
  "What is the capital of France?"
  "1+1=2, 2+2=4, 3+3="
  "写一句话描述春天。"
  "列出三种常见的水果，用逗号分隔。"
)
i=0
for p in "${prompts[@]}"; do
  i=$((i + 1))
  (
    curl -s --max-time 240 "http://127.0.0.1:$PORT/generate" \
      -H 'Content-Type: application/json' \
      -d "{\"text\":\"$p\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":32}}" \
      -o "$TMP/$i.json" -w '%{http_code} %{time_total}' > "$TMP/$i.meta" 2>&1
  ) &
done
wait

ok=0
for n in 1 2 3 4; do
  meta=$(cat "$TMP/$n.meta" 2>/dev/null || echo "000 0")
  text=$(python3 - "$TMP/$n.json" <<'PY' 2>/dev/null
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("text", ""))
except Exception:
    print("")
PY
)
  http=${meta%% *}
  if [ "$http" = "200" ] && [ -n "$text" ]; then
    ok=$((ok + 1))
    echo "[verify] #$n ✅ http=200 time=$(echo "$meta" | awk '{print $2}')s text=${text:0:60}"
  else
    echo "[verify] #$n ❌ http=$http meta=$meta"
    head -c 300 "$TMP/$n.json" 2>/dev/null; echo
  fi
done

echo "[verify] 通过 $ok/4"
[ "$ok" -eq 4 ] || exit 1

# ---- 4) 日志致命错误 ----
fatal=$(grep -aEc "OutOfMemory|CUDA error|died" "$LOG" || true)
echo "[verify] 日志致命错误计数=$fatal"
[ "$fatal" = "0" ] || exit 1
echo "[verify] ✅ EB5 验证通过"
