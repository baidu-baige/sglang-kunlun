#!/usr/bin/env bash
# 安全停掉昆仑上的 sglang 服务（launcher + 它的 worker），供 run_*.sh 起服务前调用。
#
# 用法：  bash stop_kunlun.sh      # 或 source 后调 stop_kunlun
#
# ── 为什么不能直接 `pkill -9 -f "sglang.launch_server|sglang::"` ──────────────
# 两个真实的坑（2026-10-17 各踩过一次）：
#
#  1) **会打到调用者自己**。`pkill -f` 匹配的是整条命令行，而调用它的那条命令
#     本身往往就含这些字符串，于是把自己（甚至父 shell）一起杀掉，脚本半途而废。
#     所以本脚本一律用字符类（`sglang[-]kunlun-launch`、`sglang[:][:]`）书写模式，
#     让它匹配不到含字面量的自身命令行。
#
#  2) **对 launcher 用 -9 是 doc/3「最重要的一条」明令禁止的**。所以这里分级处理：
#       a. 先给 launcher 发 TERM+INT（sglang 装了优雅退出 handler）；
#       b. 但 launcher 主线程通常卡在 `waitpid` 等子进程，而 `sglang::tokenizer_worker`
#          / `sglang::detokenizer` **不响应 TERM/INT**，handler 永远走不完 ——
#          这就是"杀了也退不掉"的真因；
#       c. 于是只对**这两类纯 CPU、且处于空闲等待**的 worker 用 -9，解开 waitpid 死锁
#          （它们按设计只做分词/反分词，不发任何 XPU 调用；实测状态为 S + ep_poll/
#          futex_wait，`xpu-smi -q` 也不登记它们 —— 杀它们不会留下 stuck kernel）；
#       d. **launcher 本身一次 -9 都不发**：子进程被回收后它会在几秒内自己走完 handler；
#       e. 最后校验 8 张卡都回到 `0 MiB / 0%`，不满足就报错退出交给人看。
set -uo pipefail

PAT_LAUNCH='sglang[-]kunlun-launch'
PAT_WORKER='sglang[:][:]'
# 允许 -9 的角色：纯 CPU 的分词/反分词 worker
PAT_CPU_WORKER='sglang[:][:]tokenizer_worker|sglang[:][:]detoken'

say() { echo "[stop_kunlun] $*"; }

# 只认"真正的" sglang 进程：`pgrep -f` 匹配整条命令行，会把别的任务的 shell 包装
# （cmdline 里引用了 `sglang::` 字面量）误判成残留，从而挡住每次启动、甚至给它发 TERM。
# 用 /proc/<pid>/comm 过滤：scheduler/worker 的 comm 是 `sglang::xxx`，launcher 是 python*。
sglang_pids() {
  local p comm
  for p in $(pgrep -f "$PAT_LAUNCH|$PAT_WORKER" 2>/dev/null); do
    comm=$(cat /proc/"$p"/comm 2>/dev/null) || continue
    case "$comm" in bash|sh|dash|zsh|ksh|timeout|grep|pgrep) continue ;; esac
    echo "$p"
  done
}
alive() { [ -n "$(sglang_pids)" ]; }

# 只校验本次任务实际要用的卡（默认全 8 张）。`run_eb5_kunlun.sh` 支持 EB5_GPUS 子集，
# 但这里原来硬编码 8 张；4–7 被别的任务占着时，4 卡运行会被误判成"有卡没回到 0 MiB"。
GPUS=${STOP_GPUS:-${EB5_GPUS:-0,1,2,3,4,5,6,7}}
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
card_free() {
  local out
  out=$(xpu-smi -i "$1" 2>/dev/null) || return 1
  grep -qE '0MiB / 98304MiB' <<<"$out"
}
count_free() {
  local n=0 d
  for d in $(tr ',' ' ' <<<"$GPUS"); do card_free "$d" && n=$((n+1)); done
  echo "$n"
}

if ! alive; then
  say "没有在跑的 sglang 进程"
  exit 0
fi

say "发现残留："
for _p in $(sglang_pids); do printf "  %s %s\n" "$_p" "$(tr '\0' ' ' < /proc/$_p/cmdline 2>/dev/null | cut -c1-110)"; done

# a) 优雅信号
for p in $(sglang_pids); do
  grep -q "$PAT_LAUNCH" /proc/"$p"/cmdline 2>/dev/null || continue
  kill -TERM "$p" 2>/dev/null && say "TERM -> $p"
  kill -INT  "$p" 2>/dev/null
done

# b) 给优雅退出一点时间
for _ in $(seq 1 6); do
  sleep 5
  alive || { say "launcher 已优雅退出"; break; }
done

# c) 还活着：只 -9 空闲的纯 CPU worker，解开 waitpid 死锁
if alive; then
  say "launcher 仍在（多半卡在 waitpid 等 worker），只对空闲的纯 CPU worker 用 -9"
  for p in $(pgrep -f "$PAT_CPU_WORKER" 2>/dev/null); do
    st=$(awk '{print $3}' /proc/"$p"/stat 2>/dev/null)
    wchan=$(cat /proc/"$p"/wchan 2>/dev/null)
    case "$st:$wchan" in
      Z:*) say "  $p 已是僵尸，跳过" ;;
      S:ep_poll|S:do_epoll_wait|S:futex_wait_queue_me|S:poll_schedule_timeout|S:do_poll*)
        kill -9 "$p" 2>/dev/null && say "  -9 -> $p（空闲 CPU worker，状态 $st/$wchan）" ;;
      *) say "  ⚠️ $p 状态 $st/$wchan 不像空闲 worker，**跳过**（不冒险）" ;;
    esac
  done
fi

# d) 等 launcher 自己退出
for _ in $(seq 1 24); do
  sleep 5
  alive || break
done

if alive; then
  say "❌ 仍有进程未退出："
  for _p in $(sglang_pids); do printf "  %s %s\n" "$_p" "$(tr '\0' ' ' < /proc/$_p/cmdline 2>/dev/null | cut -c1-110)"; done
  say "不要对 launcher 用 -9（doc/3：会留 stuck kernel）。请人工确认后再处理。"
  exit 1
fi

free=$(count_free)
say "设备：$free/$NGPU 张卡空闲（0 MiB，范围 $GPUS）"
if [ "$free" -ne "$NGPU" ]; then
  say "❌ 有卡没回到 0 MiB —— 先别起下一个任务，按 doc/3 第五节处理"
  exit 1
fi
say "✅ 已清干净"
