#!/bin/bash
# usage: bash start_bench_server.sh 27b|35b [ctx]
# ctx 默认 16384（2026-09-28 用户口径升级；np=1 单槽全上下文）；f16 KV 失败自动回退 q8_0 KV
LC=/home/super/test/llamabench/llama.cpp
B=$LC/build/bin
M27=/media/super/Hard_Disk_2/huggingface_models/Qwen3.5/Qwen3.5-27B-UD-Q5_K_XL.gguf
M35=/media/super/Hard_Disk_2/huggingface_models/Qwen3.5/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
CTX=${2:-16384}
PORT=10361; M=$M27
if [ "$1" = "35b" ]; then PORT=10362; M=$M35; fi
for pid in $(pgrep -f "llama-serve[r]"); do
  if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "port ${PORT}"; then kill $pid; sleep 3; kill -9 $pid 2>/dev/null || true; fi
done
start() {
  nohup nice -n 5 $B/llama-server -m $M -ngl 99 -c $CTX -b 2048 -ub 512 --port $PORT -t 16 -np 1 --flash-attn on $1 > /tmp/jevgen_bench_${PORT}.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do sleep 3; curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && return 0; done
  return 1
}
if start ""; then echo "READY $1 ctx=$CTX (f16 kv)"; exit 0; fi
echo "f16 kv 加载失败，回退 q8_0 kv ..."
for pid in $(pgrep -f "llama-serve[r]"); do kill $pid 2>/dev/null; done; sleep 5
if start "-ctk q8_0 -ctv q8_0"; then echo "READY $1 ctx=$CTX (q8 kv fallback)"; exit 0; fi
echo "FAILED $1"; exit 1
