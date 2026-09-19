#!/bin/bash
# Symmetric-steady control: measure stock PD with the IDENTICAL client, workload
# and mid-window metric as FG. PD stack up in MODE=serve (P+D+proxy), then
# rdma_driver --local-prefill-frac 1.0 posts every request as a plain completion
# to the PROXY -> stock PD flow, same seed-1234 random-token prompts, same
# 25%->75% completion-window steady definition.
# refs today: FG steady 2034 / e2e 1852 (conc64) | PD bench-serve e2e 1841.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=$HOME/.python/vllm0.18/bin/python
O=$S/decoupled_rerun; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f run_pd_baseline 2>/dev/null
  sleep 6; true
}
cleanup

echo "######## PD stack (MODE=serve) : $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd_steady" \
MODEL=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b \
VLLM_BIN=$HOME/.python/vllm0.18/bin/vllm \
PYTHON_BIN=$PY MODE=serve \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=2400 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd_steady_stack.log" 2>&1 &

ready=0
for i in $(seq 1 120); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8010/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8020/health 2>/dev/null)
  c=$(curl -s -m 2 -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8030/v1/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && [ "$c" = "200" ] && { ready=1; echo "stack ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "PD_STACK_NOT_READY a=$a b=$b proxy=$c"
  grep -aiE 'error|Traceback' "$O/pd_steady_stack.log" | tail -4
  cleanup; echo PD_STEADY_DONE; exit 1
fi

echo "######## PD via identical client, conc=64 : $(date +%H:%M:%S) ########"
FG_D_URL=http://127.0.0.1:8030/v1/completions \
timeout -k 20 1800 $PY -u "$S/rdma_driver.py" \
  --num-prompts 256 --output-len 384 --concurrency 64 --local-prefill-frac 1.0 \
  > "$O/pd_steady_conc64.out" 2>&1
grep -aE 'ok ===|wall=|errors' "$O/pd_steady_conc64.out" | head -3
cleanup
echo "PD_STEADY_DONE"
