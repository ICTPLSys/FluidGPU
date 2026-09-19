#!/bin/bash
# fig7 v2 (online latency, GT): the §16 sweep re-run with everything learned
# since — FG row = lever-A online config (phase-aware decode-local + FULL
# decode graph + phase-route for mixed steps + pingpong), plus a PD row.
# §16 recorded (old FG design): homoL40S 7.0 / homoA100 6.2 / AF 4.5 / FG 2.1
# req/s at the 50ms median-TPOT SLO; paper targets homoL~11/AF~14/PD~15/FG~18.
# Protocol identical to §16: conversation_benchmark (med 2075/130, per-request
# output lens), Poisson rates 2..12, 192-320 prompts/point, APC OFF, same
# client (vllm bench serve) for every row.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
GPTOSS="$MODEL_GT"
exp_preflight "$GPTOSS"
DS=datasets/conversation_benchmark/requests.jsonl
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
OUT="$S/fig7_gt_online"
LOG=$OUT/logs
RUNTIME_ROOT=$PWD/fluidgpu_runtime
EXAMPLES_DIR=$PWD/fluidgpu_runtime/examples/fluidgpu_torch
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
PORT=8977
RATES="2 4 6 7 8 10 12"
mkdir -p "$OUT" "$LOG"

CC_PIECE=$($V -c "import json,sys; sys.path.insert(0,'$RUNTIME_ROOT'); from fluidgpu_torch.vllm_integration import fluid_compilation_config; print(json.dumps(fluid_compilation_config()))")
CC_FULL=$(FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_PHASE_AWARE=1 $V -c "import json,sys; sys.path.insert(0,'$RUNTIME_ROOT'); from fluidgpu_torch.vllm_integration import fluid_compilation_config; print(json.dumps(fluid_compilation_config()))")

cleanup_gpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 $p 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null
  sleep 6
}

wait_port() {
  for i in $(seq 1 130); do
    [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null)" = "200" ] && return 0
    sleep 4
  done
  return 1
}

bench_rates() {  # row port
  local row=$1 port=$2
  for R in $RATES; do
    N=192; [ "${R%%.*}" -gt 6 ] && N=320
    echo "--- $row rate=$R n=$N: $(date +%H:%M:%S) ---"
    timeout -k 30 900 $VLLM_BIN bench serve --host 127.0.0.1 --port "$port" \
      --backend vllm --model "$GPTOSS" --seed 0 \
      --dataset-name custom --dataset-path "$DS" \
      --custom-output-len -1 --skip-chat-template --ignore-eos \
      --num-prompts $N --request-rate $R \
      --save-result --result-dir "$OUT" \
      --result-filename "${row}_r${R}.json" \
      > "$LOG/${row}_r${R}.log" 2>&1
    grep -E 'Median TPOT|Output token throughput' "$LOG/${row}_r${R}.log" | head -2
  done
}

sweep_row() {  # row, then serve cmd...
  local row=$1; shift
  cleanup_gpu
  echo "######## ROW $row : $(date +%H:%M:%S) ########"
  "$@" > "$LOG/${row}_serve.log" 2>&1 &
  if ! wait_port $PORT; then
    echo "ROW_${row}_SERVE_FAILED"
    grep -aiE 'error|Traceback' "$LOG/${row}_serve.log" | tail -3
    cleanup_gpu
    return 1
  fi
  bench_rates "$row" $PORT
  cleanup_gpu
}

# ---- parity gate for the NEW fg-online combo (fail-fast) ----
# The stock single-GPU baseline is (re)generated in place, then the FG-online
# combo must reproduce its greedy texts token-for-token (8/8).
cleanup_gpu
echo "######## FG-online parity gate (8 prompts) ########"
timeout -k 30 1200 $V $DRV \
  --model $GPTOSS --dataset-jsonl $UDS \
  --num-prompts 8 --max-num-seqs 32 --max-model-len 5128 --no-prefix-cache \
  --placement none \
  --dump-texts "$OUT/parity_base8.texts.json" \
  --output-json "$OUT/parity_base8.json" > "$LOG/parity_base.log" 2>&1
cleanup_gpu
FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 1200 $V $DRV \
  --model $GPTOSS --dataset-jsonl $UDS \
  --num-prompts 8 --max-num-seqs 32 --max-model-len 5128 --no-prefix-cache \
  --placement af --phase-aware --phase-route --pingpong \
  --dump-texts "$OUT/fg_online_parity8.texts.json" \
  --output-json "$OUT/fg_online_parity8.json" > "$LOG/parity.log" 2>&1
$V - <<EOF
import json
a = json.load(open("$OUT/parity_base8.texts.json"))
b = json.load(open("$OUT/fg_online_parity8.texts.json"))
m = sum(1 for x, y in zip(a, b) if x == y)
print(f"FG_ONLINE_PARITY {m}/{len(a)}")
assert m == len(a) == 8, "FG-online parity gate FAILED"
EOF

# ---- FG (lever-A online) ----
VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=$GPU_PAIR \
PYTHONPATH="$RUNTIME_ROOT:$EXAMPLES_DIR" \
FLUIDGPU_VLLM_PLACEMENT=af FLUIDGPU_EXPERT_DEVICE=cuda:1 FLUIDGPU_HOP=copy \
FLUIDGPU_PINGPONG=1 FLUIDGPU_PHASE_AWARE=1 FLUIDGPU_PHASE_ROUTE=1 \
FLUIDGPU_FULL_DECODE_GRAPH=1 \
  sweep_row fg_leverA $VLLM_BIN serve "$GPTOSS" --port $PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --worker-cls fluidgpu_torch.vllm_integration.FluidGPUWorker \
    --compilation-config "$CC_FULL"

# ---- PD (stock stack + proxy) ----
cleanup_gpu
echo "######## ROW pd (stock PD stack) : $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$OUT/pd_stack" \
MODEL="$GPTOSS" VLLM_BIN="$VLLM_BIN" PYTHON_BIN=$V MODE=serve \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=$HCA_P DECODE_MOONCAKE_DEVICES=$HCA_D \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=7200 \
bash scripts/run_pd_baseline_vllm.sh > "$LOG/pd_stack.log" 2>&1 &
pd_ready=0
for i in $(seq 1 130); do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8030/v1/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$GPTOSS\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  [ "$c" = "200" ] && { pd_ready=1; echo "pd proxy ready (${i}x4s)"; break; }
  sleep 4
done
if [ "$pd_ready" = "1" ]; then
  bench_rates pd 8030
else
  echo "ROW_pd_SERVE_FAILED"
  grep -aiE 'error|Traceback' "$LOG/pd_stack.log" | tail -3
fi
cleanup_gpu

# ---- homo + AF reference rows (same-day) ----
VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=$GPU_P \
  sweep_row homo_l40s $VLLM_BIN serve "$GPTOSS" --port $PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching

VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=$GPU_D \
  sweep_row homo_a100 $VLLM_BIN serve "$GPTOSS" --port $PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching

VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=$GPU_PAIR \
PYTHONPATH="$RUNTIME_ROOT:$EXAMPLES_DIR" \
FLUIDGPU_VLLM_PLACEMENT=af FLUIDGPU_EXPERT_DEVICE=cuda:1 FLUIDGPU_HOP=copy \
  sweep_row af $VLLM_BIN serve "$GPTOSS" --port $PORT \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --worker-cls fluidgpu_torch.vllm_integration.FluidGPUWorker \
    --compilation-config "$CC_PIECE"

echo "FIG7_V2_ALL_DONE"
