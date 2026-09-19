#!/usr/bin/env bash
# Measure the GPT-OSS Fig.6 stock-PD arm with the same client and 25%-75%
# completion-window steady metric as the FluidGPU arm.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
set -e

MODEL="$MODEL_GT"
IN="${IN:-4096}"
OUTL="${OUTL:-684}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
O="${GT_STEADY_OUT:-$S/gt684_final_mirror/pd_steady_ms32_conc32}"

exp_preflight "$MODEL"
mkdir -p "$O"

STACK_PID=""
cleanup() {
  if [[ -n "$STACK_PID" ]] && kill -0 "$STACK_PID" 2>/dev/null; then
    # The stack runs in its own process group so cleanup is scoped to this
    # experiment and does not wait for MODE=serve's long sleep to return.
    kill -TERM -- "-$STACK_PID" 2>/dev/null || true
    wait "$STACK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "######## GPT-OSS PD-steady ${IN}/${OUTL}, ms=${MAX_NUM_SEQS}, conc=${CONCURRENCY}: $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/stack" \
MODEL="$MODEL" VLLM_BIN="$VB" PYTHON_BIN="$PY" MODE=serve \
PREFILL_GPUS="$GPU_P" DECODE_GPUS="$GPU_D" \
PREFILL_MOONCAKE_DEVICES="$HCA_P" DECODE_MOONCAKE_DEVICES="$HCA_D" \
DECODE_MAX_SEQS="$MAX_NUM_SEQS" PREFILL_MAX_SEQS="$MAX_NUM_SEQS" \
TIMEOUT_SECONDS=1800 \
setsid bash scripts/run_pd_baseline_vllm.sh > "$O/stack.log" 2>&1 &
STACK_PID=$!

ready=0
for i in $(seq 1 150); do
  if ! kill -0 "$STACK_PID" 2>/dev/null; then
    echo "PD stack exited before becoming ready" >&2
    tail -40 "$O/stack.log" >&2
    exit 1
  fi
  body=$(curl -s -m 8 -X POST http://127.0.0.1:8030/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null || true)
  if [[ "$body" == *'"choices"'* ]]; then
    ready=1
    echo "PD proxy ready (${i}x4s)"
    break
  fi
  sleep 4
done

if [[ "$ready" != "1" ]]; then
  echo "PD stack did not become ready" >&2
  tail -40 "$O/stack.log" >&2
  exit 1
fi

FG_D_URL=http://127.0.0.1:8030/v1/completions \
timeout -k 20 2400 "$PY" -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" \
  --num-prompts "$NUM_PROMPTS" --input-len "$IN" --output-len "$OUTL" \
  --concurrency "$CONCURRENCY" --local-prefill-frac 1.0 \
  > "$O/pd_steady.out" 2>&1

grep -aE 'ok ===|wall=|errors' "$O/pd_steady.out" | head -3
echo "GPTOSS20B_PD_STEADY_DONE"
