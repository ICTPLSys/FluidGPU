#!/usr/bin/env bash
# Sweep the GPT-OSS 4096/684 FluidGPU hybrid at a fixed server batch depth.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
set -e

MODEL="$MODEL_GT"
IN="${IN:-4096}"
OUTL="${OUTL:-684}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
PHIS="${PHIS:-0.00 0.06 0.12 0.18 0.24}"
O="${FG_SWEEP_OUT:-$S/gt684_hybrid_ms${MAX_NUM_SEQS}}"

exp_preflight "$MODEL"
mkdir -p "$O"

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "######## GPT-OSS FG ${IN}/${OUTL}, ms=${MAX_NUM_SEQS}, conc=${CONCURRENCY}: $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE="$HCA_P" VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES="$GPU_P" setsid "$VB" serve "$MODEL" --port 8100 \
  --tensor-parallel-size 1 --max_num_batched_tokens 16384 \
  --max-num-seqs "$MAX_NUM_SEQS" --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_P\"}}" \
  > "$O/prefill.log" 2>&1 &
PIDS+=("$!")

MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE="$HCA_D" \
CUDA_VISIBLE_DEVICES="$GPU_D" setsid "$VB" serve "$MODEL" --port 8200 \
  --tensor-parallel-size 1 --max_num_batched_tokens 16384 \
  --max-num-seqs "$MAX_NUM_SEQS" --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_D\"}}" \
  > "$O/decode.log" 2>&1 &
PIDS+=("$!")

ready=0
for i in $(seq 1 150); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null || true)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8200/health 2>/dev/null || true)
  if [[ "$a" == "200" && "$b" == "200" ]]; then
    ready=1
    echo "FG pair ready (${i}x4s)"
    break
  fi
  sleep 4
done
if [[ "$ready" != "1" ]]; then
  echo "FG pair did not become ready" >&2
  tail -40 "$O/prefill.log" >&2
  tail -40 "$O/decode.log" >&2
  exit 1
fi

timeout -k 10 240 "$PY" -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" \
  --num-prompts 4 --input-len "$IN" --output-len 16 --concurrency 2 \
  > "$O/smoke.out" 2>&1
grep -aE 'ok ===|wall=|errors' "$O/smoke.out" | head -3

for phi in $PHIS; do
  echo "######## phi=$phi: $(date +%H:%M:%S) ########"
  timeout -k 20 1800 "$PY" -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --input-len "$IN" --output-len "$OUTL" \
    --concurrency "$CONCURRENCY" --local-prefill-frac "$phi" \
    --local-url http://127.0.0.1:8100/v1/completions \
    > "$O/h_phi${phi}.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/h_phi${phi}.out" | head -3
done

echo "GPTOSS20B_FG_SWEEP_DONE"
