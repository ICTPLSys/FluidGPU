#!/usr/bin/env bash
set -euo pipefail

# PD disaggregation baseline launcher using vLLM disaggregated serving.
# This script launches:
#   1) prefill server group(s)
#   2) decode server group(s)
#   3) mooncake connector proxy
# and optionally runs a random benchmark against the proxy endpoint.

MODEL="${MODEL:-openai/gpt-oss-20b}"
MODE="${MODE:-bench}" # bench | serve
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
PROXY_PORT="${PROXY_PORT:-8030}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# All PD traffic (bootstrap registration, health checks, proxy, benchmark) is
# machine-local. Python HTTP clients honor HTTP(S)_PROXY (curl does not for
# http), so a configured proxy silently blackholes the Mooncake bootstrap
# registration (engine core registers via http://<lan-ip>:<bootstrap-port>)
# and the API server times out waiting for the core. Strip proxies here.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="${NO_PROXY}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use ';' to separate server groups and ',' to separate TP ranks inside one group.
PREFILL_GPUS="${PREFILL_GPUS:-2}"
DECODE_GPUS="${DECODE_GPUS:-0,1}"
PREFILL_PORTS="${PREFILL_PORTS:-8010}"
BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-8998}"
DECODE_PORTS="${DECODE_PORTS:-8020}"

# Known-good Mooncake RDMA device pins can be overridden by env.
prefill_hca_placeholder="${FLUIDGPU_PREFILL_HCA_PLACEHOLDER:-<set-prefill-ib-hca>}"
decode_hca_placeholder="${FLUIDGPU_DECODE_HCA_PLACEHOLDER:-<set-decode-ib-hca>}"
PREFILL_MOONCAKE_DEVICES="${PREFILL_MOONCAKE_DEVICES:-$prefill_hca_placeholder}"
DECODE_MOONCAKE_DEVICES="${DECODE_MOONCAKE_DEVICES:-$decode_hca_placeholder}"

VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/mooncake_connector_proxy.py}"

BENCH_DATASET_NAME="${BENCH_DATASET_NAME:-random}"
BENCH_RANDOM_INPUT_LEN="${BENCH_RANDOM_INPUT_LEN:-4096}"
BENCH_RANDOM_OUTPUT_LEN="${BENCH_RANDOM_OUTPUT_LEN:-684}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-320}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-32}"
BENCH_RESULT_FILENAME="${BENCH_RESULT_FILENAME:-bench_result.json}"

LOG_ROOT="${LOG_ROOT:-artifacts/baselines/pd_vllm}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${FLUIDGPU_RUN_OUTPUT_DIR:-}" ]]; then
  RUN_DIR="${FLUIDGPU_RUN_OUTPUT_DIR}"
else
  RUN_DIR="${LOG_ROOT}/${RUN_ID}"
fi

PIDS=()

fail() {
  echo "run_pd_baseline_vllm.sh: $*" >&2
  exit 1
}

cleanup() {
  set +e
  echo "run_pd_baseline_vllm.sh: cleaning up..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait || true
}

trap cleanup EXIT INT TERM

check_required_files() {
  if [[ "$VLLM_BIN" == */* ]]; then
    [[ -x "$VLLM_BIN" ]] || fail "vLLM binary is not executable: $VLLM_BIN"
  else
    command -v "$VLLM_BIN" >/dev/null 2>&1 || fail "vLLM binary not found in PATH: $VLLM_BIN"
  fi
  if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || fail "Python binary is not executable: $PYTHON_BIN"
  else
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python binary not found in PATH: $PYTHON_BIN"
  fi
  [[ -f "$PROXY_SCRIPT" ]] || fail "Mooncake proxy script not found: $PROXY_SCRIPT"
  if [[ "$PREFILL_MOONCAKE_DEVICES" == "$prefill_hca_placeholder" || "$DECODE_MOONCAKE_DEVICES" == "$decode_hca_placeholder" ]]; then
    fail "Set PREFILL_MOONCAKE_DEVICES/DECODE_MOONCAKE_DEVICES for your machine before launching."
  fi
}

check_array_lengths() {
  local name="$1"
  local expected="$2"
  local actual="$3"
  [[ "$expected" -eq "$actual" ]] || fail "${name} count mismatch: expected ${expected}, got ${actual}"
}

count_group_size() {
  local group="$1"
  [[ -n "$group" ]] || fail "empty GPU group"
  local arr=()
  IFS=',' read -ra arr <<< "$group"
  echo "${#arr[@]}"
}

uniform_tp_size() {
  local name="$1"
  shift
  local group
  local size
  local expected=""
  for group in "$@"; do
    size="$(count_group_size "$group")"
    if [[ -z "$expected" ]]; then
      expected="$size"
      continue
    fi
    [[ "$size" -eq "$expected" ]] || fail "${name} has mixed TP sizes: $*"
  done
  echo "$expected"
}

validate_topology() {
  local prefill_tp
  local decode_tp
  prefill_tp="$(uniform_tp_size "PREFILL_GPUS" "${PREFILL_GPU_ARRAY[@]}")"
  decode_tp="$(uniform_tp_size "DECODE_GPUS" "${DECODE_GPU_ARRAY[@]}")"
  [[ "$prefill_tp" -eq "$decode_tp" ]] || fail "Mooncake requires matching TP sizes (prefill=${prefill_tp}, decode=${decode_tp})"
}

wait_for_server() {
  local port="$1"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "run_pd_baseline_vllm.sh: server on port ${port} is ready"
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= TIMEOUT_SECONDS )); then
      return 1
    fi
    sleep 1
  done
}

launch_prefill_servers() {
  local i
  for i in "${!PREFILL_GPU_ARRAY[@]}"; do
    local gpu_group="${PREFILL_GPU_ARRAY[$i]}"
    local port="${PREFILL_PORT_ARRAY[$i]}"
    local bootstrap_port="${BOOTSTRAP_PORT_ARRAY[$i]}"
    local mooncake_device="${PREFILL_MOONCAKE_DEVICE_ARRAY[$i]}"
    local tp_size
    tp_size="$(count_group_size "$gpu_group")"

    MC_MS_AUTO_DISC=0 \
    MOONCAKE_DEVICE="$mooncake_device" \
    VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port" \
    CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
    CUDA_VISIBLE_DEVICES="$gpu_group" \
    "$VLLM_BIN" serve "$MODEL" \
      --port "$port" \
      --tensor-parallel-size "$tp_size" \
      --max_num_batched_tokens "${PREFILL_MAX_BATCHED_TOKENS:-16384}" \
      --max-num-seqs "${PREFILL_MAX_SEQS:-32}" \
      --no-enable-prefix-caching \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"device_name\":\"${mooncake_device}\"}}" \
      >"${RUN_DIR}/prefill_$((i + 1)).log" 2>&1 &
    PIDS+=("$!")
    PROXY_ARGS+=(--prefill "http://0.0.0.0:${port}" "$bootstrap_port")
  done
}

launch_decode_servers() {
  local i
  for i in "${!DECODE_GPU_ARRAY[@]}"; do
    local gpu_group="${DECODE_GPU_ARRAY[$i]}"
    local port="${DECODE_PORT_ARRAY[$i]}"
    local mooncake_device="${DECODE_MOONCAKE_DEVICE_ARRAY[$i]}"
    local tp_size
    tp_size="$(count_group_size "$gpu_group")"

    MC_MS_AUTO_DISC=0 \
    MOONCAKE_DEVICE="$mooncake_device" \
    CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
    CUDA_VISIBLE_DEVICES="$gpu_group" \
    "$VLLM_BIN" serve "$MODEL" \
      --port "$port" \
      --tensor-parallel-size "$tp_size" \
      --max_num_batched_tokens "${DECODE_MAX_BATCHED_TOKENS:-16384}" \
      --max-num-seqs "${DECODE_MAX_SEQS:-32}" \
      --no-enable-prefix-caching \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"device_name\":\"${mooncake_device}\"}}" \
      >"${RUN_DIR}/decode_$((i + 1)).log" 2>&1 &
    PIDS+=("$!")
    PROXY_ARGS+=(--decode "http://0.0.0.0:${port}")
  done
}

launch_proxy() {
  CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
  "$PYTHON_BIN" "$PROXY_SCRIPT" "${PROXY_ARGS[@]}" --port "$PROXY_PORT" \
    >"${RUN_DIR}/proxy.log" 2>&1 &
  PIDS+=("$!")
}

run_benchmark() {
  local dataset_args=()
  if [[ "$BENCH_DATASET_NAME" == "custom" ]]; then
    # Expects jsonl lines {"prompt": ..., "output_tokens": N}; per-request
    # output lengths are honored (paper fig6 spliced sets ship this schema).
    [[ -n "${BENCH_DATASET_PATH:-}" ]] || fail "BENCH_DATASET_NAME=custom requires BENCH_DATASET_PATH"
    # --custom-output-len defaults to 256 and would OVERRIDE the per-request
    # output_tokens column; -1 restores the per-request values.
    dataset_args=(--dataset-path "$BENCH_DATASET_PATH" --skip-chat-template
                  --custom-output-len -1)
  else
    dataset_args=(--random-input-len "$BENCH_RANDOM_INPUT_LEN"
                  --random-output-len "$BENCH_RANDOM_OUTPUT_LEN")
  fi
  if [[ "${BENCH_IGNORE_EOS:-0}" == "1" ]]; then
    dataset_args+=(--ignore-eos)
  fi
  if [[ -n "${BENCH_MAX_CONCURRENCY:-}" ]]; then
    dataset_args+=(--max-concurrency "$BENCH_MAX_CONCURRENCY")
  fi
  CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
  "$VLLM_BIN" bench serve --port "$PROXY_PORT" \
    --backend vllm \
    --model "$MODEL" \
    --seed "$(date +%s)" \
    --dataset-name "$BENCH_DATASET_NAME" \
    "${dataset_args[@]}" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --request-rate "$BENCH_REQUEST_RATE" \
    --save-result \
    --result-dir "$RUN_DIR" \
    --result-filename "$BENCH_RESULT_FILENAME" \
    | tee "${RUN_DIR}/benchmark.log"
}

write_benchmark_summary() {
  local result_path="${RUN_DIR}/${BENCH_RESULT_FILENAME}"
  [[ -f "${result_path}" ]] || fail "missing benchmark result json: ${result_path}"
  "$PYTHON_BIN" - "${result_path}" "${RUN_DIR}/summary.json" "${RUN_DIR}/log.jsonl" "${MODEL}" "${BENCH_REQUEST_RATE}" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
model = sys.argv[4]
request_rate = float(sys.argv[5])
raw = json.loads(result_path.read_text())
if isinstance(raw, list):
    if not raw:
        raise SystemExit(f"empty benchmark result: {result_path}")
    raw = raw[-1]
if not isinstance(raw, dict):
    raise SystemExit(f"unexpected benchmark result payload: {type(raw).__name__}")

def pick_float(*names):
    for name in names:
        value = raw.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

def pick_int(*names):
    for name in names:
        value = raw.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None

throughput_req_s = pick_float("request_throughput", "throughput_req_s")
throughput_tok_s = pick_float("output_throughput", "throughput_tok_s")
latency_ms_avg = pick_float("mean_e2el_ms", "latency_ms_avg")
requests = pick_int("completed", "num_prompts", "requests")
generated_tokens = pick_int("total_output", "generated_tokens")
elapsed_s = pick_float("elapsed_s", "duration_s", "benchmark_duration_s")

if requests is None and throughput_req_s is not None and elapsed_s is not None:
    requests = int(round(throughput_req_s * elapsed_s))
if elapsed_s is None and requests is not None and throughput_req_s and throughput_req_s > 0:
    elapsed_s = requests / throughput_req_s
if generated_tokens is None and throughput_tok_s is not None and elapsed_s is not None:
    generated_tokens = int(round(throughput_tok_s * elapsed_s))

summary = {
    "status": "ok",
    "model": model,
    "request_rate": request_rate,
    "requests": requests,
    "generated_tokens": generated_tokens,
    "elapsed_s": elapsed_s,
    "throughput_req_s": throughput_req_s,
    "throughput_tok_s": throughput_tok_s,
    "latency_ms_avg": latency_ms_avg,
    "source": "vllm bench serve",
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
log_path.write_text(json.dumps(summary) + "\n")
PY
}

main() {
  command -v curl >/dev/null 2>&1 || fail "curl is required"
  check_required_files
  mkdir -p "$RUN_DIR"

  IFS=';' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
  IFS=';' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
  IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
  IFS=',' read -ra BOOTSTRAP_PORT_ARRAY <<< "$BOOTSTRAP_PORTS"
  IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
  IFS=';' read -ra PREFILL_MOONCAKE_DEVICE_ARRAY <<< "$PREFILL_MOONCAKE_DEVICES"
  IFS=';' read -ra DECODE_MOONCAKE_DEVICE_ARRAY <<< "$DECODE_MOONCAKE_DEVICES"

  check_array_lengths "PREFILL_PORTS" "${#PREFILL_GPU_ARRAY[@]}" "${#PREFILL_PORT_ARRAY[@]}"
  check_array_lengths "BOOTSTRAP_PORTS" "${#PREFILL_GPU_ARRAY[@]}" "${#BOOTSTRAP_PORT_ARRAY[@]}"
  check_array_lengths "PREFILL_MOONCAKE_DEVICES" "${#PREFILL_GPU_ARRAY[@]}" "${#PREFILL_MOONCAKE_DEVICE_ARRAY[@]}"
  check_array_lengths "DECODE_PORTS" "${#DECODE_GPU_ARRAY[@]}" "${#DECODE_PORT_ARRAY[@]}"
  check_array_lengths "DECODE_MOONCAKE_DEVICES" "${#DECODE_GPU_ARRAY[@]}" "${#DECODE_MOONCAKE_DEVICE_ARRAY[@]}"
  validate_topology

  PROXY_ARGS=()
  launch_prefill_servers
  launch_decode_servers
  launch_proxy

  local port
  for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
    wait_for_server "$port" || fail "timeout waiting for server on port ${port}"
  done

  echo "run_pd_baseline_vllm.sh: PD baseline servers are ready"
  echo "run_pd_baseline_vllm.sh: logs at ${RUN_DIR}"
  echo "run_pd_baseline_vllm.sh: proxy endpoint http://127.0.0.1:${PROXY_PORT}"

  if [[ "$MODE" == "bench" ]]; then
    run_benchmark
    write_benchmark_summary
    echo "run_pd_baseline_vllm.sh: benchmark finished"
  else
    echo "run_pd_baseline_vllm.sh: MODE=serve, keeping processes alive; Ctrl+C to stop"
    while true; do sleep 3600; done
  fi
}

main "$@"
