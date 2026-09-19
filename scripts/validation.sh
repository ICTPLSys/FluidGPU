#!/usr/bin/env bash
set -euo pipefail

requests=100
policy="fluidgpu"
pipeline="off"
rps=""
model="${FLUIDGPU_VALIDATION_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
model_root="${FLUIDGPU_MODEL_DIR:-$HOME/.cache/fluidgpu/models}"
monitor_window_ms=300
monitor_beta=1.5
comm_bw_gbps=""
device_id=0
max_new_tokens=16
allow_transformers_mismatch=0
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requests) requests="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --pipeline) pipeline="$2"; shift 2 ;;
    --rps) rps="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --monitor-window-ms) monitor_window_ms="$2"; shift 2 ;;
    --monitor-beta) monitor_beta="$2"; shift 2 ;;
    --comm-bw-gbps) comm_bw_gbps="$2"; shift 2 ;;
    --device-id) device_id="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --allow-transformers-mismatch) allow_transformers_mismatch=1; shift ;;
    *) echo "validation.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

fail() {
  echo "validation.sh: $*" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail "python3 is not available"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available; NVIDIA driver/GPU is required"
if ! nvidia-smi -i "$device_id" >/dev/null 2>&1; then
  fail "CUDA device ${device_id} is not visible to nvidia-smi"
fi
python3 - <<'PY' || fail "Python environment must provide torch with CUDA support"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
PY

ts="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$out_dir" ]]; then
  out_dir="${FLUIDGPU_RUN_OUTPUT_DIR:-artifacts/validation/${ts}}"
fi
model_path="${model_root}/${model//\//--}"
model_arg="$model"
if [[ -d "$model_path" ]]; then
  model_arg="$model_path"
fi

cmd=(
  python3 fluidgpu_runtime/examples/fluidgpu_torch/llm_generate.py
  --single-gpu
  --model "$model_arg"
  --requests "$requests"
  --max-new-tokens "$max_new_tokens"
  --device-id "$device_id"
  --diagnostics-output "$out_dir/diagnostics.pt"
  --log-jsonl "$out_dir/log.jsonl"
  --summary-json "$out_dir/summary.json"
)
if [[ "$allow_transformers_mismatch" -eq 1 ]]; then
  cmd+=(--allow-transformers-mismatch)
fi
if [[ -n "$comm_bw_gbps" ]]; then
  echo "validation.sh: --comm-bw-gbps is recorded by experiment metadata; generation uses the local CUDA device" >&2
fi

mkdir -p "$out_dir"
if [[ ! -d "$model_path" && -z "${HF_TOKEN:-}" ]]; then
  fail "model weights not found at ${model_path} and HF_TOKEN is not set; run scripts/fetch_weights.sh or export HF_TOKEN"
fi

printf '%s\n' "${cmd[*]}" >"${out_dir}/command.txt"
"${cmd[@]}" >"${out_dir}/generation_stdout.log" 2>"${out_dir}/generation_stderr.log"
echo "validation.sh: wrote ${out_dir}"
