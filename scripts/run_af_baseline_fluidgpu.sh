#!/usr/bin/env bash
set -euo pipefail

# CUDA-graph decode is the validated fast path on this artifact; all
# policies share it. Override with FLUIDGPU_CUDA_GRAPHS=0 to compare eager.
export FLUIDGPU_CUDA_GRAPHS="${FLUIDGPU_CUDA_GRAPHS:-1}"

# AF (attention/FFN split) baseline launcher on fluidgpu_torch two-rank runtime.
# This script provides a real executable AF baseline path in AD:
#   - launches rank0/rank1 with the same profile
#   - writes logs and diagnostics under artifacts/baselines/af_fluidgpu/<timestamp>/
# Environment limitations (weights, GPUs, NIC, NCCL) may still prevent success.

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PROFILE="${PROFILE:-}"
PROMPT="${PROMPT:-Explain GPU disaggregation in three sentences.}"
REQUESTS="${REQUESTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-}"
DATASET_JSONL="${DATASET_JSONL:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29540}"
RANK0_GPU="${RANK0_GPU:-0}"
RANK1_GPU="${RANK1_GPU:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
ALLOW_TRANSFORMERS_MISMATCH="${ALLOW_TRANSFORMERS_MISMATCH:-0}"
COMM_TRANSPORT="${COMM_TRANSPORT:-rdma}"
MODEL_ROOT="${FLUIDGPU_MODEL_DIR:-$HOME/.cache/fluidgpu/models}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_ROOT="${LOG_ROOT:-artifacts/baselines/af_fluidgpu}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${FLUIDGPU_RUN_OUTPUT_DIR:-}" ]]; then
  RUN_DIR="${FLUIDGPU_RUN_OUTPUT_DIR}"
else
  RUN_DIR="${LOG_ROOT}/${RUN_ID}"
fi
RANK0_PID=""
RANK1_PID=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ARG="${MODEL}"

fail() {
  echo "run_af_baseline_fluidgpu.sh: $*" >&2
  exit 1
}

cleanup() {
  set +e
  if [[ -n "${RANK0_PID}" ]] && kill -0 "${RANK0_PID}" >/dev/null 2>&1; then
    kill "${RANK0_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${RANK1_PID}" ]] && kill -0 "${RANK1_PID}" >/dev/null 2>&1; then
    kill "${RANK1_PID}" >/dev/null 2>&1 || true
  fi
  wait || true
}

trap cleanup EXIT INT TERM

check_env() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "python is not available: ${PYTHON_BIN}"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
  [[ -f "${PROFILE}" ]] || fail "profile file not found: ${PROFILE}"
  nvidia-smi -i "${RANK0_GPU}" >/dev/null 2>&1 || fail "rank0 gpu is not visible: ${RANK0_GPU}"
  nvidia-smi -i "${RANK1_GPU}" >/dev/null 2>&1 || fail "rank1 gpu is not visible: ${RANK1_GPU}"
  "${PYTHON_BIN}" - <<'PY' || fail "python env must provide torch with CUDA support"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
PY
}

resolve_model_arg() {
  local model_path="${MODEL_ROOT}/${MODEL//\//--}"
  if [[ -d "${model_path}" ]]; then
    MODEL_ARG="${model_path}"
    return
  fi
  case "${MODEL}" in
    meta-llama/*)
      [[ -n "${HF_TOKEN:-}" ]] || fail "model weights not found at ${model_path} and HF_TOKEN is not set for gated model ${MODEL}"
      ;;
  esac
  MODEL_ARG="${MODEL}"
}

resolve_base_profile() {
  if [[ -n "${PROFILE}" ]]; then
    return
  fi
  case "${MODEL}" in
    openai/gpt-oss-20b)
      PROFILE="fluidgpu_runtime/profiles/gpt_oss_20b_kernel_group_split.json"
      ;;
    meta-llama/Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct)
      # AF needs attention/FFN sub-layer granularity; the whole-layer profile
      # (llama31_8b_instruct_layers.json) cannot be mapped to the AF policy.
      PROFILE="fluidgpu_runtime/profiles/llama31_8b_milp_a100_l40s.json"
      ;;
    Qwen/Qwen2.5-1.5B)
      PROFILE="fluidgpu_runtime/profiles/qwen25_1p5b_kernel_group_split.json"
      ;;
    Qwen/Qwen2.5-VL-7B-Instruct)
      PROFILE="fluidgpu_runtime/profiles/qwen25_vl_7b_kernel_group_split.json"
      ;;
    Qwen/Qwen3-235B-A22B)
      PROFILE="prebuilt:qwen3_235b_a22b_moe_fine_split"
      ;;
    mistralai/Mamba-Codestral-7B-v0.1)
      PROFILE="fluidgpu_runtime/profiles/mamba_codestral_7b_layers.json"
      ;;
    stabilityai/stable-diffusion-3.5-medium)
      PROFILE="fluidgpu_runtime/profiles/sd35_medium_layers.json"
      ;;
    *)
      fail "no default AF profile mapping for model '${MODEL}'; set PROFILE=/abs/path/to/profile.json"
      ;;
  esac
}

prepare_profile() {
  mkdir -p "${RUN_DIR}"
  resolve_model_arg
  resolve_base_profile

  local base_profile="${PROFILE}"
  if [[ "${base_profile}" == prebuilt:* ]]; then
    local prebuilt_name="${base_profile#prebuilt:}"
    local generated_profile="${RUN_DIR}/base_profile.json"
    PYTHONPATH="${REPO_ROOT}/fluidgpu_runtime:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      fluidgpu_runtime/examples/fluidgpu_torch/prebuilt_profile.py \
      --name "${prebuilt_name}" \
      --output-profile "${generated_profile}" \
      --model "${MODEL_ARG}"
    base_profile="${generated_profile}"
  fi
  [[ -f "${base_profile}" ]] || fail "profile file not found: ${base_profile}"

  local effective_profile="${RUN_DIR}/effective_profile.json"
  PYTHONPATH="${REPO_ROOT}/fluidgpu_runtime:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    fluidgpu_runtime/examples/fluidgpu_torch/profile_variant.py \
    --input-profile "${base_profile}" \
    --output-profile "${effective_profile}" \
    --policy af \
    --model "${MODEL_ARG}"
  PROFILE="${effective_profile}"
}

build_cmd() {
  local rank="$1"
  local cmd=(
    "${PYTHON_BIN}" fluidgpu_runtime/examples/fluidgpu_torch/llm_generate.py
    --model "${MODEL_ARG}"
    --profile "${PROFILE}"
    --prompt "${PROMPT}"
    --requests "${REQUESTS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --comm-transport "${COMM_TRANSPORT}"
  )
  if [[ -n "${REQUEST_RATE}" ]]; then
    cmd+=(--request-rate "${REQUEST_RATE}")
  fi
  if [[ -n "${DATASET_JSONL}" ]]; then
    cmd+=(--dataset-jsonl "${DATASET_JSONL}")
  fi
  if [[ "${ALLOW_TRANSFORMERS_MISMATCH}" == "1" ]]; then
    cmd+=(--allow-transformers-mismatch)
  fi
  # RDMA transport needs a per-rank IB device; honor the same env contract as
  # run_fluidgpu_e2e_two_rank.sh (RANK0_NCCL_IB_HCA / RANK1_NCCL_IB_HCA).
  if [[ "${rank}" == "0" && -n "${RANK0_NCCL_IB_HCA:-}" ]]; then
    cmd+=(--nccl-ib-hca "${RANK0_NCCL_IB_HCA}")
  elif [[ "${rank}" == "1" && -n "${RANK1_NCCL_IB_HCA:-}" ]]; then
    cmd+=(--nccl-ib-hca "${RANK1_NCCL_IB_HCA}")
  fi
  if [[ "${rank}" == "0" ]]; then
    cmd+=(--diagnostics-output "${RUN_DIR}/rank0_diagnostics.pt")
    cmd+=(--log-jsonl "${RUN_DIR}/log.jsonl")
    cmd+=(--summary-json "${RUN_DIR}/summary.json")
  fi
  printf '%q ' "${cmd[@]}"
}

main() {
  cd "${REPO_ROOT}"
  prepare_profile
  check_env

  local rank0_cmd
  local rank1_cmd
  rank0_cmd="$(build_cmd 0)"
  rank1_cmd="$(build_cmd 1)"
  printf '%s\n' "${rank0_cmd}" > "${RUN_DIR}/rank0.command.txt"
  printf '%s\n' "${rank1_cmd}" > "${RUN_DIR}/rank1.command.txt"

  echo "run_af_baseline_fluidgpu.sh: starting rank1 on GPU ${RANK1_GPU}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${RANK1_GPU}" \
  FLUIDGPU_RANK=1 \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  PYTHONPATH="${PWD}/fluidgpu_runtime:${PYTHONPATH:-}" \
  bash -lc "${rank1_cmd}" > "${RUN_DIR}/rank1.log" 2>&1 &
  RANK1_PID=$!

  sleep 2

  echo "run_af_baseline_fluidgpu.sh: starting rank0 on GPU ${RANK0_GPU}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${RANK0_GPU}" \
  FLUIDGPU_RANK=0 \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  PYTHONPATH="${PWD}/fluidgpu_runtime:${PYTHONPATH:-}" \
  bash -lc "${rank0_cmd}" > "${RUN_DIR}/rank0.log" 2>&1 &
  RANK0_PID=$!

  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "${RANK0_PID}" >/dev/null 2>&1 && ! kill -0 "${RANK1_PID}" >/dev/null 2>&1; then
      break
    fi
    if (( "$(date +%s)" - start_ts >= TIMEOUT_SECONDS )); then
      fail "timeout waiting for AF baseline run to complete"
    fi
    sleep 2
  done

  wait "${RANK0_PID}"
  wait "${RANK1_PID}"
  [[ -f "${RUN_DIR}/summary.json" ]] || fail "missing summary.json in ${RUN_DIR}"
  [[ -f "${RUN_DIR}/log.jsonl" ]] || fail "missing log.jsonl in ${RUN_DIR}"

  echo "run_af_baseline_fluidgpu.sh: finished"
  echo "run_af_baseline_fluidgpu.sh: logs at ${RUN_DIR}"
}

main "$@"
