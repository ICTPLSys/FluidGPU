#!/usr/bin/env bash
set -euo pipefail

figure="manual"
requests=100
policy="${POLICY:-fluidgpu}"
model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_root="${FLUIDGPU_MODEL_DIR:-$HOME/.cache/fluidgpu/models}"
rank0_gpu="${RANK0_GPU:-0}"
rank1_gpu="${RANK1_GPU:-1}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
comm_transport="${COMM_TRANSPORT:-auto}"
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-}"
nccl_ib_hca="${NCCL_IB_HCA:-}"
nccl_ib_gid_index="${NCCL_IB_GID_INDEX:-}"
max_new_tokens="${MAX_NEW_TOKENS:-16}"
max_seq_len="${MAX_SEQ_LEN:-2048}"
request_rate=""
allow_transformers_mismatch="${ALLOW_TRANSFORMERS_MISMATCH:-0}"
prompt="${PROMPT:-Explain GPU disaggregation in three sentences.}"
python_bin="${PYTHON_BIN:-python3}"
profile_override="${PROFILE:-}"
dataset_jsonl="${DATASET_JSONL:-}"
out_dir=""
timeout_seconds="${TIMEOUT_SECONDS:-1800}"

# Widen the NCCL watchdog budget for long, GIL-shared pipelined runs. The
# default PG and every worker group (comm.new_worker_backend) read this; without
# it the budget is the 10-minute NCCL default and long Splitwise runs can trip a
# paired-p2p timeout. Any externally-set value wins.
export FLUIDGPU_NCCL_TIMEOUT_S="${FLUIDGPU_NCCL_TIMEOUT_S:-1200}"

rank0_pid=""
rank1_pid=""

fail() {
  echo "run_fluidgpu_e2e_two_rank.sh: $*" >&2
  exit 1
}

cleanup() {
  set +e
  if [[ -n "${rank0_pid}" ]] && kill -0 "${rank0_pid}" >/dev/null 2>&1; then
    kill "${rank0_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${rank1_pid}" ]] && kill -0 "${rank1_pid}" >/dev/null 2>&1; then
    kill "${rank1_pid}" >/dev/null 2>&1 || true
  fi
  wait || true
}

trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --figure) figure="$2"; shift 2 ;;
    --requests) requests="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --model-root) model_root="$2"; shift 2 ;;
    --rank0-gpu) rank0_gpu="$2"; shift 2 ;;
    --rank1-gpu) rank1_gpu="$2"; shift 2 ;;
    --master-addr) master_addr="$2"; shift 2 ;;
    --master-port) master_port="$2"; shift 2 ;;
    --comm-transport) comm_transport="$2"; shift 2 ;;
    --nccl-socket-ifname) nccl_socket_ifname="$2"; shift 2 ;;
    --nccl-ib-hca) nccl_ib_hca="$2"; shift 2 ;;
    --nccl-ib-gid-index) nccl_ib_gid_index="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --max-seq-len) max_seq_len="$2"; shift 2 ;;
    --request-rate) request_rate="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --profile) profile_override="$2"; shift 2 ;;
    --dataset-jsonl) dataset_jsonl="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --allow-transformers-mismatch) allow_transformers_mismatch=1; shift ;;
    --strict-transformers-version) allow_transformers_mismatch=0; shift ;;
    *) fail "unknown argument $1" ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

check_env() {
  command -v "${python_bin}" >/dev/null 2>&1 || fail "python is not available: ${python_bin}"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available; NVIDIA driver/GPU is required"
  nvidia-smi -i "${rank0_gpu}" >/dev/null 2>&1 || fail "rank0 GPU is not visible: ${rank0_gpu}"
  nvidia-smi -i "${rank1_gpu}" >/dev/null 2>&1 || fail "rank1 GPU is not visible: ${rank1_gpu}"
  "${python_bin}" - <<'PY' || fail "python env must provide torch with CUDA support"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() >= 1, "torch sees no CUDA devices"
PY
  if [[ "${comm_transport}" == "rdma" && -z "${nccl_ib_hca}" \
        && ( -z "${RANK0_NCCL_IB_HCA:-}" || -z "${RANK1_NCCL_IB_HCA:-}" ) ]]; then
    fail "COMM_TRANSPORT=rdma requires NCCL_IB_HCA (or RANK0_NCCL_IB_HCA + RANK1_NCCL_IB_HCA, or --nccl-ib-hca) for this machine"
  fi
}

profile_for_model() {
  if [[ -n "${profile_override}" ]]; then
    [[ -f "${profile_override}" ]] || fail "profile file not found: ${profile_override}"
    echo "${profile_override}"
    return
  fi
  case "${model}" in
    Qwen/Qwen2.5-1.5B)
      echo "fluidgpu_runtime/profiles/qwen25_1p5b_kernel_group_split.json"
      ;;
    Qwen/Qwen2.5-VL-7B-Instruct)
      echo "fluidgpu_runtime/profiles/qwen25_vl_7b_kernel_group_split.json"
      ;;
    Qwen/Qwen3-235B-A22B)
      echo "prebuilt:qwen3_235b_a22b_moe_fine_split"
      ;;
    mistralai/Mamba-Codestral-7B-v0.1)
      echo "fluidgpu_runtime/profiles/mamba_codestral_7b_layers.json"
      ;;
    meta-llama/Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct)
      # MILP plan from on-machine A100/L40S measurements (profiles/llama31_*_tasks.csv);
      # regenerate via profile_llm_tasks.py + schedule_llm_layers.py --solver milp.
      echo "fluidgpu_runtime/profiles/llama31_8b_milp_a100_l40s.json"
      ;;
    openai/gpt-oss-20b)
      # MILP plan from on-machine A100/L40S measurements (profiles/gptoss_*_tasks.csv).
      echo "fluidgpu_runtime/profiles/gpt_oss_20b_milp_a100_l40s.json"
      ;;
    stabilityai/stable-diffusion-3.5-medium)
      echo "fluidgpu_runtime/profiles/sd35_medium_layers.json"
      ;;
    *)
      fail "no fluidgpu_torch end-to-end profile for model '${model}'; provide PROFILE=/path/to/profile.json"
      ;;
  esac
}

resolve_model_arg() {
  local model_path="${model_root}/${model//\//--}"
  if [[ -d "${model_path}" ]]; then
    echo "${model_path}"
    return
  fi
  case "${model}" in
    meta-llama/*)
      [[ -n "${HF_TOKEN:-}" ]] || fail "model weights not found at ${model_path} and HF_TOKEN is not set for gated model ${model}"
      ;;
  esac
  echo "${model}"
}

wait_for_ranks() {
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    local rank0_running=0
    local rank1_running=0
    kill -0 "${rank0_pid}" >/dev/null 2>&1 && rank0_running=1 || true
    kill -0 "${rank1_pid}" >/dev/null 2>&1 && rank1_running=1 || true
    if [[ "${rank0_running}" -eq 0 && "${rank1_running}" -eq 0 ]]; then
      break
    fi
    if (( "$(date +%s)" - start_ts >= timeout_seconds )); then
      fail "timeout waiting for two-rank run; see ${out_dir}/rank0.log and ${out_dir}/rank1.log"
    fi
    sleep 2
  done

  set +e
  wait "${rank0_pid}"
  local rank0_status=$?
  wait "${rank1_pid}"
  local rank1_status=$?
  set -e
  if [[ "${rank0_status}" -ne 0 || "${rank1_status}" -ne 0 ]]; then
    fail "two-rank run failed: rank0=${rank0_status}, rank1=${rank1_status}; see ${out_dir}"
  fi
}

main() {
  check_env
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -z "${out_dir}" ]]; then
    out_dir="${FLUIDGPU_RUN_OUTPUT_DIR:-artifacts/logs/${figure}/${ts}}"
  fi
  mkdir -p "${out_dir}"

  local base_profile
  local effective_profile
  local model_arg
  base_profile="$(profile_for_model)"
  model_arg="$(resolve_model_arg)"
  if [[ "${base_profile}" == prebuilt:* ]]; then
    local prebuilt_name="${base_profile#prebuilt:}"
    local generated_profile="${out_dir}/base_profile.json"
    PYTHONPATH="${repo_root}/fluidgpu_runtime:${PYTHONPATH:-}" "${python_bin}" \
      fluidgpu_runtime/examples/fluidgpu_torch/prebuilt_profile.py \
      --name "${prebuilt_name}" \
      --output-profile "${generated_profile}" \
      --model "${model_arg}"
    base_profile="${generated_profile}"
  fi
  [[ -f "${base_profile}" ]] || fail "profile file not found: ${base_profile}"
  effective_profile="${out_dir}/effective_profile.json"
  PYTHONPATH="${repo_root}/fluidgpu_runtime:${PYTHONPATH:-}" "${python_bin}" \
    fluidgpu_runtime/examples/fluidgpu_torch/profile_variant.py \
    --input-profile "${base_profile}" \
    --output-profile "${effective_profile}" \
    --policy "${policy}" \
    --model "${model_arg}"

  # Pipelined request processing (paper §III-C): FLUIDGPU_PIPELINE_MODE
  #   off      -> synchronous single-worker llm_generate (default)
  #   naive    -> AsyncLLMEngine, FLUIDGPU_PIPELINE_WORKERS concurrent streams
  #   priority -> naive + priority-aware stream scheduling
  local pipeline_mode="${FLUIDGPU_PIPELINE_MODE:-off}"
  local pipeline_workers="${FLUIDGPU_PIPELINE_WORKERS:-2}"
  printf '%s\n' "${pipeline_mode}" > "${out_dir}/pipeline_mode.txt"
  local base_cmd
  if [[ "${pipeline_mode}" == "off" ]]; then
    base_cmd=(
      "${python_bin}" fluidgpu_runtime/examples/fluidgpu_torch/llm_generate.py
      --model "${model_arg}"
      --profile "${effective_profile}"
      --prompt "${prompt}"
      --requests "${requests}"
      --max-new-tokens "${max_new_tokens}"
      --max-seq-len "${max_seq_len}"
      --master-addr "${master_addr}"
      --master-port "${master_port}"
      --comm-transport "${comm_transport}"
      --log-jsonl "${out_dir}/log.jsonl"
      --summary-json "${out_dir}/summary.json"
    )
    if [[ -n "${dataset_jsonl}" ]]; then
      base_cmd+=(--dataset-jsonl "${dataset_jsonl}")
    fi
    if [[ -n "${request_rate}" ]]; then
      base_cmd+=(--request-rate "${request_rate}")
    fi
  else
    [[ "${pipeline_mode}" == "naive" || "${pipeline_mode}" == "priority" ]] \
      || fail "invalid FLUIDGPU_PIPELINE_MODE '${pipeline_mode}' (off|naive|priority)"
    [[ -z "${request_rate}" ]] \
      || fail "FLUIDGPU_PIPELINE_MODE=${pipeline_mode} does not support --request-rate"
    base_cmd=(
      "${python_bin}" fluidgpu_runtime/examples/fluidgpu_torch/async_llm_generate.py
      --model "${model_arg}"
      --profile "${effective_profile}"
      --prompt "${prompt}"
      --requests "${requests}"
      --max-new-tokens "${max_new_tokens}"
      --max-seq-len "${max_seq_len}"
      --master-addr "${master_addr}"
      --master-port "${master_port}"
      --comm-transport "${comm_transport}"
      --num-workers "${pipeline_workers}"
      --quiet-requests
      --summary-json "${out_dir}/summary.json"
    )
    if [[ -n "${dataset_jsonl}" ]]; then
      base_cmd+=(--dataset-jsonl "${dataset_jsonl}")
    fi
    if [[ "${pipeline_mode}" == "priority" ]]; then
      base_cmd+=(--priority-scheduling)
    fi
    # FLUIDGPU_DUAL_PLAN=1 (fluidgpu policy only): concurrent workers get
    # complementary placements — worker0 the planned profile, worker1 its
    # homo-left variant — so requests occupy different GPUs at the same
    # pipeline stage (paper zig-zag concurrency) instead of phase-aligning
    # on one GPU while the other idles.
    if [[ "${FLUIDGPU_DUAL_PLAN:-0}" == "1" && "${policy}" == "fluidgpu" ]]; then
      dual_profile="${out_dir}/effective_profile_dual.json"
      PYTHONPATH="${repo_root}/fluidgpu_runtime:${PYTHONPATH:-}" "${python_bin}" \
        fluidgpu_runtime/examples/fluidgpu_torch/profile_variant.py \
        --input-profile "${effective_profile}" \
        --output-profile "${dual_profile}" \
        --policy homo-left \
        --model "${model_arg}"
      base_cmd+=(--worker-profiles "${effective_profile},${dual_profile}")
    fi
  fi
  if [[ -n "${nccl_socket_ifname}" ]]; then
    base_cmd+=(--nccl-socket-ifname "${nccl_socket_ifname}")
  fi
  if [[ -n "${nccl_ib_gid_index}" ]]; then
    base_cmd+=(--nccl-ib-gid-index "${nccl_ib_gid_index}")
  fi
  if [[ "${allow_transformers_mismatch}" == "1" ]]; then
    base_cmd+=(--allow-transformers-mismatch)
  fi

  # FLUIDGPU_CUDA_GRAPHS=1 captures decode segments as CUDA graphs (dense
  # attn/mlp profiles only) — the launch-overhead fix that makes pipelining pay.
  if [[ "${FLUIDGPU_CUDA_GRAPHS:-0}" == "1" ]]; then
    base_cmd+=(--cuda-graph-decode)
  fi

  # Per-rank HCA pinning for same-host cross-machine simulation (paper
  # topology: each rank owns its own NIC). RANK{0,1}_NCCL_IB_HCA override the
  # shared NCCL_IB_HCA; with neither set, behavior is unchanged.
  local rank0_hca="${RANK0_NCCL_IB_HCA:-${nccl_ib_hca}}"
  local rank1_hca="${RANK1_NCCL_IB_HCA:-${nccl_ib_hca}}"

  local rank0_cmd=("${base_cmd[@]}")
  if [[ "${pipeline_mode}" == "off" ]]; then
    rank0_cmd+=(--diagnostics-output "${out_dir}/rank0_diagnostics.pt")
  fi
  local rank1_cmd=("${base_cmd[@]}")
  if [[ -n "${rank0_hca}" ]]; then
    rank0_cmd+=(--nccl-ib-hca "${rank0_hca}")
  fi
  if [[ -n "${rank1_hca}" ]]; then
    rank1_cmd+=(--nccl-ib-hca "${rank1_hca}")
  fi
  printf '%q ' "${rank0_cmd[@]}" > "${out_dir}/rank0.command.txt"
  printf '\n' >> "${out_dir}/rank0.command.txt"
  printf '%q ' "${rank1_cmd[@]}" > "${out_dir}/rank1.command.txt"
  printf '\n' >> "${out_dir}/rank1.command.txt"
  printf '%s\n' "${policy}" > "${out_dir}/policy.txt"
  printf '%s\n' "${model}" > "${out_dir}/model.txt"

  echo "run_fluidgpu_e2e_two_rank.sh: starting rank1 on GPU ${rank1_gpu}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${rank1_gpu}" \
  FLUIDGPU_RANK=1 \
  MASTER_ADDR="${master_addr}" \
  MASTER_PORT="${master_port}" \
  PYTHONPATH="${repo_root}/fluidgpu_runtime:${PYTHONPATH:-}" \
  "${rank1_cmd[@]}" > "${out_dir}/rank1.log" 2>&1 &
  rank1_pid=$!

  sleep 2

  echo "run_fluidgpu_e2e_two_rank.sh: starting rank0 on GPU ${rank0_gpu}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${rank0_gpu}" \
  FLUIDGPU_RANK=0 \
  MASTER_ADDR="${master_addr}" \
  MASTER_PORT="${master_port}" \
  PYTHONPATH="${repo_root}/fluidgpu_runtime:${PYTHONPATH:-}" \
  "${rank0_cmd[@]}" > "${out_dir}/rank0.log" 2>&1 &
  rank0_pid=$!

  wait_for_ranks
  echo "run_fluidgpu_e2e_two_rank.sh: wrote ${out_dir}"
}

main "$@"
