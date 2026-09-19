#!/usr/bin/env bash
set -euo pipefail

# Automated numerical-parity gate for the disaggregated + CUDA-graph decode path.
#
# The paper's load-bearing correctness claim is that splitting a transformer
# layer into cross-rank kernel groups and replaying decode via CUDA graphs still
# produces the SAME tokens as single-GPU execution. This script is the gate that
# proves it: it produces BOTH artifacts the comparator needs in one invocation --
#   1. a single-GPU baseline (HF forward on one GPU), and
#   2. the two-rank disaggregated candidate WITH --cuda-graph-decode,
# then runs parity_compare.py, which hard-asserts
#   token_match_ratio >= MIN_TOKEN_MATCH (default 1.0) and cosine >= MIN_COSINE
#   (default 0.999), exiting non-zero on any divergence.
#
# Wire it into CI / repro.sh so a regression in the split/mask/KV/graph path is
# caught automatically instead of silently passing the throughput chain.
#
# Env knobs (all optional):
#   MODEL, PROFILE, TOKENS, MIN_COSINE, MIN_TOKEN_MATCH,
#   RANK0_GPU, RANK1_GPU, RANK0_NCCL_IB_HCA, RANK1_NCCL_IB_HCA, COMM_TRANSPORT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PROFILE="${PROFILE:-fluidgpu_runtime/profiles/qwen25_1p5b_kernel_group_split.json}"
TOKENS="${TOKENS:-16}"
MIN_COSINE="${MIN_COSINE:-0.999}"
MIN_TOKEN_MATCH="${MIN_TOKEN_MATCH:-1.0}"
PROMPT="${PROMPT:-Explain GPU disaggregation in three sentences.}"
RANK0_GPU="${RANK0_GPU:-0}"
RANK1_GPU="${RANK1_GPU:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29570}"
COMM_TRANSPORT="${COMM_TRANSPORT:-auto}"
MODEL_ROOT="${FLUIDGPU_MODEL_DIR:-$HOME/.cache/fluidgpu/models}"
OUT_DIR="${OUT_DIR:-artifacts/validation/parity/$(basename "${MODEL}")}"
ALLOW_MISMATCH="${ALLOW_TRANSFORMERS_MISMATCH:-1}"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/fluidgpu_runtime:${PYTHONPATH:-}"
export FLUIDGPU_NCCL_TIMEOUT_S="${FLUIDGPU_NCCL_TIMEOUT_S:-300}"

# Resolve a local model directory if one is cached (offline AE machines).
# The profile's model field must match the --model argument (validate_profile
# asserts equality), so rewrite it alongside the resolution.
model_arg="${MODEL}"
cached="${MODEL_ROOT}/${MODEL//\//--}"
if [[ -d "${cached}" ]]; then
  model_arg="${cached}"
  resolved_profile="${OUT_DIR}/profile.json"
  MODEL_ARG="${model_arg}" PROFILE_IN="${PROFILE}" PROFILE_OUT="${resolved_profile}" \
  "${PYTHON_BIN}" - <<'PY'
import json, os
p = json.load(open(os.environ["PROFILE_IN"]))
p["model"] = os.environ["MODEL_ARG"]
json.dump(p, open(os.environ["PROFILE_OUT"], "w"), indent=1)
PY
  PROFILE="${resolved_profile}"
fi

gen=(fluidgpu_runtime/examples/fluidgpu_torch/llm_generate.py
  --model "${model_arg}" --prompt "${PROMPT}" --max-new-tokens "${TOKENS}"
  --requests 1 --seed 0)
[[ "${ALLOW_MISMATCH}" == "1" ]] && gen+=(--allow-transformers-mismatch)

baseline="${OUT_DIR}/baseline_single_gpu.pt"
candidate="${OUT_DIR}/candidate_two_rank_graph.pt"

echo "verify_parity: [1/3] single-GPU baseline on GPU ${RANK0_GPU} -> ${baseline}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${RANK0_GPU}" \
  "${PYTHON_BIN}" "${gen[@]}" --single-gpu --world-size 2 \
  --diagnostics-output "${baseline}" > "${OUT_DIR}/baseline.log" 2>&1

echo "verify_parity: [2/3] two-rank disaggregated + CUDA-graph candidate -> ${candidate}"
cand=("${gen[@]}" --profile "${PROFILE}" --cuda-graph-decode
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}"
  --comm-transport "${COMM_TRANSPORT}")
[[ -n "${RANK1_NCCL_IB_HCA:-}" ]] && rank1_hca=(--nccl-ib-hca "${RANK1_NCCL_IB_HCA}") || rank1_hca=()
[[ -n "${RANK0_NCCL_IB_HCA:-}" ]] && rank0_hca=(--nccl-ib-hca "${RANK0_NCCL_IB_HCA}") || rank0_hca=()

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${RANK1_GPU}" FLUIDGPU_RANK=1 \
  MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
  "${PYTHON_BIN}" "${cand[@]}" "${rank1_hca[@]}" > "${OUT_DIR}/rank1.log" 2>&1 &
rank1_pid=$!
sleep 2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${RANK0_GPU}" FLUIDGPU_RANK=0 \
  MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
  "${PYTHON_BIN}" "${cand[@]}" "${rank0_hca[@]}" \
  --diagnostics-output "${candidate}" > "${OUT_DIR}/rank0.log" 2>&1
wait "${rank1_pid}"

echo "verify_parity: [3/3] comparing (min-cosine=${MIN_COSINE} min-token-match=${MIN_TOKEN_MATCH})"
"${PYTHON_BIN}" fluidgpu_runtime/examples/fluidgpu_torch/parity_compare.py \
  --baseline "${baseline}" --candidate "${candidate}" \
  --token-count "${TOKENS}" --min-cosine "${MIN_COSINE}" \
  --min-token-match-ratio "${MIN_TOKEN_MATCH}" | tee "${OUT_DIR}/parity.txt"

echo "verify_parity: PASS (${MODEL}) -> ${OUT_DIR}/parity.txt"
