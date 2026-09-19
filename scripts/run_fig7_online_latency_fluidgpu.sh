#!/usr/bin/env bash
set -euo pipefail

# CUDA-graph decode is the validated fast path on this artifact; all
# policies share it. Override with FLUIDGPU_CUDA_GRAPHS=0 to compare eager.
export FLUIDGPU_CUDA_GRAPHS="${FLUIDGPU_CUDA_GRAPHS:-1}"

requests=100
policy="fluidgpu"
rps=""
model="${MODEL:-openai/gpt-oss-20b}"
max_new_tokens=16
prompt="${PROMPT:-Explain GPU disaggregation in three sentences.}"
out_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requests) requests="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --rps) rps="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --rank0-gpu|--rank1-gpu|--master-addr|--master-port|--comm-transport|--nccl-socket-ifname|--nccl-ib-hca|--nccl-ib-gid-index|--profile)
      extra_args+=("$1" "$2"); shift 2 ;;
    --allow-transformers-mismatch|--strict-transformers-version)
      extra_args+=("$1"); shift ;;
    *) echo "run_fig7_online_latency_fluidgpu.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

cmd=(
  bash scripts/run_fluidgpu_e2e_two_rank.sh
  --figure fig7
  --model "${model}"
  --policy "${policy}"
  --requests "${requests}"
  --max-new-tokens "${max_new_tokens}"
  --prompt "${prompt}"
)
if [[ -n "${rps}" ]]; then
  cmd+=(--request-rate "${rps}")
fi
if [[ -n "${out_dir}" ]]; then
  cmd+=(--output-dir "${out_dir}")
fi
cmd+=("${extra_args[@]}")

exec "${cmd[@]}"
