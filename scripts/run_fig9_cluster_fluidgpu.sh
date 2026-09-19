#!/usr/bin/env bash
set -euo pipefail

requests=100
policy="fluidgpu"
model="${MODEL:-openai/gpt-oss-20b}"
cluster_label=""
max_new_tokens=16
prompt="${PROMPT:-Explain heterogeneous cluster scheduling trade-offs.}"
out_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster-label) cluster_label="$2"; shift 2 ;;
    --requests) requests="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --rank0-gpu|--rank1-gpu|--master-addr|--master-port|--comm-transport|--nccl-socket-ifname|--nccl-ib-hca|--nccl-ib-gid-index|--profile)
      extra_args+=("$1" "$2"); shift 2 ;;
    --allow-transformers-mismatch|--strict-transformers-version)
      extra_args+=("$1"); shift ;;
    *) echo "run_fig9_cluster_fluidgpu.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${cluster_label}" ]]; then
  prompt="${prompt} Cluster label: ${cluster_label}."
fi

cmd=(
  bash scripts/run_fluidgpu_e2e_two_rank.sh
  --figure fig9
  --model "${model}"
  --policy "${policy}"
  --requests "${requests}"
  --max-new-tokens "${max_new_tokens}"
  --prompt "${prompt}"
)
if [[ -n "${out_dir}" ]]; then
  cmd+=(--output-dir "${out_dir}")
fi
cmd+=("${extra_args[@]}")

exec "${cmd[@]}"
