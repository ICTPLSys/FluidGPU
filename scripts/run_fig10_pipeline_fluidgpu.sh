#!/usr/bin/env bash
set -euo pipefail

# CUDA-graph decode is the validated fast path on this artifact; all
# policies share it. Override with FLUIDGPU_CUDA_GRAPHS=0 to compare eager.
export FLUIDGPU_CUDA_GRAPHS="${FLUIDGPU_CUDA_GRAPHS:-1}"

requests=100
pipeline="off"
policy="fluidgpu"
model="${MODEL:-openai/gpt-oss-20b}"
max_new_tokens=16
prompt="${PROMPT:-Summarize pipeline scheduling priorities for GPT-oss serving.}"
out_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requests) requests="$2"; shift 2 ;;
    --pipeline) pipeline="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --rank0-gpu|--rank1-gpu|--master-addr|--master-port|--comm-transport|--nccl-socket-ifname|--nccl-ib-hca|--nccl-ib-gid-index|--profile|--model-root)
      extra_args+=("$1" "$2"); shift 2 ;;
    --allow-transformers-mismatch|--strict-transformers-version)
      extra_args+=("$1"); shift ;;
    *) echo "run_fig10_pipeline_fluidgpu.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  out_dir="${FLUIDGPU_RUN_OUTPUT_DIR:-artifacts/logs/fig10/${ts}}"
fi
mkdir -p "${out_dir}"

printf '%s\n' "${pipeline}" > "${out_dir}/pipeline.txt"
printf '%s\n' "${model}" > "${out_dir}/model.txt"

cmd=(
  bash scripts/run_fluidgpu_e2e_two_rank.sh
  --figure fig10
  --model "${model}"
  --policy "${policy}"
  --requests "${requests}"
  --max-new-tokens "${max_new_tokens}"
  --prompt "${prompt}"
  --output-dir "${out_dir}"
)
cmd+=("${extra_args[@]}")

printf '%q ' "${cmd[@]}" > "${out_dir}/wrapper.command.txt"
printf '\n' >> "${out_dir}/wrapper.command.txt"

export FLUIDGPU_PIPELINE_MODE="${pipeline}"
exec "${cmd[@]}"
