#!/usr/bin/env bash
set -euo pipefail

# CUDA-graph decode is the validated fast path on this artifact; all
# policies share it. Override with FLUIDGPU_CUDA_GRAPHS=0 to compare eager.
export FLUIDGPU_CUDA_GRAPHS="${FLUIDGPU_CUDA_GRAPHS:-1}"

requests=100
policy="fluidgpu"
model="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
max_new_tokens=16
prompt="${PROMPT:-Explain GPU disaggregation in three sentences.}"
out_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requests) requests="$2"; shift 2 ;;
    --policy) policy="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --output-dir) out_dir="$2"; shift 2 ;;
    --rank0-gpu|--rank1-gpu|--master-addr|--master-port|--comm-transport|--nccl-socket-ifname|--nccl-ib-hca|--nccl-ib-gid-index|--profile|--dataset-jsonl)
      extra_args+=("$1" "$2"); shift 2 ;;
    --allow-transformers-mismatch|--strict-transformers-version)
      extra_args+=("$1"); shift ;;
    *) echo "run_fig6_offline_throughput_fluidgpu.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

# Reproduce the documented Fig.6 configuration by default (EXPERIMENTS.md):
# the FluidGPU policy drives the pipelined async engine with 2 workers and the
# per-step host-synchronized decode loop (FLUIDGPU_SYNC_DECODE=1) — the
# validated-stable pipeline configuration; more workers or the sync-free loop
# intermittently deadlock NCCL across the per-worker process groups (runbook
# §8, multi-communicator ordering hazard). Homogeneous baselines run
# single-stream. An explicit env override wins.
if [[ "${policy}" == "fluidgpu" ]]; then
  export FLUIDGPU_PIPELINE_MODE="${FLUIDGPU_PIPELINE_MODE:-naive}"
  export FLUIDGPU_PIPELINE_WORKERS="${FLUIDGPU_PIPELINE_WORKERS:-2}"
  export FLUIDGPU_SYNC_DECODE="${FLUIDGPU_SYNC_DECODE:-1}"
  case "${model}" in
    # gpt-oss's per-phase plan leaves the A100 idle through the decode phase
    # (decode lives on the L40S); complementary per-worker placement keeps
    # both GPUs busy and lifts fig6 from +21% to +71% over the best homo
    # baseline (runbook §11). Llama's zig-zag already uses both GPUs every
    # layer, so dual-plan does not help there.
    *gpt-oss*) export FLUIDGPU_DUAL_PLAN="${FLUIDGPU_DUAL_PLAN:-1}" ;;
  esac
else
  export FLUIDGPU_PIPELINE_MODE="${FLUIDGPU_PIPELINE_MODE:-off}"
fi

cmd=(
  bash scripts/run_fluidgpu_e2e_two_rank.sh
  --figure fig6
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
