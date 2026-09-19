#!/usr/bin/env bash
set -euo pipefail

fig=""
all=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fig) fig="$2"; shift 2 ;;
    --all) all=1; shift ;;
    *) echo "repro.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

spec_for_fig() {
  case "$1" in
    2) echo "experiments/fig2_kernel_heterogeneity.yaml" ;;
    3) echo "experiments/fig3_coarse_granularity.yaml" ;;
    10) echo "experiments/fig10_pipeline.yaml" ;;
    11) echo "experiments/fig11_monitor_sensitivity.yaml" ;;
    12a) echo "experiments/fig12a_slow_network.yaml" ;;
    12b) echo "experiments/fig12b_milp_scalability.yaml" ;;
    # `exit` here would only leave the command substitution, so signal with a
    # return code and let run_fig report it.
    *) return 1 ;;
  esac
}

run_spec() {
  local spec="$1"
  local args=(python3 -m fluidgpu_torch.orchestrator run "$spec")
  PYTHONPATH="${PWD}/fluidgpu_runtime:${PYTHONPATH:-}" "${args[@]}"
}

run_fig() {
  local spec
  if ! spec="$(spec_for_fig "$1")"; then
    echo "repro.sh: unknown figure $1 (try 2 3 10 11 12a 12b)" >&2
    exit 2
  fi
  run_spec "$spec"
}

if [[ "$all" -eq 1 ]]; then
  for item in 2 3 10 11 12a 12b; do
    run_fig "$item"
  done
elif [[ -n "$fig" ]]; then
  run_fig "$fig"
else
  echo "repro.sh: provide --fig <n> or --all" >&2
  exit 2
fi
