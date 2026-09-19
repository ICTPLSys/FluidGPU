#!/usr/bin/env bash
set -euo pipefail

# Exact-solve demo of the MILP policy planner. The default size (300 kernels,
# 2 GPUs ~= 1804 binaries) fits the size-limited license that ships inside
# `pip install gurobipy` — no license file is required. Licensed users can
# scale it up (e.g. DEMO_KERNELS=1500). The DP solver remains the artifact's
# default planner (license-free, cross-checked to the same optimum).
DEMO_KERNELS="${DEMO_KERNELS:-300}"
DEMO_GPUS="${DEMO_GPUS:-2}"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="artifacts/validation/planner/${ts}"
mkdir -p "$out_dir"
PYTHONPATH="${PWD}/fluidgpu_runtime:${PYTHONPATH:-}" python3 \
  fluidgpu_runtime/examples/fluidgpu_torch/milp_planner_probe.py \
  --kernels "${DEMO_KERNELS}" \
  --gpus "${DEMO_GPUS}" \
  --mode throughput \
  --seed 0 \
  --summary-json "${out_dir}/summary.json" \
  --solve-exact
echo "demo_planner.sh: wrote ${out_dir}/summary.json"
