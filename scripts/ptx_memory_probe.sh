#!/usr/bin/env bash
set -euo pipefail

execute="required"
arch="sm_80"
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute="${2:?missing value for --execute}"
      shift 2
      ;;
    --arch)
      arch="${2:?missing value for --arch}"
      shift 2
      ;;
    --output-dir)
      out_dir="${2:?missing value for --output-dir}"
      shift 2
      ;;
    *)
      echo "ptx_memory_probe.sh: unknown argument $1" >&2
      exit 2
      ;;
  esac
done

case "$execute" in
  required) ;;
  *)
    echo "ptx_memory_probe.sh: --execute must be required" >&2
    exit 2
    ;;
esac

if [[ -z "$out_dir" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  out_dir="artifacts/validation/ptx_memory_probe/${ts}"
fi

command -v python3 >/dev/null 2>&1 || { echo "ptx_memory_probe.sh: python3 is not available" >&2; exit 1; }
command -v ptxas >/dev/null 2>&1 || { echo "ptx_memory_probe.sh: ptxas is not available; install the CUDA toolkit" >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "ptx_memory_probe.sh: nvcc is not available; install the CUDA toolkit" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ptx_memory_probe.sh: nvidia-smi is not available; NVIDIA driver/GPU is required" >&2; exit 1; }

mkdir -p "$out_dir"
PYTHONPATH="${PWD}/fluidgpu_runtime:${PYTHONPATH:-}" python3 \
  fluidgpu_runtime/examples/fluidgpu_torch/ptx_memory_probe.py \
  --output-dir "$out_dir" \
  --execute "$execute" \
  --arch "$arch"
echo "ptx_memory_probe.sh: wrote ${out_dir}/report.json"
