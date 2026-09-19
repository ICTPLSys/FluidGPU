#!/bin/bash
# Shared environment resolution for the Fig.6/7/8 rerun entry points.
#
# Every machine-specific value is an override with the authors' evaluation
# host as the default, so the experiments run unmodified on the AE machine and
# are portable by exporting a handful of variables. Source this file, do not
# execute it.
#
#   FLUIDGPU_PYTHON     python interpreter (default: `python` on PATH)
#   FLUIDGPU_VLLM       vllm CLI            (default: `vllm` on PATH)
#   FLUIDGPU_MODEL_GT   gpt-oss-20b weights (HF id or local dir)
#   FLUIDGPU_MODEL_LM   Llama-3.1-8B-Instruct weights
#   FLUIDGPU_GPU_P      CUDA index of the prefill / weaker GPU   (L40S here)
#   FLUIDGPU_GPU_D      CUDA index of the decode / stronger GPU  (A100 here)
#   FLUIDGPU_GPU_X      CUDA index of the third GPU, Fig.9 only  (A100 #0)
#   FLUIDGPU_HCA_P      IB HCA on the prefill side
#   FLUIDGPU_HCA_D      IB HCA on the decode side
#   FLUIDGPU_RERUN_OUT  output root for reruns
#
# Indices are interpreted under CUDA_DEVICE_ORDER=PCI_BUS_ID, which is
# exported below; check yours with `nvidia-smi --query-gpu=index,name`.

# Repo root: this file lives at <repo>/scripts/vllm_exp/.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUIDGPU_REPO="$(cd "${EXP_DIR}/../.." && pwd)"
cd "$FLUIDGPU_REPO" || { echo "exp_env: cannot cd to $FLUIDGPU_REPO" >&2; exit 1; }

PY="${FLUIDGPU_PYTHON:-$(command -v python || command -v python3)}"
VB="${FLUIDGPU_VLLM:-$(command -v vllm)}"
[ -x "$PY" ] || { echo "exp_env: no python found; set FLUIDGPU_PYTHON" >&2; exit 1; }
[ -x "$VB" ] || { echo "exp_env: no vllm CLI found; set FLUIDGPU_VLLM" >&2; exit 1; }
# Aliases retained for the experiment scripts.
V="$PY"
VLLM_BIN="$VB"

MODEL_GT="${FLUIDGPU_MODEL_GT:-$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b}"
MODEL_LM="${FLUIDGPU_MODEL_LM:-$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct}"

GPU_P="${FLUIDGPU_GPU_P:-2}"
GPU_D="${FLUIDGPU_GPU_D:-1}"
GPU_X="${FLUIDGPU_GPU_X:-0}"
GPU_PAIR="${GPU_D},${GPU_P}"
HCA_P="${FLUIDGPU_HCA_P:-mlx5_5}"
HCA_D="${FLUIDGPU_HCA_D:-mlx5_2}"

S="${FLUIDGPU_RERUN_OUT:-${FLUIDGPU_REPO}/artifacts/validation/vllm_graphs/reruns}"
mkdir -p "$S"

# A local Mooncake/vLLM mesh must never be routed through an HTTP proxy.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

exp_preflight() {
  local missing=0
  for m in "$@"; do
    case "$m" in
      /*) [ -e "$m" ] || { echo "exp_env: model path not found: $m" >&2; missing=1; } ;;
    esac
  done
  command -v nvidia-smi >/dev/null || { echo "exp_env: nvidia-smi not found" >&2; missing=1; }
  [ "$missing" -eq 0 ] || {
    echo "exp_env: set FLUIDGPU_MODEL_GT / FLUIDGPU_MODEL_LM to your weight dirs" >&2
    exit 1
  }
}
