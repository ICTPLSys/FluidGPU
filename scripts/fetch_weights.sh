#!/usr/bin/env bash
set -euo pipefail

target="${FLUIDGPU_MODEL_DIR:-$HOME/.cache/fluidgpu/models}"
models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "openai/gpt-oss-20b"
  "Qwen/Qwen3-235B-A22B"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "mistralai/Mamba-Codestral-7B-v0.1"
  "stabilityai/stable-diffusion-3.5-medium"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    *) echo "fetch_weights.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

fail() {
  echo "fetch_weights.sh: $*" >&2
  exit 1
}

command -v huggingface-cli >/dev/null 2>&1 || fail "huggingface-cli is not available; install huggingface_hub in the active environment"
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  fail "Hugging Face authentication is required for gated model weights; run huggingface-cli login or export HF_TOKEN"
fi

mkdir -p "$target"
for model in "${models[@]}"; do
  dest="${target}/${model//\//--}"
  if [[ -d "$dest" ]]; then
    echo "exists: $model -> $dest"
    continue
  fi
  echo "download: $model -> $dest"
  huggingface-cli download "$model" --local-dir "$dest"
done
