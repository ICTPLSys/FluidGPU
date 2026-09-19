#!/usr/bin/env bash
set -euo pipefail

target="${FLUIDGPU_DATASET_DIR:-$PWD/datasets}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    *) echo "fetch_datasets.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

fail() {
  echo "fetch_datasets.sh: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is not available"
command -v git >/dev/null 2>&1 || fail "git is not available"
command -v unzip >/dev/null 2>&1 || fail "unzip is not available"
command -v huggingface-cli >/dev/null 2>&1 || fail "huggingface-cli is not available; install huggingface_hub in the active environment"

mkdir -p "${target}/coco" "${target}/parti-prompts" "${target}/splitwise"

download_file() {
  local label="$1"
  local url="$2"
  local output="$3"
  if [[ -s "$output" ]]; then
    echo "exists: $label -> $output"
    return
  fi
  echo "download: $label -> $output"
  curl --fail --location --retry 3 --output "$output" "$url"
}

download_file "COCO val2017 images" \
  "http://images.cocodataset.org/zips/val2017.zip" \
  "${target}/coco/val2017.zip"
download_file "COCO annotations" \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  "${target}/coco/annotations_trainval2017.zip"

if [[ ! -d "${target}/coco/val2017" ]]; then
  echo "extract: COCO val2017 images"
  unzip -q "${target}/coco/val2017.zip" -d "${target}/coco"
fi
if [[ ! -f "${target}/coco/annotations/captions_val2017.json" ]]; then
  echo "extract: COCO annotations"
  unzip -q "${target}/coco/annotations_trainval2017.zip" -d "${target}/coco"
fi

if [[ -e "${target}/parti-prompts/.cache" || -n "$(find "${target}/parti-prompts" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "exists: PartiPrompts -> ${target}/parti-prompts"
else
  echo "download: PartiPrompts -> ${target}/parti-prompts"
  huggingface-cli download nateraw/parti-prompts --repo-type dataset --local-dir "${target}/parti-prompts"
fi

if [[ -d "${target}/azure/.git" ]]; then
  echo "update: Azure conversation trace -> ${target}/azure"
  git -C "${target}/azure" pull --ff-only
elif [[ -e "${target}/azure" ]]; then
  fail "${target}/azure exists but is not a git checkout"
else
  echo "clone: Azure conversation trace -> ${target}/azure"
  git clone --depth 1 https://github.com/Azure/AzurePublicDataset "${target}/azure"
fi

download_file "Splitwise coding trace (Azure LLM Inference 2023)" \
  "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv" \
  "${target}/splitwise/AzureLLMInferenceTrace_code.csv"
download_file "Splitwise conversation trace (Azure LLM Inference 2023)" \
  "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv" \
  "${target}/splitwise/AzureLLMInferenceTrace_conv.csv"
