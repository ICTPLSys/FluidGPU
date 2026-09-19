#!/usr/bin/env bash
set -euo pipefail

target="${FLUIDGPU_DATASET_DIR:-$PWD/datasets}"
task="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    *) echo "preprocess_datasets.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

fail() {
  echo "preprocess_datasets.sh: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || fail "missing ${label}: ${path}; run scripts/fetch_datasets.sh first"
}

preprocess_coco() {
  require_path "${target}/coco/val2017" "COCO val2017 image directory"
  require_path "${target}/coco/annotations/captions_val2017.json" "COCO captions annotation file"
  mkdir -p "${target}/coco_512"
  python3 scripts/preprocess/coco_512.py \
    --images "${target}/coco/val2017" \
    --annotations "${target}/coco/annotations/captions_val2017.json" \
    --output "${target}/coco_512"
}

preprocess_parti() {
  require_path "${target}/parti-prompts" "PartiPrompts dataset directory"
  mkdir -p "${target}/parti_prompts_1024"
  python3 scripts/preprocess/parti_prompts.py \
    --input "${target}/parti-prompts" \
    --resolution 1024 \
    --denoising-steps 28 \
    --output "${target}/parti_prompts_1024/prompts.jsonl"
}

preprocess_conversations() {
  require_path "${target}/splitwise" "Splitwise-style source directory"
  require_path "${target}/azure" "Azure conversation trace directory"
  mkdir -p "${target}/conversation_benchmark"
  python3 scripts/preprocess/conversation_requests.py \
    --splitwise "${target}/splitwise" \
    --azure "${target}/azure" \
    --median-input-tokens 1020 \
    --median-output-tokens 129 \
    --output "${target}/conversation_benchmark/requests.jsonl"
}

case "$task" in
  all)
    preprocess_coco
    preprocess_parti
    preprocess_conversations
    ;;
  coco) preprocess_coco ;;
  parti) preprocess_parti ;;
  conversations) preprocess_conversations ;;
  *) echo "preprocess_datasets.sh: task must be all, coco, parti, or conversations" >&2; exit 2 ;;
esac
