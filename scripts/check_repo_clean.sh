#!/usr/bin/env bash
set -euo pipefail

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  status="$(git status --porcelain)"
  if [[ -n "$status" ]]; then
    echo "$status"
    echo "check_repo_clean: repository has uncommitted changes" >&2
    exit 1
  fi
fi

size_kb="$(
  du -sk \
    --exclude=./artifacts \
    --exclude=./datasets \
    --exclude=./models \
    . | awk '{print $1}'
)"
limit_kb=$((2 * 1024 * 1024))
if (( size_kb > limit_kb )); then
  echo "check_repo_clean: workspace exceeds 2 GB (${size_kb} KB)" >&2
  exit 1
fi
echo "check_repo_clean: PASS"
