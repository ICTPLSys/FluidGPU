#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "verify_env: FAIL: $*" >&2
  exit 1
}

info() {
  echo "verify_env: INFO: $*"
}

check_contains() {
  local label="$1"
  local expected="$2"
  local output="$3"
  if [[ "$output" != *"$expected"* ]]; then
    fail "$label does not contain '$expected'"
  fi
}

# Software stack
python_version="$(python3 --version 2>&1 || true)"
check_contains "python3 --version" "3.12.13" "$python_version"

nvcc_version="$(nvcc --version 2>&1 || true)"
check_contains "nvcc --version" "release 12.8" "$nvcc_version"

ptxas --version >/dev/null 2>&1 || fail "ptxas is not available"
# Gurobi may be present as the full distribution (gurobi_cl) or as the pip
# gurobipy wheel (no CLI). Accept either, as long as the version is 13.0.1.
if command -v gurobi_cl >/dev/null 2>&1; then
  gurobi_version="$(gurobi_cl --version 2>&1 || true)"
  check_contains "gurobi_cl --version" "13.0.1" "$gurobi_version"
else
  gurobi_version="$(python3 -c 'import gurobipy; print(".".join(str(v) for v in gurobipy.gurobi.version()))' 2>&1 || true)"
  check_contains "gurobipy version" "13.0.1" "$gurobi_version"
  info "gurobi_cl not found; verified via gurobipy ${gurobi_version} (pip wheel)"
fi

# GPU detection.
# FluidGPU's heterogeneous pairs span two RDMA-connected hosts, so we do NOT require both
# sides of a pair on the same machine. We only check that this host holds one side of a
# supported pair, and remind the operator which GPU must be present on the peer host.
#
# Supported pairs (paper §V-A):
#   Pair 1: A100         <-RDMA->  L40S
#   Pair 2: H100         <-RDMA->  RTX Pro 6000
#   Pair 3: B200         <-RDMA->  H100
nvidia_smi="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
[[ -n "$nvidia_smi" ]] || fail "nvidia-smi reports no GPUs"

nvidia_smi_lower="$(echo "$nvidia_smi" | tr '[:upper:]' '[:lower:]')"
has() { [[ "$nvidia_smi_lower" == *"$1"* ]]; }

matched=0
if has "a100"; then
  info "local GPU: A100 detected -> peer host must have L40S (Pair 1)"
  matched=1
fi
if has "l40s"; then
  info "local GPU: L40S detected -> peer host must have A100 (Pair 1)"
  matched=1
fi
if has "rtx pro 6000"; then
  info "local GPU: RTX Pro 6000 detected -> peer host must have H100 (Pair 2)"
  matched=1
fi
if has "b200"; then
  info "local GPU: B200 detected -> peer host must have H100 (Pair 3)"
  matched=1
fi
if has "h100"; then
  info "local GPU: H100 detected -> peer host must have RTX Pro 6000 (Pair 2) or B200 (Pair 3)"
  matched=1
fi
if [[ "$matched" -eq 0 ]]; then
  fail "no supported heterogeneous GPU detected; expected one of {A100, L40S, H100, RTX Pro 6000, B200}. nvidia-smi reported: $nvidia_smi"
fi

# RDMA (required on both hosts of a pair; this host's own NIC must have at least one ACTIVE port)
if command -v ibv_devinfo >/dev/null 2>&1; then
  # grep consumes all output to avoid a pipefail/SIGPIPE race with early exit.
  ibv_devinfo 2>/dev/null | grep "PORT_ACTIVE" >/dev/null || fail "no ACTIVE InfiniBand port on this host (both hosts of a pair need RDMA up)"
else
  fail "ibv_devinfo is not available"
fi

# ----------------------------------------------------------------------------
# Repo fingerprint
# Emit a deterministic identifier of the artifact tree so reviewers can cite
# the exact version they executed (paste these lines into any issue report).
# ----------------------------------------------------------------------------

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
info "repo path: ${repo_root}"

# Git state (if the artifact was cloned instead of unpacked from a tarball)
if command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
  git_sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
  git_desc="$(git -C "$repo_root" describe --always --dirty --tags 2>/dev/null || echo unknown)"
  info "git commit: ${git_sha}"
  info "git describe: ${git_desc}"
else
  info "git commit: (not a git checkout; see tree-hash below)"
fi

# Deterministic tree hash over source files (excludes caches and fetched data).
# Uses sha256sum on a sorted, NUL-delimited file list so ordering is stable
# across filesystems.
if command -v sha256sum >/dev/null 2>&1; then
  tree_hash="$(cd "$repo_root" && find . \
      -type f \
      -not -path './.git/*' \
      -not -path './artifacts/*' \
      -not -path './datasets/*' \
      -not -path './models/*' \
      -not -path './**/__pycache__/*' \
      -not -path './**/.pytest_cache/*' \
      -not -path './**/*.egg-info/*' \
      -not -path './cpp_runtime/build/*' \
      -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}')"
  info "tree sha256: ${tree_hash}"
else
  info "tree sha256: (sha256sum unavailable)"
fi

info "host: $(uname -srmo 2>/dev/null || uname -a)"

echo "verify_env: PASS"
