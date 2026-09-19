#!/usr/bin/env bash
set -euo pipefail

build_dir="cpp_runtime/build"
cuda_arch="${FLUIDGPU_CUDA_ARCH:-80}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="${2:?missing value for --build-dir}"
      shift 2
      ;;
    --cuda-arch)
      cuda_arch="${2:?missing value for --cuda-arch}"
      shift 2
      ;;
    *)
      echo "build_cpp_runtime.sh: unknown argument $1" >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "build_cpp_runtime.sh: $*" >&2
  exit 1
}

[[ -d cpp_runtime/include ]] || fail "missing cpp_runtime/include; the FluidGPU C++ runtime sources appear to be absent"
command -v cmake >/dev/null 2>&1 || fail "cmake is not available"
command -v nvcc >/dev/null 2>&1 || fail "nvcc is not available; install the CUDA toolkit"
command -v g++ >/dev/null 2>&1 || fail "g++ is not available"

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
[[ -d "$cuda_home" ]] || fail "CUDA_HOME does not exist: ${cuda_home}"
[[ -f "$cuda_home/include/cuda_runtime.h" ]] || fail "missing CUDA header: ${cuda_home}/include/cuda_runtime.h"

header_probe="$(mktemp -d)"
cleanup() {
  rm -rf "$header_probe"
}
trap cleanup EXIT

cat >"${header_probe}/probe.cpp" <<'EOF'
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
int main() { return 0; }
EOF
g++ -std=c++17 -c "${header_probe}/probe.cpp" -o "${header_probe}/probe.o" \
  || fail "missing RDMA headers; install libibverbs-dev and librdmacm-dev"

[[ -f cpp_runtime/third_party/nlohmann/json.hpp ]] \
  || fail "missing vendored cpp_runtime/third_party/nlohmann/json.hpp; restore the artifact archive"

# grep must consume all of ldconfig's output: with pipefail, `grep -q` exiting
# early sends ldconfig SIGPIPE and the pipeline spuriously fails.
ldconfig -p 2>/dev/null | grep 'libibverbs\.so' >/dev/null || fail "libibverbs runtime library not found"
ldconfig -p 2>/dev/null | grep 'librdmacm\.so' >/dev/null || fail "librdmacm runtime library not found"

cmake -S cpp_runtime -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_dir" --target fluidgpu_cpp_runtime -j"$(nproc)"

echo "build_cpp_runtime.sh: wrote ${build_dir}/libfluidgpu_cpp_runtime.a"
