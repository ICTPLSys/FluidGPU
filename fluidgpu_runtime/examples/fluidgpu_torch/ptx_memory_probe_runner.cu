#include <cuda.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

void check_driver(CUresult status, const char* label) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = "unknown";
  const char* message = "unknown";
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  std::fprintf(stderr, "%s failed: %s: %s\n", label, name, message);
  std::exit(1);
}

void check_runtime(cudaError_t status, const char* label) {
  if (status == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(status));
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <instrumented.cubin> <access_count>\n", argv[0]);
    return 2;
  }

  const char* cubin_path = argv[1];
  const int access_count = std::atoi(argv[2]);
  if (access_count <= 0) {
    std::fprintf(stderr, "access_count must be positive\n");
    return 2;
  }

  check_driver(cuInit(0), "cuInit");
  check_runtime(cudaSetDevice(0), "cudaSetDevice");
  check_runtime(cudaFree(nullptr), "cudaFree");

  CUmodule module = nullptr;
  CUfunction producer = nullptr;
  CUfunction consumer = nullptr;
  check_driver(cuModuleLoad(&module, cubin_path), "cuModuleLoad");
  check_driver(
      cuModuleGetFunction(&producer, module, "producer_kernel"),
      "cuModuleGetFunction producer_kernel");
  check_driver(
      cuModuleGetFunction(&consumer, module, "consumer_kernel"),
      "cuModuleGetFunction consumer_kernel");

  float* intermediate = nullptr;
  float* final_output = nullptr;
  std::uint64_t* ranges_device = nullptr;
  check_runtime(cudaMalloc(&intermediate, sizeof(float)), "cudaMalloc intermediate");
  check_runtime(cudaMalloc(&final_output, sizeof(float)), "cudaMalloc final_output");
  check_runtime(
      cudaMalloc(&ranges_device, sizeof(std::uint64_t) * static_cast<size_t>(access_count) * 2),
      "cudaMalloc ranges");

  std::vector<std::uint64_t> ranges_host(static_cast<size_t>(access_count) * 2);
  for (int i = 0; i < access_count; ++i) {
    ranges_host[static_cast<size_t>(i) * 2] = std::numeric_limits<std::uint64_t>::max();
    ranges_host[static_cast<size_t>(i) * 2 + 1] = 0;
  }
  check_runtime(
      cudaMemcpy(
          ranges_device,
          ranges_host.data(),
          ranges_host.size() * sizeof(std::uint64_t),
          cudaMemcpyHostToDevice),
      "cudaMemcpy ranges init");

  void* producer_args[] = {&intermediate, &ranges_device};
  check_driver(
      cuLaunchKernel(producer, 1, 1, 1, 1, 1, 1, 0, nullptr, producer_args, nullptr),
      "cuLaunchKernel producer_kernel");

  void* consumer_args[] = {&intermediate, &final_output, &ranges_device};
  check_driver(
      cuLaunchKernel(consumer, 1, 1, 1, 1, 1, 1, 0, nullptr, consumer_args, nullptr),
      "cuLaunchKernel consumer_kernel");
  check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check_runtime(
      cudaMemcpy(
          ranges_host.data(),
          ranges_device,
          ranges_host.size() * sizeof(std::uint64_t),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy ranges result");

  const std::uint64_t expected_min[] = {
      reinterpret_cast<std::uint64_t>(intermediate),
      reinterpret_cast<std::uint64_t>(intermediate),
      reinterpret_cast<std::uint64_t>(final_output),
  };
  const std::uint64_t expected_max[] = {
      reinterpret_cast<std::uint64_t>(intermediate) + sizeof(float) - 1,
      reinterpret_cast<std::uint64_t>(intermediate) + sizeof(float) - 1,
      reinterpret_cast<std::uint64_t>(final_output) + sizeof(float) - 1,
  };

  std::printf("{\"status\":\"ok\",\"range_count\":%d,\"ranges\":[", access_count);
  for (int i = 0; i < access_count; ++i) {
    const std::uint64_t min_addr = ranges_host[static_cast<size_t>(i) * 2];
    const std::uint64_t max_addr = ranges_host[static_cast<size_t>(i) * 2 + 1];
    const bool has_expected = i < 3;
    const std::uint64_t expected_min_addr = has_expected ? expected_min[i] : 0;
    const std::uint64_t expected_max_addr = has_expected ? expected_max[i] : 0;
    const bool match = has_expected && min_addr == expected_min_addr && max_addr == expected_max_addr;
    if (i != 0) {
      std::printf(",");
    }
    std::printf(
        "{\"access_id\":%d,\"min_addr\":%" PRIu64 ",\"max_addr\":%" PRIu64
        ",\"expected_min_addr\":%" PRIu64 ",\"expected_max_addr\":%" PRIu64
        ",\"match\":%s}",
        i,
        min_addr,
        max_addr,
        expected_min_addr,
        expected_max_addr,
        match ? "true" : "false");
  }
  std::printf("]}\n");

  check_runtime(cudaFree(ranges_device), "cudaFree ranges");
  check_runtime(cudaFree(final_output), "cudaFree final_output");
  check_runtime(cudaFree(intermediate), "cudaFree intermediate");
  check_driver(cuModuleUnload(module), "cuModuleUnload");
  return 0;
}
