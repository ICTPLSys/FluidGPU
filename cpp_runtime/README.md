# FluidGPU C++ Runtime

This module packages the FluidGPU C++/CUDA runtime sources as a standalone static library target.

Source layout:

- `cpp_runtime/include/common.*`
- `cpp_runtime/include/comm_backend.*`
- `cpp_runtime/include/runtime.*`
- `cpp_runtime/include/logger.h`
- `cpp_runtime/include/utask.h`

Build:

```bash
bash scripts/build_cpp_runtime.sh
```

The build requires CUDA Toolkit, CMake, a C++ compiler, `libibverbs-dev`, and `librdmacm-dev`. The `nlohmann/json` single header is vendored under `third_party/nlohmann/` (v3.11.3, MIT-licensed) and used by default, so no extra JSON system package is needed. To force a system-installed `nlohmann_json` instead, pass `-DFLUIDGPU_USE_SYSTEM_NLOHMANN_JSON=ON` to CMake.

Missing dependencies cause the build script or CMake configure step to fail with an explicit reason.
