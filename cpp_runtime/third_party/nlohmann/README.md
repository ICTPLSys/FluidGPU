# Vendored nlohmann/json (single header)

This directory vendors the single-header `json.hpp` from
[nlohmann/json](https://github.com/nlohmann/json).

- Version: v3.11.3
- Upstream source: https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
- SHA-256: `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`
- License: MIT (see `LICENSE.MIT` in this directory and the SPDX header at the top of `json.hpp`)

The FluidGPU CMake build (`cpp_runtime/CMakeLists.txt`) uses this vendored
header by default so that the artifact builds without requiring the
`nlohmann-json3-dev` system package or network access to GitHub. A
system-installed `nlohmann_json` CMake package or header path is still accepted
if `FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON=ON` is passed to CMake.
