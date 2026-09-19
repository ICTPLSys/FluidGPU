cmake_minimum_required(VERSION 3.22)

project(fluidgpu_cpp_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

option(FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON
  "Prefer a system-installed nlohmann_json over the vendored single header" OFF)

set(FLUIDGPU_VENDORED_NLOHMANN_JSON_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann")

if(FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON)
  find_package(nlohmann_json CONFIG QUIET)
  if(nlohmann_json_FOUND)
    set(FLUIDGPU_NLOHMANN_JSON_TARGET nlohmann_json::nlohmann_json)
  else()
    find_path(FLUIDGPU_NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NOT FLUIDGPU_NLOHMANN_JSON_INCLUDE_DIR)
      message(FATAL_ERROR
        "FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON=ON but nlohmann/json.hpp not found. "
        "Install nlohmann-json3-dev, provide an nlohmann_json CMake package, "
        "or set FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON=OFF to use the vendored header.")
    endif()
  endif()
else()
  if(NOT EXISTS "${FLUIDGPU_VENDORED_NLOHMANN_JSON_DIR}/json.hpp")
    message(FATAL_ERROR
      "Vendored nlohmann/json header missing at "
      "${FLUIDGPU_VENDORED_NLOHMANN_JSON_DIR}/json.hpp. "
      "Restore the archive or set FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON=ON.")
  endif()
  set(FLUIDGPU_NLOHMANN_JSON_INCLUDE_DIR "${FLUIDGPU_VENDORED_NLOHMANN_JSON_DIR}/..")
endif()

find_path(FLUIDGPU_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(FLUIDGPU_IBVERBS_LIBRARY ibverbs)
find_library(FLUIDGPU_RDMACM_LIBRARY rdmacm)
if(NOT FLUIDGPU_IBVERBS_INCLUDE_DIR OR NOT FLUIDGPU_IBVERBS_LIBRARY OR NOT FLUIDGPU_RDMACM_LIBRARY)
  message(FATAL_ERROR "RDMA dependencies not found. Install libibverbs-dev and librdmacm-dev.")
endif()

add_library(fluidgpu_cpp_runtime STATIC
  include/common.cu
  include/comm_backend.cu
  include/runtime.cu
)
add_library(FluidGPU::cpp_runtime ALIAS fluidgpu_cpp_runtime)

target_include_directories(fluidgpu_cpp_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FLUIDGPU_IBVERBS_INCLUDE_DIR}
)
if(FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON AND nlohmann_json_FOUND)
  # System CMake package provides include paths via the imported target below.
else()
  target_include_directories(fluidgpu_cpp_runtime PUBLIC ${FLUIDGPU_NLOHMANN_JSON_INCLUDE_DIR})
endif()

target_link_libraries(fluidgpu_cpp_runtime
  PUBLIC
    CUDA::cudart
    ${FLUIDGPU_IBVERBS_LIBRARY}
    ${FLUIDGPU_RDMACM_LIBRARY}
)
if(FLUIDGPU_USE_SYSTEM_NLOHMANN_JSON AND nlohmann_json_FOUND)
  target_link_libraries(fluidgpu_cpp_runtime PUBLIC ${FLUIDGPU_NLOHMANN_JSON_TARGET})
endif()

set_target_properties(fluidgpu_cpp_runtime PROPERTIES
  OUTPUT_NAME fluidgpu_cpp_runtime
  CUDA_SEPARABLE_COMPILATION ON
)
