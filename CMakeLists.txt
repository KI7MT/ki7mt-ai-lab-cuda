# =============================================================================
# ki7mt-ai-lab-cuda - Sovereign CUDA Engine for WSPR Propagation Analysis
# =============================================================================
#
# CMake build configuration for Blackwell GPU (sm_120)
# Dependencies: CUDA 13.1+, clickhouse-cpp
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   cmake --build . -j$(nproc)
#
# =============================================================================

cmake_minimum_required(VERSION 3.28)

# Project definition with CUDA language
project(ki7mt-ai-lab-cuda
    VERSION 2.0.6
    DESCRIPTION "Sovereign CUDA Engine for WSPR Propagation Signature Analysis"
    LANGUAGES CXX CUDA
)

# =============================================================================
# C++ Standard (C++20)
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CUDA Configuration - Blackwell Direct (sm_120)
# =============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Blackwell directly - no PTX JIT needed
set(CMAKE_CUDA_ARCHITECTURES 120)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# =============================================================================
# Find Dependencies
# =============================================================================

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")

# clickhouse-cpp (for ClickHouse integration)
find_package(clickhouse-cpp CONFIG QUIET)
if(clickhouse-cpp_FOUND)
    message(STATUS "Found clickhouse-cpp: ${clickhouse-cpp_VERSION}")
    set(HAVE_CLICKHOUSE TRUE)
else()
    message(STATUS "clickhouse-cpp not found - ClickHouse integration disabled")
    set(HAVE_CLICKHOUSE FALSE)
endif()

# =============================================================================
# Include Directories
# =============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# =============================================================================
# Library: Signature Engine
# =============================================================================
add_library(wspr_signature_engine STATIC
    src/engine/signature_kernel.cu
)

target_include_directories(wspr_signature_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(wspr_signature_engine
    CUDA::cudart
)

set_target_properties(wspr_signature_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Library: Legacy HAL (existing kernels)
# =============================================================================
add_library(wspr_cuda_hal STATIC
    src/cuda/bridge.cu
    src/cuda/bulk_kernels.cu
    src/cuda/kernels.cu
)

target_include_directories(wspr_cuda_hal PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cuda
)

target_link_libraries(wspr_cuda_hal
    CUDA::cudart
)

set_target_properties(wspr_cuda_hal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Executable: CUDA Check Utility (disabled - use Makefile build for production)
# =============================================================================
# Note: wspr-cuda-check requires the legacy Makefile build system due to
# CUDA separable compilation conflicts with the C main() entry point.
# Use 'make' in the project root for the production HAL build.
#
# add_executable(wspr-cuda-check src/wspr-cuda-check.c)
# target_link_libraries(wspr-cuda-check wspr_cuda_hal CUDA::cudart)

# =============================================================================
# Executable: Signature Engine Test
# =============================================================================
add_executable(signature-engine-test
    src/engine/test_signature.cu
)

target_link_libraries(signature-engine-test
    wspr_signature_engine
    CUDA::cudart
)

# =============================================================================
# Optional: ClickHouse Data Loader
# =============================================================================
if(HAVE_CLICKHOUSE)
    add_executable(signature-loader
        src/engine/clickhouse_loader.cpp
    )

    target_link_libraries(signature-loader
        wspr_signature_engine
        clickhouse-cpp::clickhouse-cpp
        CUDA::cudart
    )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS wspr_signature_engine wspr_cuda_hal signature-engine-test
    ARCHIVE DESTINATION lib64
    LIBRARY DESTINATION lib64
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/ki7mt
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== ki7mt-ai-lab-cuda Configuration ===")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "CUDA Arch:        sm_${CMAKE_CUDA_ARCHITECTURES} (Blackwell)")
message(STATUS "C++ Standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:    C++${CMAKE_CUDA_STANDARD}")
message(STATUS "ClickHouse:       ${HAVE_CLICKHOUSE}")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "")
