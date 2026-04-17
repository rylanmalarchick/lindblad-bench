cmake_minimum_required(VERSION 3.16)
project(lindblad_bench C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# Base flags: all warnings, treat as errors, fast math
set(BASE_FLAGS "-Wall -Wextra -Werror -ffast-math")

# CUSTOM_C_FLAGS: when set, overrides ALL flag logic (used by compiler_flags.sh)
set(CUSTOM_C_FLAGS "" CACHE STRING "Override all compiler flags (for benchmarking)")

if(CUSTOM_C_FLAGS)
    set(CMAKE_C_FLAGS "${CUSTOM_C_FLAGS}")
else()
    # Two build targets: scalar-only (for assembly comparison) and auto-vectorized
    # Use: cmake -DVECTORIZE=OFF .. to disable auto-vectorization
    option(VECTORIZE "Enable compiler auto-vectorization (-O3 -march=native)" ON)

    if(VECTORIZE)
        set(OPT_FLAGS "-O3 -march=native")
    else()
        set(OPT_FLAGS "-O2 -fno-tree-vectorize")
    endif()

    set(CMAKE_C_FLAGS "${BASE_FLAGS} ${OPT_FLAGS}")
endif()

option(ENABLE_OPENMP "Enable OpenMP for batch-parallel APIs and benchmarks" OFF)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

option(ENABLE_CUDA "Enable CUDA validation targets and benchmarks" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 89)
    endif()
endif()

# ---------------------------------------------------------------------------
# Library: lindblad_bench (static)
# ---------------------------------------------------------------------------
add_library(lindblad_bench STATIC
    src/expm.c
    src/lindblad.c
    src/propagate.c
    src/evolve.c
)

target_include_directories(lindblad_bench PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

if(ENABLE_OPENMP)
    target_link_libraries(lindblad_bench PUBLIC OpenMP::OpenMP_C)
endif()

if(ENABLE_CUDA)
    add_library(lindblad_cuda STATIC
        cuda/propagate_batch_wrapper.c
        cuda/propagate_batch_kernel.cu
    )
    target_include_directories(lindblad_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/cuda
    )
    target_link_libraries(lindblad_cuda PUBLIC
        lindblad_bench
        CUDA::cudart
        m
    )
    set_target_properties(lindblad_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
foreach(bench propagate evolve expm grape batch)
    add_executable(bench_${bench} benchmarks/bench_${bench}.c)
    target_link_libraries(bench_${bench} lindblad_bench m)
    target_include_directories(bench_${bench} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
endforeach()

if(ENABLE_CUDA)
    add_executable(bench_cuda benchmarks/bench_cuda.c)
    target_link_libraries(bench_cuda lindblad_cuda)
    target_include_directories(bench_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/cuda
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

foreach(test expm propagate evolve batch grape_build)
    add_executable(test_${test} tests/test_${test}.c)
    target_link_libraries(test_${test} lindblad_bench m)
    target_include_directories(test_${test} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME ${test} COMMAND test_${test})
endforeach()

if(ENABLE_CUDA)
    add_executable(test_cuda tests/test_cuda.c)
    target_link_libraries(test_cuda lindblad_cuda)
    target_include_directories(test_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME cuda COMMAND test_cuda)
endif()
