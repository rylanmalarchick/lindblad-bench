cmake_minimum_required(VERSION 3.16)
project(lindblad_bench C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# Base flags: all warnings, treat as errors, fast math
set(BASE_FLAGS "-Wall -Wextra -Werror -ffast-math")

# Two build targets: scalar-only (for assembly comparison) and auto-vectorized
# Use: cmake -DVECTORIZE=OFF .. to disable auto-vectorization
option(VECTORIZE "Enable compiler auto-vectorization (-O3 -march=native)" ON)

if(VECTORIZE)
    set(OPT_FLAGS "-O3 -march=native")
else()
    set(OPT_FLAGS "-O2 -fno-tree-vectorize")
endif()

set(CMAKE_C_FLAGS "${BASE_FLAGS} ${OPT_FLAGS}")

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

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
foreach(bench propagate evolve expm)
    add_executable(bench_${bench} benchmarks/bench_${bench}.c)
    target_link_libraries(bench_${bench} lindblad_bench m)
    target_include_directories(bench_${bench} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

foreach(test expm propagate)
    add_executable(test_${test} tests/test_${test}.c)
    target_link_libraries(test_${test} lindblad_bench m)
    target_include_directories(test_${test} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME ${test} COMMAND test_${test})
endforeach()
