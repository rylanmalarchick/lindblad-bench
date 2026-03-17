#!/usr/bin/env bash
# compiler_flags.sh — Reproduce all compiler flag experiments.
#
# Builds bench_propagate with 4 flag sets, runs each, and dumps
# the assembly for the hot loop (lb_propagate_step) from each build.
#
# Output: benchmarks/compiler_results.csv
#         analysis/asm/  (one .s file per flag set)
#
# Usage:
#   bash analysis/compiler_flags.sh [n_reps]
#
# Requires: gcc, objdump, cmake

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_REPS="${1:-50000}"
ASM_DIR="$ROOT/analysis/asm"
mkdir -p "$ASM_DIR"

CSV="$ROOT/benchmarks/compiler_results.csv"
echo "flags,variant,d,ns_per_step,gflops,gbytes_s" > "$CSV"

declare -A FLAG_SETS
FLAG_SETS[scalar]="-O2 -fno-tree-vectorize -fno-unroll-loops"
FLAG_SETS[O3]="-O3"
FLAG_SETS[O3_native]="-O3 -march=native"
FLAG_SETS[O3_native_fast]="-O3 -march=native -ffast-math"

for label in scalar O3 O3_native O3_native_fast; do
    flags="${FLAG_SETS[$label]}"
    build_dir="$ROOT/build-$label"
    echo ""
    echo "=== $label: $flags ==="

    cmake -B "$build_dir" -S "$ROOT" \
        -DCUSTOM_C_FLAGS="$flags -DNDEBUG" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -Wno-dev -DCMAKE_VERBOSE_MAKEFILE=OFF \
        > /dev/null 2>&1
    cmake --build "$build_dir" --target bench_propagate -- -j"$(nproc)" > /dev/null 2>&1

    # Dump assembly for lb_propagate_step
    objdump -d --no-show-raw-insn "$build_dir/bench_propagate" \
        | awk '/^[0-9a-f]+ <lb_propagate_step>:/{p=1} p{print} /^$/{if(p)exit}' \
        > "$ASM_DIR/lb_propagate_step_${label}.s" 2>/dev/null || true

    # Run and parse output
    output=$("$build_dir/bench_propagate" "$N_REPS" 2>/dev/null)
    echo "$output"

    # Extract per-d results for each variant (AoS, SoA, AVX2)
    current_d=""
    aos_ns="" aos_gf="" aos_gb=""
    soa_ns="" soa_gf="" soa_gb=""
    avx_ns="" avx_gf="" avx_gb=""

    while IFS= read -r line; do
        # Match "d = X"
        if [[ "$line" =~ d\ =\ ([0-9]+) ]]; then
            current_d="${BASH_REMATCH[1]}"
            aos_ns="" aos_gf="" aos_gb=""
            soa_ns="" soa_gf="" soa_gb=""
            avx_ns="" avx_gf="" avx_gb=""
        fi

        # AoS metrics
        if [[ "$line" =~ \[AoS\]\ ns/step:\ *([0-9.]+) ]]; then aos_ns="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[AoS\]\ GFLOP/s:\ *([0-9.]+) ]]; then aos_gf="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[AoS\]\ GB/s\ *:\ *([0-9.]+) ]]; then
            aos_gb="${BASH_REMATCH[1]}"
            echo "$label,AoS,$current_d,$aos_ns,$aos_gf,$aos_gb" >> "$CSV"
        fi

        # SoA metrics
        if [[ "$line" =~ \[SoA\]\ ns/step:\ *([0-9.]+) ]]; then soa_ns="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[SoA\]\ GFLOP/s:\ *([0-9.]+) ]]; then soa_gf="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[SoA\]\ GB/s\ *:\ *([0-9.]+) ]]; then
            soa_gb="${BASH_REMATCH[1]}"
            echo "$label,SoA,$current_d,$soa_ns,$soa_gf,$soa_gb" >> "$CSV"
        fi

        # AVX2 metrics
        if [[ "$line" =~ \[AVX\]\ ns/step:\ *([0-9.]+) ]]; then avx_ns="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[AVX\]\ GFLOP/s:\ *([0-9.]+) ]]; then avx_gf="${BASH_REMATCH[1]}"; fi
        if [[ "$line" =~ \[AVX\]\ GB/s\ *:\ *([0-9.]+) ]]; then
            avx_gb="${BASH_REMATCH[1]}"
            echo "$label,AVX2,$current_d,$avx_ns,$avx_gf,$avx_gb" >> "$CSV"
        fi
    done <<< "$output"
done

echo ""
echo "Results written to $CSV"
echo "Assembly dumps in $ASM_DIR/"
