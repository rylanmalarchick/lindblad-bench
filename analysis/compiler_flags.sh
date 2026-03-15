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
echo "flags,d,ns_per_step,gflops,gbytes_s" > "$CSV"

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
        -DCMAKE_C_FLAGS="$flags -DNDEBUG" \
        -DVECTORIZE=OFF \
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

    # Extract per-d results and append to CSV
    while IFS= read -r line; do
        d_match=$(echo "$line" | grep -oP '(?<=d = )\d+' || true)
        if [[ -n "$d_match" ]]; then
            current_d="$d_match"
        fi
        ns_match=$(echo "$line" | grep -oP '[\d.]+ (?=ns)' | head -1 || true)
        gf_match=$(echo "$line" | grep -oP '[\d.]+ (?=GFLOP)' | head -1 || true)
        gb_match=$(echo "$line" | grep -oP '[\d.]+ (?=GB/s)' | head -1 || true)
        if [[ -n "$gb_match" && -n "$current_d" ]]; then
            echo "$label,$current_d,$ns_match,$gf_match,$gb_match" >> "$CSV"
        fi
    done <<< "$output"
done

echo ""
echo "Results written to $CSV"
echo "Assembly dumps in $ASM_DIR/"
