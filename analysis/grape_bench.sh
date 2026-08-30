#!/usr/bin/env bash
# grape_bench.sh — Run the C GRAPE-style benchmark and write per-trial CSV rows.
#
# Builds bench_grape in build-grape/ (Release, no OpenMP) and writes
# benchmarks/grape_c_results_raw_<host>.csv with one row per (d, trial).
# Medians are computed by analysis/make_jcp_assets.py.
#
# Usage:
#   bash analysis/grape_bench.sh
#
# Optional environment overrides:
#   N_TRIALS=5
#   HOST_TAG=myhost
#   CSV=/path/to/raw.csv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build-grape"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
N_TRIALS="${N_TRIALS:-5}"
CSV="${CSV:-$ROOT/benchmarks/grape_c_results_raw_${HOST_TAG}.csv}"

GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=OFF >/dev/null
cmake --build "$BUILD_DIR" --target bench_grape >/dev/null

echo "host=$HOST_TAG commit=$GIT_COMMIT trials=$N_TRIALS -> $CSV"
HOST_TAG="$HOST_TAG" GIT_COMMIT="$GIT_COMMIT" \
    "$BUILD_DIR/bench_grape" --csv --trials "$N_TRIALS" > "$CSV"
echo "wrote $(($(wc -l < "$CSV") - 1)) rows"
