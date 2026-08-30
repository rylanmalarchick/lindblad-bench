#!/usr/bin/env bash
# stream_probe.sh — Read-bandwidth ceilings per working-set size for the cost model.
#
# Builds bench_stream with OpenMP and runs it at 1 thread and at the host's
# full thread list, writing benchmarks/probes_stream_<host>.csv (one row per
# trial, size, thread count).
#
# Usage: bash analysis/stream_probe.sh
# Env:   HOST_TAG, THREADS_LIST="1 8", TRIALS=5, CSV=path

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build-omp-jcp"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
TRIALS="${TRIALS:-5}"
CSV="${CSV:-$ROOT/benchmarks/probes_stream_${HOST_TAG}.csv}"
GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
case "$HOST_TAG" in
    theLittleMachine) DEFAULT_THREADS="1 8 16 24" ;;
    theMachine|desktop) DEFAULT_THREADS="1 6 12" ;;
    *) DEFAULT_THREADS="1 $(nproc)" ;;
esac
THREADS_LIST="${THREADS_LIST:-$DEFAULT_THREADS}"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=ON >/dev/null
cmake --build "$BUILD_DIR" --target bench_stream >/dev/null

echo "machine,git_commit,trial,size_kb,threads,layout,reps,bytes,gbs_per_thread,gbs_aggregate" > "$CSV"
for t in $THREADS_LIST; do
    OMP_NUM_THREADS="$t" OMP_PROC_BIND=true OMP_PLACES=cores \
    HOST_TAG="$HOST_TAG" GIT_COMMIT="$GIT_COMMIT" \
        "$BUILD_DIR/bench_stream" --csv --trials "$TRIALS" | tail -n +2 >> "$CSV"
    OMP_NUM_THREADS="$t" OMP_PROC_BIND=true OMP_PLACES=cores \
    HOST_TAG="$HOST_TAG" GIT_COMMIT="$GIT_COMMIT" \
        "$BUILD_DIR/bench_stream" --csv --shared --trials "$TRIALS" | tail -n +2 >> "$CSV"
done
echo "wrote $(($(wc -l < "$CSV") - 1)) rows to $CSV"
