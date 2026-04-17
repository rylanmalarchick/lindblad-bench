#!/usr/bin/env bash
# omp_scaling.sh — Collect batch/OpenMP CPU scaling results in CSV form.
#
# Builds bench_batch with OpenMP enabled, runs a thread-count and batch-size
# sweep for multiple trials, writes raw machine-tagged results, and then
# summarizes medians into benchmarks/cpu_batch_results.csv.
#
# Usage:
#   bash analysis/omp_scaling.sh
#
# Optional environment overrides:
#   N_REPS=200
#   THREADS_LIST="1 2 4 8"
#   BATCH_SIZES="1 8 32 128"
#   TRIALS=5
#   RAW_CSV=/path/to/raw.csv
#   CSV=/path/to/summary.csv
#   HOST_TAG=myhost
#   QUIET=1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build-openmp"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
RAW_CSV="${RAW_CSV:-$ROOT/benchmarks/cpu_batch_results_raw.csv}"
CSV="${CSV:-$ROOT/benchmarks/cpu_batch_results.csv}"
N_REPS="${N_REPS:-200}"
TRIALS="${TRIALS:-5}"
QUIET="${QUIET:-0}"

default_threads() {
    case "$HOST_TAG" in
        theLittleMachine)
            echo "1 2 4 8 16 24 32"
            ;;
        theMachine|desktop)
            echo "1 2 4 6 12"
            ;;
        *)
            echo "1 2 4 $(nproc)"
            ;;
    esac
}

THREADS_LIST="${THREADS_LIST:-$(default_threads)}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32 128}"
GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
if [[ -n "$(git -C "$ROOT" status --short)" ]]; then
    GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    else
        echo "uv not found; set UV_BIN or add uv to PATH" >&2
        exit 1
    fi
fi

export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"

mkdir -p "$(dirname "$CSV")"
echo "machine,git_commit,trial,d,threads,batch_size,n_reps,ns_per_batch,ns_per_state_step,mstate_steps_s,gflops,gbytes_s" > "$RAW_CSV"

cmake -S "$ROOT" -B "$BUILD_DIR" -DENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" --target bench_batch -- -j"$(nproc)" > /dev/null

for trial in $(seq 1 "$TRIALS"); do
    for threads in $THREADS_LIST; do
        for batch_size in $BATCH_SIZES; do
            if [[ "$QUIET" != "1" ]]; then
                echo ""
                echo "=== trial=$trial threads=$threads batch_size=$batch_size ==="
            fi
            output=$(OMP_NUM_THREADS="$threads" "$BUILD_DIR/bench_batch" "$N_REPS" "$batch_size")
            if [[ "$QUIET" != "1" ]]; then
                echo "$output"
            fi

            while IFS= read -r line; do
                [[ "$line" == RESULT,* ]] || continue
                csv_line=$(printf '%s\n' "$line" | awk -F'[=,]' -v machine="$HOST_TAG" -v git_commit="$GIT_COMMIT" -v trial="$trial" '
                    {
                        for (i = 2; i < NF; i += 2) {
                            key = $i;
                            value = $(i + 1);
                            v[key] = value;
                        }
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                               machine, git_commit, trial, v["d"], v["threads"],
                               v["batch_size"], v["n_reps"], v["ns_per_batch"],
                               v["ns_per_state_step"], v["mstate_steps_s"],
                               v["gflops"], v["gbytes_s"];
                    }')
                echo "$csv_line" >> "$RAW_CSV"
            done <<< "$output"
        done
    done
done

"$UV_BIN" run python "$ROOT/analysis/summarize_cpu_batch.py" "$RAW_CSV" "$CSV"

echo ""
echo "Raw results written to $RAW_CSV"
echo "Median summary written to $CSV"
