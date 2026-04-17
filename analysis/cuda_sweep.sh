#!/usr/bin/env bash
# cuda_sweep.sh — Collect CUDA batch benchmark results in CSV form.
#
# Builds bench_cuda, runs multiple trials over a batch-size sweep, writes raw
# machine-tagged results, then summarizes medians into a second CSV.
#
# Usage:
#   bash analysis/cuda_sweep.sh
#
# Optional environment overrides:
#   N_REPS=50
#   TRIALS=5
#   BATCH_SIZES="1 8 32 128"
#   RAW_CSV=/path/to/raw.csv
#   CSV=/path/to/summary.csv
#   HOST_TAG=myhost
#   GPU_NAME="RTX 4070 Laptop GPU"
#   QUIET=1
#   UV_BIN=/path/to/uv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build-cuda"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
RAW_CSV="${RAW_CSV:-$ROOT/benchmarks/cuda_batch_results_raw.csv}"
CSV="${CSV:-$ROOT/benchmarks/cuda_batch_results.csv}"
N_REPS="${N_REPS:-50}"
TRIALS="${TRIALS:-5}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32 128}"
QUIET="${QUIET:-0}"

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

GPU_NAME="${GPU_NAME:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)}"

mkdir -p "$(dirname "$CSV")"
echo "machine,gpu_name,git_commit,trial,d,batch_size,n_reps,host_ns_per_state_step,host_gflops,host_gbytes_s,kernel_ns_per_state_step,kernel_gflops,kernel_gbytes_s" > "$RAW_CSV"

cmake -S "$ROOT" -B "$BUILD_DIR" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" --target bench_cuda -- -j"$(nproc)" > /dev/null

for trial in $(seq 1 "$TRIALS"); do
    for batch_size in $BATCH_SIZES; do
        if [[ "$QUIET" != "1" ]]; then
            echo ""
            echo "=== trial=$trial batch_size=$batch_size ==="
        fi
        output=$("$BUILD_DIR/bench_cuda" "$N_REPS" "$batch_size")
        if [[ "$QUIET" != "1" ]]; then
            echo "$output"
        fi

        while IFS= read -r line; do
            [[ "$line" == RESULT,* ]] || continue
            csv_line=$(printf '%s\n' "$line" | awk -F'[=,]' \
                -v machine="$HOST_TAG" \
                -v gpu_name="$GPU_NAME" \
                -v git_commit="$GIT_COMMIT" \
                -v trial="$trial" '
                {
                    for (i = 2; i < NF; i += 2) {
                        key = $i;
                        value = $(i + 1);
                        v[key] = value;
                    }
                    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                           machine, gpu_name, git_commit, trial, v["d"],
                           v["batch_size"], v["n_reps"],
                           v["host_ns_per_state_step"], v["host_gflops"], v["host_gbytes_s"],
                           v["kernel_ns_per_state_step"], v["kernel_gflops"], v["kernel_gbytes_s"];
                }')
            echo "$csv_line" >> "$RAW_CSV"
        done <<< "$output"
    done
done

"$UV_BIN" run python "$ROOT/analysis/summarize_cuda_batch.py" "$RAW_CSV" "$CSV"

echo ""
echo "Raw results written to $RAW_CSV"
echo "Median summary written to $CSV"
