# CPU Batch Scaling Results

This note freezes the first validated CPU threading dataset for the JCP paper workstream.

## Provenance

- Code state: `3eafeb0-dirty`
- Local host: `theLittleMachine` (`i9-13980HX`)
- Remote host: `theMachine` (`Ryzen 5 1600`)
- Benchmark path: `benchmarks/bench_batch.c`
- Collection script: `analysis/omp_scaling.sh`
- Summary script: `analysis/summarize_cpu_batch.py`
- Timing protocol:
  - `CLOCK_MONOTONIC`
  - warm-up before timed runs
  - `TRIALS=5`
  - medians reported
  - batch sizes `1, 8, 32, 128`
- Output files:
  - Intel summary: `benchmarks/cpu_batch_results_intel.csv`
  - Intel raw: `benchmarks/cpu_batch_results_raw_intel.csv`
  - Ryzen summary: `benchmarks/cpu_batch_results_ryzen.csv`
  - Ryzen raw: `benchmarks/cpu_batch_results_raw_ryzen.csv`

## Important interpretation rule

`lb_propagate_step_batch` only enters the OpenMP loop when `batch_size > 1`.

That means:

- `batch_size = 1` is effectively a serial control case.
- Differences across thread counts at `batch_size = 1` are measurement noise, not real scaling.
- Only `batch_size >= 8` should be used for threading claims.

## Measured scaling shape

### i9-13980HX

- `d=3` does not benefit from threading.
- Best `d=3` throughput was still the serial path:
  - `batch=8`: `7.86 GFLOP/s` at `1` thread
  - `batch=32`: `10.97 GFLOP/s` at `1` thread
  - `batch=128`: `14.49 GFLOP/s` at `1` thread
- `d=9` benefits from batching plus moderate threading, but saturates early:
  - `batch=8`: best at `8` threads, `30.89 GFLOP/s`, `1.78x` faster than 1-thread
  - `batch=32`: best at `8` threads, `52.80 GFLOP/s`, `2.93x`
  - `batch=128`: best at `8` threads, `87.82 GFLOP/s`, `4.85x`
- `d=27` keeps scaling to larger thread counts:
  - `batch=8`: best at `8` threads, `64.39 GFLOP/s`, `4.56x`
  - `batch=32`: best at `32` threads, `116.65 GFLOP/s`, `8.30x`
  - `batch=128`: best at `32` threads, `159.29 GFLOP/s`, `11.28x`

### Ryzen 5 1600

- `d=3` again does not reward threading; the serial path remains best.
- `d=9` scales, but tops out much earlier than on the i9:
  - `batch=8`: best at `4` threads, `35.84 GFLOP/s`, `3.51x`
  - `batch=32`: best at `6` threads, `51.15 GFLOP/s`, `4.96x`
  - `batch=128`: best at `6` threads, `58.72 GFLOP/s`, `5.68x`
- `d=27` scales through the full Ryzen core count:
  - `batch=8`: best at `12` threads, `32.08 GFLOP/s`, `3.79x`
  - `batch=32`: best at `12` threads, `44.43 GFLOP/s`, `5.23x`
  - `batch=128`: best at `12` threads, `47.20 GFLOP/s`, `5.63x`

## Cross-host comparison

- At `d=9, batch=128`, the i9 peak (`87.82 GFLOP/s`) is about `1.50x` the Ryzen peak (`58.72 GFLOP/s`).
- At `d=27, batch=128`, the i9 peak (`159.29 GFLOP/s`) is about `3.37x` the Ryzen peak (`47.20 GFLOP/s`).
- The gap widens sharply at `d=27`, which is consistent with the i9's larger cache and substantially higher memory subsystem capability.

## Draft paper claims supported by this dataset

- Shared-memory parallelism is scientifically honest only when applied over independent batched states, not over sequential time steps of one trajectory.
- For very small cache-resident kernels (`d=3`), OpenMP overhead dominates and the serial path remains best.
- For `d=9`, batching helps, but useful CPU threading saturates at a moderate thread count on the i9 and at the physical-core limit on the Ryzen.
- For `d=27`, batching unlocks substantial throughput scaling on both hosts, but the modern Intel platform pulls away strongly at high batch size.

## Claims not yet supported

- P-core versus E-core pinning on the i9.
- Any statement about optimal CPU thread placement beyond the unrestricted scheduler.
- Any CPU versus GPU or CPU versus FPGA conclusion.
- Any end-to-end GRAPE conclusion.
