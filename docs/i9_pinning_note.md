# i9 P-core / E-core Pinning Note

Date: 2026-04-16  
Scope: auxiliary scheduler sensitivity check for the local `i9-13980HX`

## Why this was run

The main OpenMP batch sweep on the local i9 used:

- `OMP_PROC_BIND=true`
- `OMP_PLACES=cores`

but did **not** use explicit `taskset` or `numactl` pinning to force either the performance-core or efficiency-core subset.

Because the i9-13980HX is a hybrid `8P + 16E` design, an auxiliary taskset check was run to see whether the main thread-scaling conclusions were likely to be distorted by the scheduler.

## Core-class cpusets on this machine

From sysfs:

- `/sys/devices/cpu_core/cpus` = `0-15`
- `/sys/devices/cpu_atom/cpus` = `16-31`

These were used as the `taskset` masks for P-core-only and E-core-only runs.

## Measurement protocol

- executable: `build_jcp/bench_batch`
- batch size: `128`
- repetitions per timed run: `100`
- trials per configuration: `5`
- thread counts checked: `8`, `16`
- placements checked:
  - default scheduler placement with OpenMP core binding
  - P-core-only `taskset -c 0-15`
  - E-core-only `taskset -c 16-31`

Raw data:

- `/home/rylan/dev/projects/lindblad-bench/benchmarks/pinning_i9_aux_data.csv`

## Median summary

### `d=9`

- unpinned, `16` threads: `15.91 GFLOP/s`
- P-core-only, `16` threads: `14.92 GFLOP/s`
- E-core-only, `16` threads: `9.93 GFLOP/s`

### `d=27`

- unpinned, `16` threads: `12.17 GFLOP/s`
- P-core-only, `16` threads: `11.70 GFLOP/s`
- E-core-only, `16` threads: `7.25 GFLOP/s`

## Interpretation

- E-core-only placement is materially worse for the larger kernel.
- On this machine, the default scheduler plus OpenMP core binding was not dramatically worse than explicit P-core-only placement for the tested `16`-thread case.
- Therefore the main qualitative paper conclusion does **not** depend on an unmeasured scheduler assumption:
  - `d=3` does not benefit much from threading,
  - larger batched kernels do,
  - and E-core-only placement is a weaker operating point for the larger problem sizes.

## Paper-safe wording

Safe:

- say that the main sweep did not use explicit class pinning
- say that `OMP_PROC_BIND=true` and `OMP_PLACES=cores` were used
- say that an auxiliary taskset check found E-core-only placement substantially worse for `d=27`
- say that default versus P-core-only placement did not change the qualitative conclusions

Avoid:

- do not claim the published i9 results are purely P-core numbers
- do not claim explicit P-core pinning would uniformly improve every reported i9 point
