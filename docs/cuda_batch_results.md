# CUDA Batch Results

This note freezes the first validated CUDA benchmark dataset for the JCP paper workstream.

## Provenance

- Code state: `3eafeb0-dirty`
- Local GPU platform: `theLittleMachine` + `NVIDIA GeForce RTX 4070 Laptop GPU`
- Remote GPU platform: `theMachine` + `NVIDIA GeForce GTX 1070 Ti`
- Benchmark path: `benchmarks/bench_cuda.c`
- Collection script: `analysis/cuda_sweep.sh`
- Summary script: `analysis/summarize_cuda_batch.py`
- Timing protocol:
  - `TRIALS=5`
  - medians reported
  - batch sizes `1, 8, 32, 128`
  - host-visible timing from `lb_cuda_propagate_step_batch`
  - kernel-only timing from `lb_cuda_raw_batch_context_t` with resident device buffers
- Output files:
  - RTX 4070 summary: `benchmarks/cuda_batch_results_rtx4070.csv`
  - RTX 4070 raw: `benchmarks/cuda_batch_results_raw_rtx4070.csv`
  - GTX 1070 Ti summary: `benchmarks/cuda_batch_results_gtx1070ti.csv`
  - GTX 1070 Ti raw: `benchmarks/cuda_batch_results_raw_gtx1070ti.csv`

## Interpretation rules

- `host-visible` timing is a platform metric, not a pure GPU metric.
- `host-visible` includes:
  - CPU-side packing,
  - host-to-device transfer,
  - kernel launch,
  - device-to-host transfer.
- `kernel-only` timing isolates the resident CUDA kernel after data upload.
- Cross-host `host-visible` comparisons therefore mix host CPU effects with GPU effects.
- Cross-host `kernel-only` comparisons are the cleaner GPU-to-GPU comparison.

## Main measured outcome

At the measured batch sizes, the GPUs do **not** beat the best CPU in host-visible wall time.

However:

- kernel-only GPU time does beat the best CPU at larger batches for `d=9` and `d=27`,
- the host-visible gap shrinks as batch size increases,
- the crossover is therefore being blocked by launch/transfer/host overhead, not by device math throughput alone.

That is the core GPU result for the paper.

## CPU vs GPU crossover, using the best measured CPU time as reference

Reference CPU baseline is the best `median_ns_per_state_step` from `cpu_batch_results_intel.csv` at the same `(d, batch_size)`.

### RTX 4070 platform

- `d=3`:
  - GPU host-visible is slower than CPU at every measured batch.
  - Even kernel-only is still slower than CPU until `batch=128`, where it is still about `1.12x` slower.
- `d=9`:
  - `batch=32`: kernel-only is still slightly slower than CPU (`1.06x` CPU time ratio).
  - `batch=128`: kernel-only becomes faster than CPU (`0.86x`), but host-visible is still `2.67x` slower.
- `d=27`:
  - `batch=8`: kernel-only is faster than CPU (`0.53x`), but host-visible is `2.95x` slower.
  - `batch=32`: kernel-only is faster (`0.77x`), host-visible still `1.94x` slower.
  - `batch=128`: kernel-only is faster (`0.92x`), host-visible still `1.48x` slower.

### GTX 1070 Ti platform

- `d=3`:
  - Host-visible and kernel-only timing are both worse than CPU until the very largest batch, where kernel-only is only roughly parity.
- `d=9`:
  - `batch=32`: kernel-only is faster than CPU (`0.65x`).
  - `batch=128`: kernel-only is faster than CPU (`0.71x`).
  - host-visible remains slower than CPU throughout.
- `d=27`:
  - `batch=1`: kernel-only is already faster than CPU (`0.80x`), but host-visible is much slower.
  - `batch=8` and `batch=32`: kernel-only remains faster than CPU.
  - `batch=128`: both kernel-only and host-visible degrade relative to `batch=32`; raw trials show this is a real median effect in the current dataset, not a single outlier.

## Cross-GPU comparison

### Host-visible

- At `d=9, batch=32` and `d=9, batch=128`, the RTX 4070 platform is about `1.5x` faster than the GTX 1070 Ti platform in host-visible GFLOP/s.
- This does **not** mean the 4070 GPU itself is always faster for the kernel.
- The local host CPU is much stronger than the Ryzen 1600 desktop host, and host-visible timing includes CPU-side work.

### Kernel-only

- At `d=9, batch=32` and `d=9, batch=128`, the GTX 1070 Ti actually outperforms the RTX 4070 in the measured kernel-only GFLOP/s.
- At `d=27, batch=32`, the two are close, with the GTX still slightly ahead in the current medians.
- At `d=27, batch=128`, the RTX 4070 pulls clearly ahead, with about `2.75x` the GTX 1070 Ti kernel-only GFLOP/s.

This is an important measured result:

- a newer GPU does not automatically win on a very small dense kernel,
- the operating point depends strongly on problem size and batching,
- and host-visible platform performance can disagree with kernel-only GPU performance.

## Raw-data notes

- GTX 1070 Ti `d=27, batch=128` shows visible trial-to-trial spread in both host-visible and kernel-only timing.
- GTX 1070 Ti `d=9, batch=128` is stable across trials on the kernel-only metric and consistently faster than the RTX 4070 kernel-only metric in this implementation.
- RTX 4070 results are noticeably tighter across trials than the GTX 1070 Ti results.

## Draft paper claims supported by this dataset

- GPU acceleration is batch-size dependent in this problem regime.
- For the measured implementation, kernel-only GPU execution can outperform the best CPU before host-visible execution does.
- End-to-end platform performance and kernel-only GPU performance must be reported separately.
- A modern GPU does not automatically dominate an older GPU on these small dense kernels.

## Claims not yet supported

- Any claim that GPUs beat CPUs in full end-to-end wall time for the measured `(d, batch)` range.
- Any claim that Ada is uniformly faster than Pascal for this kernel.
- Any claim about optimized CUDA kernels beyond the current straightforward one-thread-per-output-row implementation.
