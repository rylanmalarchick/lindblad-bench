# Claim Audit

Date: 2026-04-16  
Scope: manuscript claim control for `paper/main_jcp.tex`

Purpose: keep synthetic or drifted claims out of the paper by tying each important statement to a measured artifact.

## Direct measurement claims

### CPU

Claim:

- i9 peak CPU throughput reaches `87.8 GFLOP/s` at `d=9` and `159.3 GFLOP/s` at `d=27`
- Ryzen peak CPU throughput reaches `58.7 GFLOP/s` at `d=9` and `47.2 GFLOP/s` at `d=27`
- best measured single-state `d=3` latencies are `309.5 ns` on the i9 and `457.6 ns` on the Ryzen
- the main i9 OpenMP sweep used `OMP_PROC_BIND=true` and `OMP_PLACES=cores`
- auxiliary `taskset` checks show E-core-only placement is materially weaker for larger kernels, while default placement stays within `4%` of P-core-only at the tested `d=27`, `16`-thread point

Source:

- `benchmarks/cpu_batch_results_intel.csv`
- `benchmarks/cpu_batch_results_ryzen.csv`
- `paper/generated/table_cpu_summary.tex`
- `benchmarks/pinning_i9_aux_data.csv`
- `docs/i9_pinning_note.md`

Status:

- direct measurement

### GPU

Claim:

- kernel-only GPU timing crosses the best CPU at larger batches for `d=9` and `d=27`
- host-visible GPU timing never crosses the best CPU in the measured range
- the empirical Roofline figure uses peak observed throughput/bandwidth on each measured CPU and GPU rather than vendor-spec peaks

Source:

- `benchmarks/cuda_batch_results_rtx4070.csv`
- `benchmarks/cuda_batch_results_gtx1070ti.csv`
- `benchmarks/cpu_batch_results_intel.csv`
- `benchmarks/cpu_batch_results_ryzen.csv`
- `paper/figures_jcp/figure_gpu_cpu_ratio.pdf`
- `paper/figures_jcp/figure_roofline_empirical.pdf`
- `analysis/make_jcp_assets.py`
- `docs/cuda_batch_results.md`

Status:

- direct measurement for the tested `(d, batch)` grid
- should not be generalized beyond the measured implementation and batch sizes
- the plotted ceilings are a derived visualization of the measured data, not direct hardware limits

### FPGA

Claim:

- the Tang Nano `d=3` kernel executes in `95 cycles` at `135 MHz`
- this corresponds to `703.7 ns` single-step latency
- the measured cycle-count spread is `0 cycles` over `500` trials
- multi-step behavior is consistent with `94n + 1` cycles

Source:

- `/home/rylan/dev/projects/tang-nano-20k/analysis/results/benchmark_compare_500.log`
- `docs/fpga_results.md`
- `paper/generated/table_fpga_summary.tex`

Status:

- `95 cycles`, `703.7 ns`, and `0 cycles over 500 trials` are direct measurements
- `94n + 1` is an inference from the measured sequence `95, 941, 9401, 94001`

### QuTiP release state

Claim:

- the measured released baseline is QuTiP `5.2.3`
- matrix-form `mesolve` was not available in that released environment

Source:

- `reference/reference.py`
- `reference/test_reference.py`
- `docs/literature_matrix.md`
- `docs/qutip_grape_results.md`

Status:

- direct environment fact for the tested machines

### GRAPE end-to-end benchmark

Claim:

- optimized C beats released QuTiP at `d=3` and `d=9`
- optimized C still loses to released QuTiP at `d=27`
- the updated speed ratios are:
  - i9: `153x`, `1.9x`, `4.0x`
  - Ryzen: `277x`, `4.1x`, `4.3x`

Source:

- `benchmarks/grape_c_intel_opt.log`
- `benchmarks/grape_c_ryzen_opt.log`
- `benchmarks/qutip_grape_results_intel.csv`
- `benchmarks/qutip_grape_results_ryzen.csv`
- `paper/generated/table_grape_summary.tex`

Status:

- direct measurement

## Implementation claims

Claim:

- the GRAPE benchmark now reuses invariant superoperator pieces and reusable Pad\'e scratch

Source:

- `include/lindblad_bench.h`
- `src/lindblad.c`
- `src/expm.c`
- `src/propagate.c`
- `benchmarks/bench_grape.c`
- `tests/test_grape_build.c`

Status:

- direct implementation fact

Claim:

- the reusable construction path matches the baseline numerically

Source:

- `tests/test_grape_build.c`
- local `ctest` pass in `build_jcp`
- desktop `ctest` pass in `~/dev/research/lindblad-bench/build_jcp`

Status:

- direct test-backed claim

## Bounded interpretation claims

Claim:

- small dense Lindblad propagation splits into CPU, GPU, and FPGA operating regimes

Source:

- synthesis of CPU, GPU, FPGA, and GRAPE measurements

Status:

- interpretation, but tightly bounded by the measured hardware and tested problem sizes

Claim:

- propagator construction is the next bottleneck for the current `d=27` control workload

Source:

- `benchmarks/grape_c_*_opt.log`
- `paper/figures_jcp/figure_grape_breakdown.pdf`

Status:

- interpretation from direct timing breakdown
- this is specific to the current implementation, tested schedule, and dense propagator strategy

Claim:

- the FPGA is useful as a deterministic-latency architecture rather than a throughput accelerator

Source:

- FPGA timing log
- CPU `d=3` single-state and batched comparisons
- jitter cross-check note in `docs/fpga_results.md`

Status:

- interpretation, but strongly supported by the measured cycle-exact repeatability and the measured CPU speed advantage

## Claims intentionally avoided in the manuscript

- No claim that GPU beats CPU in full end-to-end wall time for the measured range
- No claim that the RTX 4070 uniformly beats the GTX 1070 Ti
- No claim that the FPGA beats the CPU in throughput
- No claim that released QuTiP matrix-form `mesolve` was benchmarked
- No claim that the current C implementation accelerates GRAPE at all tested dimensions
- No claim that the local i9 exposes AVX-512

## Final audit conclusion

The current manuscript claims are traceable to measured artifacts in the repo.

The main places where interpretation occurs are clearly bounded:

- `94n + 1` FPGA behavior is an inference from measured cycle counts
- operating-regime language is a synthesis of direct timing results
- the `d=27` bottleneck claim is specific to the current dense-propagator implementation

No central manuscript claim currently depends on invented data, unsupported hardware features, or unreleased software behavior.
