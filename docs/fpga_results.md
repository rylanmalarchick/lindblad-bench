# FPGA Results Note

Date: 2026-04-16  
Scope: Tang Nano 20K `d=3` Lindblad propagator hardware results for JCP manuscript integration

## Build and provenance

- FPGA repo: `/home/rylan/dev/projects/tang-nano-20k`
- Bitstream used for hardware measurements: `matvec_pll_top`
- Actual programmed clock: `135 MHz`
- Permanent logs:
  - P&R/resource log: `/home/rylan/dev/projects/tang-nano-20k/analysis/results/matvec_pll_top_pack.log`
  - Functional validation log: `/home/rylan/dev/projects/tang-nano-20k/analysis/results/matvec_host_validation.log`
  - Hardware timing log: `/home/rylan/dev/projects/tang-nano-20k/analysis/results/benchmark_compare_500.log`
  - CPU jitter summary: `/home/rylan/dev/projects/tang-nano-20k/analysis/results/c_jitter_summary.log`

## Post-route implementation numbers

From `matvec_pll_top_pack.log`:

- `LUT4`: `1357 / 20736` (`6%`)
- `DFF`: `1114 / 15552` (`7%`)
- `RAM16SDP4`: `72 / 648` (`11%`)
- `MULT18X18`: `4 / 48` (`8%`)
- `rPLL`: `1 / 2`
- `IOB`: `11 / 384`
- Post-route max frequency for `clk_fast`: `206.06 MHz`

Paper-safe interpretation:

- The design is small relative to device capacity.
- The demonstrated clock in hardware was `135 MHz`; `206.06 MHz` is a post-route timing limit, not a measured deployed operating point.

## Functional validation

From `matvec_host_validation.log`:

- The `d=3` propagator loaded successfully over UART.
- `20` successive single-step FPGA updates matched the Python Q1.15 reference bit-exactly.
- A `20`-step `N` command also matched the Q1.15 reference bit-exactly.

Paper-safe statement:

- The programmed FPGA kernel reproduces the fixed-point reference trajectory exactly for the tested `d=3` transmon propagator.

## Hardware timing

From `benchmark_compare_500.log`:

- `500` single-step trials produced `Unique counts: [95]`
- Single-step measured counter value: `95 cycles`
- Multi-step counter values:
  - `1` step: `95 cycles`
  - `10` steps: `941 cycles`
  - `100` steps: `9401 cycles`
  - `1000` steps: `94001 cycles`

At the actual programmed `135 MHz` clock:

- Single-step command latency: `95 / 135 MHz = 703.7 ns`
- Steady-state kernel latency inferred from the large-`N` limit: `94 / 135 MHz = 696.3 ns`

Inference from the measured sequence `95, 941, 9401, 94001`:

- The hardware behaves like `94n + 1` cycles for an `N`-step request.
- The extra `+1` cycle is best interpreted as command/setup overhead around a `94`-cycle matvec pipeline.

This is an inference from the measured counts, not a directly instrumented internal signal trace.

## Jitter and transport

Measured FPGA counter behavior:

- Cycle-count spread across `500` single-step trials: `0 cycles`
- This demonstrates cycle-exact repeatability for the on-chip kernel under the tested command path.

Host-visible wall time:

- Single-step wall time over USB/UART: approximately `6.0 ms`
- The UART return payload alone is about `3.125 ms` at `115200` baud for `36` bytes with `8N1`
- Therefore the host-visible measurement is transport-dominated, not compute-dominated

Paper-safe statement:

- The result to emphasize is deterministic on-chip kernel latency, not USB-attached wall-clock latency.
- Any real-time-control claim should assume integration into a local FPGA control path rather than the current USB/UART lab interface.

## CPU comparison at `d=3`

Comparison data source:

- `/home/rylan/dev/projects/tang-nano-20k/analysis/results/benchmark_compare_500.log`
- CPU datasets originate from `/home/rylan/dev/projects/lindblad-bench/benchmarks/cpu_batch_results_intel.csv`
- CPU datasets originate from `/home/rylan/dev/projects/lindblad-bench/benchmarks/cpu_batch_results_ryzen.csv`

Latency-fair comparison (`batch=1` or single-state path):

- i9-13980HX best single-state CPU result: `309.5 ns/step`
- Ryzen 5 1600 best single-state CPU result: `457.6 ns/step`
- FPGA single-step command result: `703.7 ns/step`

Ratios:

- i9 single-state CPU is about `2.3x` faster than the FPGA single-step path
- Ryzen single-state CPU is about `1.5x` faster than the FPGA single-step path

Throughput-optimized batched CPU comparison:

- i9 best batched CPU result: `44.7 ns/state-step`
- Ryzen best batched CPU result: `24.7 ns/state-step`

Paper-safe interpretation:

- This FPGA design is not a throughput winner against either CPU.
- Its value proposition is deterministic fixed latency and hardware-embedded execution, not higher state-step throughput.

## CPU jitter cross-check

From `c_jitter_summary.log` on the local machine:

- `10000` trials of a userspace 9x9 complex matvec
- Mean: `41.9 ns`
- Standard deviation: `94.1 ns`
- Min: `38.0 ns`
- Max: `9449.0 ns`
- Range: `9411.0 ns`

Caution:

- This is a wall-clock userspace microbenchmark, so measurement overhead and OS scheduling noise are part of the result.
- It should be used as qualitative evidence that software timing is not cycle-exact under Linux userspace, not as a direct apples-to-apples throughput comparison with the FPGA hardware counter.

## Manuscript integration guidance

Recommended claims:

- The Tang Nano implementation establishes a deterministic-latency operating point for `d=3` Lindblad propagation.
- The hardware counter shows zero trial-to-trial spread across `500` single-step trials.
- The deployed `135 MHz` design executes a single command in `95` cycles and asymptotically `94` cycles per step for multi-step requests.
- CPUs remain faster in both single-state latency and batched throughput for this kernel.
- The FPGA result is therefore best positioned as a complementary architecture for embedded feedback/control paths rather than a general-purpose acceleration replacement for CPU simulation.

Claims to avoid:

- Do not claim the FPGA beats the CPU in throughput.
- Do not present USB/UART wall time as the control-loop latency of the kernel itself.
- Do not present `206.06 MHz` as an achieved experimental clock.
- Do not generalize beyond the tested `d=3` fixed-point kernel without additional measurements.
