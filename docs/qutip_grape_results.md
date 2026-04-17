# QuTiP and GRAPE Results Note

Date: 2026-04-16  
Scope: released QuTiP baseline status, microbenchmark results, and the optimized GRAPE-style application benchmark

## Released QuTiP baseline status

Environment facts established during testing:

- Installed release via `uv` on both hosts: `QuTiP 5.2.3`
- The harness probe for `options={"matrix_form": True}` returns unsupported on this released version
- Relevant unit tests:
  - `reference/test_reference.py`
  - `reference/test_grape_reference.py`

Implication for the paper:

- The fair released-software baseline is QuTiP `5.2.3` default `mesolve`
- The latest QuTiP documentation describes a matrix-form solver, but that path was not available in the released `5.2.3` package tested here
- The manuscript should therefore distinguish between released-baseline measurement and forward-looking discussion of documented-but-unreleased solver modes

## QuTiP microbenchmark results

Artifacts:

- Intel: `/home/rylan/dev/projects/lindblad-bench/benchmarks/qutip_results_intel.csv`
- Ryzen: `/home/rylan/dev/projects/lindblad-bench/benchmarks/qutip_results_ryzen.csv`

Measured QuTiP `mesolve` single-step timings:

### i9-13980HX

- `d=3`: `1.032 ms/step`
- `d=9`: `1.774 ms/step`
- `d=27`: `106.627 ms/step`

### Ryzen 5 1600

- `d=3`: `1.205 ms/step`
- `d=9`: `2.932 ms/step`
- `d=27`: `200.033 ms/step`

High-level interpretation:

- Released QuTiP is orders of magnitude slower than the optimized C propagation kernel in the isolated small-dense propagation regime
- This is a kernel result only; it does not by itself predict the end-to-end GRAPE result

## QuTiP GRAPE-style benchmark harness

Harness:

- `/home/rylan/dev/projects/lindblad-bench/reference/grape_reference.py`

Tests:

- `/home/rylan/dev/projects/lindblad-bench/reference/test_grape_reference.py`

What it measures:

- One GRAPE-like landscape point as a piecewise-constant control sequence
- Deterministic drive schedule from a fixed seed
- Segment-by-segment constant-H QuTiP evolution
- End-to-end time per trajectory plus equivalent `ns/step`

Output artifacts:

- Intel: `/home/rylan/dev/projects/lindblad-bench/benchmarks/qutip_grape_results_intel.csv`
- Ryzen: `/home/rylan/dev/projects/lindblad-bench/benchmarks/qutip_grape_results_ryzen.csv`

## Optimized C GRAPE benchmark

The current benchmark is not the original naive version. It now includes two real implementation improvements:

- reusable Lindbladian decomposition:
  - static coherent term,
  - drive coherent term,
  - dissipative term
- reusable Pad\'e matrix-exponential scratch space across segment builds

Validation:

- `tests/test_grape_build.c`
- local `ctest` passed
- desktop `ctest` passed after syncing the mirror to the current source state

Current benchmark logs:

- Intel optimized run: `/home/rylan/dev/projects/lindblad-bench/benchmarks/grape_c_intel_opt.log`
- Ryzen optimized run: `/home/rylan/dev/projects/lindblad-bench/benchmarks/grape_c_ryzen_opt.log`

These replace the older `run1` / `run2` GRAPE numbers as the manuscript source of truth.

## End-to-end GRAPE comparison

### i9-13980HX

- QuTiP:
  - `d=3`: `107.388 ms/trajectory`
  - `d=9`: `101.792 ms/trajectory`
  - `d=27`: `1903.071 ms/trajectory`
- Optimized C:
  - `d=3`: `0.700 ms/landscape point`
  - `d=9`: `54.561 ms/landscape point`
  - `d=27`: `7697.947 ms/landscape point`

Result:

- `d=3`: C is about `153x` faster than QuTiP
- `d=9`: C is about `1.9x` faster than QuTiP
- `d=27`: QuTiP is about `4.0x` faster than C

### Ryzen 5 1600

- QuTiP:
  - `d=3`: `148.607 ms/trajectory`
  - `d=9`: `189.568 ms/trajectory`
  - `d=27`: `3457.720 ms/trajectory`
- Optimized C:
  - `d=3`: `0.537 ms/landscape point`
  - `d=9`: `46.338 ms/landscape point`
  - `d=27`: `14724.661 ms/landscape point`

Result:

- `d=3`: C is about `277x` faster than QuTiP
- `d=9`: C is about `4.1x` faster than QuTiP
- `d=27`: QuTiP is about `4.3x` faster than C

## What changed after the optimization

Compared to the older benchmark state:

- `d=3` improved substantially
- `d=9` improved substantially
- `d=27` improved, but not enough to change the sign of the comparison

This is exactly the important application-level finding:

- the GRAPE result became stronger where repeated construction overhead was avoidable,
- but the largest tested case is still dominated by propagator construction rather than propagation,
- so the optimization sharpened the thesis instead of undermining it

## Paper-safe interpretation

Safe:

- Released QuTiP `5.2.3` is much slower than the optimized C propagation kernel in isolated propagation measurements
- In the optimized GRAPE-style benchmark, C wins strongly at `d=3`, wins clearly at `d=9`, and still loses at `d=27`
- The `d=27` loss persists even after removing avoidable repeated-construction overhead through decomposition and workspace reuse
- The application benchmark therefore identifies propagator construction as the next bottleneck

Avoid:

- Do not claim the current C implementation universally accelerates GRAPE for all tested dimensions
- Do not imply that fast propagation-step timings alone predict end-to-end control-runtime improvement
- Do not imply that QuTiP matrix-form `mesolve` was measured in the released environment when it was not available in `5.2.3`

## Manuscript implication

The GRAPE section should cover all three dimensions:

- `d=3`: strong application-level C win
- `d=9`: moderate but real application-level C win
- `d=27`: honest loss for C, explained by propagator construction remaining dominant

That all-three-d presentation is now the correct and defensible thesis for the application section.
