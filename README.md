# lindblad-bench

**Cache hierarchy and vectorization analysis of Lindblad master equation simulation for near-term quantum control.**

https://arxiv.org/abs/2603.18052

## Overview

For open quantum system simulation at system sizes d = 3–27 — the regime relevant to near-term quantum control (single- and few-qubit transmon gates) — the dominant performance bottleneck is **memory bandwidth at cache hierarchy boundaries**, not arithmetic throughput. This repository provides:

- A bare-metal C implementation of the Lindblad propagator (`src/`)
- A Roofline model characterization across d = 3, 9, 27 (`analysis/roofline.py`)
- Cache-pressure analysis: working set sizes vs L1/L2/L3 boundaries (`analysis/cache_math.py`)
- Compiler-generated assembly analysis at multiple flag levels (`godbolt/`)
- QuTiP reference implementation for validation and baseline comparison (`reference/`)
- Timing benchmarks comparing scalar C, auto-vectorized C, and QuTiP (`benchmarks/`)

## Background

The Lindblad master equation governs the evolution of an open quantum system:

```
dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
```

For a d-level system, ρ is a d×d complex matrix. The superoperator L acts on
the vectorized density matrix vec(ρ) ∈ ℂ^(d²), making each propagation step
a (d²)×(d²) matrix-vector multiply. At d=3 (single transmon), this is a 9×9
system. At d=9 (two-qubit), 81×81. At d=27 (three-qubit with leakage), 729×729.

The working set sizes — 1.3 KB, 105 KB, 8.5 MB — straddle the L1/L2/L3
boundaries of modern CPUs, making this an ideal case study in cache-hierarchy
performance analysis.

## Build

```bash
# Auto-vectorized build (default)
cmake -B build && cmake --build build

# Scalar-only build (for assembly comparison)
cmake -B build-scalar -DVECTORIZE=OFF && cmake --build build-scalar

# Run tests
cd build && ctest --output-on-failure

# Run a benchmark
./build/bench_propagate
```

## Repository Structure

```
src/            Bare-metal C library (expm, lindblad, propagate, evolve)
include/        Public API header
benchmarks/     Timing harness (POSIX clock_gettime)
reference/      QuTiP reference implementation
analysis/       Cache math, Roofline model, speedup plots
godbolt/        Compiler Explorer permalink catalog
python/         CPython C extension (zero-copy NumPy interface)
tests/          Correctness tests (trace preservation, unitarity)
paper/          LaTeX source
```

## Dependencies

**C library:** C11, POSIX, `-lm`. No external dependencies.

**Python analysis:**
```bash
pip install -r reference/requirements.txt
```

**Paper:** LaTeX with `revtex4-2` (PRA style) or `IEEEtran`.

## Citation

```bibtex
@misc{malarchick2026lindblad,
  title  = {Cache Hierarchy and Vectorization Analysis of Lindblad Master Equation
             Simulation for Near-Term Quantum Control},
  author = {Malarchick, Rylan},
  year   = {2026},
  note   = {arXiv preprint, arXiv:XXXX.XXXXX}
}
```

## License

MIT
