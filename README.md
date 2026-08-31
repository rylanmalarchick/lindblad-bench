# lindblad-bench

Benchmarks and cost models for small dense Lindblad propagation on CPUs, GPUs,
and an FPGA. The sizes are the ones near-term transmon control uses: d = 3, 9,
and 27.

Preprint: https://arxiv.org/abs/2603.18052

## What the repository contains

- A C11 library for dense Lindbladian construction, a Pade [13/13] matrix
  exponential, and propagation (`src/`, `include/`).
- Timing benchmarks for the propagation step, batched OpenMP propagation, a
  CUDA kernel, a GRAPE-style application benchmark, and bandwidth and launch
  probes (`benchmarks/`, `cuda/`).
- A QuTiP reference implementation and the QuTiP side of the application
  benchmark (`reference/`).
- Raw per-trial timing data as CSV files, the FPGA hardware logs, and a
  fixed-point accuracy study (`benchmarks/`, `benchmarks/fpga/`).
- Analysis scripts that turn the raw data into figures, tables, and the
  idealized cost models (`analysis/`).
- A CPython extension with NumPy bindings and a QuTiP cross-validation test
  (`python/`).
- Correctness tests for the C library (`tests/`).

The FPGA design itself lives in a separate repository. This repository holds
its measurement logs.

## Background

The Lindblad master equation gives the evolution of an open quantum system:

```
dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
```

For a d-level system, ρ is a d×d complex matrix. The superoperator L acts on
the vectorized density matrix vec(ρ) of length d². One propagation step is
then a dense (d²)×(d²) complex matrix-vector product. At d = 3 (one transmon)
the product is 9×9. At d = 9 (two transmons) it is 81×81. At d = 27 (three
transmons with leakage) it is 729×729.

The working sets are about 1.5 KB, 105 KB, and 8.3 MB. These sizes sit on the
L1, L2, and L3 boundaries of current CPUs. At these sizes cache placement,
thread fork/join cost, kernel launch cost, and transfer latency decide the
result, not peak arithmetic rate.

## Build

```bash
# Default build: -O3 -march=native, auto-vectorized
cmake -B build && cmake --build build

# Scalar build, for assembly comparison
cmake -B build-scalar -DVECTORIZE=OFF && cmake --build build-scalar

# OpenMP build, for the batched benchmarks and the stream probe
cmake -B build-omp -DENABLE_OPENMP=ON && cmake --build build-omp

# CUDA build, for the GPU benchmark and the device probe
cmake -B build-cuda -DENABLE_CUDA=ON && cmake --build build-cuda
```

Run the tests:

```bash
cd build && ctest --output-on-failure
```

## Run the benchmarks

Each script writes a host-tagged CSV file under `benchmarks/`. Set `HOST_TAG`
to name the host.

```bash
bash analysis/omp_scaling.sh      # CPU thread and batch sweep
bash analysis/cuda_sweep.sh       # GPU sweep, three timing models
bash analysis/grape_bench.sh      # C side of the GRAPE-style benchmark
bash analysis/stream_probe.sh     # read bandwidth and fork/join ceilings
./build-cuda/bench_cuda_probe --csv   # launch, PCIe, and gather ceilings
```

The QuTiP side of the application benchmark needs `qutip==5.2.3`:

```bash
uv venv .venv-qutip --python 3.12
uv pip install --python .venv-qutip/bin/python "qutip==5.2.3" numpy scipy pandas matplotlib
.venv-qutip/bin/python reference/grape_reference.py --bench-all --solver-mode all
```

## Regenerate figures and tables

```bash
.venv-qutip/bin/python analysis/make_jcp_assets.py   # figures and tables from the CSVs
.venv-qutip/bin/python analysis/cost_model.py        # idealized vs measured cost tables
.venv-qutip/bin/python analysis/fixed_point_accuracy.py   # Q1.15 and Q1.31 drift study
```

## Cross-validation against QuTiP

The Python extension links the C library into NumPy. Build the static library
position-independent first:

```bash
cmake -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON && cmake --build build
pip install numpy qutip pytest
pip install -e python/
pytest python/test_crossval.py -m crossval
```

The test compares `lb_evolve` with `qutip.mesolve` and with the exact QuTiP
propagator.

## Repository layout

```
src/            C library: expm, lindblad, propagate, evolve
include/        Public API header
benchmarks/     Timing harnesses, raw CSV data, FPGA logs
cuda/           CUDA kernel, wrapper, and device probe
reference/      QuTiP reference and GRAPE-style benchmark
analysis/       Sweep scripts, figure and table generation, cost model
godbolt/        Compiler Explorer permalink catalog
python/         CPython extension and QuTiP cross-validation
tests/          Correctness tests
docs/           Benchmark result notes
```

## Dependencies

C library: C11, POSIX, `-lm`. No external dependencies.

OpenMP and CUDA builds: a compiler with OpenMP support, and CUDA 12 with an
NVIDIA driver.

Python analysis:

```bash
pip install -r reference/requirements.txt
```

## Citation

```bibtex
@misc{malarchick2026lindblad,
  title  = {Cache Hierarchy and Vectorization Analysis of Lindblad Master Equation
             Simulation for Near-Term Quantum Control},
  author = {Malarchick, Rylan},
  year   = {2026},
  note   = {arXiv preprint, arXiv:2603.18052}
}
```

## License

MIT
