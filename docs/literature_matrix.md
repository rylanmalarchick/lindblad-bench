# Literature Matrix

Date: 2026-04-16  
Purpose: claim control for implementation and manuscript writing

## Core performance-model references

### Roofline model

- Source: Williams, Waterman, Patterson, `Roofline: An Insightful Visual Performance Model for Multicore Architectures`
- URL: https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/
- Use:
  - justify arithmetic-intensity framing,
  - justify ridge-point language,
  - justify bandwidth-vs-compute distinction.
- Safe claim:
  - Roofline is the primary analytical frame for CPU and GPU kernels.
- Do not claim:
  - that the original Roofline paper alone justifies every FPGA conclusion; FPGA discussion should also use latency and determinism framing.

## Open quantum systems software references

### QuTiP 2011

- Source: Johansson, Nation, Nori, `QuTiP: An open-source Python framework for the dynamics of open quantum systems`
- URL: https://arxiv.org/abs/1110.0573
- Use:
  - establish QuTiP as a standard baseline,
  - justify the importance of comparison against a widely used open-system toolkit.
- Safe claim:
  - QuTiP is a canonical open-source baseline for open quantum system dynamics.

### QuTiP 2 2012

- Source: Johansson, Nation, Nori, `QuTiP 2: A Python framework for the dynamics of open quantum systems`
- URL: https://arxiv.org/abs/1211.6518
- Use:
  - support discussion of time-dependent Hamiltonians and collapse operators,
  - motivate why QuTiP remains relevant as a general reference implementation.
- Safe claim:
  - QuTiP includes efficient solvers for arbitrary time-dependent Hamiltonians and collapse operators.

### QuTiP 5 matrix-form solver

- Source: QuTiP 5.3 documentation, `Lindblad Master Equation Solver`
- URL: https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-master.html
- Use:
  - constrain the fairness of the QuTiP baseline,
  - justify benchmarking both default `mesolve` and `options={"matrix_form": True}` for dense moderate-size systems.
- Safe claim:
  - QuTiP documents a matrix-form Lindblad solver as an alternative to the default superoperator formulation and states that it can be faster for denser systems.
- Manuscript implication:
  - the paper should not compare only against QuTiP's default superoperator path if `matrix_form=True` is applicable to the tested problem.

### HOQST

- Source: Chen, Lidar, `HOQST: Hamiltonian Open Quantum System Toolkit`
- URL: https://arxiv.org/abs/2011.14046
- Use:
  - show that speedups over QuTiP already exist in adjacent regimes,
  - prevent overclaiming novelty as “first open-system speedup over QuTiP.”
- Safe claim:
  - prior work has reported order-of-magnitude speedups over QuTiP for some time-dependent Hamiltonian problems.

### Dynamiqs

- Source: Dynamiqs official documentation
- URL: https://www.dynamiqs.org/stable/documentation/getting_started/whatis.html
- Use:
  - support optional GPU-software comparison discussion,
  - support statement that modern GPU-accelerated Lindblad software exists.
- Safe claim:
  - Dynamiqs is a JAX-based library for GPU-accelerated and differentiable quantum simulation with Lindblad solvers.
- Caution:
  - if used as a benchmark, version and backend must be recorded carefully because the software is still under active development.

## Lindblad-performance references

### Meyerov et al. 2019

- Source: `Transforming the Lindblad Equation into a System of Linear Equations: Performance Optimization and Parallelization of an Algorithm`
- URL: https://arxiv.org/abs/1912.01491
- Use:
  - show there is prior Lindblad performance work,
  - distinguish current paper by problem regime and architecture mix.
- Safe claim:
  - prior work studies parallel cluster-scale implementation of Lindblad simulation in a transformed real-ODE setting.
- Do not claim:
  - that no prior Lindblad performance work exists.

## Physics context references

### Transmon reference

- Source: Koch et al., `Charge insensitive qubit design derived from the Cooper pair box`
- URL: https://arxiv.org/abs/cond-mat/0703002
- Use:
  - justify transmon modeling context,
  - justify leakage-aware low-dimensional Hilbert-space discussion.
- Safe claim:
  - transmons achieve reduced charge-noise sensitivity while retaining enough anharmonicity for selective control.

### GRAPE context

- Source to cite in manuscript: Khaneja et al. 2005
- Current note:
  - not yet pinned to a browsed primary-source page in this planning pass.
- Use:
  - motivate gradient-based optimal control loops as application context.
- Action:
  - verify and pin exact primary-source citation before final manuscript writing.

## FPGA latency references

### FPGA feedback platform

- Source: Gebauer et al., `State preparation of a fluxonium qubit with feedback from a custom FPGA-based platform`
- URL: https://arxiv.org/abs/1912.06814
- Use:
  - justify sub-microsecond and low-microsecond control latency relevance.
- Safe claim:
  - the paper reports `428 ns` platform latency and a roughly `1.5 us` readout-and-drive reset sequence.

### FPGA control/readout electronics

- Source: Yang et al., `FPGA-based electronic system for the control and readout of superconducting quantum processors`
- URL: https://arxiv.org/abs/2110.07965
- Use:
  - justify that low-latency FPGA feedback/control electronics are an active and relevant systems target.
- Safe claim:
  - the paper reports `125 ns` feedback/feedforward latency and approximately `5 ps` jitter in synchronous control.

### Real-time FPGA decoding

- Source: Caune et al., `Demonstrating real-time and low-latency quantum error correction with superconducting qubits`
- URL: https://arxiv.org/abs/2410.05202
- Use:
  - justify why deterministic sub-microsecond to few-microsecond classical latency matters in superconducting-qubit experiments.
- Safe claim:
  - the paper reports mean decoding time per round below `1 us`.

## Hardware references

### Tang Nano 20K

- Source: Sipeed wiki
- URL: https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html
- Use:
  - cite board and FPGA-chip facts,
  - cite LUT, BSRAM, DSP, PLL counts,
  - cite onboard BL616 debugger / UART / JTAG path.

### Tang Nano 20K datasheet

- Source: Sipeed datasheet PDF
- URL: https://dl.sipeed.com/fileList/TANG/Nano_20K/1_Datasheet/Sipeed%20Tang%20nano%2020K%20Datasheet%20V1.3-en_US.pdf
- Use:
  - cross-check board specs and I/O path,
  - support citations for BSRAM and multiplier counts.

## Venue references

### JCP scope

- Source: Elsevier JCP page
- URL: https://shop.elsevier.com/journals/journal-of-computational-physics/0021-9991
- Use:
  - keep manuscript framed around comparison, reproducibility, robustness, and computational complexity.
- Safe claim:
  - JCP asks for comparison when addressing problems previously covered by other approaches.

### CPC scope

- Source: Elsevier computational physics journals page
- URL: https://shop.elsevier.com/journals/subjects/physical-sciences-and-engineering/physics/interdisciplinary-physics/computational-physics
- Use:
  - explain why the prior submission was desk-rejected on scope grounds.

## Claim-control summary

### Claims currently supported

- Small dense Lindblad propagation can be framed naturally with Roofline analysis.
- QuTiP is a legitimate baseline.
- GPU-accelerated Lindblad-capable software exists.
- Prior Lindblad performance work exists, but in a different regime.
- FPGA latency is relevant to superconducting-qubit feedback systems.
- Tang Nano 20K hardware facts can be cited from official sources.

### Claims that still need more support before manuscript submission

- Exact novelty sentence for “first” or “to our knowledge” wording.
- Exact GRAPE primary citation and any end-to-end control-theory framing.
- Any quantitative statement comparing our eventual results directly against Dynamiqs or HOQST.
- Any broad claim that our conclusions extend beyond the `d = 3, 9, 27` dense transmon-control regime.
- Any claim that a QuTiP comparison is fair without checking both default `mesolve` and matrix-form `mesolve` where the latter is applicable.
