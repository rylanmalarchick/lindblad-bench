# JCP Agent Blueprint

Working title: `Lindblad Propagation Across CPU, GPU, and FPGA for Near-Term Quantum Control`

Status: research-and-planning document for execution  
Date: 2026-04-16  
Primary repo: `/home/rylan/dev/projects/lindblad-bench`  
Secondary repo: `/home/rylan/dev/projects/tang-nano-20k`

## 1. Purpose

This document is the working blueprint for turning the current `lindblad-bench` preprint and `tang-nano-20k` FPGA work into a full `Journal of Computational Physics` paper.

It is not a manuscript draft. It is the agent-facing plan that should govern:

- what the paper is allowed to claim,
- what experiments are mandatory,
- what comparisons are optional,
- what current repo assets already exist,
- what technical traps to avoid while implementing the next phase.

The central requirement is to ship a comparison-driven JCP paper, not a software-package paper for CPC.

## 2. Target Thesis

### 2.1 One-sentence thesis

For the small, dense Lindblad propagation kernels that arise in near-term transmon control, peak FLOP/s is not the main decision variable; the correct hardware choice depends on operating regime:

- cache-resident CPU execution for low-overhead dense kernels,
- batched GPU execution when enough trajectories or landscape points amortize launch and transfer overheads,
- FPGA execution when deterministic low latency and jitter matter more than raw throughput.

### 2.2 Strong version of the claim

The paper should aim to show that fixed-size dense Lindblad propagation at `d = 3, 9, 27` occupies a regime that is poorly summarized by generic “GPU acceleration” or “HPC speedup” claims. The relevant questions are:

- which memory hierarchy level the propagator occupies,
- whether a given architecture can keep the kernel resident there,
- whether launch, scheduling, and transfer overhead dominate actual application time,
- whether the application cares about throughput, latency, or determinism.

### 2.3 What the paper should not claim

Do not claim:

- a universal best architecture for all Lindblad simulations,
- that FPGA throughput beats GPU or CPU in general,
- that GPU is bad for Lindblad simulation in general,
- that AVX-512 is part of the main result set on currently available hardware,
- that the results extend to large sparse many-body Lindbladians without qualification.

## 3. Why JCP, Not CPC

### 3.1 CPC outcome

The CPC desk rejection on `2026-04-13` was a scope rejection, not a quality rejection. The decision email explicitly states that the manuscript was outside CPC’s normal scope and that this was not a reflection on quality.

### 3.2 JCP fit

JCP’s current scope explicitly favors:

- original computational contributions,
- cross-disciplinary methods,
- comparisons against prior approaches when revisiting existing problem classes,
- efficacy, robustness, computational complexity, and reproducibility.

This is exactly the right framing for a paper that compares the same Lindblad kernel across:

- old x86 CPU,
- modern x86 CPU,
- old NVIDIA GPU,
- modern NVIDIA GPU,
- FPGA latency point.

### 3.3 Practical submission implication

The manuscript has to read as a computational physics study with comparative evidence, not as:

- a short microbenchmark note,
- a software release announcement,
- a pure architecture hobby project,
- or an FPGA demo paper disconnected from physics use.

## 4. Source-Backed Positioning

This section captures the external research position the paper should stand on.

### 4.1 Roofline foundation

- Williams, Waterman, Patterson, `Roofline: An Insightful Visual Performance Model for Multicore Architectures` establishes the performance model that ties arithmetic intensity to bandwidth and peak compute.
- This is the right backbone for the CPU and GPU sections.
- For FPGA, the paper should use Roofline language carefully, but the stronger framing is latency-throughput-determinism rather than pure FLOP efficiency.

### 4.2 Open quantum systems software baselines

- QuTiP 2011 and QuTiP 2 2012 establish QuTiP as a standard open-source framework for open quantum system dynamics, including master-equation solvers and time-dependent support.
- Dynamiqs’ official documentation states that it is GPU-accelerated, differentiable, JAX-based, and supports the Lindblad master equation.
- HOQST reports order-of-magnitude speedups versus QuTiP for certain time-dependent Hamiltonian problems, which matters because it shows software-level speedups over QuTiP already exist in adjacent regimes.

Implication:

- QuTiP is the mandatory baseline.
- Dynamiqs is a strong optional software comparison for the GPU discussion, but not required for the first pass if it risks derailing the core execution plan.
- The paper must clearly distinguish the present problem from general-purpose time-dependent or large-scale toolkit comparisons.

### 4.3 Existing Lindblad HPC literature

- Meyerov et al. 2019 study performance optimization and parallelization for Lindblad equations after transforming them into real ODE systems, with cluster-scale sparse and dense examples.

Implication:

- There is prior performance work on Lindblad simulation.
- The novelty here should not be stated as “first Lindblad performance paper.”
- The novelty should be stated as a different regime:
  - small dense propagators,
  - near-term transmon control sizes,
  - hardware comparison across CPU, GPU, and FPGA,
  - unified application framing via GRAPE and feedback latency.

### 4.4 Physics context

- Koch et al. 2007 remains the standard transmon reference.
- The `d = 3` single-transmon-with-leakage framing is physically reasonable and anchored in real device modeling.

### 4.5 FPGA control latency relevance

Three papers matter for the motivation:

- Gebauer et al. 2019 report `428 ns` platform latency for FPGA-based closed-loop feedback with a fluxonium qubit.
- Yang et al. 2021 report `125 ns` feedback/feedforward latency in an FPGA-based electronic system for superconducting processors.
- Caune et al. 2024 report mean decoding time per round below `1 us` in real-time FPGA-assisted superconducting-qubit experiments.

Implication:

- The FPGA section does not need to win on throughput.
- It only needs to occupy a credible low-latency deterministic operating point relevant to real-time quantum control and feedback.

### 4.6 Literature gap statement to use carefully

Safe claim:

> Based on the literature reviewed for this project, we did not identify a study that systematically characterizes fixed-size dense Lindblad propagation in the near-term transmon regime across legacy CPU, modern CPU, legacy GPU, modern GPU, and FPGA under one unified Roofline-plus-application framework.

Do not overstate this into an absolute novelty claim without further literature checking during manuscript drafting.

## 5. Current Local Reality

This section supersedes earlier assumptions.

### 5.1 Local machine

Host: `theLittleMachine`

- CPU: `13th Gen Intel Core i9-13980HX`
- Flags visible in `lscpu`: `AVX2`, `FMA`, no exposed `AVX-512`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- CUDA stack present: `nvcc` installed
- GPU compute capability: `8.9`

Implication:

- This is the modern CPU + modern GPU host.
- AVX-512 is not currently part of the runnable plan.

### 5.2 Desktop host over SSH

Host alias: `desktop`  
Resolved host: `theMachine`

- CPU: `AMD Ryzen 5 1600`, `6C/12T`, `AVX2`
- GPU: `NVIDIA GeForce GTX 1070 Ti`
- CUDA stack present: `nvcc` installed
- GPU compute capability: `6.1`

Implication:

- This is the legacy CPU + legacy GPU host.
- The earlier `Ryzen 5 1500X` assumption is incorrect and should not appear in the paper.

### 5.3 FPGA board

Local observations:

- `openFPGALoader --detect` sees a Gowin `GW2A(R)-18(C)` device.
- `/dev/ttyUSB0` and `/dev/ttyUSB1` are present.
- Both USB serial interfaces identify as `SIPEED USB_Debugger`.

Implication:

- The Tang Nano 20K board is physically connected and reachable.
- FPGA execution is not hypothetical; it can be included in the paper plan now.

## 6. Repo Audit

### 6.1 `lindblad-bench` already contains

- `benchmarks/bench_propagate.c`
- `benchmarks/bench_grape.c`
- `src/propagate.c` with:
  - AoS path,
  - SoA path,
  - explicit `AVX2` path
- `analysis/compiler_flags.sh`
- `analysis/roofline.py`
- `analysis/plot_speedup.py`
- `paper/main_cpc.tex`
- existing Intel and Ryzen CSVs in `benchmarks/`

Important current scientific consequence:

- manual `AVX2` on AoS already underperforms auto-vectorized SoA on current data.
- This is not a failure; it is a useful negative result.

### 6.2 `tang-nano-20k` already contains

- a `9x9` fixed-point complex matvec core,
- cycle counting,
- UART protocol,
- host-side validation,
- precision analysis,
- jitter comparison scripts,
- HDMI visualization work.

Important consequence:

- the FPGA section is not “start from zero.”
- it is an integration, validation, and positioning task.

### 6.3 Current repo caveats

- `tang-nano-20k/analysis/benchmark_compare.py` hardcodes `~/dev/research/lindblad-bench/benchmarks`, not the active local repo path.
- `benchmarks/qutip_results.csv` does not currently exist.
- `qutip` is not currently installed on the local host.
- `lindblad-bench` has untracked manuscript-related files.
- `tang-nano-20k` has a modified `hdmi/bloch_top.v` in the worktree.

Agent rule:

- do not revert unrelated changes.

## 7. Core Research Questions

### RQ1. CPU regime

How does the fixed-size dense propagation kernel behave across cache levels on old and modern x86 CPUs, and how much do layout, vectorization, and threading change the attainable bandwidth?

### RQ2. GPU crossover

At what problem size and batch size does GPU execution become worthwhile for this kernel, and how different is that crossover on Pascal (`1070 Ti`) versus Ada (`4070 Laptop`)?

### RQ3. FPGA operating point

Can a small fixed-point FPGA implementation provide a meaningful deterministic-latency operating point for `d = 3`, even if its throughput is not the highest among all platforms?

### RQ4. Application relevance

When wrapped in a GRAPE-like workflow, which part of the runtime dominates:

- propagator application,
- propagator construction (`expm`),
- host/device overhead,
- or control-loop orchestration?

### RQ5. Software-author guidance

What concrete recommendations can be made to authors of quantum-control software targeting this size regime?

## 8. Hypotheses

### H1

For single-trajectory propagation at `d = 3` and often `d = 9`, CPUs will beat GPUs in end-to-end wall time because launch and transfer overhead dominate the tiny dense kernel.

### H2

For sufficiently batched propagation, especially at `d = 27`, GPUs will overtake CPUs in throughput, and the `RTX 4070` should cross over earlier and more strongly than the `GTX 1070 Ti`.

### H3

The FPGA `d = 3` core will lose on raw throughput but win decisively on determinism and jitter.

### H4

In GRAPE-like workloads, optimization of the propagation kernel alone may move the bottleneck toward propagator construction (`expm`) rather than eliminating total runtime.

### H5

OpenMP gains will only be scientifically honest when parallelism is applied over independent trajectories, batch states, pulse candidates, or optimization points, not over sequential time steps of one trajectory.

This is important: a single time-ordered trajectory is not embarrassingly parallel over time.

## 9. Required Comparison Matrix

### 9.1 Problem sizes

Mandatory:

- `d = 3`
- `d = 9`
- `d = 27`

FPGA:

- `d = 3` only

### 9.2 CPU variants

Mandatory:

- scalar / non-vectorized baseline,
- auto-vectorized AoS,
- auto-vectorized SoA,
- current manual `AVX2` path,
- OpenMP or equivalent shared-memory batching path.

Optional:

- dimension-specialized kernels if needed later.

Not currently mandatory:

- `AVX-512`

### 9.3 GPU variants

One CUDA codebase, two hosts:

- `GTX 1070 Ti`
- `RTX 4070 Laptop GPU`

Measure both:

- kernel-only timing with device-resident data,
- end-to-end timing including host-device overhead.

Batch sizes should include at least:

- `1`
- `8`
- `32`
- `128`
- `512`
- `2048`

If runtime is small enough, extend further.

### 9.4 FPGA variants

Mandatory:

- current `Q1.15` implementation,
- cycle count per step,
- wall-clock per step,
- multi-step behavior,
- jitter,
- precision drift against floating-point reference,
- resource use.

Optional:

- `Q1.31` or wider fixed-point only if the precision study shows `Q1.15` is inadequate for the intended trajectory length.

## 10. Benchmarking Protocol

### 10.1 General rules

- same physical model across CPU, GPU, and FPGA where comparison is claimed,
- same `d`, `dt`, Hamiltonian convention, and collapse operators,
- warm-up before timed runs,
- repeated trials with medians reported,
- log machine info, compiler versions, CUDA version, and git commit where possible,
- keep generated data in machine-tagged CSV form.

### 10.2 CPU protocol

- pin thread counts explicitly,
- separate single-thread from threaded results,
- on the i9, separate:
  - unrestricted scheduler,
  - P-core pinning,
  - E-core-only if feasible,
  - all-core mixed mode if feasible.

### 10.3 GPU protocol

For every benchmark distinguish:

- host-visible latency,
- kernel latency,
- effective throughput under batching.

The paper must not compare:

- CPU single-step latency

against

- GPU kernel-only resident timing

without also showing the end-to-end comparison.

### 10.4 FPGA protocol

- verify bitstream and UART path before using any measured number,
- use the hardware cycle counter as the primary timing datum,
- report UART wall time separately from compute time,
- do not confuse transport latency with on-chip compute latency.

## 11. Figure and Table Plan

### Figure 1

Cross-platform operating-regime summary.

Recommended content:

- CPUs, GPUs, and FPGA on one summary chart,
- x-axis: problem size or batch size,
- y-axis: latency or throughput depending on panel,
- clear annotation of where each architecture wins.

Do not force everything into one overloaded Roofline if readability suffers.

### Figure 2

CPU Roofline and cache-placement summary for both CPUs.

### Figure 3

CPU threading scalability:

- Ryzen 1600,
- i9-13980HX,
- pinned-core variants where relevant.

### Figure 4

GPU crossover plot:

- `1070 Ti` and `4070`,
- kernel-only and end-to-end,
- batch-size sweep.

### Figure 5

FPGA latency and jitter:

- cycle counts,
- converted wall time,
- CPU vs FPGA jitter histogram or summary.

### Figure 6

End-to-end GRAPE comparison:

- C implementation,
- QuTiP baseline,
- optional Dynamiqs if added later,
- split build cost vs propagation cost.

### Tables

Mandatory tables:

- platform specs,
- benchmark variants,
- principal quantitative results,
- reproducibility checklist.

## 12. Manuscript Structure

Recommended JCP structure:

1. Introduction  
   Frame near-term quantum control, small dense Lindblad propagation, and why hardware-aware comparisons matter.

2. Related Work  
   QuTiP, Dynamiqs, HOQST, prior Lindblad HPC work, and FPGA control-latency literature.

3. Methods  
   Physical model, vectorized propagator, memory layout, benchmark methodology, hardware description.

4. CPU Results  
   Cache story, vectorization, negative AVX2 result, threading.

5. GPU Results  
   Crossover story, legacy vs modern GPU, batch dependence.

6. FPGA Results  
   `d = 3` latency, resource use, jitter, precision.

7. Application Benchmark  
   GRAPE-style workload and what actually dominates runtime.

8. Discussion  
   When to use CPU, GPU, FPGA; what software authors should do.

9. Conclusion  
   Regime map, not a one-number winner.

## 13. Execution Priorities

### Phase 0: Reproducibility freeze

- identify canonical repo paths on both machines,
- record hardware facts from `lscpu`, `nvidia-smi`, and `openFPGALoader`,
- ensure benchmark outputs are machine-tagged and scriptable.

### Phase 1: CPU paper baseline

- refresh Intel and Ryzen results,
- add OpenMP or batch-threading path,
- regenerate figures from source.

### Phase 2: CUDA comparison

- add one CUDA implementation,
- run on both GPUs,
- determine crossover behavior.

### Phase 3: FPGA validation

- verify board I/O path,
- collect cycle, jitter, and precision results,
- fix repo path issues in FPGA analysis scripts.

### Phase 4: GRAPE end-to-end

- install QuTiP,
- run matched GRAPE comparisons,
- optionally consider Dynamiqs after core comparisons are stable.

### Phase 5: manuscript rewrite

- replace CPC framing,
- write JCP comparison framing,
- integrate all figures and tables.

## 14. Critical Technical Rules For Future Implementation

### Rule 1

Do not chase AVX-512 on the current local host. It is not exposed in the current `lscpu` flags.

### Rule 2

Do not describe the desktop CPU as a `1500X`. It is a `Ryzen 5 1600`.

### Rule 3

Do not parallelize a single sequential trajectory over time steps and call it OpenMP scaling.

### Rule 4

Do not compare GPU kernel-only timings to CPU end-to-end timings without labeling them as different metrics.

### Rule 5

Do not oversell FPGA throughput. The FPGA story is latency, jitter, determinism, and control relevance.

### Rule 6

Do not revert unrelated worktree changes in either repo.

## 15. Optional Extensions

These are optional and should not block the core paper:

- Dynamiqs software comparison,
- wider fixed-point FPGA study,
- dimension-specialized CPU kernels,
- energy or power measurements,
- cache-aware or memory-level Roofline extensions,
- ARM CPU comparison if a suitable host becomes available later.

## 16. Immediate Blockers

As of this document:

- `qutip` is not installed locally,
- `benchmarks/qutip_results.csv` does not exist,
- `tang-nano-20k` analysis uses an outdated `lindblad-bench` path,
- the canonical desktop `lindblad-bench` repo path needs cleanup,
- GPU and FPGA benchmark pipelines are not yet unified into the paper’s data flow.

None of these are conceptual blockers. They are execution blockers.

## 17. Final Working Claim

If the execution goes well, the clean concluding claim should be:

> In the fixed-size dense Lindblad regime relevant to near-term transmon control, architecture choice is dictated by operating regime rather than peak theoretical compute: CPUs dominate low-overhead cache-resident propagation, GPUs dominate sufficiently batched throughput workloads, and FPGAs occupy a uniquely valuable deterministic-latency niche for real-time feedback contexts.

That is the paper this project should now be trying to prove.

## 18. Source Notes

External sources used to build this blueprint:

- JCP scope: `https://shop.elsevier.com/journals/journal-of-computational-physics/0021-9991`
- CPC scope: `https://shop.elsevier.com/journals/subjects/physical-sciences-and-engineering/physics/interdisciplinary-physics/computational-physics`
- Elsevier submission guidance: `https://www.elsevier.com/subject/next/guide-for-authors`
- Roofline paper: `https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/`
- QuTiP 2011: `https://arxiv.org/abs/1110.0573`
- QuTiP 2 2012: `https://arxiv.org/abs/1211.6518`
- HOQST: `https://arxiv.org/abs/2011.14046`
- Meyerov et al. 2019: `https://arxiv.org/abs/1912.01491`
- Dynamiqs docs: `https://www.dynamiqs.org/stable/documentation/getting_started/whatis.html`
- Transmon paper: `https://arxiv.org/abs/cond-mat/0703002`
- FPGA fluxonium feedback: `https://arxiv.org/abs/1912.06814`
- FPGA control/readout system: `https://arxiv.org/abs/2110.07965`
- Real-time low-latency QEC: `https://arxiv.org/abs/2410.05202`
- Tang Nano 20K wiki: `https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html`
- Tang Nano 20K datasheet: `https://dl.sipeed.com/fileList/TANG/Nano_20K/1_Datasheet/Sipeed%20Tang%20nano%2020K%20Datasheet%20V1.3-en_US.pdf`

Local observations included in this document came from:

- `lscpu`
- `nvidia-smi`
- `openFPGALoader --detect`
- current repo inspection in `lindblad-bench` and `tang-nano-20k`
