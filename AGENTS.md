# lindblad-bench

Bare-metal C Lindblad propagator + microarchitecture analysis. arXiv preprint target: quant-ph + cs.PF.

## Build & test

```bash
cmake -B build && cmake --build build
cd build && ctest --output-on-failure
```

Scalar-only build (for assembly comparison):
```bash
cmake -B build-scalar -DVECTORIZE=OFF && cmake --build build-scalar
```

## Key constraints

- C11 only. No external dependencies in `src/`. No heap allocation in the hot path (`lb_propagate_step`).
- All C results must validate against `reference/reference.py` (QuTiP) to Hilbert-Schmidt norm < 1e-6.
- System sizes in scope: d = 3, 9, 27. Working sets: 1.3 KB (L1), 105 KB (L2), 8.5 MB (L3).
- Timing via `clock_gettime(CLOCK_MONOTONIC)` only. No `gettimeofday`.

## Commit style

Conventional commits: `feat:`, `fix:`, `bench:`, `analysis:`, `paper:`
