# Godbolt Permalink Catalog

Compiler Explorer (https://godbolt.org) permalinks for `lb_propagate_step` and related kernels.

Methodology:
1. Paste the relevant kernel (from `src/propagate.c`) into Godbolt
2. Add `#include <complex.h>` and the minimal struct definitions needed
3. Set compiler to **x86-64 gcc 14.2** (or current latest)
4. Record the permalink and note which SIMD instructions appear in the hot loop

| Kernel | Compiler | Flags | SIMD observed | Notes | Link |
|--------|----------|-------|---------------|-------|------|
| `lb_propagate_step` (d=9 fixed-size) | gcc 14.2 | `-O2 -fno-tree-vectorize` | none | scalar baseline | _pending_ |
| `lb_propagate_step` (d=9 fixed-size) | gcc 14.2 | `-O3 -march=x86-64-v3` | `vfmadd231pd`, `vmovupd` | auto-vectorized | _pending_ |
| `lb_propagate_step` (d=9 fixed-size) | gcc 14.2 | `-O3 -march=native` | `vfmadd231pd` x4 unrolled | unrolled + AVX2 | _pending_ |
| `lb_propagate_step` (d=9 fixed-size) | clang 18 | `-O3 -march=native` | compare vs gcc | clang vs gcc | _pending_ |
| `lb_propagate_step` (d=27 dynamic)   | gcc 14.2 | `-O3 -march=native` | reduced vec (aliasing) | dynamic d barrier | _pending_ |

## What to look for

**Good signs (auto-vectorization succeeded):**
- `vmovupd` / `vloadupd` — 256-bit AVX2 loads
- `vfmadd231pd` — fused multiply-add (4 doubles at once)
- Loop unrolling: the inner `j` loop body repeated 2-4x

**Bad signs (scalar fallback):**
- `movsd` / `addsd` / `mulsd` — 64-bit scalar ops
- Presence of `__divdc3` or `__muldc3` — compiler fell back to libgcc complex multiply

## Compile-time specialization experiment

The key experiment for the paper: compare assembly when `d²` is:
1. A runtime variable (`size_t d2 = ...`)
2. A compile-time constant (`#define D2 9`)
3. A compile-time constant with `__builtin_expect`

Hypothesis: fixed `D2=9` allows the compiler to fully unroll the inner loop
(9 iterations) and emit 9 consecutive `vfmadd231pd` instructions.
With runtime `d2`, the compiler must emit a loop with a branch.

See `analysis/compiler_flags.sh` for the automated multi-build comparison.
