# Constraints

- C11 only in `src/`. No external deps. No heap in `lb_propagate_step`.
- C results must match QuTiP (`reference/reference.py`) to HS norm < 1e-6.
- System sizes: d = 3, 9, 27 only.
- Timing: `clock_gettime(CLOCK_MONOTONIC)`. No `gettimeofday`.
- Commits: `feat:`, `fix:`, `bench:`, `analysis:`, `paper:`
