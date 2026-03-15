/**
 * bench_evolve.c — End-to-end trajectory timing.
 *
 * Times lb_evolve() for a full T=1 µs trajectory at each d.
 * Includes propagator build cost (amortized) and n_steps matvec calls.
 * This is the relevant figure for GRAPE sweep throughput comparison.
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>

static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

static void run_bench(size_t d, double t_end_us, double dt_ns, int n_trials)
{
    printf("\n=== d = %zu, t_end = %.0f µs, dt = %.1f ns (%zu steps) ===\n",
           d, t_end_us, dt_ns,
           (size_t)(t_end_us * 1e3 / dt_ns));

    /* Simple diagonal Hamiltonian */
    lb_system_t sys;
    lb_system_init(&sys, d);
    lb_matrix_alloc(&sys.H, d);
    for (size_t n = 0; n < d; n++) sys.H.data[n * d + n] = (double)n + 0.0*I;

    /* Amplitude damping */
    double gamma1 = 1.0 / 50.0; /* 50 µs T1 */
    lb_matrix_t L1 = {NULL, d};
    lb_matrix_alloc(&L1, d);
    if (d >= 2) L1.data[0 * d + 1] = __builtin_sqrt(gamma1) + 0.0*I;
    lb_system_add_cop(&sys, &L1);
    lb_matrix_free(&L1);

    lb_matrix_t rho0    = {NULL, d};
    lb_matrix_t rho_out = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    lb_matrix_alloc(&rho_out, d);
    rho0.data[0] = 1.0 + 0.0*I;

    size_t n_steps = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int tr = 0; tr < n_trials; tr++) {
        lb_evolve(&sys, &rho0, t_end_us * 1e3, dt_ns, &rho_out, NULL, &n_steps);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long long total_ns = ns_diff(t0, t1);
    double ms_per_traj = (double)total_ns / (double)n_trials / 1e6;

    printf("  steps/traj   : %zu\n", n_steps);
    printf("  ms/trajectory: %.3f ms\n", ms_per_traj);
    printf("  trajs/second : %.1f\n", 1e3 / ms_per_traj);

    lb_matrix_free(&rho0);
    lb_matrix_free(&rho_out);
    lb_system_free(&sys);
}

int main(void)
{
    printf("lindblad-bench: lb_evolve end-to-end timing\n");
    run_bench(3,  1.0, 0.5, 100);
    run_bench(9,  1.0, 0.5, 20);
    run_bench(27, 1.0, 0.5, 5);
    return 0;
}
