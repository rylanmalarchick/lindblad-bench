/**
 * bench_evolve.c — End-to-end trajectory timing.
 *
 * Times lb_evolve_prop() for a full T=1 µs trajectory at each d.
 * Separates propagator build cost from trajectory integration cost.
 * This is the relevant figure for GRAPE sweep throughput comparison.
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>

static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

static void run_bench(size_t d, double t_end_us, double dt_ns, int n_trials)
{
    size_t n_steps = (size_t)ceil(t_end_us * 1e3 / dt_ns);
    size_t d2 = d * d;

    printf("\n=== d = %zu, t_end = %.0f µs, dt = %.1f ns (%zu steps) ===\n",
           d, t_end_us, dt_ns, n_steps);

    /* Build system */
    lb_system_t sys;
    lb_system_init(&sys, d);
    lb_matrix_alloc(&sys.H, d);
    for (size_t n = 0; n < d; n++) sys.H.data[n * d + n] = (double)n + 0.0*I;

    /* Amplitude damping */
    double gamma1 = 1.0 / 50.0;
    lb_matrix_t L1 = {NULL, d};
    lb_matrix_alloc(&L1, d);
    if (d >= 2) L1.data[0 * d + 1] = sqrt(gamma1) + 0.0*I;
    lb_system_add_cop(&sys, &L1);
    lb_matrix_free(&L1);

    /* ---- Time: build Lindbladian + propagator ---- */
    struct timespec t0_build, t1_build;
    clock_gettime(CLOCK_MONOTONIC, &t0_build);

    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, dt_ns, &prop);
    lb_matrix_free(&L);

    clock_gettime(CLOCK_MONOTONIC, &t1_build);
    double ms_build = (double)ns_diff(t0_build, t1_build) / 1e6;

    /* ---- Time: trajectory integration (propagator reuse) ---- */
    lb_matrix_t rho0    = {NULL, d};
    lb_matrix_t rho_out = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    lb_matrix_alloc(&rho_out, d);
    rho0.data[0] = 1.0 + 0.0*I;

    /* Warm-up */
    lb_evolve_prop(&prop, &rho0, n_steps, &rho_out, NULL);

    struct timespec t0_traj, t1_traj;
    clock_gettime(CLOCK_MONOTONIC, &t0_traj);
    for (int tr = 0; tr < n_trials; tr++) {
        lb_evolve_prop(&prop, &rho0, n_steps, &rho_out, NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1_traj);

    double ms_traj = (double)ns_diff(t0_traj, t1_traj) / (double)n_trials / 1e6;
    double ns_per_step = (double)ns_diff(t0_traj, t1_traj) / (double)n_trials / (double)n_steps;

    printf("  propagator build : %.3f ms (one-time)\n", ms_build);
    printf("  ms/trajectory    : %.3f ms (%d trials)\n", ms_traj, n_trials);
    printf("  ns/step          : %.2f ns\n", ns_per_step);
    printf("  trajs/second     : %.1f\n", 1e3 / ms_traj);
    printf("  total (build+1tr): %.3f ms\n", ms_build + ms_traj);

    lb_matrix_free(&rho0);
    lb_matrix_free(&rho_out);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
}

int main(void)
{
    printf("lindblad-bench: lb_evolve_prop end-to-end timing\n");
    printf("(propagator build timed separately from trajectory integration)\n");
    run_bench(3,  1.0, 0.5, 100);
    run_bench(9,  1.0, 0.5, 20);
    run_bench(27, 1.0, 0.5, 5);
    return 0;
}
