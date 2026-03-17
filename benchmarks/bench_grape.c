/**
 * bench_grape.c — GRAPE-style piecewise-constant propagator chain.
 *
 * Simulates the actual GRAPE inner loop:
 *   1. Build N_seg propagators (one per pulse segment with different H)
 *   2. Chain them: rho_final = P_N ... P_2 P_1 rho_0
 *
 * Reports:
 *   - Per-segment propagator build time (expm cost)
 *   - Per-trajectory chain time (N_seg matvecs)
 *   - Full GRAPE "landscape point" cost (build + chain)
 *
 * This is the timing that matters for comparing C vs QuTiP GRAPE.
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

/**
 * Build a Lindbladian with a drive-perturbed Hamiltonian.
 * H(segment) = H0 + Omega * sigma_x_truncated
 * where Omega varies per segment (simulating GRAPE pulse amplitudes).
 */
static int build_driven_lindbladian(size_t d, double omega, double gamma,
                                    lb_matrix_t *L_out)
{
    lb_system_t sys;
    lb_system_init(&sys, d);
    lb_matrix_alloc(&sys.H, d);

    /* H = diag(0, 1, ...) + omega * (|0><1| + |1><0|) */
    for (size_t n = 0; n < d; n++)
        sys.H.data[n * d + n] = (double)n + 0.0*I;
    if (d >= 2) {
        sys.H.data[0 * d + 1] += omega + 0.0*I;
        sys.H.data[1 * d + 0] += omega + 0.0*I;
    }

    /* Amplitude damping */
    if (d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        lb_matrix_alloc(&L1, d);
        L1.data[0 * d + 1] = sqrt(gamma) + 0.0*I;
        lb_system_add_cop(&sys, &L1);
        lb_matrix_free(&L1);
    }

    if (lb_build_lindbladian(&sys, L_out) != 0) {
        lb_system_free(&sys);
        return -1;
    }
    lb_system_free(&sys);
    return 0;
}

static void run_bench(size_t d, int n_segments, int steps_per_seg, int n_trials)
{
    size_t d2 = d * d;
    double gamma = 1.0 / 50.0;
    double dt = 0.5; /* ns per step */

    printf("\n=== d=%zu, %d segments × %d steps (%.1f ns total pulse) ===\n",
           d, n_segments, steps_per_seg,
           (double)(n_segments * steps_per_seg) * dt);
    printf("  P size = %.1f KB, L size = %.1f KB\n",
           (double)(d2 * d2 * 16) / 1024.0,
           (double)(d2 * d2 * 16) / 1024.0);

    /* Generate pseudo-random drive amplitudes (deterministic seed) */
    srand(42);
    double *omegas = malloc(n_segments * sizeof(double));
    for (int s = 0; s < n_segments; s++)
        omegas[s] = 0.1 * ((double)rand() / RAND_MAX - 0.5);

    /* ---- Phase 1: Build all propagators ---- */
    lb_propagator_t *props = calloc(n_segments, sizeof(lb_propagator_t));
    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int s = 0; s < n_segments; s++) {
        build_driven_lindbladian(d, omegas[s], gamma, &L);
        lb_build_propagator(&L, dt, &props[s]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_build_all = (double)ns_diff(t0, t1) / 1e6;
    double ms_build_per = ms_build_all / n_segments;

    /* ---- Phase 2: Chain propagators (the GRAPE inner loop) ---- */
    lb_matrix_t rho_cur  = {NULL, d};
    lb_matrix_t rho_next = {NULL, d};
    lb_matrix_t rho0     = {NULL, d};
    lb_matrix_alloc(&rho_cur, d);
    lb_matrix_alloc(&rho_next, d);
    lb_matrix_alloc(&rho0, d);
    rho0.data[0] = 1.0 + 0.0*I; /* |0><0| */

    /* Warm-up */
    lb_matrix_copy(&rho_cur, &rho0);
    for (int s = 0; s < n_segments; s++) {
        for (int st = 0; st < steps_per_seg; st++) {
            lb_propagate_step(&props[s], &rho_cur, &rho_next);
            double complex *tmp = rho_cur.data;
            rho_cur.data = rho_next.data;
            rho_next.data = tmp;
        }
    }

    /* Timed chain */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int tr = 0; tr < n_trials; tr++) {
        lb_matrix_copy(&rho_cur, &rho0);
        for (int s = 0; s < n_segments; s++) {
            for (int st = 0; st < steps_per_seg; st++) {
                lb_propagate_step(&props[s], &rho_cur, &rho_next);
                double complex *tmp = rho_cur.data;
                rho_cur.data = rho_next.data;
                rho_next.data = tmp;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms_chain = (double)ns_diff(t0, t1) / (double)n_trials / 1e6;
    int total_steps = n_segments * steps_per_seg;
    double ns_per_step = (double)ns_diff(t0, t1) / (double)n_trials / (double)total_steps;

    /* ---- Report ---- */
    printf("  [build] all propagators : %.3f ms (%.3f ms/segment)\n",
           ms_build_all, ms_build_per);
    printf("  [chain] ms/trajectory   : %.3f ms (%d total steps)\n",
           ms_chain, total_steps);
    printf("  [chain] ns/step         : %.2f ns\n", ns_per_step);
    printf("  [total] landscape point : %.3f ms (build + 1 chain)\n",
           ms_build_all + ms_chain);
    printf("  [total] landscape pts/s : %.1f\n",
           1e3 / (ms_build_all + ms_chain));

    /* Cleanup */
    for (int s = 0; s < n_segments; s++)
        lb_propagator_free(&props[s]);
    free(props);
    free(omegas);
    lb_matrix_free(&L);
    lb_matrix_free(&rho_cur);
    lb_matrix_free(&rho_next);
    lb_matrix_free(&rho0);
}

int main(void)
{
    printf("lindblad-bench: GRAPE-style piecewise-constant propagator chain\n");
    printf("Simulates: build N_seg propagators, chain them as rho = P_N...P_1 rho0\n");

    /* d=3: 100-segment pulse, 20 steps/segment (1 us total, 10 ns/segment) */
    run_bench(3,  100, 20, 100);

    /* d=9: 50-segment pulse, 20 steps/segment */
    run_bench(9,  50,  20, 20);

    /* d=27: 20-segment pulse, 20 steps/segment */
    run_bench(27, 20,  20, 5);

    return 0;
}
