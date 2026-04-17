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

static int build_driven_problem(size_t d,
                                double gamma,
                                lb_system_t *diss_sys,
                                lb_matrix_t *H0,
                                lb_matrix_t *Hdrive)
{
    lb_system_init(diss_sys, d);
    if (lb_matrix_alloc(H0, d) != 0) return -1;
    if (lb_matrix_alloc(Hdrive, d) != 0) {
        lb_matrix_free(H0);
        return -1;
    }

    /* H0 = diag(0, 1, ...) */
    for (size_t n = 0; n < d; n++)
        H0->data[n * d + n] = (double)n + 0.0*I;
    if (d >= 2) {
        /* Truncated sigma_x-like drive term */
        Hdrive->data[0 * d + 1] = 1.0 + 0.0*I;
        Hdrive->data[1 * d + 0] = 1.0 + 0.0*I;
    }

    /* Amplitude damping */
    if (d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        if (lb_matrix_alloc(&L1, d) != 0) {
            lb_matrix_free(H0);
            lb_matrix_free(Hdrive);
            return -1;
        }
        L1.data[0 * d + 1] = sqrt(gamma) + 0.0*I;
        if (lb_system_add_cop(diss_sys, &L1) != 0) {
            lb_matrix_free(&L1);
            lb_matrix_free(H0);
            lb_matrix_free(Hdrive);
            lb_system_free(diss_sys);
            return -1;
        }
        lb_matrix_free(&L1);
    }

    return 0;
}

static void run_bench(size_t d, int n_segments, int steps_per_seg, int n_trials)
{
    size_t d2 = d * d;
    size_t n4 = d2 * d2;
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

    /* ---- One-time setup: static and drive superoperators ---- */
    lb_propagator_t *props = calloc(n_segments, sizeof(lb_propagator_t));
    lb_system_t diss_sys;
    lb_matrix_t H0 = {0};
    lb_matrix_t Hdrive = {0};
    lb_matrix_t Lcoh0 = {0};
    lb_matrix_t Ldrive = {0};
    lb_matrix_t Ldiss = {0};
    lb_matrix_t Lbase = {0};
    lb_matrix_t L = {0};
    lb_expm_workspace_t ws = {0};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (build_driven_problem(d, gamma, &diss_sys, &H0, &Hdrive) != 0) {
        fprintf(stderr, "failed to build driven problem\n");
        exit(1);
    }
    if (lb_matrix_alloc(&Lcoh0, d2) != 0 ||
        lb_matrix_alloc(&Ldrive, d2) != 0 ||
        lb_matrix_alloc(&Ldiss, d2) != 0 ||
        lb_matrix_alloc(&Lbase, d2) != 0 ||
        lb_matrix_alloc(&L, d2) != 0 ||
        lb_expm_workspace_alloc(&ws, d2) != 0) {
        fprintf(stderr, "failed to allocate GRAPE setup buffers\n");
        exit(1);
    }

    if (lb_build_coherent_superop(&H0, &Lcoh0) != 0 ||
        lb_build_coherent_superop(&Hdrive, &Ldrive) != 0 ||
        lb_build_dissipator_superop(&diss_sys, &Ldiss) != 0) {
        fprintf(stderr, "failed to build reusable superoperators\n");
        exit(1);
    }

    for (size_t i = 0; i < n4; i++) {
        Lbase.data[i] = Lcoh0.data[i] + Ldiss.data[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_setup = (double)ns_diff(t0, t1) / 1e6;

    /* ---- Phase 1: Build all segment propagators ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int s = 0; s < n_segments; s++) {
        for (size_t i = 0; i < n4; i++) {
            L.data[i] = Lbase.data[i] + omegas[s] * Ldrive.data[i];
        }
        if (lb_build_propagator_ws(&L, dt, &props[s], &ws) != 0) {
            fprintf(stderr, "failed to build propagator %d\n", s);
            exit(1);
        }
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
    printf("  [setup] invariant superops : %.3f ms\n", ms_setup);
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
    lb_expm_workspace_free(&ws);
    lb_matrix_free(&H0);
    lb_matrix_free(&Hdrive);
    lb_matrix_free(&Lcoh0);
    lb_matrix_free(&Ldrive);
    lb_matrix_free(&Ldiss);
    lb_matrix_free(&Lbase);
    lb_matrix_free(&L);
    lb_matrix_free(&rho_cur);
    lb_matrix_free(&rho_next);
    lb_matrix_free(&rho0);
    lb_system_free(&diss_sys);
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
