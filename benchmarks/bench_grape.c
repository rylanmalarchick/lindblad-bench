/**
 * bench_grape.c — GRAPE-style piecewise-constant propagator chain.
 *
 * Simulates the GRAPE inner loop for one landscape point:
 *   1. Build N_seg propagators (one per pulse segment with a different drive)
 *   2. Chain them: rho_final = P_N ... P_2 P_1 rho_0
 *
 * The physical model matches reference/grape_reference.py exactly:
 *   H0 = diag(0, 1, ..., d-1), drive = truncated sigma_x on |0>,|1>
 *   collapse ops: sqrt(1/T1)|0><1|, sqrt(gamma_phi)|1><1|, T1 = 50, T2 = 30
 *   drive schedule: omega_s = 0.1 * (frac((s+1) * phi) - 0.5), phi = 1/golden
 *   initial state |0><0|, dt = 0.5
 *
 * Each trial rebuilds every segment propagator and runs one chain. The
 * build time is split with lb_expm_stats_t into assembly, scaling, Padé
 * products, Padé element-wise work, the linear solve, and squarings.
 *
 * Output: one CSV row per (d, trial) on stdout when --csv is given, else a
 * human-readable summary. Host and commit tags come from HOST_TAG and
 * GIT_COMMIT in the environment.
 *
 * Usage:
 *   bench_grape [--csv] [--trials N]
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <complex.h>

#define GRAPE_T1 50.0
#define GRAPE_T2 30.0
#define GRAPE_DT 0.5
#define GRAPE_PHI 0.6180339887498949

static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

/* Drive amplitude for segment s. Must match build_drive_schedule() in
 * reference/grape_reference.py bit for bit. */
double lb_grape_drive_amplitude(int s)
{
    double x = (double)(s + 1) * GRAPE_PHI;
    return 0.1 * ((x - floor(x)) - 0.5);
}

static int build_driven_problem(size_t d,
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

    if (d >= 2) {
        double gamma1 = 1.0 / GRAPE_T1;
        double t_phi = 1.0 / (1.0 / GRAPE_T2 - 0.5 / GRAPE_T1);
        double gamma_phi = 1.0 / t_phi;

        lb_matrix_t Lk = {NULL, d};
        if (lb_matrix_alloc(&Lk, d) != 0) goto fail;

        /* Amplitude damping: sqrt(gamma1) |0><1| */
        Lk.data[0 * d + 1] = sqrt(gamma1) + 0.0*I;
        if (lb_system_add_cop(diss_sys, &Lk) != 0) { lb_matrix_free(&Lk); goto fail; }

        /* Pure dephasing: sqrt(gamma_phi) |1><1| */
        Lk.data[0 * d + 1] = 0.0;
        Lk.data[1 * d + 1] = sqrt(gamma_phi) + 0.0*I;
        if (lb_system_add_cop(diss_sys, &Lk) != 0) { lb_matrix_free(&Lk); goto fail; }

        lb_matrix_free(&Lk);
    }
    return 0;

fail:
    lb_matrix_free(H0);
    lb_matrix_free(Hdrive);
    lb_system_free(diss_sys);
    return -1;
}

static void csv_header(void)
{
    puts("machine,git_commit,trial,d,n_segments,steps_per_seg,"
         "setup_ms,build_ms,assemble_ms,scale_ms,pade_mul_ms,pade_axpy_ms,"
         "solve_ms,square_ms,n_squarings,chain_ms,total_ms");
}

static void run_bench(size_t d, int n_segments, int steps_per_seg,
                      int n_trials, int csv,
                      const char *machine, const char *commit)
{
    size_t d2 = d * d;
    size_t n4 = d2 * d2;
    double dt = GRAPE_DT;

    if (!csv) {
        printf("\n=== d=%zu, %d segments × %d steps (%.1f total pulse) ===\n",
               d, n_segments, steps_per_seg,
               (double)(n_segments * steps_per_seg) * dt);
        printf("  P size = %.1f KB\n", (double)(n4 * 16) / 1024.0);
    }

    double *omegas = malloc((size_t)n_segments * sizeof(double));
    for (int s = 0; s < n_segments; s++)
        omegas[s] = lb_grape_drive_amplitude(s);

    /* ---- One-time setup: static and drive superoperators ---- */
    lb_propagator_t *props = calloc((size_t)n_segments, sizeof(lb_propagator_t));
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

    if (build_driven_problem(d, &diss_sys, &H0, &Hdrive) != 0) {
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

    lb_matrix_t rho_cur  = {NULL, d};
    lb_matrix_t rho_next = {NULL, d};
    lb_matrix_t rho0     = {NULL, d};
    lb_matrix_alloc(&rho_cur, d);
    lb_matrix_alloc(&rho_next, d);
    lb_matrix_alloc(&rho0, d);
    rho0.data[0] = 1.0 + 0.0*I; /* |0><0| */
    int total_steps = n_segments * steps_per_seg;

    /* One untimed warm-up trial (build + chain), then n_trials timed trials. */
    for (int tr = -1; tr < n_trials; tr++) {
        /* ---- Phase 1: build all segment propagators ---- */
        lb_expm_stats_t st = {0};
        struct timespec ta, tb;
        double ns_assemble = 0.0;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < n_segments; s++) {
            clock_gettime(CLOCK_MONOTONIC, &ta);
            for (size_t i = 0; i < n4; i++) {
                L.data[i] = Lbase.data[i] + omegas[s] * Ldrive.data[i];
            }
            clock_gettime(CLOCK_MONOTONIC, &tb);
            ns_assemble += (double)ns_diff(ta, tb);

            if (lb_build_propagator_ws_stats(&L, dt, &props[s], &ws, &st) != 0) {
                fprintf(stderr, "failed to build propagator %d\n", s);
                exit(1);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms_build = (double)ns_diff(t0, t1) / 1e6;

        /* ---- Phase 2: chain propagators (the GRAPE inner loop) ---- */
        lb_matrix_copy(&rho_cur, &rho0);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < n_segments; s++) {
            for (int step = 0; step < steps_per_seg; step++) {
                lb_propagate_step(&props[s], &rho_cur, &rho_next);
                double complex *tmp = rho_cur.data;
                rho_cur.data = rho_next.data;
                rho_next.data = tmp;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms_chain = (double)ns_diff(t0, t1) / 1e6;

        for (int s = 0; s < n_segments; s++)
            lb_propagator_free(&props[s]);

        if (tr < 0) continue; /* warm-up */

        if (csv) {
            printf("%s,%s,%d,%zu,%d,%d,"
                   "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%.6f,%.6f\n",
                   machine, commit, tr, d, n_segments, steps_per_seg,
                   ms_setup, ms_build,
                   ns_assemble / 1e6, st.ns_scale / 1e6,
                   st.ns_pade_mul / 1e6, st.ns_pade_axpy / 1e6,
                   st.ns_solve / 1e6, st.ns_square / 1e6,
                   st.n_squarings,
                   ms_chain, ms_build + ms_chain);
        } else {
            printf("  trial %d\n", tr);
            printf("    [setup] invariant superops : %.3f ms\n", ms_setup);
            printf("    [build] all propagators    : %.3f ms (%.3f ms/segment)\n",
                   ms_build, ms_build / n_segments);
            printf("      assemble %.3f | scale %.3f | pade_mul %.3f | pade_axpy %.3f"
                   " | solve %.3f | square %.3f ms (%ld squarings)\n",
                   ns_assemble / 1e6, st.ns_scale / 1e6, st.ns_pade_mul / 1e6,
                   st.ns_pade_axpy / 1e6, st.ns_solve / 1e6, st.ns_square / 1e6,
                   st.n_squarings);
            printf("    [chain] ms/trajectory      : %.3f ms (%d steps, %.2f ns/step)\n",
                   ms_chain, total_steps, ms_chain * 1e6 / total_steps);
            printf("    [total] landscape point    : %.3f ms\n", ms_build + ms_chain);
        }
    }

    /* Cleanup */
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

int main(int argc, char **argv)
{
    int csv = 0;
    int n_trials = 5;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv = 1;
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            n_trials = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--csv] [--trials N]\n", argv[0]);
            return 2;
        }
    }
    if (n_trials < 1) n_trials = 1;

    const char *machine = getenv("HOST_TAG");
    const char *commit = getenv("GIT_COMMIT");
    if (!machine) machine = "unknown";
    if (!commit) commit = "unknown";

    if (csv) {
        csv_header();
    } else {
        printf("lindblad-bench: GRAPE-style piecewise-constant propagator chain\n");
        printf("%d timed trials per size after one warm-up\n", n_trials);
    }

    /* Segment grid matches reference/grape_reference.py bench_all(). */
    run_bench(3,  100, 20, n_trials, csv, machine, commit);
    run_bench(9,  50,  20, n_trials, csv, machine, commit);
    run_bench(27, 20,  20, n_trials, csv, machine, commit);

    return 0;
}
