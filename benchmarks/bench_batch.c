/**
 * bench_batch.c — Batch throughput benchmark for independent propagations.
 *
 * Measures lb_propagate_step_batch on a batch of independent density matrices.
 * This is the honest shared-memory parallel benchmark used for OpenMP scaling:
 * trajectories are independent, so batch parallelism does not violate the
 * sequential dependency of time steps within a single trajectory.
 *
 * Usage:
 *   ./bench_batch [n_reps] [batch_size]
 *
 * Default n_reps = 2000, batch_size = 64.
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

static void build_transmon_system(lb_system_t *sys, size_t d,
                                  double T1_us, double T2_us)
{
    lb_system_init(sys, d);
    lb_matrix_alloc(&sys->H, d);

    for (size_t n = 0; n < d; n++) {
        sys->H.data[n * d + n] = (double)n + 0.0 * I;
    }

    double gamma1 = 1.0 / T1_us;
    lb_matrix_t L1 = {NULL, d};
    lb_matrix_alloc(&L1, d);
    if (d >= 2) {
        L1.data[0 * d + 1] = sqrt(gamma1) + 0.0 * I;
    }
    lb_system_add_cop(sys, &L1);
    lb_matrix_free(&L1);

    double T_phi = 1.0 / (1.0 / T2_us - 0.5 / T1_us);
    if (T_phi > 0 && d >= 2) {
        double gamma_phi = 1.0 / T_phi;
        lb_matrix_t L2 = {NULL, d};
        lb_matrix_alloc(&L2, d);
        L2.data[1 * d + 1] = sqrt(gamma_phi) + 0.0 * I;
        lb_system_add_cop(sys, &L2);
        lb_matrix_free(&L2);
    }
}

static int alloc_batch(lb_matrix_t *xs, size_t batch_size, size_t d)
{
    for (size_t i = 0; i < batch_size; i++) {
        if (lb_matrix_alloc(&xs[i], d) != 0) {
            for (size_t j = 0; j < i; j++) {
                lb_matrix_free(&xs[j]);
            }
            return -1;
        }
    }
    return 0;
}

static void free_batch(lb_matrix_t *xs, size_t batch_size)
{
    for (size_t i = 0; i < batch_size; i++) {
        lb_matrix_free(&xs[i]);
    }
}

static void init_batch_inputs(lb_matrix_t *rho_batch, size_t batch_size, size_t d)
{
    for (size_t b = 0; b < batch_size; b++) {
        double p0 = 0.55 + 0.25 * sin(0.1 * (double)b);
        double p1 = 0.35 + 0.15 * cos(0.2 * (double)b);
        double p2 = 1.0 - p0 - p1;

        if (d >= 1) {
            rho_batch[b].data[0 * d + 0] = p0 + 0.0 * I;
        }
        if (d >= 2) {
            rho_batch[b].data[1 * d + 1] = p1 + 0.0 * I;
            rho_batch[b].data[0 * d + 1] = 0.02 * (double)(b + 1) + 0.01 * I;
            rho_batch[b].data[1 * d + 0] = 0.02 * (double)(b + 1) - 0.01 * I;
        }
        if (d >= 3) {
            rho_batch[b].data[2 * d + 2] = p2 + 0.0 * I;
        }
    }
}

static void run_bench(size_t d, long n_reps, size_t batch_size)
{
    size_t d2 = d * d;
    size_t state_steps = (size_t)n_reps * batch_size;
    int ret = 0;

    printf("\n=== d = %zu (batch = %zu, P size = %.1f KB) ===\n",
           d, batch_size, (double)(d2 * d2 * 16) / 1024.0);

    lb_system_t sys;
    build_transmon_system(&sys, d, 50.0, 30.0);

    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, 0.5, &prop);
    lb_matrix_free(&L);
    lb_system_free(&sys);

    lb_matrix_t *rho_in = calloc(batch_size, sizeof(lb_matrix_t));
    lb_matrix_t *rho_out = calloc(batch_size, sizeof(lb_matrix_t));
    if (!rho_in || !rho_out || alloc_batch(rho_in, batch_size, d) != 0 ||
        alloc_batch(rho_out, batch_size, d) != 0) {
        fprintf(stderr, "batch allocation failed\n");
        if (rho_in) {
            free_batch(rho_in, batch_size);
        }
        if (rho_out) {
            free_batch(rho_out, batch_size);
        }
        free(rho_in);
        free(rho_out);
        lb_propagator_free(&prop);
        exit(1);
    }
    init_batch_inputs(rho_in, batch_size, d);

    for (int w = 0; w < 10; w++) {
        ret = lb_propagate_step_batch(&prop, rho_in, rho_out, batch_size);
        if (ret != 0) {
            fprintf(stderr, "batch warm-up failed\n");
            goto done;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long r = 0; r < n_reps; r++) {
        ret = lb_propagate_step_batch(&prop, rho_in, rho_out, batch_size);
        if (ret != 0) {
            fprintf(stderr, "batch timing loop failed\n");
            goto done;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long long total_ns = ns_diff(t0, t1);
    double s_total = (double)total_ns * 1e-9;
    double ns_per_batch = (double)total_ns / (double)n_reps;
    double ns_per_state_step = (double)total_ns / (double)state_steps;

    double flops_per_state_step = 8.0 * (double)d2 * (double)d2;
    double bytes_per_state_step = (double)(d2 * d2 + 2 * d2) * 16.0;
    double gflops = (flops_per_state_step * (double)state_steps) / (s_total * 1e9);
    double gbytes_s = (bytes_per_state_step * (double)state_steps) / (s_total * 1e9);
    double mstate_steps_s = ((double)state_steps / s_total) / 1e6;

#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif

    printf("  threads      : %d\n", threads);
    printf("  n_reps       : %ld\n", n_reps);
    printf("  total        : %.3f ms\n", (double)total_ns / 1e6);
    printf("  ns/batch     : %.2f ns\n", ns_per_batch);
    printf("  ns/state-step: %.2f ns\n", ns_per_state_step);
    printf("  Mstate-steps/s: %.3f\n", mstate_steps_s);
    printf("  GFLOP/s      : %.3f\n", gflops);
    printf("  GB/s         : %.3f\n", gbytes_s);
    printf("  arith intens : %.4f FLOP/byte\n",
           flops_per_state_step / bytes_per_state_step);
    printf("RESULT,d=%zu,threads=%d,batch_size=%zu,n_reps=%ld,ns_per_batch=%.2f,ns_per_state_step=%.2f,mstate_steps_s=%.6f,gflops=%.6f,gbytes_s=%.6f\n",
           d, threads, batch_size, n_reps, ns_per_batch, ns_per_state_step,
           mstate_steps_s, gflops, gbytes_s);

done:
    free_batch(rho_in, batch_size);
    free_batch(rho_out, batch_size);
    free(rho_in);
    free(rho_out);
    lb_propagator_free(&prop);

    if (ret != 0) {
        exit(1);
    }
}

int main(int argc, char **argv)
{
    long n_reps = 2000;
    size_t batch_size = 64;

    if (argc > 1) {
        n_reps = atol(argv[1]);
    }
    if (argc > 2) {
        batch_size = (size_t)atol(argv[2]);
    }
    if (n_reps <= 0 || batch_size == 0) {
        fprintf(stderr, "n_reps must be > 0 and batch_size must be > 0\n");
        return 1;
    }

    printf("lindblad-bench: batch propagation benchmark\n");
    printf("Measures lb_propagate_step_batch over independent states\n");
    printf("n_reps = %ld, batch_size = %zu\n", n_reps, batch_size);

    run_bench(3, n_reps, batch_size);
    run_bench(9, n_reps, batch_size);
    run_bench(27, n_reps, batch_size);

    return 0;
}
