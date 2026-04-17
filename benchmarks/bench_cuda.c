/**
 * bench_cuda.c — CUDA batch propagation benchmark.
 *
 * Reports two timing regimes for the same batched propagation kernel:
 *   1. Host-visible wall time via lb_cuda_propagate_step_batch
 *   2. Device-resident kernel time via lb_cuda_raw_batch_context_t
 *
 * Usage:
 *   ./bench_cuda [n_reps] [batch_size]
 *
 * Default n_reps = 200, batch_size = 64.
 */

#include "lindblad_bench.h"
#include "lindblad_cuda.h"
#include "propagate_batch_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <complex.h>

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

static void pack_batch(const lb_matrix_t *rho_batch,
                       size_t batch_size,
                       size_t d,
                       double *packed)
{
    size_t d2 = d * d;
    size_t state_bytes = d2 * sizeof(double complex);
    for (size_t b = 0; b < batch_size; b++) {
        memcpy((char *)packed + b * state_bytes, rho_batch[b].data, state_bytes);
    }
}

static int run_bench(size_t d, long n_reps, size_t batch_size)
{
    size_t d2 = d * d;
    size_t state_steps = (size_t)n_reps * batch_size;
    double flops_per_state_step = 8.0 * (double)d2 * (double)d2;
    double bytes_per_state_step = (double)(d2 * d2 + 2 * d2) * 16.0;
    int ret = 0;

    long long host_total_ns = 0;
    double host_s_total = 0.0;
    double host_ns_per_state_step = 0.0;
    double host_gflops = 0.0;
    double host_gbytes_s = 0.0;
    float kernel_ms_total = 0.0f;
    double kernel_total_ns = 0.0;
    double kernel_s_total = 0.0;
    double kernel_ns_per_state_step = 0.0;
    double kernel_gflops = 0.0;
    double kernel_gbytes_s = 0.0;
    lb_cuda_raw_batch_context_t ctx = {0};

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

    lb_matrix_t *rho_in = (lb_matrix_t *)calloc(batch_size, sizeof(lb_matrix_t));
    lb_matrix_t *rho_out = (lb_matrix_t *)calloc(batch_size, sizeof(lb_matrix_t));
    size_t packed_bytes = batch_size * d2 * sizeof(double complex);
    double *rho_in_packed = (double *)malloc(packed_bytes);
    double *rho_out_packed = (double *)malloc(packed_bytes);
    if (!rho_in || !rho_out || !rho_in_packed || !rho_out_packed ||
        alloc_batch(rho_in, batch_size, d) != 0 ||
        alloc_batch(rho_out, batch_size, d) != 0) {
        fprintf(stderr, "CUDA benchmark allocation failed\n");
        ret = 1;
        goto done;
    }
    init_batch_inputs(rho_in, batch_size, d);

    for (int w = 0; w < 5; w++) {
        if (lb_cuda_propagate_step_batch(&prop, rho_in, rho_out, batch_size) != 0) {
            fprintf(stderr, "CUDA host-visible warm-up failed\n");
            ret = 1;
            goto done;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long r = 0; r < n_reps; r++) {
        if (lb_cuda_propagate_step_batch(&prop, rho_in, rho_out, batch_size) != 0) {
            fprintf(stderr, "CUDA host-visible timing loop failed\n");
            ret = 1;
            goto done;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    host_total_ns = ns_diff(t0, t1);
    host_s_total = (double)host_total_ns * 1e-9;
    host_ns_per_state_step = (double)host_total_ns / (double)state_steps;
    host_gflops = (flops_per_state_step * (double)state_steps) / (host_s_total * 1e9);
    host_gbytes_s = (bytes_per_state_step * (double)state_steps) / (host_s_total * 1e9);

    pack_batch(rho_in, batch_size, d, rho_in_packed);

    if (lb_cuda_raw_batch_init(&ctx, (const double *)prop.P.data, d2, batch_size) != 0 ||
        lb_cuda_raw_batch_upload_input(&ctx, rho_in_packed) != 0) {
        fprintf(stderr, "CUDA raw context setup failed\n");
        ret = 1;
        goto done;
    }

    for (int w = 0; w < 5; w++) {
        if (lb_cuda_raw_batch_launch(&ctx) != 0) {
            fprintf(stderr, "CUDA kernel warm-up failed\n");
            ret = 1;
            goto done;
        }
    }

    for (long r = 0; r < n_reps; r++) {
        float kernel_ms = 0.0f;
        if (lb_cuda_raw_batch_launch_timed(&ctx, &kernel_ms) != 0) {
            fprintf(stderr, "CUDA kernel timing loop failed\n");
            ret = 1;
            goto done;
        }
        kernel_ms_total += kernel_ms;
    }

    if (lb_cuda_raw_batch_download_output(&ctx, rho_out_packed) != 0) {
        fprintf(stderr, "CUDA kernel output download failed\n");
        ret = 1;
        goto done;
    }

    kernel_total_ns = (double)kernel_ms_total * 1e6;
    kernel_s_total = kernel_total_ns * 1e-9;
    kernel_ns_per_state_step = kernel_total_ns / (double)state_steps;
    kernel_gflops = (flops_per_state_step * (double)state_steps) / (kernel_s_total * 1e9);
    kernel_gbytes_s = (bytes_per_state_step * (double)state_steps) / (kernel_s_total * 1e9);

    printf("  n_reps             : %ld\n", n_reps);
    printf("  [host] total       : %.3f ms\n", (double)host_total_ns / 1e6);
    printf("  [host] ns/state    : %.2f ns\n", host_ns_per_state_step);
    printf("  [host] GFLOP/s     : %.3f\n", host_gflops);
    printf("  [host] GB/s        : %.3f\n", host_gbytes_s);
    printf("  [kernel] total     : %.3f ms\n", kernel_total_ns / 1e6);
    printf("  [kernel] ns/state  : %.2f ns\n", kernel_ns_per_state_step);
    printf("  [kernel] GFLOP/s   : %.3f\n", kernel_gflops);
    printf("  [kernel] GB/s      : %.3f\n", kernel_gbytes_s);
    printf("  arith intens       : %.4f FLOP/byte\n",
           flops_per_state_step / bytes_per_state_step);
    printf("RESULT,d=%zu,batch_size=%zu,n_reps=%ld,host_ns_per_state_step=%.2f,host_gflops=%.6f,host_gbytes_s=%.6f,kernel_ns_per_state_step=%.2f,kernel_gflops=%.6f,kernel_gbytes_s=%.6f\n",
           d, batch_size, n_reps, host_ns_per_state_step, host_gflops, host_gbytes_s,
           kernel_ns_per_state_step, kernel_gflops, kernel_gbytes_s);

done:
    lb_cuda_raw_batch_free(&ctx);
    if (rho_in) {
        free_batch(rho_in, batch_size);
    }
    if (rho_out) {
        free_batch(rho_out, batch_size);
    }
    free(rho_in);
    free(rho_out);
    free(rho_in_packed);
    free(rho_out_packed);
    lb_propagator_free(&prop);
    return ret;
}

int main(int argc, char **argv)
{
    long n_reps = 200;
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

    printf("lindblad-bench: CUDA batch propagation benchmark\n");
    printf("n_reps = %ld, batch_size = %zu\n", n_reps, batch_size);

    if (run_bench(3, n_reps, batch_size) != 0) {
        return 1;
    }
    if (run_bench(9, n_reps, batch_size) != 0) {
        return 1;
    }
    if (run_bench(27, n_reps, batch_size) != 0) {
        return 1;
    }

    return 0;
}
