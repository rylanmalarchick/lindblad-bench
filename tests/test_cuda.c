/**
 * test_cuda.c — CUDA batch propagation correctness tests.
 *
 * Tests:
 *   1. lb_cuda_propagate_step_batch matches CPU batch propagation at d=3
 *   2. lb_cuda_propagate_step_batch matches CPU batch propagation at d=9
 *   3. lb_cuda_propagate_step_batch matches CPU batch propagation at d=27
 */

#include "lindblad_bench.h"
#include "lindblad_cuda.h"
#include "propagate_batch_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#define TOL 1e-12
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int check(const char *name, int cond)
{
    printf("  %-55s %s\n", name, cond ? PASS : FAIL);
    return cond;
}

static void build_amp_damp(lb_system_t *sys, size_t d, double gamma)
{
    lb_system_init(sys, d);
    lb_matrix_alloc(&sys->H, d);
    for (size_t n = 0; n < d; n++) {
        sys->H.data[n * d + n] = (double)n + 0.0 * I;
    }

    if (d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        lb_matrix_alloc(&L1, d);
        L1.data[0 * d + 1] = sqrt(gamma) + 0.0 * I;
        lb_system_add_cop(sys, &L1);
        lb_matrix_free(&L1);
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

static int run_cuda_match_test(size_t d, size_t batch_size)
{
    lb_system_t sys;
    build_amp_damp(&sys, d, 1.0 / 50.0);

    const size_t d2 = d * d;
    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, 0.5, &prop);
    lb_matrix_free(&L);

    lb_matrix_t rho_in[batch_size];
    lb_matrix_t rho_out_cpu[batch_size];
    lb_matrix_t rho_out_gpu[batch_size];
    if (alloc_batch(rho_in, batch_size, d) != 0 ||
        alloc_batch(rho_out_cpu, batch_size, d) != 0 ||
        alloc_batch(rho_out_gpu, batch_size, d) != 0) {
        return check("batch allocation", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        rho_in[b].data[0 * d + 0] = 0.6 + 0.0 * I;
        if (d >= 2) {
            rho_in[b].data[1 * d + 1] = 0.4 + 0.0 * I;
            rho_in[b].data[0 * d + 1] = 0.01 * (double)(b + 1) + 0.02 * I;
            rho_in[b].data[1 * d + 0] = 0.01 * (double)(b + 1) - 0.02 * I;
        }
    }

    if (lb_propagate_step_batch(&prop, rho_in, rho_out_cpu, batch_size) != 0) {
        return check("CPU batch propagation", 0);
    }
    if (lb_cuda_propagate_step_batch(&prop, rho_in, rho_out_gpu, batch_size) != 0) {
        return check("CUDA batch propagation", 0);
    }

    int ok = 1;
    for (size_t b = 0; b < batch_size; b++) {
        if (lb_hs_norm(&rho_out_cpu[b], &rho_out_gpu[b]) > TOL) {
            ok = 0;
        }
    }

    free_batch(rho_in, batch_size);
    free_batch(rho_out_cpu, batch_size);
    free_batch(rho_out_gpu, batch_size);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
    return ok;
}

static int run_cuda_raw_match_test(size_t d, size_t batch_size)
{
    lb_system_t sys;
    build_amp_damp(&sys, d, 1.0 / 50.0);

    const size_t d2 = d * d;
    const size_t state_bytes = d2 * sizeof(double complex);
    const size_t batch_bytes = batch_size * state_bytes;
    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, 0.5, &prop);
    lb_matrix_free(&L);

    lb_matrix_t rho_in[batch_size];
    lb_matrix_t rho_out_cpu[batch_size];
    lb_matrix_t rho_out_gpu[batch_size];
    if (alloc_batch(rho_in, batch_size, d) != 0 ||
        alloc_batch(rho_out_cpu, batch_size, d) != 0 ||
        alloc_batch(rho_out_gpu, batch_size, d) != 0) {
        return check("batch allocation", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        rho_in[b].data[0 * d + 0] = 0.5 + 0.0 * I;
        if (d >= 2) {
            rho_in[b].data[1 * d + 1] = 0.5 + 0.0 * I;
            rho_in[b].data[0 * d + 1] = 0.02 * (double)(b + 1) + 0.01 * I;
            rho_in[b].data[1 * d + 0] = 0.02 * (double)(b + 1) - 0.01 * I;
        }
    }

    if (lb_propagate_step_batch(&prop, rho_in, rho_out_cpu, batch_size) != 0) {
        return check("CPU batch propagation", 0);
    }

    double *rho_in_packed = malloc(batch_bytes);
    double *rho_out_packed = malloc(batch_bytes);
    if (!rho_in_packed || !rho_out_packed) {
        free(rho_in_packed);
        free(rho_out_packed);
        return check("raw packed allocation", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        memcpy((char *)rho_in_packed + b * state_bytes, rho_in[b].data, state_bytes);
    }

    lb_cuda_raw_batch_context_t ctx = {0};
    if (lb_cuda_raw_batch_init(&ctx, (const double *)prop.P.data, d2, batch_size) != 0 ||
        lb_cuda_raw_batch_upload_input(&ctx, rho_in_packed) != 0 ||
        lb_cuda_raw_batch_launch(&ctx) != 0 ||
        lb_cuda_raw_batch_download_output(&ctx, rho_out_packed) != 0) {
        lb_cuda_raw_batch_free(&ctx);
        free(rho_in_packed);
        free(rho_out_packed);
        return check("CUDA raw batch context", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        memcpy(rho_out_gpu[b].data, (const char *)rho_out_packed + b * state_bytes,
               state_bytes);
    }

    int ok = 1;
    for (size_t b = 0; b < batch_size; b++) {
        if (lb_hs_norm(&rho_out_cpu[b], &rho_out_gpu[b]) > TOL) {
            ok = 0;
        }
    }

    lb_cuda_raw_batch_free(&ctx);
    free(rho_in_packed);
    free(rho_out_packed);
    free_batch(rho_in, batch_size);
    free_batch(rho_out_cpu, batch_size);
    free_batch(rho_out_gpu, batch_size);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
    return ok;
}

int main(void)
{
    printf("=== test_cuda ===\n");
    int failures = 0;

    puts("test_cuda_matches_cpu_d3:");
    failures += !check("CUDA matches CPU at d=3", run_cuda_match_test(3, 7));

    puts("test_cuda_matches_cpu_d9:");
    failures += !check("CUDA matches CPU at d=9", run_cuda_match_test(9, 5));

    puts("test_cuda_matches_cpu_d27:");
    failures += !check("CUDA matches CPU at d=27", run_cuda_match_test(27, 3));

    puts("test_cuda_raw_context_matches_cpu_d9:");
    failures += !check("CUDA raw context matches CPU at d=9", run_cuda_raw_match_test(9, 5));

    printf("\n%d / 4 tests passed\n", 4 - failures);
    return failures > 0 ? 1 : 0;
}
