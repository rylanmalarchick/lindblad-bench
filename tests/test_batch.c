/**
 * test_batch.c — Batch propagation / evolution correctness tests.
 *
 * Tests:
 *   1. lb_propagate_step_batch matches repeated lb_propagate_step
 *   2. lb_evolve_prop_batch matches repeated lb_evolve_prop
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

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
    for (size_t n = 0; n < d; n++)
        sys->H.data[n * d + n] = (double)n + 0.0*I;

    if (d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        lb_matrix_alloc(&L1, d);
        L1.data[0 * d + 1] = sqrt(gamma) + 0.0*I;
        lb_system_add_cop(sys, &L1);
        lb_matrix_free(&L1);
    }
}

static int alloc_batch(lb_matrix_t *xs, size_t batch_size, size_t d)
{
    for (size_t i = 0; i < batch_size; i++) {
        if (lb_matrix_alloc(&xs[i], d) != 0) {
            for (size_t j = 0; j < i; j++) lb_matrix_free(&xs[j]);
            return -1;
        }
    }
    return 0;
}

static void free_batch(lb_matrix_t *xs, size_t batch_size)
{
    for (size_t i = 0; i < batch_size; i++) lb_matrix_free(&xs[i]);
}

static int test_propagate_step_batch_matches_serial(void)
{
    puts("test_propagate_step_batch_matches_serial:");

    const size_t d = 3;
    const size_t batch_size = 4;
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
    lb_matrix_t rho_out_serial[batch_size];
    lb_matrix_t rho_out_batch[batch_size];
    if (alloc_batch(rho_in, batch_size, d) != 0 ||
        alloc_batch(rho_out_serial, batch_size, d) != 0 ||
        alloc_batch(rho_out_batch, batch_size, d) != 0) {
        return check("batch allocation", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        double p = 1.0 / (double)batch_size;
        rho_in[b].data[0 * d + 0] = p + 0.0*I;
        rho_in[b].data[1 * d + 1] = (1.0 - p) + 0.0*I;
        rho_in[b].data[0 * d + 1] = 0.1 * (double)(b + 1) + 0.0*I;
        rho_in[b].data[1 * d + 0] = 0.1 * (double)(b + 1) + 0.0*I;
    }

    for (size_t b = 0; b < batch_size; b++)
        lb_propagate_step(&prop, &rho_in[b], &rho_out_serial[b]);

    if (lb_propagate_step_batch(&prop, rho_in, rho_out_batch, batch_size) != 0) {
        free_batch(rho_in, batch_size);
        free_batch(rho_out_serial, batch_size);
        free_batch(rho_out_batch, batch_size);
        lb_propagator_free(&prop);
        lb_system_free(&sys);
        return check("lb_propagate_step_batch returned success", 0);
    }

    int ok = 1;
    for (size_t b = 0; b < batch_size; b++) {
        if (lb_hs_norm(&rho_out_serial[b], &rho_out_batch[b]) > TOL) ok = 0;
    }

    free_batch(rho_in, batch_size);
    free_batch(rho_out_serial, batch_size);
    free_batch(rho_out_batch, batch_size);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
    return check("lb_propagate_step_batch matches serial loop", ok);
}

static int test_evolve_prop_batch_matches_serial(void)
{
    puts("test_evolve_prop_batch_matches_serial:");

    const size_t d = 3;
    const size_t batch_size = 3;
    const size_t n_steps = 20;
    lb_system_t sys;
    build_amp_damp(&sys, d, 1.0 / 50.0);

    const size_t d2 = d * d;
    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, 0.5, &prop);
    lb_matrix_free(&L);

    lb_matrix_t rho0[batch_size];
    lb_matrix_t rho_out_serial[batch_size];
    lb_matrix_t rho_out_batch[batch_size];
    if (alloc_batch(rho0, batch_size, d) != 0 ||
        alloc_batch(rho_out_serial, batch_size, d) != 0 ||
        alloc_batch(rho_out_batch, batch_size, d) != 0) {
        return check("batch allocation", 0);
    }

    for (size_t b = 0; b < batch_size; b++) {
        rho0[b].data[b * d + b] = 1.0 + 0.0*I;
    }

    for (size_t b = 0; b < batch_size; b++)
        lb_evolve_prop(&prop, &rho0[b], n_steps, &rho_out_serial[b], NULL);

    if (lb_evolve_prop_batch(&prop, rho0, batch_size, n_steps, rho_out_batch) != 0) {
        free_batch(rho0, batch_size);
        free_batch(rho_out_serial, batch_size);
        free_batch(rho_out_batch, batch_size);
        lb_propagator_free(&prop);
        lb_system_free(&sys);
        return check("lb_evolve_prop_batch returned success", 0);
    }

    int ok = 1;
    for (size_t b = 0; b < batch_size; b++) {
        if (lb_hs_norm(&rho_out_serial[b], &rho_out_batch[b]) > TOL) ok = 0;
    }

    free_batch(rho0, batch_size);
    free_batch(rho_out_serial, batch_size);
    free_batch(rho_out_batch, batch_size);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
    return check("lb_evolve_prop_batch matches serial loop", ok);
}

int main(void)
{
    printf("=== test_batch ===\n");
    int failures = 0;
    failures += !test_propagate_step_batch_matches_serial();
    failures += !test_evolve_prop_batch_matches_serial();
    printf("\n%d / 2 tests passed\n", 2 - failures);
    return failures > 0 ? 1 : 0;
}
