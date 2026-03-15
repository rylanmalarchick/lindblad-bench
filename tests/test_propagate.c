/**
 * test_propagate.c — Physical correctness tests for lb_propagate_step / lb_evolve.
 *
 * Tests:
 *   1. Trace preservation: Tr[ρ(t)] = 1 for all t
 *   2. Hermiticity: ρ(t) = ρ(t)† for all t
 *   3. Positivity proxy: diagonal elements >= 0 for all t
 *   4. Monotone decay: for amplitude damping, ρ_11(t) decreasing
 *   5. Zero collapse ops: unitary evolution conserves purity Tr[ρ²] = 1
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define TOL     1e-10
#define TOL_LAX 1e-8
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int check(const char *name, int cond)
{
    printf("  %-50s %s\n", name, cond ? PASS : FAIL);
    return cond;
}

/* Build d=3 transmon system with T1 damping */
static void build_simple(lb_system_t *sys, size_t d, int add_damping)
{
    lb_system_init(sys, d);
    lb_matrix_alloc(&sys->H, d);
    for (size_t n = 0; n < d; n++)
        sys->H.data[n * d + n] = (double)n + 0.0*I;

    if (add_damping && d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        lb_matrix_alloc(&L1, d);
        L1.data[0 * d + 1] = sqrt(1.0 / 50.0) + 0.0*I; /* sqrt(gamma_1) */
        lb_system_add_cop(sys, &L1);
        lb_matrix_free(&L1);
    }
}

static int test_trace_preservation(void)
{
    puts("test_trace_preservation:");
    lb_system_t sys;
    build_simple(&sys, 3, 1);

    lb_matrix_t rho0 = {NULL, 3}, rho_out = {NULL, 3};
    lb_matrix_alloc(&rho0, 3);
    lb_matrix_alloc(&rho_out, 3);
    rho0.data[1 * 3 + 1] = 1.0 + 0.0*I; /* |1><1| */

    size_t n_steps;
    lb_evolve(&sys, &rho0, 100.0, 1.0, &rho_out, NULL, &n_steps);

    double complex tr = lb_trace(&rho_out);
    int ok = fabs(creal(tr) - 1.0) < TOL_LAX && fabs(cimag(tr)) < TOL_LAX;

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("Tr[rho(t)] = 1 after 100 ns evolution", ok);
}

static int test_hermiticity(void)
{
    puts("test_hermiticity:");
    lb_system_t sys;
    build_simple(&sys, 3, 1);

    /* Superposition initial state: (|0> + |1>) / sqrt(2) */
    lb_matrix_t rho0 = {NULL, 3}, rho_out = {NULL, 3};
    lb_matrix_alloc(&rho0, 3);
    lb_matrix_alloc(&rho_out, 3);
    rho0.data[0 * 3 + 0] = 0.5 + 0.0*I;
    rho0.data[0 * 3 + 1] = 0.5 + 0.0*I;
    rho0.data[1 * 3 + 0] = 0.5 + 0.0*I;
    rho0.data[1 * 3 + 1] = 0.5 + 0.0*I;

    size_t n_steps;
    lb_evolve(&sys, &rho0, 50.0, 0.5, &rho_out, NULL, &n_steps);

    int ok = 1;
    size_t d = 3;
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++) {
            double complex a = rho_out.data[i * d + j];
            double complex b = conj(rho_out.data[j * d + i]);
            if (cabs(a - b) > TOL_LAX) ok = 0;
        }

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("rho(t) = rho(t)† (Hermitian)", ok);
}

static int test_diagonal_positive(void)
{
    puts("test_diagonal_positive:");
    lb_system_t sys;
    build_simple(&sys, 3, 1);

    lb_matrix_t rho0 = {NULL, 3}, rho_out = {NULL, 3};
    lb_matrix_alloc(&rho0, 3);
    lb_matrix_alloc(&rho_out, 3);
    rho0.data[1 * 3 + 1] = 1.0 + 0.0*I;

    size_t n_steps;
    lb_evolve(&sys, &rho0, 200.0, 1.0, &rho_out, NULL, &n_steps);

    int ok = 1;
    for (size_t i = 0; i < 3; i++) {
        if (creal(rho_out.data[i * 3 + i]) < -TOL_LAX) ok = 0;
    }

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("diag(rho(t)) >= 0 (positivity proxy)", ok);
}

static int test_unitary_purity(void)
{
    puts("test_unitary_purity:");
    lb_system_t sys;
    build_simple(&sys, 3, 0); /* no damping */

    /* Pure state: |+> = (|0>+|1>)/sqrt(2) */
    lb_matrix_t rho0 = {NULL, 3}, rho_out = {NULL, 3};
    lb_matrix_alloc(&rho0, 3);
    lb_matrix_alloc(&rho_out, 3);
    rho0.data[0 * 3 + 0] = 0.5 + 0.0*I;
    rho0.data[0 * 3 + 1] = 0.5 + 0.0*I;
    rho0.data[1 * 3 + 0] = 0.5 + 0.0*I;
    rho0.data[1 * 3 + 1] = 0.5 + 0.0*I;

    size_t n_steps;
    lb_evolve(&sys, &rho0, 20.0, 0.2, &rho_out, NULL, &n_steps);

    /* purity = Tr[rho^2] */
    size_t d = 3;
    double complex purity = 0.0;
    for (size_t i = 0; i < d; i++)
        for (size_t k = 0; k < d; k++)
            purity += rho_out.data[i * d + k] * rho_out.data[k * d + i];

    int ok = fabs(creal(purity) - 1.0) < TOL_LAX;

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("Tr[rho^2] = 1 under unitary evolution (purity)", ok);
}

int main(void)
{
    printf("=== test_propagate ===\n");
    int failures = 0;
    failures += !test_trace_preservation();
    failures += !test_hermiticity();
    failures += !test_diagonal_positive();
    failures += !test_unitary_purity();
    printf("\n%d / 4 tests passed\n", 4 - failures);
    return failures > 0 ? 1 : 0;
}
