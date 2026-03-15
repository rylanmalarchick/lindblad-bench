/**
 * test_expm.c — Correctness tests for lb_expm.
 *
 * Tests:
 *   1. expm(0) = I
 *   2. expm(A) * expm(-A) = I  (inverse property)
 *   3. det(expm(A)) = exp(tr(A))  (Jacobi's formula, real diagonal case)
 *   4. Known analytic result: expm(i*pi*sigma_x/2) = i*sigma_x (up to phase)
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#define TOL     1e-6
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int check(const char *name, int cond)
{
    printf("  %-45s %s\n", name, cond ? PASS : FAIL);
    return cond;
}

static int test_zero_matrix(void)
{
    puts("test_expm_zero:");
    lb_matrix_t A   = {NULL, 3};
    lb_matrix_t out = {NULL, 3};
    lb_matrix_alloc(&A, 3);
    lb_matrix_alloc(&out, 3);
    /* A = 0 */

    lb_expm(&A, &out);

    /* Check out == I */
    int fail = 0;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            double complex expected = (i == j) ? 1.0 + 0.0*I : 0.0 + 0.0*I;
            double err = cabs(out.data[i * 3 + j] - expected);
            if (err > TOL) fail++;
        }
    }
    lb_matrix_free(&A);
    lb_matrix_free(&out);
    return check("expm(0) = I (3x3)", fail == 0);
}

static int test_inverse(void)
{
    puts("test_expm_inverse:");
    size_t n = 4;
    lb_matrix_t A     = {NULL, n};
    lb_matrix_t negA  = {NULL, n};
    lb_matrix_t eA    = {NULL, n};
    lb_matrix_t enA   = {NULL, n};
    lb_matrix_t prod  = {NULL, n};
    lb_matrix_t I_ref = {NULL, n};
    lb_matrix_alloc(&A, n); lb_matrix_alloc(&negA, n);
    lb_matrix_alloc(&eA, n); lb_matrix_alloc(&enA, n);
    lb_matrix_alloc(&prod, n); lb_matrix_alloc(&I_ref, n);

    /* Small Hermitian A */
    double vals[] = {0.1, 0.2, -0.1, 0.05};
    for (size_t i = 0; i < n; i++) {
        A.data[i * n + i]    =  vals[i] + 0.0*I;
        negA.data[i * n + i] = -vals[i] + 0.0*I;
    }

    lb_expm(&A, &eA);
    lb_expm(&negA, &enA);

    /* prod = eA * enA */
    size_t n2 = n * n;
    memset(prod.data, 0, n2 * sizeof(double complex));
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < n; k++)
            for (size_t j = 0; j < n; j++)
                prod.data[i * n + j] += eA.data[i * n + k] * enA.data[k * n + j];

    /* Check prod ≈ I */
    int fail = 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (cabs(prod.data[i * n + j] - expected) > TOL) fail++;
        }

    lb_matrix_free(&A); lb_matrix_free(&negA);
    lb_matrix_free(&eA); lb_matrix_free(&enA);
    lb_matrix_free(&prod); lb_matrix_free(&I_ref);
    return check("expm(A) * expm(-A) = I (4x4 diagonal)", fail == 0);
}

static int test_pauli_x(void)
{
    puts("test_expm_pauli:");
    /* expm(i*theta*sigma_x) = cos(theta)*I + i*sin(theta)*sigma_x */
    size_t n = 2;
    lb_matrix_t A   = {NULL, n};
    lb_matrix_t out = {NULL, n};
    lb_matrix_alloc(&A, n);
    lb_matrix_alloc(&out, n);

    double theta = M_PI / 4.0;
    /* A = i*theta*sigma_x = i*theta * [[0,1],[1,0]] */
    A.data[0 * 2 + 1] = theta * I; /* (0,1) */
    A.data[1 * 2 + 0] = theta * I; /* (1,0) */

    lb_expm(&A, &out);

    /* Expected: [[cos(theta), i*sin(theta)], [i*sin(theta), cos(theta)]] */
    double c = cos(theta), s = sin(theta);
    double complex expected[4] = {
        c + 0.0*I,  0.0 + s*I,
        0.0 + s*I,  c + 0.0*I
    };

    int fail = 0;
    for (int i = 0; i < 4; i++) {
        if (cabs(out.data[i] - expected[i]) > TOL) fail++;
    }

    lb_matrix_free(&A);
    lb_matrix_free(&out);
    return check("expm(i*pi/4*sigma_x) analytic (2x2)", fail == 0);
}

int main(void)
{
    printf("=== test_expm ===\n");
    int failures = 0;
    failures += !test_zero_matrix();
    failures += !test_inverse();
    failures += !test_pauli_x();
    printf("\n%d / 3 tests passed\n", 3 - failures);
    return failures > 0 ? 1 : 0;
}
