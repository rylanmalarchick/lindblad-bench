/**
 * test_grape_build.c — tests for reusable GRAPE construction helpers.
 *
 * Tests:
 *   1. coherent + dissipator decomposition matches full Lindbladian build
 *   2. workspace-backed propagator build matches the baseline propagator build
 */

#include "lindblad_bench.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-10
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int check(const char *name, int cond)
{
    printf("  %-55s %s\n", name, cond ? PASS : FAIL);
    return cond;
}

static double hs_diff(const lb_matrix_t *A, const lb_matrix_t *B)
{
    double acc = 0.0;
    size_t n2 = A->dim * A->dim;
    for (size_t i = 0; i < n2; i++) {
        double complex d = A->data[i] - B->data[i];
        acc += creal(d) * creal(d) + cimag(d) * cimag(d);
    }
    return sqrt(acc);
}

static int build_driven_system(size_t d, double omega, double gamma, lb_system_t *sys)
{
    lb_system_init(sys, d);
    if (lb_matrix_alloc(&sys->H, d) != 0) return -1;

    for (size_t n = 0; n < d; n++) {
        sys->H.data[n * d + n] = (double)n + 0.0 * I;
    }
    if (d >= 2) {
        sys->H.data[0 * d + 1] += omega + 0.0 * I;
        sys->H.data[1 * d + 0] += omega + 0.0 * I;
    }

    if (d >= 2) {
        lb_matrix_t L1 = {NULL, d};
        if (lb_matrix_alloc(&L1, d) != 0) {
            lb_system_free(sys);
            return -1;
        }
        L1.data[0 * d + 1] = sqrt(gamma) + 0.0 * I;
        if (lb_system_add_cop(sys, &L1) != 0) {
            lb_matrix_free(&L1);
            lb_system_free(sys);
            return -1;
        }
        lb_matrix_free(&L1);
    }

    return 0;
}

static int test_lindbladian_decomposition(void)
{
    puts("test_lindbladian_decomposition:");

    const size_t d = 3;
    const size_t d2 = d * d;
    const double gamma = 1.0 / 50.0;
    const double omega = 0.0375;

    lb_system_t sys;
    if (build_driven_system(d, omega, gamma, &sys) != 0) return 0;

    lb_matrix_t H0 = {NULL, d};
    lb_matrix_t Hdrive = {NULL, d};
    lb_matrix_t Lfull = {NULL, d2};
    lb_matrix_t Lcoh0 = {NULL, d2};
    lb_matrix_t Ldrive = {NULL, d2};
    lb_matrix_t Ldiss = {NULL, d2};
    lb_matrix_t Lsum = {NULL, d2};
    lb_matrix_t *bufs[] = {&H0, &Hdrive, &Lfull, &Lcoh0, &Ldrive, &Ldiss, &Lsum};

    int ok = 0;
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        size_t dim = (i < 2) ? d : d2;
        if (lb_matrix_alloc(bufs[i], dim) != 0) goto done;
    }

    for (size_t n = 0; n < d; n++) {
        H0.data[n * d + n] = (double)n + 0.0 * I;
    }
    if (d >= 2) {
        Hdrive.data[0 * d + 1] = 1.0 + 0.0 * I;
        Hdrive.data[1 * d + 0] = 1.0 + 0.0 * I;
    }

    if (lb_build_lindbladian(&sys, &Lfull) != 0) goto done;
    if (lb_build_coherent_superop(&H0, &Lcoh0) != 0) goto done;
    if (lb_build_coherent_superop(&Hdrive, &Ldrive) != 0) goto done;
    if (lb_build_dissipator_superop(&sys, &Ldiss) != 0) goto done;

    for (size_t i = 0; i < d2 * d2; i++) {
        Lsum.data[i] = Lcoh0.data[i] + omega * Ldrive.data[i] + Ldiss.data[i];
    }

    ok = hs_diff(&Lfull, &Lsum) < TOL;

done:
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        lb_matrix_free(bufs[i]);
    }
    lb_system_free(&sys);
    return check("coherent + dissipator decomposition matches full build", ok);
}

static int test_propagator_workspace_matches(void)
{
    puts("test_propagator_workspace_matches:");

    const size_t d = 3;
    const size_t d2 = d * d;
    const double gamma = 1.0 / 50.0;
    const double omega = -0.021;
    const double dt = 0.5;

    lb_system_t sys;
    if (build_driven_system(d, omega, gamma, &sys) != 0) return 0;

    lb_matrix_t L = {NULL, d2};
    lb_propagator_t base = {0};
    lb_propagator_t ws_prop = {0};
    lb_expm_workspace_t ws = {0};
    int ok = 0;

    if (lb_matrix_alloc(&L, d2) != 0) goto done;
    if (lb_build_lindbladian(&sys, &L) != 0) goto done;
    if (lb_build_propagator(&L, dt, &base) != 0) goto done;
    if (lb_expm_workspace_alloc(&ws, d2) != 0) goto done;
    if (lb_build_propagator_ws(&L, dt, &ws_prop, &ws) != 0) goto done;

    ok = hs_diff(&base.P, &ws_prop.P) < TOL;

done:
    lb_matrix_free(&L);
    lb_propagator_free(&base);
    lb_propagator_free(&ws_prop);
    lb_expm_workspace_free(&ws);
    lb_system_free(&sys);
    return check("workspace propagator build matches baseline", ok);
}

int main(void)
{
    printf("=== test_grape_build ===\n");
    int failures = 0;
    failures += !test_lindbladian_decomposition();
    failures += !test_propagator_workspace_matches();
    printf("\n%d / 2 tests passed\n", 2 - failures);
    return failures ? 1 : 0;
}
