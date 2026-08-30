/**
 * test_grape_build.c — tests for reusable GRAPE construction helpers.
 *
 * Tests:
 *   1. coherent + dissipator decomposition matches full Lindbladian build
 *   2. workspace-backed propagator build matches the baseline propagator build
 *   3. decomposition still holds with two collapse operators (T1 + dephasing)
 *   4. stats-instrumented build returns the same propagator and a consistent
 *      squaring count
 *   5. drive schedule matches the pinned values shared with grape_reference.py
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

/* Mirror of lb_grape_drive_amplitude() in benchmarks/bench_grape.c. */
static double drive_amplitude(int s)
{
    double x = (double)(s + 1) * 0.6180339887498949;
    return 0.1 * ((x - floor(x)) - 0.5);
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

static int add_dephasing(lb_system_t *sys, double gamma_phi)
{
    size_t d = sys->d;
    lb_matrix_t L2 = {NULL, d};
    if (lb_matrix_alloc(&L2, d) != 0) return -1;
    L2.data[1 * d + 1] = sqrt(gamma_phi) + 0.0 * I;
    int rc = lb_system_add_cop(sys, &L2);
    lb_matrix_free(&L2);
    return rc;
}

static int test_two_cop_decomposition(void)
{
    puts("test_two_cop_decomposition:");

    const size_t d = 3;
    const size_t d2 = d * d;
    const double T1 = 50.0, T2 = 30.0;
    const double gamma1 = 1.0 / T1;
    const double gamma_phi = 1.0 / (1.0 / (1.0 / T2 - 0.5 / T1));
    const double omega = 0.0118;

    lb_system_t sys;
    if (build_driven_system(d, omega, gamma1, &sys) != 0) return 0;
    if (add_dephasing(&sys, gamma_phi) != 0) { lb_system_free(&sys); return 0; }

    lb_matrix_t H0 = {NULL, d}, Hdrive = {NULL, d};
    lb_matrix_t Lfull = {NULL, d2}, Lcoh0 = {NULL, d2}, Ldrive = {NULL, d2};
    lb_matrix_t Ldiss = {NULL, d2}, Lsum = {NULL, d2};
    lb_matrix_t *bufs[] = {&H0, &Hdrive, &Lfull, &Lcoh0, &Ldrive, &Ldiss, &Lsum};
    int ok = 0;
    int two_cops = 0;

    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (lb_matrix_alloc(bufs[i], (i < 2) ? d : d2) != 0) goto done;
    }
    for (size_t n = 0; n < d; n++) H0.data[n * d + n] = (double)n + 0.0 * I;
    Hdrive.data[0 * d + 1] = 1.0 + 0.0 * I;
    Hdrive.data[1 * d + 0] = 1.0 + 0.0 * I;

    if (lb_build_lindbladian(&sys, &Lfull) != 0) goto done;
    if (lb_build_coherent_superop(&H0, &Lcoh0) != 0) goto done;
    if (lb_build_coherent_superop(&Hdrive, &Ldrive) != 0) goto done;
    if (lb_build_dissipator_superop(&sys, &Ldiss) != 0) goto done;
    for (size_t i = 0; i < d2 * d2; i++) {
        Lsum.data[i] = Lcoh0.data[i] + omega * Ldrive.data[i] + Ldiss.data[i];
    }
    two_cops = (sys.n_cops == 2);
    ok = two_cops && hs_diff(&Lfull, &Lsum) < TOL;

done:
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) lb_matrix_free(bufs[i]);
    lb_system_free(&sys);
    check("system holds two collapse operators", two_cops);
    return check("two-cop decomposition matches full build", ok);
}

static int test_stats_build_matches(void)
{
    puts("test_stats_build_matches:");

    const size_t d = 3;
    const size_t d2 = d * d;
    const double gamma = 1.0 / 50.0;
    const double omega = 0.0354;
    const double dt = 0.5;

    lb_system_t sys;
    if (build_driven_system(d, omega, gamma, &sys) != 0) return 0;

    lb_matrix_t L = {NULL, d2};
    lb_propagator_t plain = {0};
    lb_propagator_t timed = {0};
    lb_expm_workspace_t ws = {0};
    lb_expm_stats_t st = {0};
    int same = 0, counts = 0, timers = 0;

    if (lb_matrix_alloc(&L, d2) != 0) goto done;
    if (lb_build_lindbladian(&sys, &L) != 0) goto done;
    if (lb_expm_workspace_alloc(&ws, d2) != 0) goto done;
    if (lb_build_propagator_ws(&L, dt, &plain, &ws) != 0) goto done;
    if (lb_build_propagator_ws_stats(&L, dt, &timed, &ws, &st) != 0) goto done;

    same = hs_diff(&plain.P, &timed.P) == 0.0;

    /* Expected squarings: ceil(log2(||L dt||_1 / 0.25)), the THETA_13 in expm.c */
    double norm = 0.0;
    for (size_t j = 0; j < d2; j++) {
        double col = 0.0;
        for (size_t i = 0; i < d2; i++) col += cabs(L.data[i * d2 + j] * dt);
        if (col > norm) norm = col;
    }
    long expected_s = (norm > 0.25) ? (long)ceil(log2(norm / 0.25)) : 0;
    counts = (st.n_builds == 1) && (st.n_squarings == expected_s);

    timers = st.ns_pade_mul > 0.0 && st.ns_solve > 0.0 && st.ns_scale > 0.0
          && (expected_s == 0 || st.ns_square > 0.0);

done:
    lb_matrix_free(&L);
    lb_propagator_free(&plain);
    lb_propagator_free(&timed);
    lb_expm_workspace_free(&ws);
    lb_system_free(&sys);
    check("stats build returns bit-identical propagator", same);
    check("stats squaring count matches norm rule", counts);
    return check("stats timers are populated", timers) && same && counts;
}

static int test_drive_schedule_pinned(void)
{
    puts("test_drive_schedule_pinned:");
    /* Values from reference/grape_reference.py build_drive_schedule(5). */
    const double expected[5] = {
        0.01180339887498949,
        -0.026393202250021022,
        0.03541019662496847,
        -0.002786404500042039,
        -0.040983005625052554,
    };
    int ok = 1;
    for (int s = 0; s < 5; s++) {
        if (fabs(drive_amplitude(s) - expected[s]) > 1e-15) ok = 0;
    }
    return check("first five drive amplitudes match Python", ok);
}

int main(void)
{
    printf("=== test_grape_build ===\n");
    int failures = 0;
    failures += !test_lindbladian_decomposition();
    failures += !test_propagator_workspace_matches();
    failures += !test_two_cop_decomposition();
    failures += !test_stats_build_matches();
    failures += !test_drive_schedule_pinned();
    printf("\n%d / 5 tests passed\n", 5 - failures);
    return failures ? 1 : 0;
}
