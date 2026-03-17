/**
 * test_evolve.c — End-to-end evolution tests and analytic comparisons.
 *
 * Tests:
 *   1. Amplitude damping steady state: |1><1| -> |0><0| as t -> inf
 *   2. lb_evolve_prop matches lb_evolve for the same system
 *   3. Trajectory storage: stored states are consistent with final state
 *   4. Purity decay: Tr[rho^2] < 1 under dissipation from a pure state
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#define TOL     1e-6
#define TOL_LAX 1e-4
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int check(const char *name, int cond)
{
    printf("  %-55s %s\n", name, cond ? PASS : FAIL);
    return cond;
}

/* Build d=2 system with pure amplitude damping (T1 = 50 us) */
static void build_amp_damp(lb_system_t *sys, size_t d, double gamma)
{
    lb_system_init(sys, d);
    lb_matrix_alloc(&sys->H, d);
    /* H = diag(0, 1, ...) */
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

static int test_steady_state(void)
{
    puts("test_steady_state:");
    /* For amplitude damping |0><1| with rate gamma, starting from |1><1|,
     * the steady state is |0><0|. Evolve long enough to approach it. */
    size_t d = 2;
    double gamma = 1.0 / 50.0;
    lb_system_t sys;
    build_amp_damp(&sys, d, gamma);

    lb_matrix_t rho0 = {NULL, d}, rho_out = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    lb_matrix_alloc(&rho_out, d);
    rho0.data[1 * d + 1] = 1.0 + 0.0*I; /* |1><1| */

    /* Evolve for 5*T1 = 250 us — population should be ~e^{-5} ≈ 0.0067 */
    size_t n_steps;
    lb_evolve(&sys, &rho0, 250.0, 0.5, &rho_out, NULL, &n_steps);

    double p0 = creal(rho_out.data[0 * d + 0]); /* ground state pop */
    double p1 = creal(rho_out.data[1 * d + 1]); /* excited state pop */
    double expected_p1 = exp(-gamma * 250.0);

    int ok = fabs(p1 - expected_p1) < TOL_LAX && fabs(p0 - (1.0 - expected_p1)) < TOL_LAX;

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("steady state: |1> -> |0> under amp damping (5*T1)", ok);
}

static int test_evolve_prop_matches(void)
{
    puts("test_evolve_prop_matches:");
    /* lb_evolve and lb_evolve_prop should produce identical results */
    size_t d = 3;
    double gamma = 1.0 / 50.0;
    double dt = 0.5;
    double t_end = 50.0;

    lb_system_t sys;
    build_amp_damp(&sys, d, gamma);

    lb_matrix_t rho0 = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    rho0.data[1 * d + 1] = 1.0 + 0.0*I;

    /* Method 1: lb_evolve */
    lb_matrix_t rho_a = {NULL, d};
    lb_matrix_alloc(&rho_a, d);
    size_t n_steps;
    lb_evolve(&sys, &rho0, t_end, dt, &rho_a, NULL, &n_steps);

    /* Method 2: lb_evolve_prop */
    size_t d2 = d * d;
    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    lb_propagator_t prop = {0};
    lb_build_propagator(&L, dt, &prop);
    lb_matrix_free(&L);

    lb_matrix_t rho_b = {NULL, d};
    lb_matrix_alloc(&rho_b, d);
    lb_evolve_prop(&prop, &rho0, n_steps, &rho_b, NULL);

    double diff = lb_hs_norm(&rho_a, &rho_b);
    int ok = diff < 1e-12;

    lb_matrix_free(&rho0); lb_matrix_free(&rho_a); lb_matrix_free(&rho_b);
    lb_propagator_free(&prop);
    lb_system_free(&sys);
    return check("lb_evolve_prop matches lb_evolve (HS norm < 1e-12)", ok);
}

static int test_trajectory_storage(void)
{
    puts("test_trajectory_storage:");
    size_t d = 2;
    double gamma = 1.0 / 50.0;
    double dt = 1.0;
    double t_end = 10.0;
    size_t expected_steps = 10;

    lb_system_t sys;
    build_amp_damp(&sys, d, gamma);

    lb_matrix_t rho0 = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    rho0.data[1 * d + 1] = 1.0 + 0.0*I;

    /* Allocate trajectory storage: n_steps + 1 states */
    lb_matrix_t *traj = calloc(expected_steps + 1, sizeof(lb_matrix_t));
    for (size_t i = 0; i <= expected_steps; i++)
        lb_matrix_alloc(&traj[i], d);

    lb_matrix_t rho_out = {NULL, d};
    lb_matrix_alloc(&rho_out, d);

    size_t n_steps;
    lb_evolve(&sys, &rho0, t_end, dt, &rho_out, traj, &n_steps);

    /* Check: last trajectory state matches rho_out */
    double diff = lb_hs_norm(&traj[n_steps], &rho_out);
    int ok_final = diff < 1e-12;

    /* Check: initial trajectory state matches rho0 */
    double diff0 = lb_hs_norm(&traj[0], &rho0);
    int ok_init = diff0 < 1e-12;

    /* Check: trace preserved at every stored step */
    int ok_trace = 1;
    for (size_t i = 0; i <= n_steps; i++) {
        double complex tr = lb_trace(&traj[i]);
        if (fabs(creal(tr) - 1.0) > 1e-8 || fabs(cimag(tr)) > 1e-8)
            ok_trace = 0;
    }

    /* Check: excited state population monotonically decreasing */
    int ok_decay = 1;
    for (size_t i = 1; i <= n_steps; i++) {
        double p1_prev = creal(traj[i-1].data[1 * d + 1]);
        double p1_cur  = creal(traj[i].data[1 * d + 1]);
        if (p1_cur > p1_prev + 1e-10) ok_decay = 0;
    }

    for (size_t i = 0; i <= expected_steps; i++)
        lb_matrix_free(&traj[i]);
    free(traj);
    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);

    int ok = ok_final && ok_init && ok_trace && ok_decay;
    check("trajectory[0] matches rho0", ok_init);
    check("trajectory[end] matches rho_out", ok_final);
    check("trace = 1 at every stored step", ok_trace);
    check("excited pop monotonically decreasing", ok_decay);
    return ok;
}

static int test_purity_decay(void)
{
    puts("test_purity_decay:");
    /* Start from pure |+> = (|0>+|1>)/sqrt(2), evolve under damping.
     * Purity Tr[rho^2] must decrease from 1 (pure) toward something < 1. */
    size_t d = 2;
    double gamma = 1.0 / 50.0;

    lb_system_t sys;
    build_amp_damp(&sys, d, gamma);

    lb_matrix_t rho0 = {NULL, d}, rho_out = {NULL, d};
    lb_matrix_alloc(&rho0, d);
    lb_matrix_alloc(&rho_out, d);
    /* |+><+| */
    rho0.data[0 * d + 0] = 0.5 + 0.0*I;
    rho0.data[0 * d + 1] = 0.5 + 0.0*I;
    rho0.data[1 * d + 0] = 0.5 + 0.0*I;
    rho0.data[1 * d + 1] = 0.5 + 0.0*I;

    size_t n_steps;
    lb_evolve(&sys, &rho0, 25.0, 0.5, &rho_out, NULL, &n_steps);

    /* Compute Tr[rho^2] */
    double complex purity = 0.0;
    for (size_t i = 0; i < d; i++)
        for (size_t k = 0; k < d; k++)
            purity += rho_out.data[i * d + k] * rho_out.data[k * d + i];

    /* Purity should be < 1 but > 0.5 (not yet fully decayed) */
    double p = creal(purity);
    int ok = p < 1.0 - 1e-6 && p > 0.5;

    lb_matrix_free(&rho0); lb_matrix_free(&rho_out);
    lb_system_free(&sys);
    return check("purity decays: 0.5 < Tr[rho^2] < 1 after 25 us", ok);
}

int main(void)
{
    printf("=== test_evolve ===\n");
    int failures = 0;
    failures += !test_steady_state();
    failures += !test_evolve_prop_matches();
    failures += !test_trajectory_storage();
    failures += !test_purity_decay();
    printf("\n%d / 4 tests passed\n", 4 - failures);
    return failures > 0 ? 1 : 0;
}
