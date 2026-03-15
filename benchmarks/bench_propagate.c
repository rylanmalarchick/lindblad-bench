/**
 * bench_propagate.c — Timing benchmark for lb_propagate_step.
 *
 * This is the Roofline subject: a (d²)×(d²) complex matvec.
 * Reports wall time, estimated FLOP/s, and achieved bandwidth.
 *
 * Run at d = 3, 9, 27 to observe L1 -> L2 -> L3 cache transitions.
 *
 * Usage:
 *   ./bench_propagate [n_reps]
 *   Default n_reps = 10000
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <complex.h>

/* Nanoseconds between two timespecs */
static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

/* Build a trivial transmon-like system at dimension d.
 * H = diag(0, 1, 2*(1 + alpha)) for d=3 (harmonic + anharmonicity).
 * Collapse ops: sqrt(1/T1)*|0><1|, sqrt(1/T2)*sigma_z (truncated to d). */
static void build_transmon_system(lb_system_t *sys, size_t d,
                                  double T1_us, double T2_us)
{
    lb_system_init(sys, d);
    lb_matrix_alloc(&sys->H, d);

    /* Diagonal Hamiltonian: E_n = n for simplicity */
    for (size_t n = 0; n < d; n++) {
        sys->H.data[n * d + n] = (double)n + 0.0 * I;
    }

    /* Amplitude damping: L1 = sqrt(1/T1) * |0><1| */
    double gamma1 = 1.0 / T1_us;
    lb_matrix_t L1 = {NULL, d};
    lb_matrix_alloc(&L1, d);
    if (d >= 2) L1.data[0 * d + 1] = sqrt(gamma1) + 0.0 * I;
    lb_system_add_cop(sys, &L1);
    lb_matrix_free(&L1);

    /* Pure dephasing: L2 = sqrt(1/T_phi) * |1><1| */
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

static void run_bench(size_t d, long n_reps)
{
    size_t d2 = d * d;
    printf("\n=== d = %zu (d² = %zu, P size = %.1f KB) ===\n",
           d, d2, (double)(d2 * d2 * 16) / 1024.0);

    /* Build system and propagator */
    lb_system_t sys;
    build_transmon_system(&sys, d, 50.0, 30.0); /* IQM Garnet-like T1/T2 */

    lb_matrix_t L = {NULL, d2};
    lb_matrix_alloc(&L, d2);
    lb_build_lindbladian(&sys, &L);

    double dt = 0.5; /* ns, arbitrary fixed step */
    lb_propagator_t prop = {0};
    lb_build_propagator(&L, dt, &prop);
    lb_matrix_free(&L);
    lb_system_free(&sys);

    /* Initial state: |0><0| */
    lb_matrix_t rho_in  = {NULL, d};
    lb_matrix_t rho_out = {NULL, d};
    lb_matrix_alloc(&rho_in,  d);
    lb_matrix_alloc(&rho_out, d);
    rho_in.data[0] = 1.0 + 0.0 * I; /* |0><0| */

    /* Warm-up */
    for (int w = 0; w < 10; w++) {
        lb_propagate_step(&prop, &rho_in, &rho_out);
    }

    /* Timed loop */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long r = 0; r < n_reps; r++) {
        lb_propagate_step(&prop, &rho_in, &rho_out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long long total_ns = ns_diff(t0, t1);
    double ns_per_step = (double)total_ns / (double)n_reps;
    double s_total     = (double)total_ns * 1e-9;

    /* FLOPs: d^4 complex multiply-adds = 2*d^4 real FMA = 8*d^4 real ops */
    double flops_per_step = 8.0 * (double)d2 * (double)d2;
    double gflops = (flops_per_step * (double)n_reps) / (s_total * 1e9);

    /* Bandwidth: read P (d^4 * 16 B) + read vec_in (d^2 * 16 B) + write vec_out */
    double bytes_per_step = (double)(d2 * d2 + 2 * d2) * 16.0;
    double gbytes_s = (bytes_per_step * (double)n_reps) / (s_total * 1e9);

    printf("  n_reps       : %ld\n", n_reps);
    printf("  total time   : %.3f ms\n", (double)total_ns / 1e6);
    printf("  ns / step    : %.2f ns\n", ns_per_step);
    printf("  throughput   : %.3f GFLOP/s\n", gflops);
    printf("  bandwidth    : %.3f GB/s\n", gbytes_s);
    printf("  arith intens : %.4f FLOP/byte\n",
           flops_per_step / bytes_per_step);

    lb_matrix_free(&rho_in);
    lb_matrix_free(&rho_out);
    lb_propagator_free(&prop);
}

int main(int argc, char **argv)
{
    long n_reps = 10000;
    if (argc > 1) n_reps = atol(argv[1]);

    printf("lindblad-bench: lb_propagate_step timing\n");
    printf("Compiler flags: see CMakeLists.txt\n");
    printf("n_reps = %ld\n", n_reps);

    run_bench(3,  n_reps);   /* d=3:  9×9   matvec, P in L1  */
    run_bench(9,  n_reps);   /* d=9:  81×81 matvec, P in L2  */
    run_bench(27, n_reps);   /* d=27: 729×729 matvec, P in L3 */

    return 0;
}
