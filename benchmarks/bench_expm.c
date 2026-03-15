/**
 * bench_expm.c — Isolated benchmark for the matrix exponential.
 *
 * Measures lb_expm() cost at d² = 9, 81, 729 (d = 3, 9, 27).
 * expm is called once per propagator build, so its cost amortizes over
 * many timesteps in long trajectories — but it still matters for GRAPE
 * parameter sweeps where a new propagator is needed per iteration.
 */

#include "lindblad_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>

static long long ns_diff(struct timespec a, struct timespec b)
{
    return (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
         + (b.tv_nsec - a.tv_nsec);
}

static void run_bench(size_t n, int n_reps)
{
    printf("\n=== expm: n = %zu (%zu×%zu matrix, %.1f KB) ===\n",
           n, n, n, (double)(n * n * 16) / 1024.0);

    /* Random Hermitian matrix as test input */
    lb_matrix_t A   = {NULL, n};
    lb_matrix_t out = {NULL, n};
    lb_matrix_alloc(&A, n);
    lb_matrix_alloc(&out, n);

    /* Fill with small-norm Hermitian matrix (so expm is well-conditioned) */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            double re = 0.01 * ((double)rand() / RAND_MAX - 0.5);
            double im = (i == j) ? 0.0 : 0.01 * ((double)rand() / RAND_MAX - 0.5);
            A.data[i * n + j] =  re + im * I;
            A.data[j * n + i] =  re - im * I;
        }
    }

    /* Warm-up */
    lb_expm(&A, &out);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < n_reps; r++) {
        lb_expm(&A, &out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long long total_ns = ns_diff(t0, t1);
    double us_per_call = (double)total_ns / (double)n_reps / 1e3;

    printf("  n_reps      : %d\n", n_reps);
    printf("  us / call   : %.1f µs\n", us_per_call);
    printf("  ms total    : %.2f ms\n", (double)total_ns / 1e6);

    lb_matrix_free(&A);
    lb_matrix_free(&out);
}

int main(void)
{
    srand(42);
    printf("lindblad-bench: lb_expm timing\n");
    run_bench(9,   1000);  /* d=3: Lindbladian is 9×9  */
    run_bench(81,  100);   /* d=9: 81×81               */
    run_bench(729, 5);     /* d=27: 729×729             */
    return 0;
}
