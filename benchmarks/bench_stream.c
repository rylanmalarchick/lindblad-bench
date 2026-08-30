/**
 * bench_stream.c — Read-bandwidth probe for the cost model.
 *
 * Streams a buffer of W bytes R times with a summing kernel, so the
 * measured rate is the read bandwidth of whichever cache level holds W
 * bytes. With OpenMP, each thread streams its own private buffer of W
 * bytes, which mirrors the batch-propagation pattern (independent states,
 * shared nothing except the propagator).
 *
 * Sizes default to the three propagator working sets plus a DRAM-sized
 * buffer. Each size runs one warm-up pass and n_trials timed passes.
 *
 * With --shared every thread streams the same buffer, which mirrors the
 * shared propagator P. Without it each thread has a private buffer.
 *
 * Usage:
 *   bench_stream [--csv] [--trials N] [--reps R] [--shared]
 * Environment:
 *   HOST_TAG, GIT_COMMIT   tags for the CSV rows
 *   SIZES_KB="1.5 105 8300 65536"   space-separated buffer sizes in KB
 *   OMP_NUM_THREADS        thread count (1 without OpenMP)
 */

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Sum W/8 doubles with 16 independent partial sums, so the loop is bound
 * by load throughput rather than by floating-point add latency. */
static double stream_sum(const double *buf, size_t n)
{
    double acc[16] = {0.0};
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        for (int k = 0; k < 16; k++) acc[k] += buf[i + k];
    }
    for (; i < n; i++) acc[0] += buf[i];
    double s = 0.0;
    for (int k = 0; k < 16; k++) s += acc[k];
    return s;
}

/* Cost of one empty OpenMP parallel region, the fork/join term. */
static double fork_join_ns(int reps)
{
#ifdef _OPENMP
    volatile int sink = 0;
    for (int w = 0; w < 10; w++) {
        #pragma omp parallel
        { if (omp_get_thread_num() == 12345) sink++; }
    }
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        #pragma omp parallel
        { if (omp_get_thread_num() == 12345) sink++; }
    }
    return (now_ns() - t0) / reps;
#else
    (void)reps;
    return 0.0;
#endif
}

static int threads_available(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int g_shared = 0;

static void run_size(double size_kb, int reps, int n_trials, int csv,
                     const char *machine, const char *commit)
{
    size_t bytes = (size_t)(size_kb * 1024.0);
    size_t n = bytes / sizeof(double);
    if (n < 4) n = 4;
    bytes = n * sizeof(double);
    int nt = threads_available();

    double **bufs = malloc((size_t)nt * sizeof(double *));
    for (int t = 0; t < nt; t++) {
        if (g_shared && t > 0) { bufs[t] = bufs[0]; continue; }
        bufs[t] = aligned_alloc(64, ((bytes + 63) / 64) * 64);
        for (size_t i = 0; i < n; i++) bufs[t][i] = (double)(i % 7) * 0.5;
    }
    volatile double sink = 0.0;

    for (int tr = -1; tr < n_trials; tr++) {
        double t0 = now_ns();
#ifdef _OPENMP
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            double local = 0.0;
            for (int r = 0; r < reps; r++) local += stream_sum(bufs[t], n);
            #pragma omp critical
            sink += local;
        }
#else
        for (int r = 0; r < reps; r++) sink += stream_sum(bufs[0], n);
#endif
        double t1 = now_ns();
        if (tr < 0) continue;

        double total_bytes = (double)bytes * (double)reps * (double)nt;
        double gbs_aggregate = total_bytes / (t1 - t0);
        double gbs_per_thread = gbs_aggregate / nt;
        if (csv) {
            printf("%s,%s,%d,%.1f,%d,%s,%d,%zu,%.3f,%.3f\n",
                   machine, commit, tr, size_kb, nt, g_shared ? "shared" : "private",
                   reps, bytes, gbs_per_thread, gbs_aggregate);
        } else {
            printf("  size=%9.1f KB threads=%2d trial=%d  %.2f GB/s per thread, %.2f GB/s aggregate\n",
                   size_kb, nt, tr, gbs_per_thread, gbs_aggregate);
        }
    }
    if (sink == 12345.678) printf("sink %f\n", sink); /* keep the sum alive */
    for (int t = 0; t < nt; t++) if (!g_shared || t == 0) free(bufs[t]);
    free(bufs);
}

int main(int argc, char **argv)
{
    int csv = 0, n_trials = 5, reps = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) csv = 1;
        else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) n_trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--shared") == 0) g_shared = 1;
        else { fprintf(stderr, "usage: %s [--csv] [--trials N] [--reps R] [--shared]\n", argv[0]); return 2; }
    }
    const char *machine = getenv("HOST_TAG"); if (!machine) machine = "unknown";
    const char *commit = getenv("GIT_COMMIT"); if (!commit) commit = "unknown";
    const char *sizes = getenv("SIZES_KB"); if (!sizes || !sizes[0]) sizes = "1.5 105 8300 65536";

    if (csv) puts("machine,git_commit,trial,size_kb,threads,layout,reps,bytes,gbs_per_thread,gbs_aggregate");
    else printf("bench_stream: read bandwidth probe, %d threads\n", threads_available());

    /* Fork/join rows use size_kb = 0 and carry the cost in ns in the
     * gbs_per_thread column (unit differs, flagged by layout = forkjoin). */
    for (int tr = 0; tr < n_trials; tr++) {
        double ns = fork_join_ns(2000);
        if (csv) printf("%s,%s,%d,0.0,%d,forkjoin,2000,0,%.3f,0.0\n", machine, commit, tr, threads_available(), ns);
        else printf("  fork/join threads=%d trial=%d  %.1f ns per parallel region\n", threads_available(), tr, ns);
    }

    const char *p = sizes;
    for (int k = 0; k < 16 && *p; k++) {
        char *end;
        double kb = strtod(p, &end);
        if (end == p) break;
        /* Aim for about 256 MB of traffic per trial unless reps is forced. */
        int r = reps > 0 ? reps : (int)fmax(4.0, 256.0 * 1024.0 / kb);
        run_size(kb, r, n_trials, csv, machine, commit);
        p = end;
        while (*p == ' ') p++;
    }
    return 0;
}
