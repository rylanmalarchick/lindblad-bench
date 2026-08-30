/**
 * bench_cuda_probe.cu — Device-side ceilings for the GPU cost model.
 *
 * Measures, on the current device:
 *   launch_ns        empty-kernel launch-to-completion latency (cudaEvent)
 *   d2d_gbs_<KB>     device-to-device copy bandwidth (cudaMemcpy) at a size
 *   h2d_gbs_<KB>     host-to-device bandwidth, pageable host memory
 *   d2h_gbs_<KB>     device-to-host bandwidth, pageable host memory
 *   gather_gbs_<KB>  read bandwidth of a kernel that reads a buffer with the
 *                    same stride-N gather pattern as the propagation kernel
 *   stream_gbs_<KB>  read bandwidth of a coalesced read kernel, same buffer
 *
 * Sizes default to the P working sets 1.5 KB, 105 KB, 8.3 MB, plus 64 MB.
 * Each measurement repeats n_reps times after 5 warm-ups. One CSV row per
 * (trial, quantity). Trials default to 5.
 *
 * Usage: bench_cuda_probe [--csv] [--trials N] [--reps R]
 * Environment: HOST_TAG, GIT_COMMIT
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

__global__ void empty_kernel() {}

/* Coalesced: thread t reads elements t, t+stride, ... */
__global__ void stream_read(const double2 *buf, size_t n, double *sink)
{
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double acc = 0.0;
    for (size_t i = t; i < n; i += stride) acc += buf[i].x + buf[i].y;
    if (acc == 1234.5678) sink[0] = acc;
}

/* Gather: thread t owns row t of an N x N matrix and walks it (stride N
 * across the warp), exactly like propagate_batch_kernel. */
__global__ void gather_read(const double2 *buf, int N, double *sink)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const double2 *p = buf + (size_t)row * N;
    double acc = 0.0;
    for (int j = 0; j < N; j++) acc += p[j].x + p[j].y;
    if (acc == 1234.5678) sink[0] = acc;
}

static const char *g_machine = "unknown";
static const char *g_commit = "unknown";
static const char *g_gpu = "unknown";
static int g_csv = 0;

static void emit(int trial, const char *quantity, double value, const char *unit)
{
    if (g_csv) printf("%s,%s,%s,%d,%s,%.6f,%s\n", g_machine, g_gpu, g_commit, trial, quantity, value, unit);
    else printf("  trial %d  %-22s %12.3f %s\n", trial, quantity, value, unit);
}

static double event_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return (double)ms;
}

int main(int argc, char **argv)
{
    int n_trials = 5, n_reps = 200;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")) g_csv = 1;
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) n_trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) n_reps = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--csv] [--trials N] [--reps R]\n", argv[0]); return 2; }
    }
    if (getenv("HOST_TAG")) g_machine = getenv("HOST_TAG");
    if (getenv("GIT_COMMIT")) g_commit = getenv("GIT_COMMIT");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu_name = prop.name;
    g_gpu = gpu_name.c_str();

    if (g_csv) puts("machine,gpu_name,git_commit,trial,quantity,value,unit");
    else printf("bench_cuda_probe on %s\n", g_gpu);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double *d_sink = nullptr;
    cudaMalloc(&d_sink, sizeof(double));

    const double sizes_kb[] = {1.5, 105.0, 8300.0, 65536.0};
    const int n_sizes = 4;

    for (int trial = 0; trial < n_trials; trial++) {
        /* Launch latency: back-to-back empty kernels, event-bracketed. */
        for (int w = 0; w < 5; w++) empty_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < n_reps; r++) empty_kernel<<<1, 32>>>();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        emit(trial, "launch_ns", event_ms(e0, e1) * 1e6 / n_reps, "ns");

        /* Launch + sync latency: what a synchronous caller sees. */
        cudaEventRecord(e0);
        for (int r = 0; r < n_reps; r++) { empty_kernel<<<1, 32>>>(); cudaDeviceSynchronize(); }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        emit(trial, "launch_sync_ns", event_ms(e0, e1) * 1e6 / n_reps, "ns");

        for (int si = 0; si < n_sizes; si++) {
            size_t bytes = (size_t)(sizes_kb[si] * 1024.0);
            size_t n = bytes / sizeof(double2);
            if (n < 1) n = 1;
            bytes = n * sizeof(double2);
            char tag[64];
            int reps = (int)(n_reps * (bytes < (1u << 20) ? 1 : 1));

            double2 *d_a = nullptr, *d_b = nullptr;
            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMemset(d_a, 0, bytes);
            std::vector<double2> h(n);
            for (size_t i = 0; i < n; i++) { h[i].x = 0.5; h[i].y = 0.25; }

            /* D2D copy bandwidth (read + write counted as 2 x bytes). */
            for (int w = 0; w < 5; w++) cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; r++) cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            snprintf(tag, sizeof tag, "d2d_gbs_%.0fkb", sizes_kb[si]);
            emit(trial, tag, 2.0 * bytes * reps / (event_ms(e0, e1) * 1e6), "GB/s");

            /* H2D and D2H, pageable memory, as the wrapper uses. */
            for (int w = 0; w < 5; w++) cudaMemcpy(d_a, h.data(), bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; r++) cudaMemcpy(d_a, h.data(), bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            snprintf(tag, sizeof tag, "h2d_gbs_%.0fkb", sizes_kb[si]);
            emit(trial, tag, (double)bytes * reps / (event_ms(e0, e1) * 1e6), "GB/s");

            cudaEventRecord(e0);
            for (int r = 0; r < reps; r++) cudaMemcpy(h.data(), d_a, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            snprintf(tag, sizeof tag, "d2h_gbs_%.0fkb", sizes_kb[si]);
            emit(trial, tag, (double)bytes * reps / (event_ms(e0, e1) * 1e6), "GB/s");

            /* Coalesced read kernel. */
            int block = 256;
            int grid = (int)((n + block - 1) / block);
            if (grid > 4096) grid = 4096;
            for (int w = 0; w < 5; w++) stream_read<<<grid, block>>>(d_a, n, d_sink);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; r++) stream_read<<<grid, block>>>(d_a, n, d_sink);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            snprintf(tag, sizeof tag, "stream_gbs_%.0fkb", sizes_kb[si]);
            emit(trial, tag, (double)bytes * reps / (event_ms(e0, e1) * 1e6), "GB/s");

            /* Gather read kernel over the square matrix that fits in bytes. */
            int N = 1;
            while ((size_t)(N + 1) * (N + 1) <= n) N++;
            size_t sq_bytes = (size_t)N * N * sizeof(double2);
            int ggrid = (N + block - 1) / block;
            for (int w = 0; w < 5; w++) gather_read<<<ggrid, block>>>(d_a, N, d_sink);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; r++) gather_read<<<ggrid, block>>>(d_a, N, d_sink);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            snprintf(tag, sizeof tag, "gather_gbs_%.0fkb", sizes_kb[si]);
            emit(trial, tag, (double)sq_bytes * reps / (event_ms(e0, e1) * 1e6), "GB/s");

            cudaFree(d_a);
            cudaFree(d_b);
        }
    }

    cudaFree(d_sink);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}
