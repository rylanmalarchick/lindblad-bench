#include "propagate_batch_kernel.h"
#include <cuda_runtime.h>

namespace {

__global__ void propagate_batch_kernel(const double2 *__restrict__ P,
                                       const double2 *__restrict__ rho_in,
                                       double2 *__restrict__ rho_out,
                                       int d2,
                                       int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * d2;
    if (idx >= total) {
        return;
    }

    int batch = idx / d2;
    int row = idx - batch * d2;
    const double2 *P_row = P + (size_t)row * d2;
    const double2 *vec_in = rho_in + (size_t)batch * d2;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int j = 0; j < d2; j++) {
        double2 p = P_row[j];
        double2 v = vec_in[j];
        acc_re += (p.x * v.x) - (p.y * v.y);
        acc_im += (p.x * v.y) + (p.y * v.x);
    }

    rho_out[(size_t)batch * d2 + row] = make_double2(acc_re, acc_im);
}

static void free_context_storage(lb_cuda_raw_batch_context_t *ctx)
{
    if (!ctx) {
        return;
    }

    cudaFree(ctx->d_P);
    cudaFree(ctx->d_rho_in);
    cudaFree(ctx->d_rho_out);
    ctx->d_P = nullptr;
    ctx->d_rho_in = nullptr;
    ctx->d_rho_out = nullptr;
    ctx->d2 = 0;
    ctx->batch_size = 0;
}

static int launch_context(lb_cuda_raw_batch_context_t *ctx, float *kernel_ms)
{
    if (!ctx) {
        return -1;
    }

    int total = (int)(ctx->batch_size * ctx->d2);
    if (total == 0) {
        if (kernel_ms) {
            *kernel_ms = 0.0f;
        }
        return 0;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t err = cudaSuccess;
    if (kernel_ms) {
        err = cudaEventCreate(&start);
        if (err != cudaSuccess) {
            return -1;
        }
        err = cudaEventCreate(&stop);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            return -1;
        }
        err = cudaEventRecord(start);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return -1;
        }
    }

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    propagate_batch_kernel<<<grid_size, block_size>>>(
        (const double2 *)ctx->d_P,
        (const double2 *)ctx->d_rho_in,
        (double2 *)ctx->d_rho_out,
        (int)ctx->d2,
        (int)ctx->batch_size
    );

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        if (kernel_ms) {
            err = cudaEventRecord(stop);
            if (err == cudaSuccess) {
                err = cudaEventSynchronize(stop);
            }
        } else {
            err = cudaDeviceSynchronize();
        }
    }

    if (err == cudaSuccess && kernel_ms) {
        err = cudaEventElapsedTime(kernel_ms, start, stop);
    }

    if (start) {
        cudaEventDestroy(start);
    }
    if (stop) {
        cudaEventDestroy(stop);
    }

    return err == cudaSuccess ? 0 : -1;
}

}  // namespace

extern "C" int lb_cuda_raw_batch_init(lb_cuda_raw_batch_context_t *ctx,
                                      const double *P_interleaved,
                                      size_t d2,
                                      size_t batch_size)
{
    if (!ctx || !P_interleaved) {
        return -1;
    }

    size_t prop_elems = d2 * d2;
    size_t batch_elems = batch_size * d2;
    size_t prop_bytes = prop_elems * sizeof(double2);
    size_t batch_bytes = batch_elems * sizeof(double2);

    ctx->d_P = nullptr;
    ctx->d_rho_in = nullptr;
    ctx->d_rho_out = nullptr;
    ctx->d2 = d2;
    ctx->batch_size = batch_size;

    cudaError_t err = cudaMalloc((void **)&ctx->d_P, prop_bytes);
    if (err != cudaSuccess) {
        free_context_storage(ctx);
        return -1;
    }

    if (batch_bytes > 0) {
        err = cudaMalloc((void **)&ctx->d_rho_in, batch_bytes);
        if (err != cudaSuccess) {
            free_context_storage(ctx);
            return -1;
        }
        err = cudaMalloc((void **)&ctx->d_rho_out, batch_bytes);
        if (err != cudaSuccess) {
            free_context_storage(ctx);
            return -1;
        }
    }

    err = cudaMemcpy(ctx->d_P, P_interleaved, prop_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free_context_storage(ctx);
        return -1;
    }

    return 0;
}

extern "C" int lb_cuda_raw_batch_upload_input(lb_cuda_raw_batch_context_t *ctx,
                                              const double *rho_in_interleaved)
{
    if (!ctx || (!rho_in_interleaved && ctx->batch_size > 0)) {
        return -1;
    }

    size_t batch_bytes = ctx->batch_size * ctx->d2 * sizeof(double2);
    if (batch_bytes == 0) {
        return 0;
    }

    cudaError_t err = cudaMemcpy(ctx->d_rho_in, rho_in_interleaved,
                                 batch_bytes, cudaMemcpyHostToDevice);
    return err == cudaSuccess ? 0 : -1;
}

extern "C" int lb_cuda_raw_batch_launch(lb_cuda_raw_batch_context_t *ctx)
{
    return launch_context(ctx, nullptr);
}

extern "C" int lb_cuda_raw_batch_launch_timed(lb_cuda_raw_batch_context_t *ctx,
                                              float *kernel_ms)
{
    return launch_context(ctx, kernel_ms);
}

extern "C" int lb_cuda_raw_batch_download_output(const lb_cuda_raw_batch_context_t *ctx,
                                                 double *rho_out_interleaved)
{
    if (!ctx || (!rho_out_interleaved && ctx->batch_size > 0)) {
        return -1;
    }

    size_t batch_bytes = ctx->batch_size * ctx->d2 * sizeof(double2);
    if (batch_bytes == 0) {
        return 0;
    }

    cudaError_t err = cudaMemcpy(rho_out_interleaved, ctx->d_rho_out,
                                 batch_bytes, cudaMemcpyDeviceToHost);
    return err == cudaSuccess ? 0 : -1;
}

extern "C" void lb_cuda_raw_batch_free(lb_cuda_raw_batch_context_t *ctx)
{
    free_context_storage(ctx);
}

extern "C" int lb_cuda_raw_propagate_step_batch(const double *P_interleaved,
                                                const double *rho_in_interleaved,
                                                double *rho_out_interleaved,
                                                size_t d2,
                                                size_t batch_size)
{
    if (!P_interleaved || (!rho_in_interleaved && batch_size > 0) ||
        (!rho_out_interleaved && batch_size > 0)) {
        return -1;
    }

    lb_cuda_raw_batch_context_t ctx = {0};
    if (lb_cuda_raw_batch_init(&ctx, P_interleaved, d2, batch_size) != 0) {
        return -1;
    }
    if (lb_cuda_raw_batch_upload_input(&ctx, rho_in_interleaved) != 0 ||
        lb_cuda_raw_batch_launch(&ctx) != 0 ||
        lb_cuda_raw_batch_download_output(&ctx, rho_out_interleaved) != 0) {
        lb_cuda_raw_batch_free(&ctx);
        return -1;
    }

    lb_cuda_raw_batch_free(&ctx);
    return 0;
}
