#ifndef PROPAGATE_BATCH_KERNEL_H
#define PROPAGATE_BATCH_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *d_P;
    void *d_rho_in;
    void *d_rho_out;
    size_t d2;
    size_t batch_size;
} lb_cuda_raw_batch_context_t;

int lb_cuda_raw_batch_init(lb_cuda_raw_batch_context_t *ctx,
                           const double *P_interleaved,
                           size_t d2,
                           size_t batch_size);

int lb_cuda_raw_batch_upload_input(lb_cuda_raw_batch_context_t *ctx,
                                   const double *rho_in_interleaved);

int lb_cuda_raw_batch_launch(lb_cuda_raw_batch_context_t *ctx);

int lb_cuda_raw_batch_launch_timed(lb_cuda_raw_batch_context_t *ctx,
                                   float *kernel_ms);

int lb_cuda_raw_batch_download_output(const lb_cuda_raw_batch_context_t *ctx,
                                      double *rho_out_interleaved);

void lb_cuda_raw_batch_free(lb_cuda_raw_batch_context_t *ctx);

int lb_cuda_raw_propagate_step_batch(const double *P_interleaved,
                                     const double *rho_in_interleaved,
                                     double *rho_out_interleaved,
                                     size_t d2,
                                     size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* PROPAGATE_BATCH_KERNEL_H */
