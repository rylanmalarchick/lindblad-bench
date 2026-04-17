#ifndef LINDBLAD_CUDA_H
#define LINDBLAD_CUDA_H

#include "lindblad_bench.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply one propagation step to a batch of independent density matrices on a
 * CUDA device. This entry point performs the full host-visible operation:
 * host-to-device transfer, kernel launch, device-to-host transfer.
 */
int lb_cuda_propagate_step_batch(const lb_propagator_t *prop,
                                 const lb_matrix_t *rho_in_batch,
                                 lb_matrix_t *rho_out_batch,
                                 size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* LINDBLAD_CUDA_H */
