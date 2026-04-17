#include "lindblad_cuda.h"
#include "propagate_batch_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <complex.h>

int lb_cuda_propagate_step_batch(const lb_propagator_t *prop,
                                 const lb_matrix_t *rho_in_batch,
                                 lb_matrix_t *rho_out_batch,
                                 size_t batch_size)
{
    if (!prop || (!rho_in_batch && batch_size > 0) || (!rho_out_batch && batch_size > 0)) {
        return -1;
    }

    size_t d = prop->d;
    size_t d2 = d * d;
    size_t state_bytes = d2 * sizeof(double complex);
    size_t batch_bytes = batch_size * state_bytes;

    for (size_t b = 0; b < batch_size; b++) {
        if (rho_in_batch[b].dim != d || rho_out_batch[b].dim != d) {
            return -1;
        }
    }

    double *rho_in_packed = NULL;
    double *rho_out_packed = NULL;
    if (batch_size > 0) {
        rho_in_packed = (double *)malloc(batch_bytes);
        rho_out_packed = (double *)malloc(batch_bytes);
        if (!rho_in_packed || !rho_out_packed) {
            free(rho_in_packed);
            free(rho_out_packed);
            return -1;
        }
    }

    for (size_t b = 0; b < batch_size; b++) {
        memcpy((char *)rho_in_packed + b * state_bytes, rho_in_batch[b].data, state_bytes);
    }

    int ret = lb_cuda_raw_propagate_step_batch((const double *)prop->P.data,
                                               rho_in_packed,
                                               rho_out_packed,
                                               d2,
                                               batch_size);
    if (ret == 0) {
        for (size_t b = 0; b < batch_size; b++) {
            memcpy(rho_out_batch[b].data, (const char *)rho_out_packed + b * state_bytes,
                   state_bytes);
        }
    }

    free(rho_in_packed);
    free(rho_out_packed);
    return ret;
}
