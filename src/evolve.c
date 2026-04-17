/**
 * evolve.c — Full trajectory integration.
 *
 * Evolves ρ(0) to ρ(t_end) by repeated application of the precomputed
 * propagator: ρ(t + dt) = reshape(P · vec(ρ(t))).
 *
 * The propagator is built once (O(d^6) cost from expm) and reused for
 * every timestep (O(d^4) cost per step). For long trajectories this
 * amortizes the expm cost to negligible.
 */

#include "propagate.h"
#include "expm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * lb_evolve_prop — evolve with a pre-built propagator
 *
 * This is the GRAPE-friendly entry point: caller builds L and P once,
 * then calls this for each trajectory / segment / initial state.
 * -------------------------------------------------------------------------- */
int lb_evolve_prop(const lb_propagator_t *prop,
                   const lb_matrix_t *rho0,
                   size_t n_steps,
                   lb_matrix_t *rho_out,
                   lb_matrix_t *store_trajectory)
{
    size_t d  = prop->d;
    int ret   = 0;

    lb_matrix_t rho_cur  = {NULL, d};
    lb_matrix_t rho_next = {NULL, d};
    if (lb_matrix_alloc(&rho_cur,  d) != 0) { return -1; }
    if (lb_matrix_alloc(&rho_next, d) != 0) { lb_matrix_free(&rho_cur); return -1; }

    memcpy(rho_cur.data, rho0->data, d * d * sizeof(double complex));

    if (store_trajectory) {
        memcpy(store_trajectory[0].data, rho_cur.data,
               d * d * sizeof(double complex));
    }

    for (size_t step = 0; step < n_steps; step++) {
        if (lb_propagate_step(prop, &rho_cur, &rho_next) != 0) {
            ret = -1;
            goto done;
        }

        double complex *tmp = rho_cur.data;
        rho_cur.data        = rho_next.data;
        rho_next.data       = tmp;

        if (store_trajectory) {
            memcpy(store_trajectory[step + 1].data, rho_cur.data,
                   d * d * sizeof(double complex));
        }
    }

    memcpy(rho_out->data, rho_cur.data, d * d * sizeof(double complex));

done:
    lb_matrix_free(&rho_cur);
    lb_matrix_free(&rho_next);
    return ret;
}

int lb_evolve_prop_batch(const lb_propagator_t *prop,
                         const lb_matrix_t *rho0_batch,
                         size_t batch_size,
                         size_t n_steps,
                         lb_matrix_t *rho_out_batch)
{
    if (!prop || (!rho0_batch && batch_size > 0) || (!rho_out_batch && batch_size > 0)) {
        return -1;
    }

#ifdef _OPENMP
    int ret = 0;
    #pragma omp parallel for if (batch_size > 1) schedule(static) reduction(|:ret)
    for (ptrdiff_t b = 0; b < (ptrdiff_t)batch_size; b++) {
        if (rho0_batch[b].dim != prop->d || rho_out_batch[b].dim != prop->d) {
            ret = 1;
            continue;
        }
        if (lb_evolve_prop(prop, &rho0_batch[b], n_steps, &rho_out_batch[b], NULL) != 0) {
            ret = 1;
        }
    }

    return ret ? -1 : 0;
#else
    for (size_t b = 0; b < batch_size; b++) {
        if (rho0_batch[b].dim != prop->d || rho_out_batch[b].dim != prop->d) {
            return -1;
        }
        if (lb_evolve_prop(prop, &rho0_batch[b], n_steps, &rho_out_batch[b], NULL) != 0) {
            return -1;
        }
    }

    return 0;
#endif
}

/* --------------------------------------------------------------------------
 * lb_evolve — convenience wrapper that builds L and P internally
 * -------------------------------------------------------------------------- */
int lb_evolve(const lb_system_t *sys,
              const lb_matrix_t *rho0,
              double t_end,
              double dt,
              lb_matrix_t *rho_out,
              lb_matrix_t *store_trajectory,
              size_t *n_steps_out)
{
    size_t d  = sys->d;
    size_t d2 = d * d;

    /* Build Lindbladian superoperator */
    lb_matrix_t L = {NULL, d2};
    if (lb_matrix_alloc(&L, d2) != 0) return -1;
    if (lb_build_lindbladian(sys, &L) != 0) { lb_matrix_free(&L); return -1; }

    /* Build propagator */
    lb_propagator_t prop = {0};
    if (lb_build_propagator(&L, dt, &prop) != 0) {
        lb_matrix_free(&L);
        return -1;
    }
    lb_matrix_free(&L);

    size_t n_steps = (size_t)ceil(t_end / dt);
    int ret = lb_evolve_prop(&prop, rho0, n_steps, rho_out, store_trajectory);

    if (n_steps_out) *n_steps_out = n_steps;
    lb_propagator_free(&prop);
    return ret;
}
