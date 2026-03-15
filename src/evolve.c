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
    int ret   = 0;

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

    /* Allocate working state */
    lb_matrix_t rho_cur  = {NULL, d};
    lb_matrix_t rho_next = {NULL, d};
    if (lb_matrix_alloc(&rho_cur,  d) != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&rho_next, d) != 0) { ret = -1; goto done; }

    /* Copy initial state */
    memcpy(rho_cur.data, rho0->data, d * d * sizeof(double complex));

    size_t n_steps = (size_t)ceil(t_end / dt);

    /* Store t=0 state if trajectory requested */
    if (store_trajectory) {
        memcpy(store_trajectory[0].data, rho_cur.data,
               d * d * sizeof(double complex));
    }

    /* Integration loop */
    for (size_t step = 0; step < n_steps; step++) {
        if (lb_propagate_step(&prop, &rho_cur, &rho_next) != 0) {
            ret = -1;
            goto done;
        }

        /* Swap buffers (no copy) */
        double complex *tmp = rho_cur.data;
        rho_cur.data        = rho_next.data;
        rho_next.data       = tmp;

        if (store_trajectory) {
            memcpy(store_trajectory[step + 1].data, rho_cur.data,
                   d * d * sizeof(double complex));
        }
    }

    /* Write final state */
    memcpy(rho_out->data, rho_cur.data, d * d * sizeof(double complex));
    if (n_steps_out) *n_steps_out = n_steps;

done:
    lb_matrix_free(&rho_cur);
    lb_matrix_free(&rho_next);
    lb_propagator_free(&prop);
    return ret;
}
