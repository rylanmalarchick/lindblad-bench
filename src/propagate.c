/**
 * propagate.c — Propagator construction and single-step density matrix update.
 *
 * The propagator P = expm(L * dt) is built once (lb_build_propagator) and
 * reused for every timestep when H and collapse operators are time-independent.
 *
 * The hot path is lb_propagate_step:
 *   vec(ρ_out) = P · vec(ρ_in)
 *   then reshape back to d×d.
 *
 * This is a (d²)×(d²) complex matvec — the innermost loop of any Lindblad
 * trajectory simulation. At d=3: 9×9, at d=9: 81×81, at d=27: 729×729.
 * Working sets: 1.3 KB (L1), 105 KB (L2), 8.5 MB (L3).
 */

#include "propagate.h"
#include "expm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* --------------------------------------------------------------------------
 * Propagator build
 * -------------------------------------------------------------------------- */

int lb_build_propagator(const lb_matrix_t *L, double dt, lb_propagator_t *prop)
{
    size_t d2 = L->dim;

    prop->dt = dt;
    prop->d  = (size_t)round(sqrt((double)d2)); /* d² -> d */

    /* Allocate propagator matrix */
    if (lb_matrix_alloc(&prop->P, d2) != 0) return -1;

    /* Scale L by dt: Ldt = L * dt */
    lb_matrix_t Ldt = {NULL, d2};
    if (lb_matrix_alloc(&Ldt, d2) != 0) {
        lb_matrix_free(&prop->P);
        return -1;
    }
    for (size_t i = 0; i < d2 * d2; i++) {
        Ldt.data[i] = L->data[i] * dt;
    }

    /* P = expm(L * dt) */
    int ret = lb_expm(&Ldt, &prop->P);
    lb_matrix_free(&Ldt);
    return ret;
}

void lb_propagator_free(lb_propagator_t *prop)
{
    lb_matrix_free(&prop->P);
    prop->dt = 0.0;
    prop->d  = 0;
}

/* --------------------------------------------------------------------------
 * lb_propagate_step — hot path
 *
 * Vectorize ρ_in (d×d -> d² vector), multiply by P (d²×d²), reshape to d×d.
 *
 * This is the function analyzed on Godbolt and in the Roofline model.
 * Arithmetic intensity: (2*d^4 FLOPs) / (d^4 * 16 bytes) = 1/8 FLOP/byte
 * at the limit where P fits in cache; bandwidth-bound when it doesn't.
 * -------------------------------------------------------------------------- */
int lb_propagate_step(const lb_propagator_t *prop,
                      const lb_matrix_t *rho_in,
                      lb_matrix_t *rho_out)
{
    size_t d  = prop->d;
    size_t d2 = d * d;

    /* Vectorize rho_in: vec[i*d + j] = rho_in[i][j] (row-major, already) */
    const double complex *vec_in = rho_in->data; /* row-major = already vectorized */

    /* Matvec: vec_out = P * vec_in */
    double complex *P       = prop->P.data;
    double complex *vec_out = rho_out->data;

    /* NOTE: This loop is the Godbolt / Roofline subject.
     * With -O3 -march=native, GCC/Clang auto-vectorize the inner loop
     * using 256-bit AVX2 registers (2 complex doubles per ymm register).
     * The outer loop stride and aliasing must be resolvable at compile time
     * for full vectorization — see godbolt/README.md for analysis. */
    for (size_t i = 0; i < d2; i++) {
        double complex acc = 0.0;
        for (size_t j = 0; j < d2; j++) {
            acc += P[i * d2 + j] * vec_in[j];
        }
        vec_out[i] = acc;
    }

    /* rho_out is already shaped d×d (row-major, same layout as ρ matrix).
     * No reshape needed: row-major vectorization == row-major matrix. */
    return 0;
}
