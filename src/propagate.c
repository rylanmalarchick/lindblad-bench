/**
 * propagate.c — Propagator construction and single-step density matrix update.
 *
 * The propagator P = expm(L * dt) is built once (lb_build_propagator) and
 * reused for every timestep when H and collapse operators are time-independent.
 *
 * The hot path is lb_propagate_step:
 *   vec(ρ_out) = P · vec(ρ_in)
 *   then reshape back to d×d.
 */

#include "propagate.h"
#include "expm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

int lb_propagate_step_soa(const lb_matrix_soa_t *P_soa,
                          const lb_matrix_soa_t *rho_in,
                          lb_matrix_soa_t *rho_out)
{
    size_t d2 = P_soa->dim; /* P is d2 x d2, rho is effectively d x d so size is d2 */

    const double *P_re = P_soa->real;
    const double *P_im = P_soa->imag;
    const double *vec_in_re = rho_in->real;
    const double *vec_in_im = rho_in->imag;
    double *vec_out_re = rho_out->real;
    double *vec_out_im = rho_out->imag;

    /* Matvec for SoA: (A + iB)(x + iy) = (Ax - By) + i(Ay + Bx) */
    for (size_t i = 0; i < d2; i++) {
        double acc_re = 0.0;
        double acc_im = 0.0;
        for (size_t j = 0; j < d2; j++) {
            double p_re = P_re[i * d2 + j];
            double p_im = P_im[i * d2 + j];
            double v_re = vec_in_re[j];
            double v_im = vec_in_im[j];

            acc_re += (p_re * v_re) - (p_im * v_im);
            acc_im += (p_re * v_im) + (p_im * v_re);
        }
        vec_out_re[i] = acc_re;
        vec_out_im[i] = acc_im;
    }

    return 0;
}

#ifdef __AVX2__
int lb_propagate_step_avx2(const lb_propagator_t *prop,
                           const lb_matrix_t *rho_in,
                           lb_matrix_t *rho_out)
{
    size_t d = prop->d;
    size_t d2 = d * d;

    const double *P = (const double *)prop->P.data;
    const double *vec_in = (const double *)rho_in->data;
    double *vec_out = (double *)rho_out->data;

    /*
     * Manual AVX2 for complex matrix-vector multiplication.
     * We load 2 complex numbers at a time into a 256-bit register (4 doubles).
     * [P_re1, P_im1, P_re2, P_im2] and [v_re1, v_im1, v_re2, v_im2]
     */
    for (size_t i = 0; i < d2; i++) {
        __m256d acc = _mm256_setzero_pd();

        size_t j = 0;
        /* Process 2 complex elements per iteration */
        for (; j + 1 < d2; j += 2) {
            /* Use unaligned loads — row offsets may not be 32-byte aligned
             * when d² is odd (e.g. d=3, d²=9). No penalty on Haswell+. */
            __m256d p_vec = _mm256_loadu_pd(&P[(i * d2 + j) * 2]);
            __m256d v_vec = _mm256_loadu_pd(&vec_in[j * 2]);

            /* Broadcast real/imag parts of P elements */
            __m256d p_re = _mm256_movedup_pd(p_vec);       /* [re0, re0, re1, re1] */
            __m256d p_im = _mm256_shuffle_pd(p_vec, p_vec, 0xF); /* [im0, im0, im1, im1] */

            /* Swap real/imag in v for cross-product terms */
            __m256d v_swap = _mm256_shuffle_pd(v_vec, v_vec, 0x5); /* [im, re, im, re] */

            __m256d prod1 = _mm256_mul_pd(p_re, v_vec);
            __m256d prod2 = _mm256_mul_pd(p_im, v_swap);

            /* addsub: even lanes subtract (re), odd lanes add (im) */
            acc = _mm256_add_pd(acc, _mm256_addsub_pd(prod1, prod2));
        }

        /* Horizontal reduce: acc = [re0, im0, re1, im1] */
        double acc_arr[4];
        _mm256_storeu_pd(acc_arr, acc);

        double total_re = acc_arr[0] + acc_arr[2];
        double total_im = acc_arr[1] + acc_arr[3];

        /* Remainder scalar loop */
        for (; j < d2; j++) {
            double p_re = P[(i * d2 + j) * 2];
            double p_im = P[(i * d2 + j) * 2 + 1];
            double v_re = vec_in[j * 2];
            double v_im = vec_in[j * 2 + 1];

            total_re += (p_re * v_re) - (p_im * v_im);
            total_im += (p_re * v_im) + (p_im * v_re);
        }

        vec_out[i * 2] = total_re;
        vec_out[i * 2 + 1] = total_im;
    }

    return 0;
}
#endif
