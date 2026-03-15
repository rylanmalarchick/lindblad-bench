/**
 * expm.c — Matrix exponential via Padé [13/13] with scaling and squaring.
 *
 * Algorithm: Higham 2005, "The Scaling and Squaring Method for the Matrix
 * Exponential Revisited", SIAM J. Matrix Anal. Appl. 26(4):1179–1193.
 * DOI: 10.1137/04061101X
 *
 * Padé coefficients b_k for order 13 (Table 1, Higham 2005):
 *   Numerator and denominator share the same |b_k| with alternating signs
 *   in the denominator. See lb_expm() for the construction.
 */

#include "expm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* Padé [13/13] coefficients (Higham 2005, Table 1) */
static const double PADE_COEFS[14] = {
    1.0,
    0.5,
    0.12,
    1.833333333333333e-2,
    1.992063492063492e-3,
    1.630434782608696e-4,
    1.035196687370600e-5,
    5.175983436853002e-7,
    2.043151129664400e-8,
    6.306659613335500e-10,
    1.483575475919960e-11,
    2.529153491597966e-13,
    2.810170546428544e-15,
    1.544088392870150e-17,
};

/* Scaling threshold: use s squarings when ||A||_1 / 2^s <= theta_13 */
static const double THETA_13 = 5.371920351148152;

/* --------------------------------------------------------------------------
 * Internal matrix helpers (implementations used by expm and lindblad)
 * -------------------------------------------------------------------------- */

int mat_mul(const lb_matrix_t *A, const lb_matrix_t *B, lb_matrix_t *out)
{
    size_t n = A->dim;
    /* out must be pre-allocated and not alias A or B */
    memset(out->data, 0, n * n * sizeof(double complex));
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double complex aik = A->data[i * n + k];
            if (aik == 0.0) continue; /* sparse skip */
            for (size_t j = 0; j < n; j++) {
                out->data[i * n + j] += aik * B->data[k * n + j];
            }
        }
    }
    return 0;
}

void mat_add_scaled(lb_matrix_t *out,
                    double complex alpha, const lb_matrix_t *A,
                    double complex beta,  const lb_matrix_t *B)
{
    size_t n = out->dim * out->dim;
    for (size_t i = 0; i < n; i++) {
        out->data[i] = alpha * A->data[i] + beta * B->data[i];
    }
}

void mat_identity(lb_matrix_t *out)
{
    size_t n = out->dim;
    memset(out->data, 0, n * n * sizeof(double complex));
    for (size_t i = 0; i < n; i++) {
        out->data[i * n + i] = 1.0 + 0.0 * I;
    }
}

double mat_one_norm(const lb_matrix_t *A)
{
    size_t n = A->dim;
    double max_col = 0.0;
    for (size_t j = 0; j < n; j++) {
        double col_sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            col_sum += cabs(A->data[i * n + j]);
        }
        if (col_sum > max_col) max_col = col_sum;
    }
    return max_col;
}

int mat_solve(lb_matrix_t *A, lb_matrix_t *B)
{
    /* Gaussian elimination with partial pivoting: A*X = B -> X stored in B.
     * Both A and B are dim×dim. A is destroyed in place. */
    size_t n = A->dim;

    for (size_t col = 0; col < n; col++) {
        /* Find pivot */
        size_t pivot = col;
        double max_val = cabs(A->data[col * n + col]);
        for (size_t row = col + 1; row < n; row++) {
            double v = cabs(A->data[row * n + col]);
            if (v > max_val) { max_val = v; pivot = row; }
        }
        if (max_val < 1e-15) return -1; /* singular */

        /* Swap rows in A and B */
        if (pivot != col) {
            for (size_t j = 0; j < n; j++) {
                double complex tmp = A->data[col * n + j];
                A->data[col * n + j] = A->data[pivot * n + j];
                A->data[pivot * n + j] = tmp;

                tmp = B->data[col * n + j];
                B->data[col * n + j] = B->data[pivot * n + j];
                B->data[pivot * n + j] = tmp;
            }
        }

        double complex diag = A->data[col * n + col];
        for (size_t row = col + 1; row < n; row++) {
            double complex factor = A->data[row * n + col] / diag;
            for (size_t j = col; j < n; j++) {
                A->data[row * n + j] -= factor * A->data[col * n + j];
            }
            for (size_t j = 0; j < n; j++) {
                B->data[row * n + j] -= factor * B->data[col * n + j];
            }
        }
    }

    /* Back substitution */
    for (int col = (int)n - 1; col >= 0; col--) {
        double complex diag = A->data[col * n + col];
        for (size_t j = 0; j < n; j++) {
            B->data[col * n + j] /= diag;
        }
        for (int row = col - 1; row >= 0; row--) {
            double complex factor = A->data[row * n + col];
            for (size_t j = 0; j < n; j++) {
                B->data[row * n + j] -= factor * B->data[col * n + j];
            }
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * Padé [13/13] numerator and denominator
 *
 * Given A, compute:
 *   U = A * (b_13 A^12 + b_11 A^10 + ... + b_1 I)   (odd Padé terms)
 *   V =     (b_12 A^12 + b_10 A^10 + ... + b_0 I)   (even Padé terms)
 *   expm(A) ≈ (V + U)^{-1} (V - U)
 *
 * See Higham 2005, Algorithm 2.3.
 * -------------------------------------------------------------------------- */
static int pade13(const lb_matrix_t *A, lb_matrix_t *out)
{
    size_t n = A->dim;

    /* We need: A2, A4, A6, U, V, tmp */
    lb_matrix_t A2 = {NULL, n};
    lb_matrix_t A4 = {NULL, n};
    lb_matrix_t A6 = {NULL, n};
    lb_matrix_t U  = {NULL, n};
    lb_matrix_t V  = {NULL, n};
    lb_matrix_t W  = {NULL, n}; /* scratch */
    lb_matrix_t *bufs[] = {&A2, &A4, &A6, &U, &V, &W};
    int allocated = 0;

    for (int i = 0; i < 6; i++) {
        if (lb_matrix_alloc(bufs[i], n) != 0) goto cleanup;
        allocated++;
    }

    /* A2 = A*A, A4 = A2*A2, A6 = A4*A2 */
    mat_mul(A, A, &A2);
    mat_mul(&A2, &A2, &A4);
    mat_mul(&A4, &A2, &A6);

    /* V = b12*A6 + b10*A4 + b8*A2 (even, high) — stored in W first */
    /* Then multiply by A6 to get b12*A^12 + b10*A^10 + b8*A^8 */

    /* Build V = b[12]*A6 + b[10]*A4 + b[8]*A2 + b[6]*I (using W as scratch) */
    mat_identity(&V);
    for (size_t i = 0; i < n * n; i++) {
        V.data[i] = PADE_COEFS[12] * A6.data[i]
                  + PADE_COEFS[10] * A4.data[i]
                  + PADE_COEFS[8]  * A2.data[i];
    }
    /* V = A6 * V + b6*A6 + b4*A4 + b2*A2 + b0*I */
    mat_mul(&A6, &V, &W);
    mat_identity(&V);
    for (size_t i = 0; i < n * n; i++) {
        V.data[i] = W.data[i]
                  + PADE_COEFS[6] * A6.data[i]
                  + PADE_COEFS[4] * A4.data[i]
                  + PADE_COEFS[2] * A2.data[i];
    }
    /* V += b0*I */
    for (size_t i = 0; i < n; i++) V.data[i * n + i] += PADE_COEFS[0];

    /* Build U = b[13]*A6 + b[11]*A4 + b[9]*A2 (odd, high) */
    for (size_t i = 0; i < n * n; i++) {
        W.data[i] = PADE_COEFS[13] * A6.data[i]
                  + PADE_COEFS[11] * A4.data[i]
                  + PADE_COEFS[9]  * A2.data[i];
    }
    /* U = A6 * W + b7*A6 + b5*A4 + b3*A2 + b1*I */
    mat_mul(&A6, &W, &U);
    for (size_t i = 0; i < n * n; i++) {
        U.data[i] += PADE_COEFS[7] * A6.data[i]
                   + PADE_COEFS[5] * A4.data[i]
                   + PADE_COEFS[3] * A2.data[i];
    }
    /* U += b1*I */
    for (size_t i = 0; i < n; i++) U.data[i * n + i] += PADE_COEFS[1];
    /* U = A * U */
    mat_mul(A, &U, &W);
    memcpy(U.data, W.data, n * n * sizeof(double complex));

    /* expm(A) = (V - U)^{-1} (V + U)
     * Store numerator (V+U) in out, denominator (V-U) in W, then solve. */
    for (size_t i = 0; i < n * n; i++) {
        out->data[i] = V.data[i] + U.data[i]; /* numerator  */
        W.data[i]   = V.data[i] - U.data[i]; /* denominator */
    }
    /* Solve W * out = out  ->  out = W^{-1} * (V+U) */
    if (mat_solve(&W, out) != 0) { allocated = -1; goto cleanup; }

cleanup:
    for (int i = 0; i < 6; i++) lb_matrix_free(bufs[i]);
    return (allocated < 0) ? -1 : 0;
}

/* --------------------------------------------------------------------------
 * Public: lb_expm
 * -------------------------------------------------------------------------- */
int lb_expm(const lb_matrix_t *A, lb_matrix_t *out)
{
    size_t n = A->dim;
    double norm = mat_one_norm(A);

    /* Determine number of squarings s such that ||A/2^s||_1 <= theta_13 */
    int s = 0;
    if (norm > THETA_13) {
        s = (int)ceil(log2(norm / THETA_13));
    }

    /* Scale: As = A / 2^s */
    lb_matrix_t As = {NULL, n};
    if (lb_matrix_alloc(&As, n) != 0) return -1;
    double scale = 1.0 / (double)(1 << s);
    for (size_t i = 0; i < n * n; i++) {
        As.data[i] = A->data[i] * scale;
    }

    /* Compute Padé approximant of expm(As) */
    if (pade13(&As, out) != 0) {
        lb_matrix_free(&As);
        return -1;
    }
    lb_matrix_free(&As);

    /* Squaring phase: out = out^(2^s) */
    lb_matrix_t tmp = {NULL, n};
    if (lb_matrix_alloc(&tmp, n) != 0) return -1;
    for (int i = 0; i < s; i++) {
        mat_mul(out, out, &tmp);
        memcpy(out->data, tmp.data, n * n * sizeof(double complex));
    }
    lb_matrix_free(&tmp);

    return 0;
}
