/**
 * expm.h — Internal header for matrix exponential computation.
 *
 * Implements Padé [13/13] approximant with scaling and squaring.
 * Reference: Higham 2005, DOI: 10.1137/04061101X
 *
 * Not part of the public API — use lb_expm() from lindblad_bench.h.
 */

#ifndef EXPM_H
#define EXPM_H

#include "lindblad_bench.h"

/**
 * Compute matrix product: out = A * B.
 * All three must be dim×dim. out must not alias A or B.
 */
int mat_mul(const lb_matrix_t *A, const lb_matrix_t *B, lb_matrix_t *out);

/**
 * Compute matrix sum: out = alpha*A + beta*B, in-place into out.
 * All three must be dim×dim.
 */
void mat_add_scaled(lb_matrix_t *out,
                    double complex alpha, const lb_matrix_t *A,
                    double complex beta,  const lb_matrix_t *B);

/**
 * Set out = identity matrix (1 on diagonal, 0 elsewhere).
 */
void mat_identity(lb_matrix_t *out);

/**
 * Compute the 1-norm of a matrix (max column sum of absolute values).
 * Used in Higham's norm estimation for Padé degree selection.
 */
double mat_one_norm(const lb_matrix_t *A);

/**
 * Solve the linear system A*X = B for X, in-place (B is overwritten with X).
 * Uses Gaussian elimination with partial pivoting.
 * Returns 0 on success, -1 if A is singular.
 */
int mat_solve(lb_matrix_t *A, lb_matrix_t *B);

/**
 * Compute out = expm(A) using caller-owned reusable scratch space.
 * ws must have been allocated for A->dim by lb_expm_workspace_alloc().
 */
int lb_expm_ws(const lb_matrix_t *A, lb_matrix_t *out, lb_expm_workspace_t *ws);

#endif /* EXPM_H */
