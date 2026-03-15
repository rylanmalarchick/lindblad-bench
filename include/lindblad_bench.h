/**
 * lindblad_bench.h — Public API for the lindblad-bench C library.
 *
 * Bare-metal Lindblad master equation simulation targeting near-term quantum
 * control use cases (d = 3–27, single- and few-qubit transmon systems).
 *
 * Physical model:
 *   dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
 *
 * Vectorized form:
 *   d(vec ρ)/dt = L · vec(ρ),  L ∈ ℂ^(d²×d²)
 *
 * Each propagation step:
 *   vec(ρ(t+dt)) = expm(L·dt) · vec(ρ(t))
 *
 * References:
 *   Lindblad 1976, DOI: 10.1007/BF01608499
 *   Gorini, Kossakowski, Sudarshan 1976, DOI: 10.1063/1.522979
 *   Higham 2005 (expm), DOI: 10.1137/04061101X
 */

#ifndef LINDBLAD_BENCH_H
#define LINDBLAD_BENCH_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Types
 * -------------------------------------------------------------------------- */

/**
 * Complex matrix stored in row-major order.
 * data[i*cols + j] = element (i, j).
 * All matrices are square: rows == cols == dim.
 */
typedef struct {
    double complex *data; /* heap-allocated, caller owns */
    size_t dim;           /* matrix dimension */
} lb_matrix_t;

/**
 * A Lindblad system: one Hamiltonian + up to LB_MAX_COPS collapse operators.
 * All operators live in Hilbert space of dimension d (d-level system).
 * The Lindbladian superoperator acts on vec(ρ) ∈ ℂ^(d²).
 */
#define LB_MAX_COPS 8

typedef struct {
    lb_matrix_t H;               /* Hamiltonian (d×d, Hermitian) */
    lb_matrix_t cops[LB_MAX_COPS]; /* collapse operators L_k (d×d) */
    size_t n_cops;               /* number of active collapse operators */
    size_t d;                    /* Hilbert space dimension */
} lb_system_t;

/**
 * Precomputed propagator for a fixed timestep dt.
 * P = expm(L * dt), where L is the d²×d² Lindbladian superoperator.
 * Reuse across timesteps when H and cops are time-independent.
 */
typedef struct {
    lb_matrix_t P;   /* propagator matrix (d²×d²) */
    double dt;       /* timestep this propagator was built for */
    size_t d;        /* Hilbert space dimension */
} lb_propagator_t;

/* --------------------------------------------------------------------------
 * Matrix allocation / deallocation
 * -------------------------------------------------------------------------- */

/** Allocate a dim×dim zero matrix. Returns 0 on success, -1 on OOM. */
int lb_matrix_alloc(lb_matrix_t *m, size_t dim);

/** Free matrix data. Safe to call on zero-initialized struct. */
void lb_matrix_free(lb_matrix_t *m);

/** Copy src into dst (must be same dimension). */
int lb_matrix_copy(lb_matrix_t *dst, const lb_matrix_t *src);

/* --------------------------------------------------------------------------
 * System construction
 * -------------------------------------------------------------------------- */

/** Initialize a system struct (zeroes all fields). */
void lb_system_init(lb_system_t *sys, size_t d);

/** Free all matrices owned by sys. */
void lb_system_free(lb_system_t *sys);

/** Add a collapse operator. Returns -1 if LB_MAX_COPS exceeded. */
int lb_system_add_cop(lb_system_t *sys, const lb_matrix_t *L_k);

/* --------------------------------------------------------------------------
 * Lindbladian construction
 * -------------------------------------------------------------------------- */

/**
 * Build the Lindbladian superoperator L (d²×d²) from sys.
 *
 * L = -i(H⊗I - I⊗H^T) + Σ_k (L_k⊗L_k* - ½(L_k†L_k⊗I) - ½(I⊗L_k^T L_k*))
 *
 * out must be pre-allocated with dimension d² × d².
 */
int lb_build_lindbladian(const lb_system_t *sys, lb_matrix_t *out);

/* --------------------------------------------------------------------------
 * Matrix exponential
 * -------------------------------------------------------------------------- */

/**
 * Compute out = expm(A) using Padé [13/13] with scaling and squaring.
 *
 * Algorithm: Higham 2005, "The Scaling and Squaring Method for the Matrix
 * Exponential Revisited", DOI: 10.1137/04061101X.
 *
 * A and out must be square matrices of the same dimension.
 * Returns 0 on success, -1 on allocation failure.
 */
int lb_expm(const lb_matrix_t *A, lb_matrix_t *out);

/* --------------------------------------------------------------------------
 * Propagator
 * -------------------------------------------------------------------------- */

/**
 * Build the propagator P = expm(L * dt) from a Lindbladian superoperator L.
 * Caches the result in prop for reuse across timesteps.
 */
int lb_build_propagator(const lb_matrix_t *L, double dt, lb_propagator_t *prop);

/** Free propagator memory. */
void lb_propagator_free(lb_propagator_t *prop);

/**
 * Apply one propagation step: rho_out = P · vec(rho_in), reshaped to d×d.
 *
 * rho_in and rho_out are d×d density matrices (NOT vectorized).
 * The vectorization/reshaping is handled internally.
 */
int lb_propagate_step(const lb_propagator_t *prop,
                      const lb_matrix_t *rho_in,
                      lb_matrix_t *rho_out);

/* --------------------------------------------------------------------------
 * Trajectory integration
 * -------------------------------------------------------------------------- */

/**
 * Evolve rho from t=0 to t=t_end using fixed timestep dt.
 * Stores the final state in rho_out.
 *
 * If store_trajectory is non-NULL, it must point to an array of
 * (n_steps+1) pre-allocated lb_matrix_t structs (d×d each).
 * Set store_trajectory=NULL to skip trajectory storage (saves memory).
 *
 * Returns number of steps taken, or -1 on error.
 */
int lb_evolve(const lb_system_t *sys,
              const lb_matrix_t *rho0,
              double t_end,
              double dt,
              lb_matrix_t *rho_out,
              lb_matrix_t *store_trajectory,
              size_t *n_steps_out);

/* --------------------------------------------------------------------------
 * Diagnostics
 * -------------------------------------------------------------------------- */

/** Compute trace of a density matrix (should be 1.0 for valid state). */
double complex lb_trace(const lb_matrix_t *rho);

/**
 * Compute Hilbert-Schmidt norm ||A - B||_HS = sqrt(Tr[(A-B)†(A-B)]).
 * Used for validating C results against QuTiP reference.
 */
double lb_hs_norm(const lb_matrix_t *A, const lb_matrix_t *B);

#ifdef __cplusplus
}
#endif

#endif /* LINDBLAD_BENCH_H */
