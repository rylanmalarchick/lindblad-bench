/**
 * lindblad.c — Lindbladian superoperator construction and system management.
 *
 * Builds the d²×d² Lindbladian L from H and collapse operators {L_k}:
 *
 *   L = -i(H⊗I - I⊗H*) + Σ_k [ L_k⊗L_k* - ½(L_k†L_k⊗I) - ½(I⊗L_k^T L_k*) ]
 *
 * References:
 *   Lindblad 1976, DOI: 10.1007/BF01608499
 *   Gorini, Kossakowski, Sudarshan 1976, DOI: 10.1063/1.522979
 */

#include "lindblad.h"
#include "expm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* --------------------------------------------------------------------------
 * Matrix allocation
 * -------------------------------------------------------------------------- */

int lb_matrix_alloc(lb_matrix_t *m, size_t dim)
{
    m->dim  = dim;
    size_t size = dim * dim * sizeof(double complex);
    size_t aligned_size = (size + 63) & ~63; /* pad to multiple of 64 */
    m->data = aligned_alloc(64, aligned_size);
    if (m->data) {
        memset(m->data, 0, aligned_size);
        return 0;
    }
    return -1;
}

void lb_matrix_free(lb_matrix_t *m)
{
    free(m->data);
    m->data = NULL;
    m->dim  = 0;
}

int lb_matrix_copy(lb_matrix_t *dst, const lb_matrix_t *src)
{
    if (dst->dim != src->dim) return -1;
    memcpy(dst->data, src->data, src->dim * src->dim * sizeof(double complex));
    return 0;
}

int lb_matrix_soa_alloc(lb_matrix_soa_t *m, size_t dim)
{
    m->dim = dim;
    size_t size = dim * dim * sizeof(double);
    size_t aligned_size = (size + 63) & ~63;
    m->real = aligned_alloc(64, aligned_size);
    m->imag = aligned_alloc(64, aligned_size);
    if (m->real && m->imag) {
        memset(m->real, 0, aligned_size);
        memset(m->imag, 0, aligned_size);
        return 0;
    }
    if (m->real) { free(m->real); m->real = NULL; }
    if (m->imag) { free(m->imag); m->imag = NULL; }
    return -1;
}

void lb_matrix_soa_free(lb_matrix_soa_t *m)
{
    free(m->real);
    free(m->imag);
    m->real = NULL;
    m->imag = NULL;
    m->dim = 0;
}

int lb_matrix_to_soa(const lb_matrix_t *aos, lb_matrix_soa_t *soa)
{
    if (aos->dim != soa->dim) return -1;
    size_t n = aos->dim * aos->dim;
    for (size_t i = 0; i < n; i++) {
        soa->real[i] = creal(aos->data[i]);
        soa->imag[i] = cimag(aos->data[i]);
    }
    return 0;
}

int lb_matrix_to_aos(const lb_matrix_soa_t *soa, lb_matrix_t *aos)
{
    if (soa->dim != aos->dim) return -1;
    size_t n = soa->dim * soa->dim;
    for (size_t i = 0; i < n; i++) {
        aos->data[i] = soa->real[i] + I * soa->imag[i];
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * System construction
 * -------------------------------------------------------------------------- */

void lb_system_init(lb_system_t *sys, size_t d)
{
    memset(sys, 0, sizeof(lb_system_t));
    sys->d = d;
}

void lb_system_free(lb_system_t *sys)
{
    lb_matrix_free(&sys->H);
    for (size_t i = 0; i < sys->n_cops; i++) {
        lb_matrix_free(&sys->cops[i]);
    }
    sys->n_cops = 0;
}

int lb_system_add_cop(lb_system_t *sys, const lb_matrix_t *L_k)
{
    if (sys->n_cops >= LB_MAX_COPS) return -1;
    lb_matrix_t *slot = &sys->cops[sys->n_cops];
    if (lb_matrix_alloc(slot, sys->d) != 0) return -1;
    memcpy(slot->data, L_k->data, sys->d * sys->d * sizeof(double complex));
    sys->n_cops++;
    return 0;
}

/* --------------------------------------------------------------------------
 * Kronecker product helpers (internal)
 * -------------------------------------------------------------------------- */

static int kron(const lb_matrix_t *A, const lb_matrix_t *B, lb_matrix_t *out)
{
    size_t da = A->dim, db = B->dim;
    size_t n  = da * db;
    if (out->dim != n) return -1;
    memset(out->data, 0, n * n * sizeof(double complex));
    for (size_t ia = 0; ia < da; ia++) {
        for (size_t ja = 0; ja < da; ja++) {
            double complex aij = A->data[ia * da + ja];
            if (aij == 0.0) continue;
            for (size_t ib = 0; ib < db; ib++) {
                for (size_t jb = 0; jb < db; jb++) {
                    size_t row = ia * db + ib;
                    size_t col = ja * db + jb;
                    out->data[row * n + col] += aij * B->data[ib * db + jb];
                }
            }
        }
    }
    return 0;
}

static void mat_conj(const lb_matrix_t *A, lb_matrix_t *out)
{
    size_t n2 = A->dim * A->dim;
    for (size_t i = 0; i < n2; i++) {
        out->data[i] = conj(A->data[i]);
    }
}

static void mat_dag(const lb_matrix_t *A, lb_matrix_t *out)
{
    size_t n = A->dim;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out->data[j * n + i] = conj(A->data[i * n + j]);
        }
    }
}

/* --------------------------------------------------------------------------
 * lb_build_lindbladian
 *
 * All matrices declared at top so goto-based cleanup is well-defined.
 * lb_matrix_free() is safe to call on zero-initialized structs (data==NULL).
 * -------------------------------------------------------------------------- */
int lb_build_lindbladian(const lb_system_t *sys, lb_matrix_t *out)
{
    size_t d  = sys->d;
    size_t d2 = d * d;

    if (out->dim != d2) return -1;
    memset(out->data, 0, d2 * d2 * sizeof(double complex));

    /* Declare all temporaries at top — zero-initialized for safe goto cleanup */
    lb_matrix_t Id    = {0};
    lb_matrix_t Hc    = {0};
    lb_matrix_t HkId  = {0};
    lb_matrix_t IdkHc = {0};
    lb_matrix_t Ldag  = {0};
    lb_matrix_t LdagL = {0};
    lb_matrix_t Lc    = {0};
    lb_matrix_t LkLc  = {0};
    lb_matrix_t LdLkI = {0};
    lb_matrix_t IkLdL = {0};
    int ret = 0;

    if (lb_matrix_alloc(&Id,    d)  != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&Hc,    d)  != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&HkId,  d2) != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&IdkHc, d2) != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&Ldag,  d)  != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&LdagL, d)  != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&Lc,    d)  != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&LkLc,  d2) != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&LdLkI, d2) != 0) { ret = -1; goto done; }
    if (lb_matrix_alloc(&IkLdL, d2) != 0) { ret = -1; goto done; }

    mat_identity(&Id);
    mat_conj(&sys->H, &Hc);

    /* Coherent part: -i(H⊗Id - Id⊗H*) */
    kron(&sys->H, &Id, &HkId);
    kron(&Id,     &Hc, &IdkHc);
    for (size_t i = 0; i < d2 * d2; i++) {
        /* Note: capital I here is the imaginary unit from <complex.h>, not our matrix */
        out->data[i] += -I * (HkId.data[i] - IdkHc.data[i]);
    }

    /* Dissipative part: Σ_k [ L_k⊗L_k* - ½(L_k†L_k⊗Id) - ½(Id⊗L_k^T L_k*) ] */
    for (size_t k = 0; k < sys->n_cops; k++) {
        const lb_matrix_t *Lop = &sys->cops[k];

        mat_dag(Lop, &Ldag);          /* L†        */
        mat_mul(&Ldag, Lop, &LdagL); /* L†L       */
        mat_conj(Lop, &Lc);           /* L*        */

        kron(Lop,    &Lc,    &LkLc);  /* L ⊗ L*   */
        kron(&LdagL, &Id,    &LdLkI); /* L†L ⊗ Id */

        /* Row-major vectorization: vec(ρM) = (I⊗M^T)vec(ρ).
         * For Hermitian M = L†L: M^T = M*, so use conj(LdagL).
         * Reuse Hc (d×d, no longer needed after coherent part). */
        mat_conj(&LdagL, &Hc);
        kron(&Id,    &Hc,    &IkLdL); /* Id ⊗ (L†L)^T */

        for (size_t i = 0; i < d2 * d2; i++) {
            out->data[i] += LkLc.data[i]
                          - 0.5 * LdLkI.data[i]
                          - 0.5 * IkLdL.data[i];
        }
    }

done:
    lb_matrix_free(&Id);
    lb_matrix_free(&Hc);
    lb_matrix_free(&HkId);
    lb_matrix_free(&IdkHc);
    lb_matrix_free(&Ldag);
    lb_matrix_free(&LdagL);
    lb_matrix_free(&Lc);
    lb_matrix_free(&LkLc);
    lb_matrix_free(&LdLkI);
    lb_matrix_free(&IkLdL);
    return ret;
}

/* --------------------------------------------------------------------------
 * Diagnostics
 * -------------------------------------------------------------------------- */

double complex lb_trace(const lb_matrix_t *rho)
{
    double complex tr = 0.0;
    size_t n = rho->dim;
    for (size_t i = 0; i < n; i++) {
        tr += rho->data[i * n + i];
    }
    return tr;
}

double lb_hs_norm(const lb_matrix_t *A, const lb_matrix_t *B)
{
    size_t n2 = A->dim * A->dim;
    double acc = 0.0;
    for (size_t i = 0; i < n2; i++) {
        double complex diff = A->data[i] - B->data[i];
        acc += creal(diff) * creal(diff) + cimag(diff) * cimag(diff);
    }
    return sqrt(acc);
}
