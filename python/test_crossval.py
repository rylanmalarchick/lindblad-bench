"""
test_crossval.py — Cross-validate the C library against QuTiP.

This is the automated form of the AGENTS.md contract:
    "C results must match QuTiP to HS norm < 1e-6."

It evolves the same transmon system with the C extension (lb_evolve) and with
QuTiP, then compares the final density matrices in Hilbert-Schmidt (Frobenius)
norm. Two independent QuTiP paths are used:

  * mesolve  — QuTiP's standard ODE solver (the reference/reference.py path).
               Its own integrator error floors agreement near ~1e-7, so it is
               checked against the documented 1e-6 contract.
  * exact    — QuTiP's Liouvillian exponentiated directly (no ODE integrator).
               This is accurate to ~machine precision and is checked at a tight
               1e-8, so it actually guards the C matrix-exponential's forward
               accuracy. (A regression to the pre-fix under-scaled expm, which
               was off by ~5e-7, fails this test.)

Dependencies (qutip, numpy, and the built C extension) are required: if any is
missing this module fails to import and pytest reports an error, never a skip
that would read as coverage. Marked `crossval` so CI can assert it ran.
"""

import numpy as np
import pytest
import qutip as qt

from lindblad_bench import evolve

# Match reference/reference.py build_transmon_system: amplitude damping (T1) and
# pure dephasing (T2) collapse operators on a diagonal Hamiltonian.
T1_US = 50.0
T2_US = 30.0

MESOLVE_THRESHOLD = 1e-6  # documented contract; floored by mesolve integrator error
EXACT_THRESHOLD = 1e-8    # tight guard on the C expm forward accuracy

pytestmark = pytest.mark.crossval


def _build_system(d):
    """Return (H, cops) as NumPy complex128 arrays."""
    H = np.diag(np.arange(d, dtype=complex))
    cops = []
    if d >= 2:
        gamma1 = 1.0 / T1_US
        L1 = np.zeros((d, d), dtype=complex)
        L1[0, 1] = np.sqrt(gamma1)
        cops.append(L1)

        t_phi = 1.0 / (1.0 / T2_US - 0.5 / T1_US)
        if t_phi > 0:
            gamma_phi = 1.0 / t_phi
            L2 = np.zeros((d, d), dtype=complex)
            L2[1, 1] = np.sqrt(gamma_phi)
            cops.append(L2)
    return H, cops


def _superposition(d):
    """Density matrix for (|0> + |1>)/sqrt(2): non-stationary under damping and
    dephasing, so it exercises coherent phase evolution, T1 decay, and T2
    coherence loss. The ground state |0><0| is a fixed point and must NOT be
    used here — it leaves the propagator untested."""
    rho0 = np.zeros((d, d), dtype=complex)
    for i in (0, 1):
        for j in (0, 1):
            rho0[i, j] = 0.5
    return rho0


def _qutip_mesolve(H, cops, rho0, t_end, dt):
    tlist = np.arange(0.0, t_end + dt, dt)
    options = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000}
    result = qt.mesolve(
        qt.Qobj(H),
        qt.Qobj(rho0),
        tlist,
        c_ops=[qt.Qobj(c) for c in cops],
        e_ops=[],
        options=options,
    )
    return result.states[-1].full()


def _qutip_exact(H, cops, rho0, t_end, dt):
    """Final state via QuTiP's Liouvillian exponentiated directly: P = expm(L*dt)
    applied ceil(t_end/dt) times. Matches lb_evolve's fixed-step propagator
    semantics exactly, with no ODE-integrator error."""
    n_steps = int(np.ceil(t_end / dt))
    liouv = qt.liouvillian(qt.Qobj(H), [qt.Qobj(c) for c in cops])
    P = (liouv * dt).expm()
    v = qt.operator_to_vector(qt.Qobj(rho0))
    for _ in range(n_steps):
        v = P * v
    return qt.vector_to_operator(v).full()


@pytest.mark.parametrize("d", [3, 9, 27])
def test_evolve_matches_qutip_mesolve(d):
    """C lb_evolve final state matches QuTiP mesolve to the 1e-6 contract.

    The initial superposition decays only partway over t_end, so the compared
    state is genuinely mid-trajectory (off-diagonal coherences and excited
    population both nonzero), not a trivial steady state."""
    t_end, dt = 30.0, 0.5
    H, cops = _build_system(d)
    rho0 = _superposition(d)

    rho_c = evolve(H, cops, rho0, t_end, dt)
    rho_qt = _qutip_mesolve(H, cops, rho0, t_end, dt)

    # Guard against re-introducing a fixed-point initial state: the dynamics
    # must actually move the state away from rho0.
    assert np.linalg.norm(rho_c - rho0, "fro") > 1e-3

    hs_norm = np.linalg.norm(rho_qt - rho_c, "fro")
    assert hs_norm < MESOLVE_THRESHOLD, (
        f"d={d}: ||rho_mesolve - rho_c||_HS = {hs_norm:.3e} >= {MESOLVE_THRESHOLD:.0e}"
    )


@pytest.mark.parametrize("d", [3, 9, 27])
def test_evolve_matches_qutip_exact(d):
    """C lb_evolve matches QuTiP's exact propagator to 1e-8.

    This is the tight guard: it pins the C matrix-exponential's forward accuracy
    against an integrator-free reference. The pre-fix under-scaled expm (~5e-7
    error) fails this assertion."""
    t_end, dt = 30.0, 0.5
    H, cops = _build_system(d)
    rho0 = _superposition(d)

    rho_c = evolve(H, cops, rho0, t_end, dt)
    rho_exact = _qutip_exact(H, cops, rho0, t_end, dt)

    hs_norm = np.linalg.norm(rho_exact - rho_c, "fro")
    assert hs_norm < EXACT_THRESHOLD, (
        f"d={d}: ||rho_exact - rho_c||_HS = {hs_norm:.3e} >= {EXACT_THRESHOLD:.0e}"
    )


def test_evolve_preserves_trace_against_qutip():
    """Both solvers keep Tr[rho] = 1; the C trace must match QuTiP's tightly."""
    d = 3
    t_end, dt = 50.0, 0.5
    H, cops = _build_system(d)
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[1, 1] = 1.0  # start in |1><1|

    rho_c = evolve(H, cops, rho0, t_end, dt)
    rho_exact = _qutip_exact(H, cops, rho0, t_end, dt)

    assert abs(np.trace(rho_c) - 1.0) < 1e-9
    assert abs(np.trace(rho_c) - np.trace(rho_exact)) < 1e-9
