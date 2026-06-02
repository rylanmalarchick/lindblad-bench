"""
validate.py — Validate C library output against QuTiP reference.

Runs both lb_evolve (via C extension) and QuTiP mesolve on the same
system and compares final density matrices. Passes if Hilbert-Schmidt
norm < 1e-6.

Usage:
    python python/lindblad_bench/validate.py [--d 3] [--t-end 100] [--dt 0.5]
"""

import argparse
import sys
import numpy as np

try:
    import qutip as qt
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False


def _system(d: int):
    """Hamiltonian, collapse operators, and a non-stationary initial state.

    rho0 is the (|0>+|1>)/sqrt(2) superposition. The ground state |0><0| is a
    fixed point of amplitude damping and would leave the propagator untested,
    so it is deliberately avoided here."""
    H = np.diag(np.arange(d, dtype=complex))
    cops = []
    if d >= 2:
        L1 = np.zeros((d, d), dtype=complex)
        L1[0, 1] = np.sqrt(1.0 / 50.0)
        cops.append(L1)
    rho0 = np.zeros((d, d), dtype=complex)
    for i in (0, 1):
        for j in (0, 1):
            rho0[i, j] = 0.5
    return H, cops, rho0


def run_qutip(d: int, t_end: float, dt: float) -> np.ndarray:
    if not HAS_QUTIP:
        raise ImportError("qutip required: pip install qutip")
    H, cops, rho0 = _system(d)
    tlist = np.arange(0, t_end + dt, dt)
    # Tight solver tolerances so the comparison reflects the C propagator's
    # accuracy, not mesolve's default integrator drift.
    options = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000}
    result = qt.mesolve(qt.Qobj(H), qt.Qobj(rho0), tlist,
                        c_ops=[qt.Qobj(c) for c in cops], e_ops=[],
                        options=options)
    return result.states[-1].full()


def run_c(d: int, t_end: float, dt: float) -> np.ndarray:
    from lindblad_bench import evolve
    H, cops, rho0 = _system(d)
    return evolve(H, cops, rho0, t_end, dt)


def validate(d: int, t_end: float, dt: float) -> bool:
    print(f"Validating d={d}, t_end={t_end}, dt={dt}...")
    rho_qutip = run_qutip(d, t_end, dt)
    rho_c     = run_c(d, t_end, dt)
    hs_norm   = np.linalg.norm(rho_qutip - rho_c, "fro")
    ok = hs_norm < 1e-6
    print(f"  ||rho_qutip - rho_c||_HS = {hs_norm:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",     type=int,   default=3)
    parser.add_argument("--t-end", type=float, default=100.0, dest="t_end")
    parser.add_argument("--dt",    type=float, default=0.5)
    parser.add_argument("--all",   action="store_true",
                        help="Validate all three d values (3, 9, 27)")
    args = parser.parse_args()

    if args.all:
        results = [validate(d, args.t_end, args.dt) for d in [3, 9, 27]]
        passed = sum(results)
        print(f"\n{passed}/3 passed")
        sys.exit(0 if all(results) else 1)
    else:
        ok = validate(args.d, args.t_end, args.dt)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
