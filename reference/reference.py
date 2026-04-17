"""
reference.py — QuTiP reference implementation of Lindblad simulation.

Two roles:
  1. Validation oracle: run alongside C library, compare to Hilbert-Schmidt norm < 1e-6
  2. Baseline benchmark: time QuTiP mesolve() for the same system/timestep as C

The harness supports multiple solver modes. The released QuTiP version
available on PyPI may not support every documented mode, so the script probes
for support before attempting to benchmark those paths.

Usage:
    # Validate C output (C must write rho_final to stdout as CSV):
    python reference/reference.py --validate --d 3

    # Benchmark QuTiP step timing for comparison:
    python reference/reference.py --bench --d 3 --n-reps 100 --solver-mode default

    # Run all three d values, probing all supported solver modes:
    python reference/reference.py --bench-all --solver-mode all
"""

import argparse
import sys
import time
import csv
import os

try:
    import numpy as np
    import qutip as qt
except ImportError:
    print("qutip and numpy required: pip install qutip numpy", file=sys.stderr)
    sys.exit(1)


_MATRIX_FORM_SUPPORTED = None


# ---------------------------------------------------------------------------
# System construction (mirrors bench_propagate.c build_transmon_system)
# ---------------------------------------------------------------------------

def build_transmon_system(d: int, T1_us: float = 50.0, T2_us: float = 30.0):
    """
    Build a d-level transmon-like system.
    H = diag(0, 1, ..., d-1) (dimensionless, scaled units)
    Collapse ops: sqrt(gamma1)*|0><1|, sqrt(gamma_phi)*|1><1|
    """
    H = qt.Qobj(np.diag(np.arange(d, dtype=complex)))

    gamma1 = 1.0 / T1_us
    cops = []

    if d >= 2:
        L1_data = np.zeros((d, d), dtype=complex)
        L1_data[0, 1] = np.sqrt(gamma1)
        cops.append(qt.Qobj(L1_data))

    T_phi = 1.0 / (1.0 / T2_us - 0.5 / T1_us)
    if T_phi > 0 and d >= 2:
        gamma_phi = 1.0 / T_phi
        L2_data = np.zeros((d, d), dtype=complex)
        L2_data[1, 1] = np.sqrt(gamma_phi)
        cops.append(qt.Qobj(L2_data))

    return H, cops


def ground_state(d: int) -> qt.Qobj:
    data = np.zeros((d, d), dtype=complex)
    data[0, 0] = 1.0
    return qt.Qobj(data)


def matrix_form_supported() -> bool:
    global _MATRIX_FORM_SUPPORTED
    if _MATRIX_FORM_SUPPORTED is not None:
        return _MATRIX_FORM_SUPPORTED

    H, cops = build_transmon_system(2)
    rho0 = ground_state(2)
    try:
        qt.mesolve(
            H,
            rho0,
            [0.0, 0.5],
            c_ops=cops,
            e_ops=[],
            options={"matrix_form": True},
        )
    except KeyError:
        _MATRIX_FORM_SUPPORTED = False
    else:
        _MATRIX_FORM_SUPPORTED = True
    return _MATRIX_FORM_SUPPORTED


def solver_options(solver_mode: str) -> dict:
    if solver_mode == "default":
        return {}
    if solver_mode == "matrix_form":
        if not matrix_form_supported():
            raise RuntimeError(
                f"Installed QuTiP {qt.__version__} does not support matrix_form"
            )
        return {"matrix_form": True}
    raise ValueError(f"Unknown solver mode: {solver_mode}")


def normalize_solver_modes(solver_mode: str) -> list[str]:
    if solver_mode == "all":
        modes = ["default"]
        if matrix_form_supported():
            modes.append("matrix_form")
        return modes
    return [solver_mode]


# ---------------------------------------------------------------------------
# Single-step propagation (validates against C lb_propagate_step)
# ---------------------------------------------------------------------------

def propagate_step_qutip(
    H,
    cops,
    rho0: qt.Qobj,
    dt: float,
    solver_mode: str = "default",
) -> qt.Qobj:
    """Evolve rho0 by one step dt using mesolve."""
    result = qt.mesolve(
        H,
        rho0,
        [0.0, dt],
        c_ops=cops,
        e_ops=[],
        options=solver_options(solver_mode),
    )
    return result.states[-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    d: int,
    dt: float = 0.5,
    t_end: float = 100.0,
    solver_mode: str = "default",
) -> None:
    """
    Evolve using QuTiP and compare to C output read from stdin.
    Prints Hilbert-Schmidt norm difference.
    """
    H, cops = build_transmon_system(d)
    rho0 = ground_state(d)

    tlist = np.arange(0, t_end + dt, dt)
    result = qt.mesolve(
        H,
        rho0,
        tlist,
        c_ops=cops,
        e_ops=[],
        options=solver_options(solver_mode),
    )
    rho_final_qutip = result.states[-1].full()

    print(f"QuTiP rho_final (d={d}, t={t_end}):")
    print(np.round(rho_final_qutip, 6))

    # Read C output from stdin if piped
    if not sys.stdin.isatty():
        lines = sys.stdin.read().strip().splitlines()
        rho_c = np.array([[complex(x) for x in row.split(",")] for row in lines])
        hs_norm = np.linalg.norm(rho_final_qutip - rho_c, "fro")
        print(f"\nHilbert-Schmidt norm ||rho_qutip - rho_c|| = {hs_norm:.2e}")
        ok = hs_norm < 1e-6
        print(f"Validation: {'PASS' if ok else 'FAIL'} (threshold 1e-6)")
        sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_single(d: int, n_reps: int, solver_mode: str = "default") -> dict:
    """Time QuTiP propagate_step and return ns_per_step."""
    H, cops = build_transmon_system(d)
    rho0 = ground_state(d)
    dt = 0.5  # ns, matches bench_propagate.c

    # Warm-up
    for _ in range(5):
        propagate_step_qutip(H, cops, rho0, dt, solver_mode=solver_mode)

    t0 = time.perf_counter_ns()
    for _ in range(n_reps):
        propagate_step_qutip(H, cops, rho0, dt, solver_mode=solver_mode)
    t1 = time.perf_counter_ns()

    ns_per_step = (t1 - t0) / n_reps
    return {
        "d": d,
        "solver_mode": solver_mode,
        "ns_per_step": ns_per_step,
        "n_reps": n_reps,
        "qutip_version": qt.__version__,
    }


def bench_all(n_reps_map: dict, outpath: str, solver_mode: str) -> None:
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    results = []
    for mode in normalize_solver_modes(solver_mode):
        for d, n_reps in n_reps_map.items():
            print(f"Benchmarking d={d}, mode={mode} ({n_reps} reps)...", flush=True)
            r = bench_single(d, n_reps, solver_mode=mode)
            print(f"  d={d}, mode={mode}: {r['ns_per_step']:.0f} ns/step "
                  f"({r['ns_per_step']/1e6:.2f} ms/step)")
            results.append(r)

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["d", "solver_mode", "ns_per_step", "n_reps", "qutip_version"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate",  action="store_true")
    parser.add_argument("--bench",     action="store_true")
    parser.add_argument("--bench-all", action="store_true", dest="bench_all")
    parser.add_argument("--d",         type=int, default=3)
    parser.add_argument("--n-reps",    type=int, default=100, dest="n_reps")
    parser.add_argument("--dt",        type=float, default=0.5)
    parser.add_argument("--outdir",    default="benchmarks")
    parser.add_argument(
        "--solver-mode",
        choices=["default", "matrix_form", "all"],
        default="default",
        help="QuTiP solver mode to benchmark or validate",
    )
    args = parser.parse_args()

    if args.validate:
        validate(args.d, dt=args.dt, solver_mode=args.solver_mode)
    elif args.bench:
        if args.solver_mode == "all":
            parser.error("--bench requires a single solver mode")
        r = bench_single(args.d, args.n_reps, solver_mode=args.solver_mode)
        print(f"d={r['d']}, mode={r['solver_mode']}: {r['ns_per_step']:.0f} ns/step")
    elif args.bench_all:
        n_reps_map = {3: 500, 9: 100, 27: 20}
        bench_all(
            n_reps_map,
            os.path.join(args.outdir, "qutip_results.csv"),
            args.solver_mode,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
