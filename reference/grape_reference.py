#!/usr/bin/env python3
"""
grape_reference.py — QuTiP GRAPE-style benchmark harness.

Models one GRAPE "landscape point" as a piecewise-constant control sequence:
  1. Build a deterministic drive schedule over N segments
  2. Evolve the density matrix through each segment with constant H
  3. Report end-to-end time per trajectory and equivalent ns/state-step

This is the QuTiP-side application benchmark that complements the C
`bench_grape` executable.
"""

import argparse
import csv
import os
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import qutip as qt


REFERENCE_PATH = Path(__file__).resolve().parent / "reference.py"
spec = importlib.util.spec_from_file_location("reference_script", REFERENCE_PATH)
reference = importlib.util.module_from_spec(spec)
sys.modules["reference_script"] = reference
assert spec.loader is not None
spec.loader.exec_module(reference)


GRAPE_PHI = 0.6180339887498949  # 1/golden ratio; matches bench_grape.c


def build_drive_schedule(n_segments: int, seed: int = 42) -> np.ndarray:
    """Deterministic low-discrepancy drive schedule.

    omega_s = 0.1 * (frac((s + 1) * phi) - 0.5). Identical to
    lb_grape_drive_amplitude() in benchmarks/bench_grape.c so the C and
    QuTiP benchmarks evolve under the same controls. `seed` is accepted for
    call compatibility and ignored.
    """
    del seed
    s = np.arange(1, n_segments + 1, dtype=np.float64)
    x = s * GRAPE_PHI
    return 0.1 * ((x - np.floor(x)) - 0.5)


def sigma_x_truncated(d: int) -> qt.Qobj:
    data = np.zeros((d, d), dtype=complex)
    if d >= 2:
        data[0, 1] = 1.0
        data[1, 0] = 1.0
    return qt.Qobj(data)


def build_segment_hamiltonians(
    d: int,
    n_segments: int,
    seed: int = 42,
) -> list[qt.Qobj]:
    H0, _ = reference.build_transmon_system(d)
    drive = sigma_x_truncated(d)
    omegas = build_drive_schedule(n_segments, seed=seed)
    return [H0 + omega * drive for omega in omegas]


def run_piecewise_propagator(
    d: int,
    n_segments: int,
    steps_per_seg: int,
    dt: float,
    seed: int = 42,
) -> qt.Qobj:
    """Algorithm-matched QuTiP path: P_s = expm(L_s dt), applied steps_per_seg
    times per segment on the vectorized state. Same operation count as the
    C benchmark; the exponential and matvec come from QuTiP/SciPy."""
    _, c_ops = reference.build_transmon_system(d)
    rho_vec = qt.operator_to_vector(reference.ground_state(d))

    for H in build_segment_hamiltonians(d, n_segments, seed=seed):
        L = qt.liouvillian(H, c_ops)
        P = (L * dt).expm()
        for _ in range(steps_per_seg):
            rho_vec = P * rho_vec

    return qt.vector_to_operator(rho_vec)


def run_piecewise_mesolve(
    d: int,
    n_segments: int,
    steps_per_seg: int,
    dt: float,
    solver_mode: str = "default",
    seed: int = 42,
) -> qt.Qobj:
    if solver_mode == "propagator":
        return run_piecewise_propagator(d, n_segments, steps_per_seg, dt, seed=seed)

    _, c_ops = reference.build_transmon_system(d)
    rho = reference.ground_state(d)
    seg_duration = steps_per_seg * dt
    options = reference.solver_options(solver_mode)

    for H in build_segment_hamiltonians(d, n_segments, seed=seed):
        result = qt.mesolve(
            H,
            rho,
            [0.0, seg_duration],
            c_ops=c_ops,
            e_ops=[],
            options=options,
        )
        rho = result.states[-1]

    return rho


def bench_piecewise_mesolve(
    d: int,
    n_segments: int,
    steps_per_seg: int,
    n_trials: int,
    dt: float,
    solver_mode: str = "default",
    seed: int = 42,
) -> dict:
    run_piecewise_mesolve(
        d,
        n_segments,
        steps_per_seg,
        dt,
        solver_mode=solver_mode,
        seed=seed,
    )

    trial_ns = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        run_piecewise_mesolve(
            d,
            n_segments,
            steps_per_seg,
            dt,
            solver_mode=solver_mode,
            seed=seed,
        )
        t1 = time.perf_counter_ns()
        trial_ns.append(t1 - t0)

    ns_median = float(np.median(trial_ns))
    total_steps = n_segments * steps_per_seg
    return {
        "d": d,
        "solver_mode": solver_mode,
        "n_segments": n_segments,
        "steps_per_seg": steps_per_seg,
        "n_trials": n_trials,
        "ms_per_trajectory": ns_median / 1e6,
        "ns_per_step": ns_median / total_steps,
        "ms_trials": ";".join(f"{t / 1e6:.6f}" for t in trial_ns),
        "qutip_version": qt.__version__,
    }


def bench_all(outpath: str, solver_mode: str, dt: float) -> None:
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    configs = {
        3: {"n_segments": 100, "steps_per_seg": 20, "n_trials": 20},
        9: {"n_segments": 50, "steps_per_seg": 20, "n_trials": 5},
        27: {"n_segments": 20, "steps_per_seg": 20, "n_trials": 5},
    }

    results = []
    for mode in reference.normalize_solver_modes(solver_mode):
        for d, cfg in configs.items():
            print(
                f"Benchmarking d={d}, mode={mode}, "
                f"{cfg['n_segments']} segments x {cfg['steps_per_seg']} steps...",
                flush=True,
            )
            result = bench_piecewise_mesolve(
                d=d,
                n_segments=cfg["n_segments"],
                steps_per_seg=cfg["steps_per_seg"],
                n_trials=cfg["n_trials"],
                dt=dt,
                solver_mode=mode,
            )
            print(
                f"  d={d}, mode={mode}: "
                f"{result['ms_per_trajectory']:.3f} ms/trajectory, "
                f"{result['ns_per_step']:.2f} ns/step"
            )
            results.append(result)

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "d",
                "solver_mode",
                "n_segments",
                "steps_per_seg",
                "n_trials",
                "ms_per_trajectory",
                "ns_per_step",
                "ms_trials",
                "qutip_version",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-all", action="store_true", dest="bench_all")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n-segments", type=int, default=100, dest="n_segments")
    parser.add_argument("--steps-per-seg", type=int, default=20, dest="steps_per_seg")
    parser.add_argument("--n-trials", type=int, default=3, dest="n_trials")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--outdir", default="benchmarks")
    parser.add_argument(
        "--solver-mode",
        choices=["default", "propagator", "matrix_form", "all"],
        default="default",
    )
    args = parser.parse_args()

    if args.bench:
        if args.solver_mode == "all":
            parser.error("--bench requires a single solver mode")
        result = bench_piecewise_mesolve(
            d=args.d,
            n_segments=args.n_segments,
            steps_per_seg=args.steps_per_seg,
            n_trials=args.n_trials,
            dt=args.dt,
            solver_mode=args.solver_mode,
        )
        print(
            f"d={result['d']}, mode={result['solver_mode']}: "
            f"{result['ms_per_trajectory']:.3f} ms/trajectory, "
            f"{result['ns_per_step']:.2f} ns/step"
        )
    elif args.bench_all:
        bench_all(
            os.path.join(args.outdir, "qutip_grape_results.csv"),
            solver_mode=args.solver_mode,
            dt=args.dt,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
