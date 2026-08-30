#!/usr/bin/env python3
"""
fixed_point_accuracy.py — Effect of Q1.15 and Q1.31 fixed-point arithmetic
on d=3 Lindblad dynamics.

Builds the d=3 transmon propagator P = expm(L dt) in double precision, then
evolves |0><0| for N steps three ways: double, Q1.15, Q1.31. The fixed-point
matvec mirrors the Tang Nano RTL: integer multiply, integer accumulate,
floor shift, saturate. Two cases: undriven (H0 only) and driven
(H0 + 0.05 sigma_x), both with the T1/T2/dt of the GRAPE benchmark.

Reports, per sampled step count:
  hs_err      Hilbert-Schmidt norm of vec(rho) - vec(rho_double)
  trace_dist  trace distance 0.5 ||rho - rho_double||_1
  trace_drift |Tr(rho) - 1|
  p1_err      |<1|rho|1> - <1|rho_double|1>|
  saturations cumulative count of saturated accumulator outputs

Usage:
    python analysis/fixed_point_accuracy.py [--steps 10000] [--out benchmarks/fixed_point_accuracy.csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

T1 = 50.0
T2 = 30.0
DT = 0.5
DRIVE = 0.05


def build_lindbladian(d: int, drive: float) -> np.ndarray:
    H = np.diag(np.arange(d, dtype=complex))
    if d >= 2 and drive != 0.0:
        H[0, 1] += drive
        H[1, 0] += drive

    gamma1 = 1.0 / T1
    t_phi = 1.0 / (1.0 / T2 - 0.5 / T1)
    gamma_phi = 1.0 / t_phi
    cops = []
    L1 = np.zeros((d, d), dtype=complex)
    L1[0, 1] = np.sqrt(gamma1)
    cops.append(L1)
    L2 = np.zeros((d, d), dtype=complex)
    L2[1, 1] = np.sqrt(gamma_phi)
    cops.append(L2)

    eye = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, eye) - np.kron(eye, H.conj()))
    for Lk in cops:
        LdL = Lk.conj().T @ Lk
        L += np.kron(Lk, Lk.conj()) - 0.5 * np.kron(LdL, eye) - 0.5 * np.kron(eye, LdL.T)
    return L


def quantize(val: np.ndarray, frac_bits: int):
    scale = 2 ** frac_bits
    hi = scale - 1
    lo = -scale
    re = np.clip(np.round(val.real * scale), lo, hi).astype(np.int64)
    im = np.clip(np.round(val.imag * scale), lo, hi).astype(np.int64)
    return re, im


def fixed_matvec(P_re, P_im, v_re, v_im, frac_bits: int):
    """Integer matvec matching the RTL: multiply, accumulate, floor shift, saturate.
    Returns (out_re, out_im, n_saturated)."""
    n = len(v_re)
    hi = 2 ** frac_bits - 1
    lo = -(2 ** frac_bits)
    out_re = np.zeros(n, dtype=np.int64)
    out_im = np.zeros(n, dtype=np.int64)
    n_sat = 0
    for i in range(n):
        acc_re = 0
        acc_im = 0
        for j in range(n):
            pr = int(P_re[i, j])
            pi = int(P_im[i, j])
            vr = int(v_re[j])
            vi = int(v_im[j])
            acc_re += pr * vr - pi * vi
            acc_im += pr * vi + pi * vr
        r = acc_re >> frac_bits
        m = acc_im >> frac_bits
        if r > hi or r < lo:
            n_sat += 1
        if m > hi or m < lo:
            n_sat += 1
        out_re[i] = max(lo, min(hi, r))
        out_im[i] = max(lo, min(hi, m))
    return out_re, out_im, n_sat


def to_complex(re, im, frac_bits: int) -> np.ndarray:
    scale = 2.0 ** frac_bits
    return re / scale + 1j * im / scale


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape(d, d)


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff = 0.5 * (diff + diff.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


def sample_points(n_steps: int) -> list[int]:
    pts = np.unique(np.round(np.logspace(0, np.log10(n_steps), 40)).astype(int))
    return [int(p) for p in pts if 1 <= p <= n_steps]


def run_case(name: str, drive: float, n_steps: int, writer: csv.DictWriter) -> None:
    d = 3
    L = build_lindbladian(d, drive)
    P = expm(L * DT)
    print(f"[{name}] max|P_ij| = {np.abs(P).max():.6f}, ||L dt||_1 = {np.abs(L * DT).sum(axis=0).max():.4f}")

    rho_ref = np.zeros(d * d, dtype=complex)
    rho_ref[0] = 1.0
    state = {}
    for bits in (15, 31):
        P_re, P_im = quantize(P, bits)
        v_re, v_im = quantize(rho_ref, bits)
        p_err = float(np.linalg.norm(P - to_complex(P_re, P_im, bits)))
        print(f"[{name}] Q1.{bits}: ||P - P_q||_HS = {p_err:.3e}")
        state[bits] = {"P_re": P_re, "P_im": P_im, "v_re": v_re, "v_im": v_im, "sat": 0, "p_err": p_err}

    samples = set(sample_points(n_steps))
    for step in range(1, n_steps + 1):
        rho_ref = P @ rho_ref
        for bits, st in state.items():
            st["v_re"], st["v_im"], n_sat = fixed_matvec(st["P_re"], st["P_im"], st["v_re"], st["v_im"], bits)
            st["sat"] += n_sat
        if step in samples:
            ref_m = unvec(rho_ref, d)
            for bits, st in state.items():
                v = to_complex(st["v_re"], st["v_im"], bits)
                m = unvec(v, d)
                writer.writerow({
                    "case": name,
                    "drive": drive,
                    "format": f"Q1.{bits}",
                    "step": step,
                    "time": step * DT,
                    "hs_err": float(np.linalg.norm(v - rho_ref)),
                    "trace_dist": trace_distance(m, ref_m),
                    "trace_drift": abs(float(np.trace(m).real) - 1.0),
                    "p1_err": abs(float(m[1, 1].real) - float(ref_m[1, 1].real)),
                    "p1_ref": float(ref_m[1, 1].real),
                    "saturations": st["sat"],
                    "p_quant_err": st["p_err"],
                })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--out", default="benchmarks/fixed_point_accuracy.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "drive", "format", "step", "time", "hs_err", "trace_dist",
              "trace_drift", "p1_err", "p1_ref", "saturations", "p_quant_err"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        run_case("undriven", 0.0, args.steps, writer)
        run_case("driven", DRIVE, args.steps, writer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
