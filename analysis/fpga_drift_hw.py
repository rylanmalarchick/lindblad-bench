#!/usr/bin/env python3
"""
fpga_drift_hw.py — Hardware-in-the-loop fixed-point drift on the Tang Nano.

Loads the d=3 Q1.15 propagator onto the board, runs N on-chip steps with the
'N' command, reads the 36-byte state back, and compares it with (a) the
Python Q1.15 model from analysis/fixed_point_accuracy.py and (b) the
double-precision reference. Two cases: undriven and driven (0.05 sigma_x),
same T1, T2, dt as the GRAPE benchmark.

The Python model must match the board bit for bit at every N. The double
comparison gives the physical error of the fixed-point path.

Usage:
    python analysis/fpga_drift_hw.py [--port /dev/ttyUSB1] [--steps 1,10,100,1000,10000]
                                     [--out benchmarks/fpga/drift_hw.csv]
Requires pyserial and the tang-nano-20k host script for the UART protocol.
"""

import argparse
import csv
import sys
import importlib.util
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
TANG = Path.home() / "dev" / "projects" / "tang-nano-20k"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fpa = load_module("fixed_point_accuracy", HERE / "fixed_point_accuracy.py")
host = load_module("matvec_host", TANG / "matvec" / "host" / "matvec_host.py")


def q15_state(v_re, v_im):
    return fpa.to_complex(np.asarray(v_re, dtype=np.int64), np.asarray(v_im, dtype=np.int64), 15)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--steps", default="1,10,100,1000,10000")
    ap.add_argument("--out", default="benchmarks/fpga/drift_hw.csv")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    if max(steps) > 65535:
        raise SystemExit("the N command carries a uint16 step count")

    import serial
    ser = serial.Serial(args.port, args.baud, timeout=60)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "drive", "steps", "hw_matches_model", "max_lsb_diff",
              "hs_err_vs_double", "trace_dist_vs_double", "trace_drift", "p1_err"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for case, drive in (("undriven", 0.0), ("driven", fpa.DRIVE)):
            L = fpa.build_lindbladian(3, drive)
            P = expm(L * fpa.DT)
            P_re, P_im = fpa.quantize(P, 15)
            rho0 = np.zeros(9, dtype=complex)
            rho0[0] = 1.0
            r0_re, r0_im = fpa.quantize(rho0, 15)

            host.send_load_p(ser, P_re.reshape(-1), P_im.reshape(-1))

            for n in steps:
                # Board: fresh state, N steps on chip, read back.
                host.send_load_rho(ser, r0_re, r0_im)
                hw_re, hw_im = host.send_n_steps(ser, n)

                # Python Q1.15 model, same start.
                v_re, v_im = r0_re.copy(), r0_im.copy()
                ref = rho0.copy()
                for _ in range(n):
                    v_re, v_im, _sat = fpa.fixed_matvec(P_re, P_im, v_re, v_im, 15)
                    ref = P @ ref

                lsb = max(int(np.max(np.abs(hw_re.astype(np.int64) - v_re))),
                          int(np.max(np.abs(hw_im.astype(np.int64) - v_im))))
                hw = q15_state(hw_re, hw_im)
                m = hw.reshape(3, 3)
                rm = ref.reshape(3, 3)
                row = {
                    "case": case,
                    "drive": drive,
                    "steps": n,
                    "hw_matches_model": int(lsb == 0),
                    "max_lsb_diff": lsb,
                    "hs_err_vs_double": float(np.linalg.norm(hw - ref)),
                    "trace_dist_vs_double": fpa.trace_distance(m, rm),
                    "trace_drift": abs(float(np.trace(m).real) - 1.0),
                    "p1_err": abs(float(m[1, 1].real) - float(rm[1, 1].real)),
                }
                w.writerow(row)
                print(f"[{case}] N={n:6d} hw==model:{bool(lsb == 0)} lsb_diff={lsb} "
                      f"td={row['trace_dist_vs_double']:.3e} tr_drift={row['trace_drift']:.3e}")
    ser.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
