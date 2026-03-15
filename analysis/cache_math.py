"""
cache_math.py — Theoretical cache pressure analysis for Lindblad simulation.

For a d-level quantum system, the Lindbladian superoperator L acts on
vec(ρ) ∈ ℂ^(d²). Each propagation step is a (d²)×(d²) complex matvec.

Working set size of the propagator matrix P = expm(L·dt):
    bytes(P) = d^4 * 16  (complex128 = 16 bytes)

This script computes working set sizes and annotates them against the
cache hierarchy of the benchmark machine.

Output: table + plot (figures/cache_breakdown.pdf)

Usage:
    python analysis/cache_math.py [--machine <name>]
"""

import argparse
import math
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Machine cache specs (bytes)
# Defaults: Intel Core i9-13980HX (theLittleMachine)
# ---------------------------------------------------------------------------
MACHINES = {
    "i9-13980HX": {
        "name": "Intel Core i9-13980HX (theLittleMachine)",
        "L1":   49_152,       # 48 KB per P-core (data)
        "L2":   1_310_720,    # 1.25 MB per P-core
        "L3":   36_700_160,   # 35 MB shared
        "peak_gflops": 128.0, # ~128 GFLOP/s (AVX2, FP64, 2 FMAs/cycle, 3.0 GHz base)
        "peak_gbytes":  80.0, # ~80 GB/s DDR5
    },
    "generic": {
        "name": "Generic (edit cache_math.py to add your machine)",
        "L1":   32_768,
        "L2":   262_144,
        "L3":   8_388_608,
        "peak_gflops": 64.0,
        "peak_gbytes": 40.0,
    },
}

def working_set_bytes(d: int) -> dict:
    """Return working set sizes (bytes) for a d-level Lindblad simulation."""
    d2 = d * d
    bytes_per_complex = 16  # complex128

    P_bytes    = d2 * d2 * bytes_per_complex   # propagator matrix
    rho_bytes  = d2 * bytes_per_complex         # density vector (input + output)
    L_bytes    = d2 * d2 * bytes_per_complex   # Lindbladian (same size as P)

    return {
        "d":        d,
        "d2":       d2,
        "P_bytes":  P_bytes,
        "rho_bytes": rho_bytes,
        "L_bytes":  L_bytes,
        "hot_bytes": P_bytes + 2 * rho_bytes,  # matvec working set
        "P_KB":     P_bytes / 1024,
        "P_MB":     P_bytes / 1024**2,
    }

def arithmetic_intensity(d: int) -> float:
    """
    Arithmetic intensity of lb_propagate_step (FLOP/byte).

    FLOPs: d^4 complex multiply-adds = 8*d^4 real FLOPs
           (each complex fma = 6 real ops: 4 mul + 2 add;
            approximated as 8 for symmetry with standard counting)
    Bytes: read P (d^4*16) + read vec_in (d^2*16) + write vec_out (d^2*16)
    """
    d2 = d * d
    flops = 8.0 * d2 * d2
    bytes_transferred = (d2 * d2 + 2 * d2) * 16.0
    return flops / bytes_transferred

def cache_level(d: int, machine: dict) -> str:
    ws = working_set_bytes(d)["hot_bytes"]
    if ws <= machine["L1"]: return "L1"
    if ws <= machine["L2"]: return "L2"
    if ws <= machine["L3"]: return "L3"
    return "DRAM"

def print_table(machine: dict) -> None:
    print(f"\nMachine: {machine['name']}")
    print(f"L1={machine['L1']//1024} KB  "
          f"L2={machine['L2']//1024} KB  "
          f"L3={machine['L3']//1024//1024} MB")
    print()
    print(f"{'d':>4}  {'d²':>6}  {'P size':>12}  {'hot WS':>12}  "
          f"{'cache':>6}  {'AI (F/B)':>10}  {'ridge (F/B)':>12}")
    print("-" * 80)

    ridge = machine["peak_gflops"] / machine["peak_gbytes"]  # FLOP/byte

    for d in [1, 2, 3, 4, 5, 6, 9, 12, 18, 27, 36]:
        ws  = working_set_bytes(d)
        ai  = arithmetic_intensity(d)
        lvl = cache_level(d, machine)

        def fmt(b):
            if b < 1024: return f"{b:.0f} B"
            if b < 1024**2: return f"{b/1024:.1f} KB"
            return f"{b/1024**2:.2f} MB"

        print(f"{d:>4}  {ws['d2']:>6}  {fmt(ws['P_bytes']):>12}  "
              f"{fmt(ws['hot_bytes']):>12}  {lvl:>6}  "
              f"{ai:>10.4f}  {ridge:>12.4f}")

    print()
    print(f"Ridge point (peak FLOP/s / peak BW) = {ridge:.3f} FLOP/byte")
    print(f"Systems with AI < ridge point are MEMORY BOUND.")
    print()
    for d in [3, 9, 27]:
        ai  = arithmetic_intensity(d)
        lvl = cache_level(d, machine)
        bound = "memory-bound" if ai < ridge else "compute-bound"
        print(f"  d={d:2d}: AI={ai:.4f} FLOP/byte → {bound} ({lvl} regime)")

def plot_roofline_skeleton(machine: dict, outpath: str) -> None:
    """
    Plot a skeleton Roofline model annotated with d=3,9,27 points.
    Actual measured FLOP/s filled in by roofline.py after benchmarking.
    """
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ridge = machine["peak_gflops"] / machine["peak_gbytes"]

    ai_range = [0.01, 100.0]
    # Roofline: min(peak_bw * AI, peak_flops)
    import numpy as np
    ais = np.logspace(-2, 2, 500)
    roof = np.minimum(machine["peak_gbytes"] * ais, machine["peak_gflops"])

    ax.loglog(ais, roof, "k-", linewidth=2, label="Roofline ceiling")
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.6,
               label=f"Ridge point ({ridge:.2f} FLOP/byte)")

    colors = {"L1": "#2196F3", "L2": "#FF9800", "L3": "#F44336", "DRAM": "#9C27B0"}
    for d in [3, 9, 27]:
        ai  = arithmetic_intensity(d)
        lvl = cache_level(d, machine)
        # Theoretical peak at this AI (upper bound)
        peak_at_ai = min(machine["peak_gbytes"] * ai, machine["peak_gflops"])
        ax.axvline(ai, color=colors[lvl], linestyle=":", alpha=0.5)
        ax.scatter([ai], [peak_at_ai * 0.5], color=colors[lvl], s=100, zorder=5,
                   label=f"d={d} ({lvl}, AI={ai:.3f})")
        ax.annotate(f"d={d}", (ai, peak_at_ai * 0.5),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Skeleton — {machine['name']}\n"
                 f"(run bench_propagate + roofline.py to fill measured points)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 10.0)
    ax.set_ylim(0.1, machine["peak_gflops"] * 2)

    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved roofline skeleton to {outpath}")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", default="i9-13980HX",
                        choices=list(MACHINES.keys()),
                        help="Machine profile to use")
    parser.add_argument("--plot", action="store_true",
                        help="Generate roofline skeleton plot")
    parser.add_argument("--outdir", default="paper/figures",
                        help="Output directory for plots")
    args = parser.parse_args()

    machine = MACHINES[args.machine]
    print_table(machine)

    if args.plot:
        plot_roofline_skeleton(machine, f"{args.outdir}/roofline_skeleton.pdf")

if __name__ == "__main__":
    main()
