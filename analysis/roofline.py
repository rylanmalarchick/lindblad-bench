"""
roofline.py — Roofline model plot from measured benchmark data.

Reads benchmark output (CSV or stdin) and overlays measured GFLOP/s
on the theoretical Roofline ceiling from cache_math.py.

Usage:
    # After running bench_propagate and capturing output:
    ./build/bench_propagate | python analysis/roofline.py --stdin

    # Or from a CSV with columns: d, gflops, gbytes
    python analysis/roofline.py --csv benchmarks/results.csv

CSV format:
    d,gflops,gbytes_s
    3,<measured>,<measured>
    9,<measured>,<measured>
    27,<measured>,<measured>
"""

import argparse
import sys
import re

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("numpy + matplotlib required: pip install numpy matplotlib", file=sys.stderr)
    sys.exit(1)

from cache_math import MACHINES, arithmetic_intensity, cache_level, working_set_bytes

COLORS = {"L1": "#2196F3", "L2": "#FF9800", "L3": "#F44336", "DRAM": "#9C27B0"}


def parse_bench_output(text: str) -> list[dict]:
    """Parse stdout from bench_propagate into list of {d, gflops, gbytes_s}."""
    results = []
    current_d = None
    for line in text.splitlines():
        m = re.search(r"d\s*=\s*(\d+)", line)
        if m:
            current_d = int(m.group(1))
        m = re.search(r"throughput\s*:\s*([\d.]+)\s*GFLOP", line)
        if m and current_d is not None:
            gflops = float(m.group(1))
        m = re.search(r"bandwidth\s*:\s*([\d.]+)\s*GB", line)
        if m and current_d is not None:
            gbytes_s = float(m.group(1))
            results.append({"d": current_d, "gflops": gflops, "gbytes_s": gbytes_s})
            current_d = None
    return results


def parse_csv(path: str) -> list[dict]:
    import csv
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "d": int(row["d"]),
                "gflops": float(row["gflops"]),
                "gbytes_s": float(row.get("gbytes_s", 0)),
            })
    return results


def plot(measured: list[dict], machine: dict, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    peak_gflops = machine["peak_gflops"]
    peak_gbytes = machine["peak_gbytes"]
    ridge = peak_gflops / peak_gbytes

    ais = np.logspace(-2, 2, 500)
    roof = np.minimum(peak_gbytes * ais, peak_gflops)
    ax.loglog(ais, roof, "k-", linewidth=2.5, label="Roofline ceiling", zorder=3)
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.7,
               label=f"Ridge ({ridge:.2f} FLOP/B)", zorder=2)

    # Measured points
    for pt in measured:
        d   = pt["d"]
        ai  = arithmetic_intensity(d)
        lvl = cache_level(d, machine)
        gf  = pt["gflops"]
        ax.scatter([ai], [gf], color=COLORS[lvl], s=120, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"d={d}\n({lvl})",
            (ai, gf),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=COLORS[lvl],
        )
        # Efficiency line from measured to ceiling
        ceiling = min(peak_gbytes * ai, peak_gflops)
        ax.plot([ai, ai], [gf, ceiling], color=COLORS[lvl],
                linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(
        f"Roofline: Lindblad Propagator (lb_propagate_step)\n"
        f"{machine['name']}",
        fontsize=11,
    )

    legend_patches = [
        mpatches.Patch(color=COLORS[lvl], label=f"{lvl} regime")
        for lvl in ["L1", "L2", "L3"]
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][:2],
              fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.005, 5.0)
    ax.set_ylim(0.5, peak_gflops * 1.5)

    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")
    plt.close(fig)


def main() -> None:
    import matplotlib.patches as mpatches  # noqa: F401 (needed in plot())
    globals()["mpatches"] = mpatches

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stdin", action="store_true",
                        help="Read bench_propagate stdout from stdin")
    parser.add_argument("--csv", help="Path to CSV with measured results")
    parser.add_argument("--machine", default="i9-13980HX",
                        choices=list(MACHINES.keys()))
    parser.add_argument("--outdir", default="paper/figures")
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read()
        measured = parse_bench_output(text)
    elif args.csv:
        measured = parse_csv(args.csv)
    else:
        print("No measured data provided. Generating skeleton plot.", file=sys.stderr)
        # Dummy data for skeleton
        measured = [
            {"d": 3,  "gflops": 0.0, "gbytes_s": 0.0},
            {"d": 9,  "gflops": 0.0, "gbytes_s": 0.0},
            {"d": 27, "gflops": 0.0, "gbytes_s": 0.0},
        ]

    machine = MACHINES[args.machine]
    plot(measured, machine, f"{args.outdir}/roofline.pdf")


if __name__ == "__main__":
    main()
