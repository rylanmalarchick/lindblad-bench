"""
plot_speedup.py — Speedup vs d plot: C scalar / C auto-vec / C AVX2 vs QuTiP.

Reads from benchmarks/compiler_results.csv (from compiler_flags.sh)
and benchmarks/qutip_results.csv (from reference/reference.py --bench).

Usage:
    python analysis/plot_speedup.py \
        --c-csv benchmarks/compiler_results.csv \
        --qutip-csv benchmarks/qutip_results.csv \
        --outdir paper/figures
"""

import argparse
import sys
import os

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_DEPS = True
except ImportError:
    print("numpy + matplotlib + pandas required.", file=sys.stderr)
    sys.exit(1)


def plot_speedup(c_csv: str, qutip_csv: str, outdir: str) -> None:
    c_df     = pd.read_csv(c_csv)
    qutip_df = pd.read_csv(qutip_csv)  # columns: d, ns_per_step

    ds = sorted(c_df["d"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = {
        "scalar":         ("C scalar (-O2, no vec)",  "#607D8B", "o"),
        "O3":             ("C auto-vec (-O3)",         "#2196F3", "s"),
        "O3_native":      ("C auto-vec + native",      "#4CAF50", "^"),
        "O3_native_fast": ("C fast-math + native",     "#FF9800", "D"),
    }

    for flag, (label, color, marker) in labels.items():
        flag_df = c_df[c_df["flags"] == flag]
        if flag_df.empty:
            continue

        speedups = []
        for d in ds:
            c_row = flag_df[flag_df["d"] == d]
            q_row = qutip_df[qutip_df["d"] == d]
            if c_row.empty or q_row.empty:
                speedups.append(np.nan)
            else:
                speedups.append(
                    q_row["ns_per_step"].values[0] / c_row["ns_per_step"].values[0]
                )
        ax.semilogy(ds, speedups, marker=marker, color=color,
                    label=label, linewidth=1.8, markersize=7)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="QuTiP baseline")
    ax.set_xlabel("System dimension d", fontsize=12)
    ax.set_ylabel("Speedup over QuTiP", fontsize=12)
    ax.set_title("Lindblad Propagator Speedup vs QuTiP\n(lb_propagate_step, single step)", fontsize=11)
    ax.set_xticks(ds)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "speedup_vs_d.pdf")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c-csv",     default="benchmarks/compiler_results.csv")
    parser.add_argument("--qutip-csv", default="benchmarks/qutip_results.csv")
    parser.add_argument("--outdir",    default="paper/figures")
    args = parser.parse_args()

    if not os.path.exists(args.c_csv):
        print(f"C results not found: {args.c_csv}\nRun: bash analysis/compiler_flags.sh",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.qutip_csv):
        print(f"QuTiP results not found: {args.qutip_csv}\nRun: python reference/reference.py --bench",
              file=sys.stderr)
        sys.exit(1)

    plot_speedup(args.c_csv, args.qutip_csv, args.outdir)


if __name__ == "__main__":
    main()
