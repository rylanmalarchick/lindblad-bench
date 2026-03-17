#!/usr/bin/env python3
"""Generate publication figures from compiler_results.csv."""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "benchmarks", "compiler_results.csv")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Machine specs (i9-13980HX)
PEAK_GFLOPS = 128.0   # AVX2 FP64, 2 FMA/cycle, ~3 GHz base
PEAK_BW_DRAM = 80.0   # GB/s DDR5

# Estimated cache bandwidths (single-core, sustained)
CACHE_BW = {"L1": 800.0, "L2": 300.0, "L3": 150.0, "DRAM": PEAK_BW_DRAM}

# Colors and markers
CACHE_COLORS = {"L1": "#2196F3", "L2": "#FF9800", "L3": "#F44336"}
VARIANT_MARKERS = {"AoS": "s", "SoA": "o", "AVX2": "^"}
D_CACHE = {3: "L1", 9: "L2", 27: "L3"}


def ai(d):
    d2 = d * d
    return 8.0 * d2 * d2 / ((d2 * d2 + 2 * d2) * 16.0)


def load_csv():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def fig_roofline(rows):
    """Bandwidth bar chart: variants x system sizes at O3_native_fast,
    with cache bandwidth ceilings as horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    dims = [3, 9, 27]
    variants = ["AoS", "SoA", "AVX2"]
    variant_colors = {"AoS": "#78909C", "SoA": "#2196F3", "AVX2": "#FF9800"}

    subset = [r for r in rows if r["flags"] == "O3_native_fast"]

    x = np.arange(len(dims))
    width = 0.22

    for i, v in enumerate(variants):
        bws = []
        for d in dims:
            match = [r for r in subset
                     if int(r["d"]) == d and r["variant"] == v]
            bws.append(float(match[0]["gbytes_s"]) if match else 0)
        ax.bar(x + (i - 1) * width, bws, width, label=v,
               color=variant_colors[v], edgecolor="black", linewidth=0.4)

    # Cache bandwidth ceilings as reference
    cache_labels = [
        ("L1", CACHE_BW["L1"], "dotted"),
        ("L2", CACHE_BW["L2"], "dashed"),
        ("L3", CACHE_BW["L3"], "dashdot"),
    ]
    for label, bw, ls in cache_labels:
        if bw < 200:  # only show if it fits in the plot range
            ax.axhline(bw, color="gray", linestyle=ls, alpha=0.4, linewidth=1)
            ax.text(len(dims) - 0.6, bw + 1, f"{label} ceiling",
                    fontsize=7, color="gray", alpha=0.7)

    ax.set_ylabel("Achieved Bandwidth (GB/s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d={d}$ ({D_CACHE[d]})"
                        for d in dims], fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=9)
    ax.set_ylim(0, 55)

    out = os.path.join(FIG_DIR, "roofline.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")
    plt.close(fig)


def ws_label(d):
    """Human-readable working set size."""
    d2 = d * d
    nbytes = (d2 * d2 + 2 * d2) * 16
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.0f} KB"
    else:
        return f"{nbytes / (1024 * 1024):.1f} MB"


def fig_flags(rows):
    """Grouped bar chart: SoA bandwidth across flag sets."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    flag_labels = [
        ("scalar", "$-$O2 scalar"),
        ("O3", "$-$O3"),
        ("O3_native", "$-$O3 native"),
        ("O3_native_fast", "$-$O3 native fast-math"),
    ]
    dims = [3, 9, 27]
    dim_colors = [CACHE_COLORS[D_CACHE[d]] for d in dims]

    x = np.arange(len(flag_labels))
    width = 0.25

    for i, d in enumerate(dims):
        bws = []
        for flag_key, _ in flag_labels:
            match = [r for r in rows
                     if r["flags"] == flag_key and r["variant"] == "SoA"
                     and int(r["d"]) == d]
            bws.append(float(match[0]["gbytes_s"]) if match else 0)
        ax.bar(x + (i - 1) * width, bws, width, label=f"$d={d}$",
               color=dim_colors[i], edgecolor="black", linewidth=0.4)

    ax.set_ylabel("Achieved Bandwidth (GB/s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in flag_labels], fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=9)
    ax.set_ylim(0, 55)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "flags_comparison.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_csv()
    fig_roofline(rows)
    fig_flags(rows)
