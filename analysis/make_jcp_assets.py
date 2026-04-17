#!/usr/bin/env python3
"""
Generate manuscript-grade figures and TeX table snippets for the JCP draft.

Inputs:
  - CPU batch CSV summaries
  - CUDA batch CSV summaries
  - optimized GRAPE benchmark logs
  - QuTiP GRAPE CSV summaries
  - FPGA hardware timing log

Outputs:
  - paper/figures_jcp/*.pdf
  - paper/generated/*.tex
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures_jcp"
GEN_DIR = PAPER_DIR / "generated"
BENCH_DIR = ROOT / "benchmarks"
FPGA_ROOT = Path("/home/rylan/dev/projects/tang-nano-20k")
FPGA_LOG = FPGA_ROOT / "analysis" / "results" / "benchmark_compare_500.log"

CPU_FILES = {
    "i9-13980HX": BENCH_DIR / "cpu_batch_results_intel.csv",
    "Ryzen 5 1600": BENCH_DIR / "cpu_batch_results_ryzen.csv",
}

GPU_FILES = {
    "RTX 4070 Laptop GPU": BENCH_DIR / "cuda_batch_results_rtx4070.csv",
    "GTX 1070 Ti": BENCH_DIR / "cuda_batch_results_gtx1070ti.csv",
}

GRAPE_LOGS = {
    "i9-13980HX": BENCH_DIR / "grape_c_intel_opt.log",
    "Ryzen 5 1600": BENCH_DIR / "grape_c_ryzen_opt.log",
}

QUTIP_GRAPE_FILES = {
    "i9-13980HX": BENCH_DIR / "qutip_grape_results_intel.csv",
    "Ryzen 5 1600": BENCH_DIR / "qutip_grape_results_ryzen.csv",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_cpu() -> pd.DataFrame:
    frames = []
    for host, path in CPU_FILES.items():
        df = pd.read_csv(path)
        df["host_label"] = host
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_gpu() -> pd.DataFrame:
    frames = []
    for gpu_label, path in GPU_FILES.items():
        df = pd.read_csv(path)
        df["gpu_label"] = gpu_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_qutip_grape() -> pd.DataFrame:
    frames = []
    for host, path in QUTIP_GRAPE_FILES.items():
        df = pd.read_csv(path)
        df["host_label"] = host
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse_grape_log(path: Path, host_label: str) -> pd.DataFrame:
    current: dict[str, float | int | str] | None = None
    rows: list[dict[str, float | int | str]] = []

    d_re = re.compile(r"=== d=(\d+),")
    setup_re = re.compile(r"\[setup\]\s+invariant superops : ([0-9.]+) ms")
    build_re = re.compile(r"\[build\]\s+all propagators : ([0-9.]+) ms")
    chain_re = re.compile(r"\[chain\]\s+ms/trajectory\s+: ([0-9.]+) ms")
    total_re = re.compile(r"\[total\]\s+landscape point : ([0-9.]+) ms")

    for line in path.read_text().splitlines():
        if match := d_re.search(line):
            if current is not None:
                rows.append(current)
            current = {"host_label": host_label, "d": int(match.group(1))}
            continue
        if current is None:
            continue
        if match := setup_re.search(line):
            current["setup_ms"] = float(match.group(1))
        elif match := build_re.search(line):
            current["build_ms"] = float(match.group(1))
        elif match := chain_re.search(line):
            current["chain_ms"] = float(match.group(1))
        elif match := total_re.search(line):
            current["total_ms"] = float(match.group(1))

    if current is not None:
        rows.append(current)

    return pd.DataFrame(rows)


def load_grape_c() -> pd.DataFrame:
    frames = []
    for host, path in GRAPE_LOGS.items():
        frames.append(parse_grape_log(path, host))
    return pd.concat(frames, ignore_index=True)


def parse_fpga_log(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    values: dict[str, float | int] = {}

    patterns = {
        "cycles_per_step": r"Cycles/step:\s+(\d+)",
        "time_per_step_ns": r"Time/step:\s+([0-9.]+) ns",
        "distinct_counts": r"Unique counts:\s+\[[0-9,\s]+\] \((\d+) distinct values\)",
        "cpu_i9_single_ns": r"CPU \(i9-13980HX\).*?Single-state time/step:\s+([0-9.]+) ns",
        "cpu_ryzen_single_ns": r"CPU \(Ryzen 5 1600\).*?Single-state time/step:\s+([0-9.]+) ns",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise RuntimeError(f"failed to parse {key} from {path}")
        raw = match.group(1)
        values[key] = int(raw) if raw.isdigit() else float(raw)

    values["steady_state_cycles"] = 94
    values["steady_state_ns"] = values["steady_state_cycles"] / 135.0 * 1e3
    values["programmed_clock_mhz"] = 135
    values["lut4_used"] = 1357
    values["lut4_total"] = 20736
    values["dsp_used"] = 4
    values["dsp_total"] = 48
    values["trial_spread_cycles"] = 0
    values["trials"] = 500
    return values


def speedup_string(c_ms: float, qutip_ms: float) -> str:
    if c_ms < qutip_ms:
        ratio = qutip_ms / c_ms
        label = "C"
    else:
        ratio = c_ms / qutip_ms
        label = "QuTiP"
    if ratio >= 10:
        ratio_str = f"{ratio:.0f}\\times"
    else:
        ratio_str = f"{ratio:.1f}\\times"
    return f"{label} (${ratio_str}$)"


def write_table(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n")


def write_tables(
    cpu: pd.DataFrame,
    gpu: pd.DataFrame,
    grape_c: pd.DataFrame,
    qutip_grape: pd.DataFrame,
    fpga: dict[str, float | int],
) -> None:
    host_order = {"i9-13980HX": 0, "Ryzen 5 1600": 1}
    host_short = {"i9-13980HX": "i9", "Ryzen 5 1600": "Ryzen"}
    cpu_rows = []
    for host in CPU_FILES:
        host_df = cpu[cpu["host_label"] == host]
        d3_latency = host_df[(host_df["d"] == 3) & (host_df["batch_size"] == 1)]["median_ns_per_state_step"].min()
        d9_peak = host_df[host_df["d"] == 9]["median_gflops"].max()
        d27_peak = host_df[host_df["d"] == 27]["median_gflops"].max()
        cpu_rows.append(
            f"{host} & {d3_latency:.1f} & {d9_peak:.1f} & {d27_peak:.1f} \\\\"
        )

    write_table(
        GEN_DIR / "table_cpu_summary.tex",
        rf"""
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
host & $d=3$ latency (ns/step) & $d=9$ peak (GFLOP/s) & $d=27$ peak (GFLOP/s) \\
\midrule
{chr(10).join(cpu_rows)}
\bottomrule
\end{{tabular}}
""",
    )

    roofline_rows = []
    roofline_specs = [
        ("i9-13980HX", cpu[cpu["host_label"] == "i9-13980HX"], "median_gflops", "median_gbytes_s"),
        ("Ryzen 5 1600", cpu[cpu["host_label"] == "Ryzen 5 1600"], "median_gflops", "median_gbytes_s"),
        ("RTX 4070 (kernel-only)", gpu[gpu["gpu_label"] == "RTX 4070 Laptop GPU"], "median_kernel_gflops", "median_kernel_gbytes_s"),
        ("GTX 1070 Ti (kernel-only)", gpu[gpu["gpu_label"] == "GTX 1070 Ti"], "median_kernel_gflops", "median_kernel_gbytes_s"),
    ]
    for label, df, perf_col, bw_col in roofline_specs:
        peak_compute = df[perf_col].max()
        peak_bandwidth = df[bw_col].max()
        d3_point = df[df["d"] == 3][perf_col].max()
        d9_point = df[df["d"] == 9][perf_col].max()
        d27_point = df[df["d"] == 27][perf_col].max()
        roofline_rows.append(
            f"{label} & {peak_bandwidth:.1f} & {peak_compute:.1f} & {d3_point:.1f} & {d9_point:.1f} & {d27_point:.1f} \\\\"
        )

    write_table(
        GEN_DIR / "table_roofline_summary.tex",
        rf"""
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
platform & peak bandwidth (GB/s) & peak compute (GFLOP/s) & $d=3$ point & $d=9$ point & $d=27$ point \\
\midrule
{chr(10).join(roofline_rows)}
\bottomrule
\end{{tabular}}
""",
    )

    merged = grape_c.merge(
        qutip_grape[["host_label", "d", "ms_per_trajectory"]],
        on=["host_label", "d"],
        how="inner",
    )
    merged["host_order"] = merged["host_label"].map(host_order)
    merged = merged.sort_values(["host_order", "d"])
    grape_rows = []
    for _, row in merged.iterrows():
        grape_rows.append(
            f"{host_short[row['host_label']]} & $d={int(row['d'])}$ & {row['total_ms']:.3f} & "
            f"{row['ms_per_trajectory']:.3f} & {speedup_string(row['total_ms'], row['ms_per_trajectory'])} \\\\"
        )

    write_table(
        GEN_DIR / "table_grape_summary.tex",
        rf"""
\begin{{tabular}}{{@{{}}llrrl@{{}}}}
\toprule
host & $d$ & C (ms) & QuTiP (ms) & winner \\
\midrule
{chr(10).join(grape_rows)}
\bottomrule
\end{{tabular}}
""",
    )

    write_table(
        GEN_DIR / "table_fpga_summary.tex",
        rf"""
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
quantity & value \\
\midrule
programmed clock & {fpga['programmed_clock_mhz']} MHz \\
single-step count & {fpga['cycles_per_step']} cycles \\
single-step latency & {fpga['time_per_step_ns']:.1f} ns \\
long-run count & $94n + 1$ cycles \\
trial-to-trial spread & {fpga['trial_spread_cycles']} cycles over {fpga['trials']} trials \\
LUT4 usage & {fpga['lut4_used']} / {fpga['lut4_total']} \\
DSP usage & {fpga['dsp_used']} / {fpga['dsp_total']} \\
\bottomrule
\end{{tabular}}
""",
    )


def arithmetic_intensity(d: int) -> float:
    d2 = d * d
    return (8.0 * d2 * d2) / (16.0 * (d2 * d2 + 2.0 * d2))


def plot_empirical_roofline(cpu: pd.DataFrame, gpu: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    point_colors = {3: "#1b9e77", 9: "#d95f02", 27: "#7570b3"}
    x = np.logspace(-2, 2, 256)

    platform_specs = [
        ("i9-13980HX", cpu[cpu["host_label"] == "i9-13980HX"], "median_gflops", "median_gbytes_s"),
        ("Ryzen 5 1600", cpu[cpu["host_label"] == "Ryzen 5 1600"], "median_gflops", "median_gbytes_s"),
        ("RTX 4070 (kernel-only)", gpu[gpu["gpu_label"] == "RTX 4070 Laptop GPU"], "median_kernel_gflops", "median_kernel_gbytes_s"),
        ("GTX 1070 Ti (kernel-only)", gpu[gpu["gpu_label"] == "GTX 1070 Ti"], "median_kernel_gflops", "median_kernel_gbytes_s"),
    ]

    for ax, (title, df, perf_col, bw_col) in zip(axes.flat, platform_specs):
        peak_compute = df[perf_col].max()
        peak_bandwidth = df[bw_col].max()
        y = np.minimum(peak_compute, peak_bandwidth * x)

        ax.plot(x, y, color="#444444", linewidth=1.4)
        ax.axhline(peak_compute, color="#444444", linewidth=0.9, linestyle=":")
        ridge_x = peak_compute / peak_bandwidth
        ax.axvline(ridge_x, color="#777777", linewidth=0.8, linestyle="--", alpha=0.8)

        for d in [3, 9, 27]:
            ai = arithmetic_intensity(d)
            perf = df[df["d"] == d][perf_col].max()
            ax.scatter(ai, perf, s=28, color=point_colors[d], zorder=3)
            ax.annotate(
                fr"$d={d}$",
                xy=(ai, perf),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=point_colors[d],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e2)
        ax.set_ylim(1e-1, max(peak_compute * 1.25, 1.0))
        ax.set_title(title)
        ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")

    fig.suptitle("Empirical Rooflines from Peak Observed Throughput/Bandwidth", y=1.02, fontsize=11)
    fig.savefig(FIG_DIR / "figure_roofline_empirical.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cpu_thread_scaling(cpu: pd.DataFrame) -> None:
    colors = {3: "#1b9e77", 9: "#d95f02", 27: "#7570b3"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))

    for ax, host in zip(axes, CPU_FILES):
        host_df = cpu[(cpu["host_label"] == host) & (cpu["batch_size"] == 128)]
        for d in [3, 9, 27]:
            df = host_df[host_df["d"] == d].sort_values("threads")
            ax.plot(
                df["threads"],
                df["median_gflops"],
                marker="o",
                linewidth=1.6,
                markersize=4.0,
                color=colors[d],
                label=fr"$d={d}$",
            )
        ax.set_title(host)
        ax.set_xlabel("Threads")
        ax.set_ylabel("GFLOP/s")
        ax.set_xticks(sorted(host_df["threads"].unique()))

    legend_handles = [
        Line2D([0], [0], color=colors[d], marker="o", linewidth=1.6, markersize=4.0, label=fr"$d={d}$")
        for d in [3, 9, 27]
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("CPU Thread Scaling at Batch Size 128", y=1.06, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIG_DIR / "figure_cpu_thread_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gpu_cpu_ratio(cpu: pd.DataFrame, gpu: pd.DataFrame) -> None:
    best_cpu = (
        cpu.groupby(["d", "batch_size"], as_index=False)["median_ns_per_state_step"]
        .min()
        .rename(columns={"median_ns_per_state_step": "best_cpu_ns"})
    )
    merged = gpu.merge(best_cpu, on=["d", "batch_size"], how="left")
    merged["host_ratio"] = merged["median_host_ns_per_state_step"] / merged["best_cpu_ns"]
    merged["kernel_ratio"] = merged["median_kernel_ns_per_state_step"] / merged["best_cpu_ns"]

    colors = {3: "#1b9e77", 9: "#d95f02", 27: "#7570b3"}
    linestyles = {"RTX 4070 Laptop GPU": "-", "GTX 1070 Ti": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))
    for ax, key, title in [
        (axes[0], "host_ratio", "Host-visible GPU time / best CPU time"),
        (axes[1], "kernel_ratio", "Kernel-only GPU time / best CPU time"),
    ]:
        for gpu_label in GPU_FILES:
            for d in [3, 9, 27]:
                df = merged[(merged["gpu_label"] == gpu_label) & (merged["d"] == d)].sort_values("batch_size")
                ax.plot(
                    df["batch_size"],
                    df[key],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.8,
                    color=colors[d],
                    linestyle=linestyles[gpu_label],
                    label=f"{gpu_label}, d={d}",
                )
        ax.axhline(1.0, color="black", linewidth=0.9, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 8, 32, 128], labels=["1", "8", "32", "128"])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Time ratio")
        ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color=colors[d], marker="o", linewidth=1.4, markersize=3.8, label=fr"$d={d}$")
        for d in [3, 9, 27]
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.4, label="RTX 4070"),
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.4, label="GTX 1070 Ti"),
        ]
    )
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG_DIR / "figure_gpu_cpu_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_grape_breakdown(grape_c: pd.DataFrame, qutip_grape: pd.DataFrame) -> None:
    merged = grape_c.merge(
        qutip_grape[["host_label", "d", "ms_per_trajectory"]],
        on=["host_label", "d"],
        how="inner",
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.9))
    colors = {"build": "#c44e52", "chain": "#4c72b0", "qutip": "#55a868"}

    for ax, host in zip(axes, CPU_FILES):
        df = merged[merged["host_label"] == host].sort_values("d")
        x = np.arange(len(df))
        width = 0.34
        ax.bar(x - width / 2, df["build_ms"], width, color=colors["build"], label="C build")
        ax.bar(x - width / 2, df["chain_ms"], width, bottom=df["build_ms"], color=colors["chain"], label="C chain")
        ax.bar(x + width / 2, df["ms_per_trajectory"], width, color=colors["qutip"], alpha=0.9, label="QuTiP total")

        ax.set_xticks(x, labels=[fr"$d={int(v)}$" for v in df["d"]])
        ax.set_yscale("log")
        ax.set_ylabel("ms per landscape point")
        ax.set_title(host)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "figure_grape_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fpga_latency(cpu: pd.DataFrame, fpga: dict[str, float | int]) -> None:
    i9_latency = cpu[(cpu["host_label"] == "i9-13980HX") & (cpu["d"] == 3) & (cpu["batch_size"] == 1)]["median_ns_per_state_step"].min()
    ryzen_latency = cpu[(cpu["host_label"] == "Ryzen 5 1600") & (cpu["d"] == 3) & (cpu["batch_size"] == 1)]["median_ns_per_state_step"].min()

    labels = ["i9 CPU", "Ryzen CPU", "FPGA single", "FPGA steady"]
    values = [i9_latency, ryzen_latency, fpga["time_per_step_ns"], fpga["steady_state_ns"]]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylabel("ns per d=3 step")
    ax.set_title("Single-state d=3 Latency")
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.text(
        2.5,
        max(values) * 1.08,
        "FPGA spread: 0 cycles over 500 trials",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(FIG_DIR / "figure_fpga_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    setup_style()

    cpu = load_cpu()
    gpu = load_gpu()
    grape_c = load_grape_c()
    qutip_grape = load_qutip_grape()
    fpga = parse_fpga_log(FPGA_LOG)

    write_tables(cpu, gpu, grape_c, qutip_grape, fpga)
    plot_cpu_thread_scaling(cpu)
    plot_gpu_cpu_ratio(cpu, gpu)
    plot_grape_breakdown(grape_c, qutip_grape)
    plot_fpga_latency(cpu, fpga)

    print("Wrote figures to", FIG_DIR)
    print("Wrote tables to", GEN_DIR)


if __name__ == "__main__":
    main()
