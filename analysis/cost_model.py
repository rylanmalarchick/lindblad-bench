#!/usr/bin/env python3
"""
cost_model.py — Idealized cost per state-step for each platform, next to the
measured value, with the mechanism that explains the gap.

Inputs (all under benchmarks/):
  probes_stream_<host>.csv        read bandwidth per working set, fork/join cost
  probes_cuda_<gpu>.csv           launch latency, PCIe and device bandwidths
  cpu_batch_results_<host>.csv    measured CPU medians
  cpu_spot_check_intel.csv        d=4, d=8 spot check (i9)
  cuda_batch_results_<gpu>.csv    measured GPU medians (kernel, resident, host)
  fpga/benchmark_compare_1000_108mhz_20260830.log   measured FPGA cycles

Model, with n = d^2 and one state-step = one n x n complex matvec:
  bytes  B(n) = 16 (n^2 + 2 n)              compulsory traffic
  flops  F(n) = 8 n^2

CPU, one thread:
  T_bw   = 16 n^2 / BW_1(W)    W = 16 n^2 working set, BW_1 from the probe at
                               the nearest size at 1 thread
  T_chain = n^2 / 2 * L_fma / f  one FMA-class op per 2 complex elements on a
                               single accumulator chain, latency L_fma cycles
  T_1    = max(T_bw, T_chain)  no loop-overhead term: the gap at small n is
                               reported, not fitted
CPU, T threads, batch b:
  T_b    = T_1 * ceil(b / T) + t_fork / b   (fork/join amortized over the batch)
  BW_1 is replaced by the shared-buffer aggregate per thread at T threads.

Transfers are modeled as latency plus size over asymptotic bandwidth, both
fitted from the probe: t_lat from the 2 KB point, BW_inf from the 64 MB point.
GPU kernel-only, batch b (idealized: coalesced reads at the device's best
measured streaming rate; the stride-n gather and low thread counts at small
b n are the gap, both quantified by the probe):
  T_k    = (t_launch + 16 n^2 b / BW_stream_max) / b
GPU resident-P:
  T_r    = T_k + (t_h2d(16 n b) + t_d2h(16 n b) + t_sync) / b
GPU host-visible (P re-uploaded every call, three cudaMalloc/cudaFree):
  T_h    = T_r + (t_h2d(16 n^2) + t_alloc) / b     t_alloc = 3 x (t_malloc + t_free),
                                                   not probed, stated in CONSTANTS
FPGA:
  cycles = n^2 + n + 4 per step at one MAC, 94 n_steps + 1 total.

Constants that are not measured by a probe are stated in CONSTANTS and
printed in the table caption source.
"""

from __future__ import annotations

import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
GEN = ROOT / "paper" / "generated"

CONSTANTS = {
    # FMA latency in cycles and sustained clock for the single-thread kernel.
    # L_fma: FMA latency in cycles. f_ghz: sustained single-core clock during
    # the runs (nominal turbo). T: the thread count used for the batched rows.
    # T: thread counts for the batched rows (the probe measured these counts).
    "i9-13980HX": {"L_fma": 4, "f_ghz": 5.0, "T": (8, 24), "stream": "theLittleMachine",
                   "cpu": "cpu_batch_results_intel.csv"},
    "Ryzen 5 1600": {"L_fma": 5, "f_ghz": 3.4, "T": (6, 12), "stream": "theMachine",
                     "cpu": "cpu_batch_results_ryzen.csv"},
}
GPUS = {
    "RTX 4070 Laptop GPU": {"probe": "probes_cuda_rtx4070.csv", "cuda": "cuda_batch_results_rtx4070.csv"},
    "GTX 1070 Ti": {"probe": "probes_cuda_gtx1070ti.csv", "cuda": "cuda_batch_results_gtx1070ti.csv"},
}
SIZES_KB = [1.5, 105.0, 8300.0, 65536.0]
T_ALLOC_NS = 3 * 2 * 3000.0   # three cudaMalloc + three cudaFree at ~3 us each
FPGA_LOG = BENCH / "fpga" / "benchmark_compare_1000_108mhz_20260830.log"
FPGA_MHZ = 108.0


def working_set_kb(n: int) -> float:
    return 16.0 * n * n / 1024.0


def nearest_size(kb: float) -> float:
    """Smallest probe size that holds the working set, with 1% slack so that
    8303.8 KB maps to the 8300 KB probe point."""
    return min(SIZES_KB, key=lambda s: abs(s - kb) if s >= 0.99 * kb else 1e12)


def load_stream(host_tag: str) -> tuple[dict, dict, dict]:
    df = pd.read_csv(BENCH / f"probes_stream_{host_tag}.csv")
    private = defaultdict(list)
    shared = defaultdict(list)
    fork = defaultdict(list)
    for _, r in df.iterrows():
        key = (float(r["size_kb"]), int(r["threads"]))
        if r["layout"] == "forkjoin":
            fork[int(r["threads"])].append(float(r["gbs_per_thread"]))
        elif r["layout"] == "shared":
            shared[key].append(float(r["gbs_per_thread"]))
        else:
            private[key].append(float(r["gbs_per_thread"]))
    med = lambda d: {k: st.median(v) for k, v in d.items()}
    return med(private), med(shared), med(fork)


def load_gpu_probe(name: str) -> dict:
    df = pd.read_csv(BENCH / GPUS[name]["probe"])
    out = defaultdict(list)
    for _, r in df.iterrows():
        out[r["quantity"]].append(float(r["value"]))
    return {k: st.median(v) for k, v in out.items()}


def cpu_rows() -> list[dict]:
    rows = []
    spot = pd.read_csv(BENCH / "cpu_spot_check_intel.csv")
    for host, c in CONSTANTS.items():
        private, shared, fork = load_stream(c["stream"])
        meas = pd.read_csv(BENCH / c["cpu"])
        meas = meas.rename(columns={"median_ns_per_state_step": "ns"})
        T_lo, T_hi = c["T"]
        points = [(d, 1, 1) for d in (3, 9, 27)]
        points += [(d, T, 128) for T in (T_lo, T_hi) for d in (3, 9, 27)]
        if host == "i9-13980HX":
            points += [(4, 1, 128), (8, 1, 128), (4, T_lo, 128), (8, T_lo, 128)]
        for d, T, b in points:
            n = d * d
            W = working_set_kb(n)
            size = nearest_size(W)
            bw = (shared if T > 1 else private).get((size, T)) or private.get((size, 1))
            f_hz = c["f_ghz"] * 1e9
            t_bw = 16.0 * n * n / (bw * 1e9) * 1e9
            t_chain = (n * n / 2.0) * c["L_fma"] / f_hz * 1e9
            t1 = max(t_bw, t_chain)
            t_fork = fork.get(T, 0.0) if T > 1 else 0.0
            pred = t1 * (-(-b // T)) / b + t_fork / b
            if d in (4, 8):
                m = spot[(spot["d"] == d) & (spot["threads"] == T) & (spot["batch_size"] == b)]
                measured = float(m["ns_per_state_step"].iloc[0]) if len(m) else float("nan")
            else:
                m = meas[(meas["d"] == d) & (meas["threads"] == T) & (meas["batch_size"] == b)]
                measured = float(m["ns"].iloc[0]) if len(m) else float("nan")
            bound = "FMA chain" if t_chain >= t_bw else "bandwidth"
            if d <= 4:
                cause = "loop overhead per row"
            elif T > 1:
                cause = f"{bound} + fork/join"
            else:
                cause = bound
            rows.append(dict(platform=host, d=d, threads=T, batch=b, predicted_ns=pred,
                             measured_ns=measured, ratio=measured / pred if pred else float("nan"),
                             cause=cause, detail=f"BW={bw:.0f} GB/s at {size:g} KB, t_bw={t_bw:.0f}, t_chain={t_chain:.0f}, t_fork={t_fork:.0f}"))
    return rows


def gpu_rows() -> list[dict]:
    rows = []
    for name, g in GPUS.items():
        p = load_gpu_probe(name)
        meas = pd.read_csv(BENCH / g["cuda"])
        # allocation cost: measured host-visible at d=3, b=1 minus everything the model knows
        for d in (3, 9, 27):
            n = d * d
            W = working_set_kb(n)
            size = nearest_size(W)
            tag = f"{size:.0f}kb"
            bw_gather = p[f"gather_gbs_{tag}"] * 1e9
            bw_stream = max(v for k, v in p.items() if k.startswith("stream_gbs_")) * 1e9
            t_launch = p["launch_ns"]
            # launch_sync_ns brackets launch + cudaDeviceSynchronize; the launch
            # part is already in T_kernel, so only the synchronize remainder is added.
            t_sync = p["launch_sync_ns"] - p["launch_ns"]
            # The probe's "2kb" point moves floor(1.5*1024/16)*16 = 1536 bytes.
            small = float(int(1.5 * 1024 / 16) * 16)
            def xfer(kind: str, nbytes: float) -> float:
                t_lat = small / (p[f"{kind}_gbs_2kb"] * 1e9) * 1e9
                bw_inf = p[f"{kind}_gbs_65536kb"] * 1e9
                return t_lat + nbytes / bw_inf * 1e9
            for b in (1, 8, 32, 128):
                m = meas[(meas["d"] == d) & (meas["batch_size"] == b)]
                if not len(m):
                    continue
                t_k = (t_launch + 16.0 * n * n * b / bw_stream * 1e9) / b
                t_r = t_k + (xfer("h2d", 16.0 * n * b) + xfer("d2h", 16.0 * n * b) + t_sync) / b
                t_h = t_r + (xfer("h2d", 16.0 * n * n) + T_ALLOC_NS) / b
                for regime, pred, col, cause in (
                    ("kernel", t_k, "median_kernel_ns_per_state_step",
                     "too few threads to fill the device" if b * n < 4096 else "stride-$n$ gather of $P$"),
                    ("resident", t_r, "median_resident_ns_per_state_step", "PCIe round trip + sync"),
                    ("host", t_h, "median_host_ns_per_state_step", "P re-upload + cudaMalloc/cudaFree"),
                ):
                    measured = float(m[col].iloc[0])
                    rows.append(dict(platform=f"{name} ({regime})", d=d, threads=0, batch=b,
                                     predicted_ns=pred, measured_ns=measured, ratio=measured / pred,
                                     cause=cause, detail=f"stream_max={bw_stream/1e9:.0f} GB/s, gather@{tag}={bw_gather/1e9:.0f} GB/s, launch={t_launch:.0f} ns, sync={t_sync:.0f} ns"))
    return rows


def fpga_rows() -> list[dict]:
    text = FPGA_LOG.read_text()
    counts = {}
    for m in re.finditer(r"(\d+) steps:\s+(\d+) cycles", text):
        counts[int(m.group(1))] = int(m.group(2))
    rows = []
    n = 9
    for steps, cyc in sorted(counts.items()):
        pred_cycles = (n * n + n + 4) * steps + 1
        pred_ns = pred_cycles / FPGA_MHZ * 1e3 / steps
        meas_ns = cyc / FPGA_MHZ * 1e3 / steps
        rows.append(dict(platform="Tang Nano 20K (on-chip)", d=3, threads=1, batch=steps,
                         predicted_ns=pred_ns, measured_ns=meas_ns, ratio=meas_ns / pred_ns,
                         cause="exact: counter model", detail=f"{cyc} cycles for {steps} steps"))
    return rows


def write_tables(rows: list[dict]) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    short = {"i9-13980HX": "i9", "Ryzen 5 1600": "Ryzen", "Tang Nano 20K (on-chip)": "Tang Nano"}

    cpu = [r for r in rows if r["platform"] in short and "Tang" not in r["platform"]]
    fpga = [r for r in rows if "Tang" in r["platform"] and r["batch"] in (1, 1000)]
    lines = []
    for r in cpu + fpga:
        lines.append(
            f"{short[r['platform']]} & {r['d']} & {r['threads']} & {r['batch']} & "
            f"{r['predicted_ns']:.0f} & {r['measured_ns']:.0f} & {r['ratio']:.2f} & {r['cause']} \\\\"
        )
    (GEN / "table_cost_model_cpu.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrl@{}}\n\\toprule\n"
        "platform & $d$ & threads & batch & idealized (ns) & measured (ns) & ratio & gap mechanism \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    gpu_lines = []
    for name in GPUS:
        for d in (3, 9, 27):
            for b in (1, 8, 32, 128):
                sel = {r["platform"].split(" (")[1][:-1]: r for r in rows if r["platform"].startswith(name) and r["d"] == d and r["batch"] == b}
                if len(sel) < 3:
                    continue
                k, rr, h = sel["kernel"], sel["resident"], sel["host"]
                gname = "RTX 4070" if "4070" in name else "GTX 1070 Ti"
                gpu_lines.append(
                    f"{gname} & {d} & {b} & {k['predicted_ns']:.0f} & {k['measured_ns']:.0f} & {k['ratio']:.1f} & "
                    f"{rr['predicted_ns']:.0f} & {rr['measured_ns']:.0f} & {rr['ratio']:.1f} & "
                    f"{h['predicted_ns']:.0f} & {h['measured_ns']:.0f} & {h['ratio']:.1f} \\\\"
                )
    (GEN / "table_cost_model_gpu.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrrrrrr@{}}\n\\toprule\n"
        " & & & \\multicolumn{3}{c}{kernel-only} & \\multicolumn{3}{c}{resident $P$} & \\multicolumn{3}{c}{host-visible} \\\\\n"
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-12}\n"
        "GPU & $d$ & batch & idealized & meas. & ratio & idealized & meas. & ratio & idealized & meas. & ratio \\\\\n"
        "\\midrule\n" + "\n".join(gpu_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    pd.DataFrame(rows).to_csv(BENCH / "cost_model_table.csv", index=False)


def write_probe_table() -> None:
    """Measured ceilings that feed the model, one row per host or GPU."""
    lines = []
    for host, c in CONSTANTS.items():
        private, shared, fork = load_stream(c["stream"])
        T = c["T"][0]
        vals = [private[(sz, 1)] for sz in SIZES_KB]
        lines.append(
            f"{host} & " + " & ".join(f"{v:.0f}" for v in vals) +
            f" & {shared[(8300.0, T)] * T:.0f} ({T}) & {fork[T] / 1e3:.1f} ({T}) \\\\"
        )
    (GEN / "table_probes_cpu.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
        "host & 1.5 KB & 105 KB & 8.3 MB & 64 MB & shared 8.3 MB, $T$ threads & fork/join ($\\mu$s) \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    glines = []
    for name in GPUS:
        p = load_gpu_probe(name)
        gname = "RTX 4070" if "4070" in name else "GTX 1070 Ti"
        glines.append(
            f"{gname} & {p['launch_ns'] / 1e3:.1f} & {p['launch_sync_ns'] / 1e3:.1f} & "
            f"{1536 / (p['h2d_gbs_2kb'] * 1e9) * 1e6:.1f} & {p['h2d_gbs_65536kb']:.1f} & {p['d2h_gbs_65536kb']:.1f} & "
            f"{p['stream_gbs_8300kb']:.0f} & {p['gather_gbs_8300kb']:.0f} & {p['stream_gbs_65536kb']:.0f} & {p['gather_gbs_65536kb']:.0f} \\\\"
        )
    (GEN / "table_probes_gpu.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n"
        "GPU & launch ($\\mu$s) & launch+sync ($\\mu$s) & copy latency ($\\mu$s) & H2D (GB/s) & D2H (GB/s) & "
        "stream 8.3 MB & gather 8.3 MB & stream 64 MB & gather 64 MB \\\\\n"
        "\\midrule\n" + "\n".join(glines) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main() -> None:
    rows = cpu_rows() + gpu_rows() + fpga_rows()
    for r in rows:
        print(f"{r['platform']:34s} d={r['d']:2d} T={r['threads']:2d} b={r['batch']:4d} "
              f"pred={r['predicted_ns']:10.0f} meas={r['measured_ns']:10.0f} ratio={r['ratio']:6.2f}  {r['cause']}  [{r['detail']}]")
    write_tables(rows)
    write_probe_table()
    print("wrote cost-model and probe tables to", GEN)


if __name__ == "__main__":
    main()
