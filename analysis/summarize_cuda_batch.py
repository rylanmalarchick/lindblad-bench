#!/usr/bin/env python3
"""Summarize raw CUDA batch benchmark results by median."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict


NUMERIC_FIELDS = [
    "host_ns_per_state_step",
    "host_gflops",
    "host_gbytes_s",
    "kernel_ns_per_state_step",
    "kernel_gflops",
    "kernel_gbytes_s",
]

GROUP_FIELDS = [
    "machine",
    "gpu_name",
    "git_commit",
    "d",
    "batch_size",
    "n_reps",
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_cuda_batch.py <raw_csv> <summary_csv>", file=sys.stderr)
        return 1

    raw_csv, summary_csv = sys.argv[1], sys.argv[2]
    groups: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {field: [] for field in NUMERIC_FIELDS}
    )

    with open(raw_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = tuple(row[field] for field in GROUP_FIELDS)
            for field in NUMERIC_FIELDS:
                groups[key][field].append(float(row[field]))

    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(GROUP_FIELDS + [f"median_{field}" for field in NUMERIC_FIELDS])

        def sort_key(item: tuple[tuple[str, ...], dict[str, list[float]]]) -> tuple:
            key, _ = item
            return (
                key[0],
                key[1],
                key[2],
                int(key[3]),
                int(key[4]),
                int(key[5]),
            )

        for key, values in sorted(groups.items(), key=sort_key):
            writer.writerow(
                list(key)
                + [statistics.median(values[field]) for field in NUMERIC_FIELDS]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
