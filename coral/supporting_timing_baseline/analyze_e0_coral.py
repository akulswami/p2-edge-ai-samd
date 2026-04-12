from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev

LOG_PATH = Path("e0_log.txt")


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("No data")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")

    deltas: list[int] = []
    rows: list[tuple[str, int, int]] = []

    with LOG_PATH.open("r", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("E0 Bootcheck Start"):
                continue
            if line.startswith("format,"):
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue
            if parts[0] != "event":
                continue

            state = parts[1]
            try:
                t_us = int(parts[2])
                delta_us = int(parts[3])
            except ValueError:
                continue

            rows.append((state, t_us, delta_us))
            deltas.append(delta_us)

    if not deltas:
        raise RuntimeError("No valid event rows parsed")

    sorted_deltas = sorted(deltas)

    print("=== Coral E0 Baseline Statistics ===")
    print(f"Samples           : {len(deltas)}")
    print(f"Mean delta (us)   : {mean(deltas):.3f}")
    print(f"Median delta (us) : {median(deltas):.3f}")
    print(f"Std dev (us)      : {pstdev(deltas):.3f}")
    print(f"Min delta (us)    : {min(deltas)}")
    print(f"Max delta (us)    : {max(deltas)}")
    print(f"P95 delta (us)    : {percentile(sorted_deltas, 0.95):.3f}")
    print(f"P99 delta (us)    : {percentile(sorted_deltas, 0.99):.3f}")
    print(f"Peak to peak (us) : {max(deltas) - min(deltas)}")

    duration_s = (rows[-1][1] - rows[0][1]) / 1_000_000.0
    print(f"Duration (s)      : {duration_s:.3f}")

    with Path("e0_coral_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "samples",
                "mean_us",
                "median_us",
                "stddev_us",
                "min_us",
                "max_us",
                "p95_us",
                "p99_us",
                "peak_to_peak_us",
                "duration_s",
            ]
        )
        writer.writerow(
            [
                len(deltas),
                round(mean(deltas), 3),
                round(median(deltas), 3),
                round(pstdev(deltas), 3),
                min(deltas),
                max(deltas),
                round(percentile(sorted_deltas, 0.95), 3),
                round(percentile(sorted_deltas, 0.99), 3),
                max(deltas) - min(deltas),
                round(duration_s, 3),
            ]
        )

    print("\nSaved summary to e0_coral_summary.csv")


if __name__ == "__main__":
    main()
