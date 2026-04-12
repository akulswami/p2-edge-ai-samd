from __future__ import annotations

import csv
from pathlib import Path

LOG_PATH = Path("e0_infer_log.txt")
OUT_PATH = Path("e0_coral_infer_summary.csv")


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("No data")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")

    rows: list[tuple[int, int, float]] = []

    with LOG_PATH.open("r", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("E0 Infer Baseline Start"):
                continue
            if line.startswith("E0 Infer Baseline End"):
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue
            if parts[0] != "infer":
                continue

            try:
                iteration = int(parts[1])
                class_id = int(parts[2])
                score = float(parts[3])
            except ValueError:
                continue

            rows.append((iteration, class_id, score))

    if not rows:
        raise RuntimeError("No valid inference rows parsed")

    ref_class = rows[0][1]
    ref_score = rows[0][2]

    ster_errors = 0
    deltas: list[float] = []

    for _, class_id, score in rows:
        if class_id != ref_class:
            ster_errors += 1
        deltas.append(abs(score - ref_score))

    deltas_sorted = sorted(deltas)

    metrics = {
        "num_samples": len(rows),
        "class_id_constant": ref_class,
        "score_constant": f"{ref_score:.6f}",
        "STER_nominal": ster_errors / len(rows),
        "delta_mean": sum(deltas) / len(rows),
        "delta_p99": percentile(deltas_sorted, 0.99),
        "delta_max": max(deltas),
        "accuracy_nominal": 1.0 if ster_errors == 0 else 1.0 - (ster_errors / len(rows)),
    }

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print("=== Coral E0 Inference Baseline Summary ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
