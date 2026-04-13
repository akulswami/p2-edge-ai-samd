"""
E4 Coral — Network I/O Stressor Experiment
BLE connections: 0 / 2 / 4 / 6 simultaneous (nRF52840 DK central)
WiFi: 2.4 GHz co-channel interference via GL-AX1800

Protocol:
  - 10 trials per BLE connection level (0/2/4/6)
  - 2000 inferences each trial (Edge TPU, MobileNetV1 int8)
  - REF_CLASS=905, REF_SCORE=0.320312 (deterministic baseline)
  - STER = fraction of inferences where class != REF_CLASS or score != REF_SCORE
  - delta = 0.0 expected (deterministic int8)

Usage:
  python3 e4_coral.py --conns <0|2|4|6>

Output: ~/e4_coral/results/e4_coral_conns<N>.csv
"""

import argparse
import sys, os, csv, time
sys.path.insert(0, os.path.expanduser("~/e1_coral"))
from coral_capture import find_coral_port, capture_inferences, compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.expanduser("~/e4_coral/results")
N_INFER     = 2000
N_TRIALS    = 10
REF_CLASS   = 905
REF_SCORE   = 0.320312

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="E4 Coral BLE/WiFi stressor")
parser.add_argument("--conns", type=int, required=True, choices=[0, 2, 4, 6],
                    help="Number of active BLE connections on nRF52840 DK")
args = parser.parse_args()

N_CONNS     = args.conns
RESULTS_CSV = os.path.join(RESULTS_DIR, f"e4_coral_conns{N_CONNS}.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n=== E4 Coral Network I/O Stressor ===")
print(f"BLE connections : {N_CONNS}")
print(f"Trials          : {N_TRIALS}")
print(f"Inferences/trial: {N_INFER}")
print(f"REF_CLASS       : {REF_CLASS}")
print(f"REF_SCORE       : {REF_SCORE}")
print(f"Output          : {RESULTS_CSV}")
print(f"=====================================\n")

# ── Detect Coral port ─────────────────────────────────────────────────────────
print("Detecting Coral serial port...")
port = find_coral_port()
if port is None:
    print("ERROR: Coral not found. Check USB connection.")
    sys.exit(1)
print(f"  Found: {port}\n")

# ── Main experiment loop ──────────────────────────────────────────────────────
fieldnames = [
    "ble_conns", "trial", "n_captured",
    "ster", "accuracy", "delta_mean", "delta_p99"
]

with open(RESULTS_CSV, "w", newline="") as f:
    csv.writer(f).writerow(fieldnames)

for trial in range(1, N_TRIALS + 1):
    print(f"Trial {trial}/{N_TRIALS} (BLE conns={N_CONNS})")

    results = capture_inferences(port, n_expected=N_INFER)

    if not results:
        print(f"  WARNING: No data captured for trial {trial} — skipping")
        continue

    n_captured = len(results)

    # STER: fraction where class != REF_CLASS
    ster_count = sum(1 for _, cls, _ in results if cls != REF_CLASS)
    ster = ster_count / n_captured

    # Accuracy: fraction where class == REF_CLASS
    acc = 1.0 - ster

    # Delta: deviation of score from REF_SCORE (deterministic int8 → expect 0.0)
    deltas = [abs(score - REF_SCORE) for _, _, score in results]
    delta_mean = float(sum(deltas) / len(deltas))
    delta_p99  = float(sorted(deltas)[int(0.99 * len(deltas))])

    print(f"  Captured={n_captured}  STER={ster:.4f}  "
          f"Acc={acc:.4f}  δ_mean={delta_mean:.6f}  δ_P99={delta_p99:.6f}")

    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            N_CONNS, trial, n_captured,
            round(ster, 6), round(acc, 6),
            round(delta_mean, 6), round(delta_p99, 6)
        ])

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n--- E4 Coral Summary (conns={N_CONNS}) ---")
with open(RESULTS_CSV) as f:
    rows = list(csv.DictReader(f))

if rows:
    sters  = [float(r["ster"])       for r in rows]
    dmeans = [float(r["delta_mean"]) for r in rows]
    dp99s  = [float(r["delta_p99"])  for r in rows]

    print(f"  Trials completed : {len(rows)}/{N_TRIALS}")
    print(f"  STER mean        : {sum(sters)/len(sters):.4f}")
    print(f"  STER max         : {max(sters):.4f}")
    print(f"  δ_mean           : {sum(dmeans)/len(dmeans):.6f}")
    print(f"  δ_P99            : {max(dp99s):.6f}")
    print(f"\nResults saved to {RESULTS_CSV}")
