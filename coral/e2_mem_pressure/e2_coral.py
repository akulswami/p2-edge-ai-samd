"""
E2 Coral — Memory Pressure Experiment
stress-ng --vm at 25/50/75/90% of host RAM while Coral runs inference.
5 trials per level, 2000 inferences each (one full firmware run per trial).
Output: ~/e2_coral/results/e2_coral.csv

BEFORE RUNNING:
  Same firmware as E1 (e0_infer_baseline) — no reflash needed.
  Confirm Coral is connected: ls /dev/ttyACM*
  Run from Ubuntu: python3 e2_coral.py
"""

import subprocess, signal, time, os, csv, sys
sys.path.insert(0, os.path.dirname(__file__))
from coral_capture import find_coral_port, capture_inferences, compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.expanduser("~/e2_coral/results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e2_coral.csv")
N_INFER     = 2000
N_TRIALS    = 5

# Nominal baseline from E0
REF_CLASS = 905
REF_SCORE = 0.320312

# Host RAM — detect at runtime, fill pct% of total
VM_LEVELS = [25, 50, 75, 90]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Detect total host RAM ─────────────────────────────────────────────────────
def get_total_ram_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    return 16384  # fallback

TOTAL_RAM_MB = get_total_ram_mb()
print(f"Host RAM detected: {TOTAL_RAM_MB} MB")

# ── Serial port ───────────────────────────────────────────────────────────────
print("Detecting Coral serial port...")
port = find_coral_port()
print(f"  Found: {port}")

# ── CSV header ────────────────────────────────────────────────────────────────
with open(RESULTS_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "vm_pct", "trial",
        "ster", "accuracy",
        "delta_mean", "delta_p99",
        "n_infer", "n_violations"
    ])

# ── Stress helper ─────────────────────────────────────────────────────────────
def start_vm_stress(pct):
    mb = int(TOTAL_RAM_MB * pct / 100)
    cmd = [
        "stress-ng", "--vm", "1",
        "--vm-bytes", f"{mb}M",
        "--vm-keep",
        "--vm-method", "all",
        "-t", "0",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_stress(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nStarting Coral E2: {len(VM_LEVELS)} levels × {N_TRIALS} trials\n")
print("NOTE: Each trial waits for Coral to complete 2000 inferences (~30-60s).")
print("Power-cycle or reset the Coral between trials to restart firmware.\n")

for vm_pct in VM_LEVELS:
    print(f"=== VM fill: {vm_pct}% ({int(TOTAL_RAM_MB * vm_pct / 100)} MB) ===")

    for trial in range(1, N_TRIALS + 1):
        stress_proc = None
        try:
            # Start stressor and let it ramp up
            stress_proc = start_vm_stress(vm_pct)
            time.sleep(3.0)

            print(f"  Trial {trial:02d}/{N_TRIALS} — reset Coral now (power cycle or nRST button), then press Enter...")
            input()

            print(f"  Capturing {N_INFER} inferences from {port}...")
            rows = capture_inferences(port, n_expected=N_INFER)

            if len(rows) < N_INFER * 0.9:
                print(f"  WARNING: only {len(rows)} rows captured (expected {N_INFER})")

            m = compute_metrics(rows, ref_class=REF_CLASS, ref_score=REF_SCORE)

            print(
                f"  Trial {trial:02d}/{N_TRIALS} | "
                f"VM={vm_pct}% | "
                f"STER={m['ster']:.4f} | "
                f"Acc={m['acc']:.4f} | "
                f"δ_mean={m['d_mean']:.6f} | "
                f"δ_P99={m['d_p99']:.6f} | "
                f"n={m['n']}"
            )

            with open(RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    vm_pct, trial,
                    f"{m['ster']:.6f}", f"{m['acc']:.6f}",
                    f"{m['d_mean']:.6f}", f"{m['d_p99']:.6f}",
                    m["n"], m["violations"]
                ])

        finally:
            stop_stress(stress_proc)
            time.sleep(2.0)

print(f"\nCoral E2 complete. Results → {RESULTS_CSV}")
