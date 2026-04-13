"""
E1 Coral — CPU Contention Experiment
stress-ng --cpu at 25/50/75/100% on Ubuntu host while Coral runs inference.
5 trials per level, 2000 inferences each (one full firmware run per trial).
Output: ~/e1_coral/results/e1_coral.csv

BEFORE RUNNING:
  1. Flash the E0 firmware to Coral (it will auto-run on boot):
       cd ~/coralmicro && cmake --build build --target flashtool
       python3 scripts/flashtool.py --build_dir build \
           --elf_path build/examples/e0_infer_baseline/e0_infer_baseline \
           --usb_ip <coral_ip>   # or use --serial_port
  2. Confirm Coral is connected: ls /dev/ttyACM*
  3. Run from Ubuntu: python3 e1_coral.py
"""

import subprocess, signal, time, os, csv, sys
sys.path.insert(0, os.path.dirname(__file__))
from coral_capture import find_coral_port, capture_inferences, compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.expanduser("~/e1_coral/results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e1_coral.csv")
N_INFER     = 2000
N_TRIALS    = 5

# Nominal baseline from E0
REF_CLASS = 905
REF_SCORE = 0.320312

# CPU load levels → stress-ng worker count
# Ubuntu host (12-thread machine assumed; workers = ceil(pct/100 * nproc))
# We use --cpu-load pct instead of worker count for precision
CPU_LEVELS = [0, 25, 50, 75, 100]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Serial port ───────────────────────────────────────────────────────────────
print("Detecting Coral serial port...")
port = find_coral_port()
print(f"  Found: {port}")

# ── CSV header ────────────────────────────────────────────────────────────────
with open(RESULTS_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "cpu_pct", "trial",
        "ster", "accuracy",
        "delta_mean", "delta_p99",
        "n_infer", "n_violations"
    ])

# ── Stress helper ─────────────────────────────────────────────────────────────
def start_cpu_stress(pct):
    if pct == 0:
        return None
    cmd = ["stress-ng", "--cpu", "0", f"--cpu-load", str(pct), "-t", "0"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_stress(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ── Main loop ────────────────────────────────────────────────────────────────
print(f"\nStarting Coral E1: {len(CPU_LEVELS)} levels × {N_TRIALS} trials\n")
print("NOTE: Each trial waits for Coral to complete 2000 inferences (~30-60s).")
print("Power-cycle or reset the Coral between trials to restart firmware.\n")

for cpu_pct in CPU_LEVELS:
    print(f"=== CPU load: {cpu_pct}% ===")

    for trial in range(1, N_TRIALS + 1):
        stress_proc = None
        try:
            # Start stressor
            stress_proc = start_cpu_stress(cpu_pct)
            if cpu_pct > 0:
                time.sleep(2.0)   # let stressor ramp up

            print(f"  Trial {trial:02d}/{N_TRIALS} — reset Coral now (power cycle or nRST button), then press Enter...")
            input()

            print(f"  Capturing {N_INFER} inferences from {port}...")
            rows = capture_inferences(port, n_expected=N_INFER)

            if len(rows) < N_INFER * 0.9:
                print(f"  WARNING: only {len(rows)} rows captured (expected {N_INFER})")

            m = compute_metrics(rows, ref_class=REF_CLASS, ref_score=REF_SCORE)

            print(
                f"  Trial {trial:02d}/{N_TRIALS} | "
                f"CPU={cpu_pct}% | "
                f"STER={m['ster']:.4f} | "
                f"Acc={m['acc']:.4f} | "
                f"δ_mean={m['d_mean']:.6f} | "
                f"δ_P99={m['d_p99']:.6f} | "
                f"n={m['n']}"
            )

            with open(RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    cpu_pct, trial,
                    f"{m['ster']:.6f}", f"{m['acc']:.6f}",
                    f"{m['d_mean']:.6f}", f"{m['d_p99']:.6f}",
                    m["n"], m["violations"]
                ])

        finally:
            stop_stress(stress_proc)
            time.sleep(1.0)

print(f"\nCoral E1 complete. Results → {RESULTS_CSV}")
