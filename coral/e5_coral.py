#!/usr/bin/env python3
"""
E5 Coral Experiment --- Combined Realistic Deployment Load
Stressors: CPU 75% + Memory 50% + BLE 4 connections + Disk fio write
Platform:  Coral Dev Board Micro, Edge TPU, TFLite int8
Protocol:  10 trials x 2000 inferences (nRST reset per trial)
Baseline:  REF_CLASS=905, REF_SCORE=0.320312
Output:    ~/e5_coral/e5_coral_results.json
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# ── import shared capture utility ─────────────────────────────────────────────
sys.path.insert(0, str(Path.home() / "e1_coral"))
from coral_capture import capture_inferences, compute_metrics

# ── paths ─────────────────────────────────────────────────────────────────────
HOME       = Path.home()
OUTPUT_DIR = HOME / "e5_coral"

# ── protocol constants ────────────────────────────────────────────────────────
N_TRIALS       = 10
N_INFER        = 2000
REF_CLASS      = 905
REF_SCORE      = 0.320312
T_STAR         = 0.05

# ── stressor config ───────────────────────────────────────────────────────────
CPU_LOAD_PCT   = 75
MEM_FILL_PCT   = 50
FIO_TARGET     = Path("/tmp/e5_coral_fio_stressor.tmp")
FIO_RUNTIME    = 1200   # seconds — longer than full experiment
FIO_SIZE       = "10G"

# ── stressor verification thresholds ─────────────────────────────────────────
CPU_VERIFY_MIN = 60.0
MEM_VERIFY_MIN = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Stressor management
# ─────────────────────────────────────────────────────────────────────────────
class StressorManager:
    def __init__(self):
        self.procs = []

    def start_cpu(self):
        cmd = [
            "stress-ng", "--cpu", "0",
            "--cpu-load", str(CPU_LOAD_PCT),
            "--timeout", str(FIO_RUNTIME)
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(("cpu_stress", p))
        print(f"  [stressor] CPU stress started (pid={p.pid}, target={CPU_LOAD_PCT}%)")

    def start_memory(self):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_kb = int(line.split()[1])
                    break
        fill_mb = int(total_kb * MEM_FILL_PCT / 100 / 1024)
        cmd = [
            "stress-ng", "--vm", "1",
            "--vm-bytes", f"{fill_mb}M",
            "--timeout", str(FIO_RUNTIME)
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(("mem_stress", p))
        print(f"  [stressor] Memory stress started (pid={p.pid}, fill={fill_mb}MB / {MEM_FILL_PCT}%)")

    def start_fio(self):
        """Write to /tmp on Ubuntu's NVMe — generates I/O pressure on host."""
        cmd = [
            "fio",
            "--name=e5_coral_disk_stressor",
            f"--filename={FIO_TARGET}",
            "--rw=write",
            "--bs=1M",
            f"--size={FIO_SIZE}",
            "--ioengine=libaio",
            "--iodepth=4",
            "--direct=1",
            f"--runtime={FIO_RUNTIME}",
            "--time_based",
            "--output=/dev/null"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(("fio", p))
        print(f"  [stressor] fio disk write started (pid={p.pid}, target={FIO_TARGET})")

    def start_all(self):
        print("[E5 Coral] Starting stressors...")
        self.start_cpu()
        self.start_memory()
        self.start_fio()
        print("[E5 Coral] Waiting 10s for stressors to reach steady state...")
        time.sleep(10)

    def verify_active(self) -> bool:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=2.0)
        mem_pct = psutil.virtual_memory().percent
        print(f"  [verify] CPU={cpu_pct:.1f}% (min={CPU_VERIFY_MIN}%)  "
              f"MEM={mem_pct:.1f}% (min={MEM_VERIFY_MIN}%)")
        ok = True
        if cpu_pct < CPU_VERIFY_MIN:
            print(f"  [WARN] CPU {cpu_pct:.1f}% below threshold")
            ok = False
        if mem_pct < MEM_VERIFY_MIN:
            print(f"  [WARN] MEM {mem_pct:.1f}% below threshold")
            ok = False
        for name, p in self.procs:
            if p.poll() is not None:
                print(f"  [WARN] Stressor '{name}' exited early")
                ok = False
        return ok

    def stop_all(self):
        print("[E5 Coral] Stopping stressors...")
        for name, p in self.procs:
            try:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except Exception:
                    p.kill()
                    p.wait(timeout=2)
                print(f"  [stressor] {name} stopped (pid={p.pid})")
            except Exception as e:
                print(f"  [stressor] {name} stop error: {e}")
        self.procs.clear()
        if FIO_TARGET.exists():
            FIO_TARGET.unlink()
            print(f"  [stressor] fio temp file removed: {FIO_TARGET}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E5 Coral --- Combined Realistic Deployment Load")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"Protocol: {N_TRIALS} trials x {N_INFER} inferences")
    print(f"REF_CLASS={REF_CLASS}  REF_SCORE={REF_SCORE}  T*={T_STAR}")
    print(f"Stressors: CPU {CPU_LOAD_PCT}% + MEM {MEM_FILL_PCT}% + BLE 4 conns + fio write")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")

    # ── confirm BLE active ────────────────────────────────────────────────────
    print("\n[pre-flight] BLE: confirm nRF DK STATUS=4/4 in serial monitor before proceeding")
    input("  Press Enter when BLE 4 connections confirmed active... ")

    # ── start stressors ───────────────────────────────────────────────────────
    stressor = StressorManager()

    try:
        stressor.start_all()

        print("\n[pre-flight] Verifying stressors...")
        ok = stressor.verify_active()
        if not ok:
            resp = input("\n[WARN] Stressor verification failed. Continue anyway? [y/N]: ")
            if resp.strip().lower() != "y":
                print("[abort] Exiting.")
                stressor.stop_all()
                sys.exit(1)

        print("\n[pre-flight] All checks passed. Starting trials...\n")
        print("NOTE: Each trial requires a manual nRST press on the Coral board.")
        print("      You will be prompted for each trial.\n")

        all_results = []

        for t in range(N_TRIALS):
            print(f"\n{'─'*60}")
            print(f"  TRIAL {t+1}/{N_TRIALS} — Press nRST on Coral when prompted")
            print(f"{'─'*60}")

            rows = capture_inferences(
                port=None,       # auto-detect
                n_expected=N_INFER,
                timeout=180
            )

            if not rows:
                print(f"  [WARN] Trial {t+1}: no data captured — skipping")
                continue

            metrics = compute_metrics(rows, ref_class=REF_CLASS, ref_score=REF_SCORE)

            result = {
                "trial":       t + 1,
                "n_captured":  metrics["n"],
                "ster":        metrics["ster"],
                "accuracy":    metrics["acc"],
                "delta_mean":  metrics["d_mean"],
                "delta_p99":   metrics["d_p99"],
                "violations":  metrics["violations"],
                "ref_class":   REF_CLASS,
                "ref_score":   REF_SCORE,
            }
            all_results.append(result)

            print(f"  [trial {t+1:02d}/{N_TRIALS}] "
                  f"STER={result['ster']:.4f}  "
                  f"Acc={result['accuracy']*100:.2f}%  "
                  f"δ_mean={result['delta_mean']:.4f}  "
                  f"δ_P99={result['delta_p99']:.4f}  "
                  f"n={result['n_captured']}")

            # save partial results after each trial
            partial_path = OUTPUT_DIR / f"e5_coral_trial_{t+1:02d}.json"
            with open(partial_path, "w") as f:
                json.dump(result, f, indent=2)

            # re-verify stressors every 3 trials
            if (t + 1) % 3 == 0 and t < N_TRIALS - 1:
                print(f"\n[mid-run verify] After trial {t+1}:")
                stressor.verify_active()

    except KeyboardInterrupt:
        print("\n[interrupted] Stopping stressors and saving partial results...")
    finally:
        stressor.stop_all()

    if not all_results:
        print("[error] No trials completed.")
        sys.exit(1)

    # ── aggregate ─────────────────────────────────────────────────────────────
    ster_vals = np.array([r["ster"] for r in all_results])
    acc_vals  = np.array([r["accuracy"] for r in all_results])
    dm_vals   = np.array([r["delta_mean"] for r in all_results])
    dp99_vals = np.array([r["delta_p99"] for r in all_results])

    summary = {
        "experiment":      "E5",
        "platform":        "Coral Dev Board Micro",
        "timestamp":       datetime.now().isoformat(),
        "n_trials":        len(all_results),
        "n_infer":         N_INFER,
        "t_star":          T_STAR,
        "ref_class":       REF_CLASS,
        "ref_score":       REF_SCORE,
        "cpu_load_pct":    CPU_LOAD_PCT,
        "mem_fill_pct":    MEM_FILL_PCT,
        "ble_connections": 4,
        "ster_mean":       float(np.mean(ster_vals)),
        "ster_std":        float(np.std(ster_vals)),
        "ster_min":        float(np.min(ster_vals)),
        "ster_max":        float(np.max(ster_vals)),
        "acc_mean":        float(np.mean(acc_vals)),
        "acc_std":         float(np.std(acc_vals)),
        "delta_mean_mean": float(np.mean(dm_vals)),
        "delta_p99_mean":  float(np.mean(dp99_vals)),
        "trials":          all_results
    }

    summary_path = OUTPUT_DIR / "e5_coral_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E5 CORAL --- FINAL SUMMARY")
    print("=" * 70)
    print(f"Trials completed : {len(all_results)}/{N_TRIALS}")
    print(f"STER mean        : {summary['ster_mean']:.4f}  "
          f"(std={summary['ster_std']:.4f}, "
          f"min={summary['ster_min']:.4f}, "
          f"max={summary['ster_max']:.4f})")
    print(f"Accuracy mean    : {summary['acc_mean']*100:.2f}%")
    print(f"δ_mean mean      : {summary['delta_mean_mean']:.4f}")
    print(f"δ_P99 mean       : {summary['delta_p99_mean']:.4f}")
    print(f"\nResults saved to : {OUTPUT_DIR}")
    print(f"Summary JSON     : {summary_path}")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
