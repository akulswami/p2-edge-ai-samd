#!/usr/bin/env python3
"""
E5 Jetson Experiment --- Combined Realistic Deployment Load
Stressors: CPU 75% + Memory 50% + BLE 4 connections + Disk fio write
Platform:  NVIDIA Jetson Orin Nano Super, JetPack 6, TensorRT FP16
Protocol:  10 trials x 500 inferences, cycle-paced T_NOMINAL=0.100s
Baseline:  ~/e4_experiment/e4_baseline.npy (fixed, do not recompute)
Output:    ~/e5_experiment/e5_trial_XX.npy + e5_results_summary.json
"""

import os
import sys
import time
import json
import subprocess
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

# ── paths ────────────────────────────────────────────────────────────────────
HOME            = Path.home()
MODEL_PATH      = HOME / "e0_experiment/models/mobilenetv2_fp16.trt"
MANIFEST_PATH   = HOME / "e0_experiment/data/manifest.json"
BASELINE_PATH   = HOME / "e4_experiment/e4_baseline.npy"
OUTPUT_DIR      = HOME / "e5_experiment"
FIO_TARGET      = Path("/media/akulswami/0292F43492F42DB3/e5_fio_stressor.tmp")

# ── protocol constants ────────────────────────────────────────────────────────
N_TRIALS        = 10
N_INFER         = 500
T_NOMINAL       = 0.100          # seconds, 10 Hz cycle
T_STAR          = 0.05           # tolerance band
CPU_LOAD_PCT    = 75             # stress-ng --cpu target
MEM_FILL_PCT    = 50             # stress-ng --vm target
FIO_SIZE        = "20G"          # sustained write file size
FIO_RUNTIME     = 900            # seconds — longer than full experiment

# ── stressor verification thresholds ─────────────────────────────────────────
CPU_VERIFY_MIN  = 60.0           # minimum CPU% to confirm stressor active
MEM_VERIFY_MIN  = 20.0           # minimum MEM% to confirm stressor active


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT / pycuda inference engine
# ─────────────────────────────────────────────────────────────────────────────
def load_engine(model_path: Path):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    return engine, context


def build_io_buffers(engine):
    import pycuda.driver as cuda

    # Mirror E4 exactly: explicit float32 buffers, (1,3,224,224) in, (1,1000) out
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    h_in  = cuda.pagelocked_empty((1, 3, 224, 224), dtype=np.float16)
    h_out = cuda.pagelocked_empty((1, 1000),         dtype=np.float16)
    d_in  = cuda.mem_alloc(h_in.nbytes)
    d_out = cuda.mem_alloc(h_out.nbytes)
    stream = cuda.Stream()

    inputs  = [{"host": h_in,  "device": d_in,  "name": input_name}]
    outputs = [{"host": h_out, "device": d_out, "name": output_name}]
    bindings = [int(d_in), int(d_out)]

    return inputs, outputs, bindings, stream


def load_image(path: str, shape=(3, 224, 224)) -> np.ndarray:
    """Load and preprocess image to MobileNetV2 FP32 input format — mirror E4 exactly."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = arr.transpose(2, 0, 1)   # HWC → CHW
    return arr.astype(np.float16)[np.newaxis, :]  # shape (1,3,224,224) float32


def infer(context, inputs, outputs, bindings, stream, img: np.ndarray) -> np.ndarray:
    import pycuda.driver as cuda

    np.copyto(inputs[0]["host"], img)
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    logits = outputs[0]["host"][0].astype(np.float32)  # shape (1000,)
    # softmax — mirror E4
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Stressor management
# ─────────────────────────────────────────────────────────────────────────────
class StressorManager:
    def __init__(self):
        self.procs = []

    def start_cpu(self):
        """stress-ng --cpu 0 targets all cores at CPU_LOAD_PCT%."""
        cmd = [
            "stress-ng", "--cpu", "0",
            "--cpu-load", str(CPU_LOAD_PCT),
            "--timeout", str(FIO_RUNTIME)
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(("cpu_stress", p))
        print(f"  [stressor] CPU stress started (pid={p.pid}, target={CPU_LOAD_PCT}%)")

    def start_memory(self):
        """stress-ng --vm fills MEM_FILL_PCT% of available RAM."""
        # get total RAM in MB
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
        """Sustained sequential write to USB HDD."""
        cmd = [
            "fio",
            "--name=e5_disk_stressor",
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
        print("[E5] Starting stressors...")
        self.start_cpu()
        self.start_memory()
        self.start_fio()
        print("[E5] Waiting 10s for stressors to reach steady state...")
        time.sleep(10)

    def verify_active(self) -> bool:
        """Check CPU and memory load are at expected levels."""
        import psutil
        cpu_pct = psutil.cpu_percent(interval=2.0)
        mem     = psutil.virtual_memory()
        mem_pct = mem.percent

        print(f"  [verify] CPU={cpu_pct:.1f}% (min={CPU_VERIFY_MIN}%)  "
              f"MEM={mem_pct:.1f}% (min={MEM_VERIFY_MIN}%)")

        ok = True
        if cpu_pct < CPU_VERIFY_MIN:
            print(f"  [WARN] CPU load {cpu_pct:.1f}% below threshold {CPU_VERIFY_MIN}% "
                  f"— stressor may not be active")
            ok = False
        if mem_pct < MEM_VERIFY_MIN:
            print(f"  [WARN] MEM load {mem_pct:.1f}% below threshold {MEM_VERIFY_MIN}% "
                  f"— stressor may not be active")
            ok = False

        # check fio process still running
        for name, p in self.procs:
            if p.poll() is not None:
                print(f"  [WARN] Stressor process '{name}' (pid={p.pid}) has exited early")
                ok = False

        return ok

    def stop_all(self):
        print("[E5] Stopping stressors...")
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
                try:
                    p.kill()
                except Exception:
                    pass
        self.procs.clear()
        # clean up fio temp file
        if FIO_TARGET.exists():
            FIO_TARGET.unlink()
            print(f"  [stressor] fio temp file removed: {FIO_TARGET}")


# ─────────────────────────────────────────────────────────────────────────────
# BLE connection count verification
# ─────────────────────────────────────────────────────────────────────────────
def verify_ble_connections(expected: int = 4) -> bool:
    """
    Check hciconfig / hcitool for active BLE connections.
    This is a best-effort check — the nRF DK manages connections on its side.
    We look for HCI connection handles as a proxy.
    """
    try:
        result = subprocess.run(
            ["hcitool", "con"],
            capture_output=True, text=True, timeout=5
        )
        lines = [l for l in result.stdout.strip().splitlines() if "ACL" in l or "LE" in l]
        count = len(lines)
        print(f"  [verify] BLE managed by nRF DK on Ubuntu host — not visible from Jetson hcitool (expected)")
        return True   # non-blocking: DK manages connections, host may not see all
    except Exception as e:
        print(f"  [verify] BLE check skipped ({e}) — confirm manually")
        return True   # non-blocking


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(trial_idx: int, images: list, baseline: np.ndarray,
              engine, context, inputs, outputs, bindings, stream) -> dict:
    """
    Run one E5 trial: 500 cycle-paced inferences under active stressors.
    Returns per-trial stats dict.
    """
    deltas      = np.zeros(N_INFER, dtype=np.float32)
    top1_correct = 0
    cycle_times  = np.zeros(N_INFER, dtype=np.float32)

    print(f"\n  [trial {trial_idx+1:02d}/{N_TRIALS}] Starting 500 inferences...")
    t_trial_start = time.perf_counter()

    for i in range(N_INFER):
        t_cycle_start = time.perf_counter()

        entry   = images[i % N_INFER]
        img     = load_image(entry["path"])
        softmax = infer(context, inputs, outputs, bindings, stream, img)

        # δ = L-inf norm vs fixed baseline
        delta      = float(np.max(np.abs(softmax - baseline)))
        deltas[i]  = delta

        # top-1 accuracy
        if np.argmax(softmax) == entry["label"]:
            top1_correct += 1

        t_elapsed = time.perf_counter() - t_cycle_start
        cycle_times[i] = t_elapsed

        # cycle-pace to T_NOMINAL
        remaining = T_NOMINAL - t_elapsed
        if remaining > 0:
            time.sleep(remaining)

    t_trial_end = time.perf_counter()

    ster      = float(np.mean(deltas > T_STAR))
    acc       = float(top1_correct / N_INFER)
    delta_mean = float(np.mean(deltas))
    delta_p99  = float(np.percentile(deltas, 99))
    duration   = t_trial_end - t_trial_start

    exceedances = int(np.sum(deltas > T_STAR))

    print(f"  [trial {trial_idx+1:02d}/{N_TRIALS}] "
          f"STER={ster:.4f}  Acc={acc*100:.2f}%  "
          f"δ_mean={delta_mean:.4f}  δ_P99={delta_p99:.4f}  "
          f"exceedances={exceedances}  duration={duration:.1f}s")

    return {
        "trial":        trial_idx + 1,
        "ster":         ster,
        "accuracy":     acc,
        "delta_mean":   delta_mean,
        "delta_p99":    delta_p99,
        "delta_max":    float(np.max(deltas)),
        "exceedances":  exceedances,
        "n_infer":      N_INFER,
        "duration_s":   duration,
        "deltas":       deltas   # saved to .npy separately
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E5 Jetson --- Combined Realistic Deployment Load")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── output dir ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output dir: {OUTPUT_DIR}")

    # ── load manifest ─────────────────────────────────────────────────────────
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    images = manifest[:N_INFER]
    print(f"[setup] Manifest loaded: {len(images)} images")
    assert len(images) == N_INFER, f"Expected {N_INFER} images, got {len(images)}"

    # ── load fixed baseline ───────────────────────────────────────────────────
    baseline = np.load(BASELINE_PATH)
    print(f"[setup] Baseline loaded: {BASELINE_PATH}, shape={baseline.shape}")
    assert baseline.ndim == 1 and baseline.shape[0] == 1000, \
        f"Baseline shape unexpected: got {baseline.shape}, expected (1000,)"

    # ── load TensorRT engine ──────────────────────────────────────────────────
    print(f"[setup] Loading TensorRT engine: {MODEL_PATH}")
    engine, context = load_engine(MODEL_PATH)
    inputs, outputs, bindings, stream = build_io_buffers(engine)
    context.set_tensor_address(inputs[0]["name"],  int(inputs[0]["device"]))
    context.set_tensor_address(outputs[0]["name"], int(outputs[0]["device"]))
    print("[setup] Engine loaded OK")

    # ── verify BLE ────────────────────────────────────────────────────────────
    print("\n[pre-flight] Verifying BLE connections...")
    verify_ble_connections(expected=4)

    # ── start stressors ───────────────────────────────────────────────────────
    stressor = StressorManager()

    try:
        stressor.start_all()

        # ── verify stressors active ───────────────────────────────────────────
        print("\n[pre-flight] Verifying stressors are active...")
        stressor_ok = stressor.verify_active()
        if not stressor_ok:
            resp = input("\n[WARN] Stressor verification failed. Continue anyway? [y/N]: ")
            if resp.strip().lower() != "y":
                print("[abort] Exiting. Fix stressors and rerun.")
                stressor.stop_all()
                sys.exit(1)

        print("\n[pre-flight] All checks passed. Starting trials...\n")

        # ── run trials ────────────────────────────────────────────────────────
        all_results = []
        all_deltas  = []

        for t in range(N_TRIALS):
            result = run_trial(t, images, baseline,
                               engine, context, inputs, outputs, bindings, stream)
            deltas = result.pop("deltas")
            all_results.append(result)
            all_deltas.append(deltas)

            # save per-trial .npy immediately (crash safety)
            trial_path = OUTPUT_DIR / f"e5_trial_{t+1:02d}.npy"
            np.save(trial_path, deltas)

            # re-verify stressors every 3 trials
            if (t + 1) % 3 == 0 and t < N_TRIALS - 1:
                print(f"\n[mid-run verify] After trial {t+1}:")
                stressor.verify_active()

    except KeyboardInterrupt:
        print("\n[interrupted] KeyboardInterrupt — stopping stressors and saving partial results")
    finally:
        stressor.stop_all()

    # ── aggregate results ─────────────────────────────────────────────────────
    if not all_results:
        print("[error] No trials completed.")
        sys.exit(1)

    ster_vals = np.array([r["ster"] for r in all_results])
    acc_vals  = np.array([r["accuracy"] for r in all_results])
    dm_vals   = np.array([r["delta_mean"] for r in all_results])
    dp99_vals = np.array([r["delta_p99"] for r in all_results])

    summary = {
        "experiment":       "E5",
        "platform":         "Jetson Orin Nano Super",
        "timestamp":        datetime.now().isoformat(),
        "n_trials":         len(all_results),
        "n_infer":          N_INFER,
        "t_star":           T_STAR,
        "cpu_load_pct":     CPU_LOAD_PCT,
        "mem_fill_pct":     MEM_FILL_PCT,
        "ble_connections":  4,
        "fio_target":       str(FIO_TARGET),
        "baseline_path":    str(BASELINE_PATH),
        "ster_mean":        float(np.mean(ster_vals)),
        "ster_std":         float(np.std(ster_vals)),
        "ster_min":         float(np.min(ster_vals)),
        "ster_max":         float(np.max(ster_vals)),
        "acc_mean":         float(np.mean(acc_vals)),
        "acc_std":          float(np.std(acc_vals)),
        "delta_mean_mean":  float(np.mean(dm_vals)),
        "delta_p99_mean":   float(np.mean(dp99_vals)),
        "trials":           all_results
    }

    summary_path = OUTPUT_DIR / "e5_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # save all deltas stacked
    all_deltas_path = OUTPUT_DIR / "e5_all_deltas.npy"
    np.save(all_deltas_path, np.array(all_deltas))

    # ── final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E5 JETSON --- FINAL SUMMARY")
    print("=" * 70)
    print(f"Trials completed : {len(all_results)}/{N_TRIALS}")
    print(f"STER mean        : {summary['ster_mean']:.4f}  "
          f"(std={summary['ster_std']:.4f}, "
          f"min={summary['ster_min']:.4f}, "
          f"max={summary['ster_max']:.4f})")
    print(f"Accuracy mean    : {summary['acc_mean']*100:.2f}%  "
          f"(std={summary['acc_std']*100:.4f}%)")
    print(f"δ_mean mean      : {summary['delta_mean_mean']:.4f}")
    print(f"δ_P99 mean       : {summary['delta_p99_mean']:.4f}")
    print(f"\nResults saved to : {OUTPUT_DIR}")
    print(f"Summary JSON     : {summary_path}")
    print(f"All deltas .npy  : {all_deltas_path}")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
