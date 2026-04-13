"""
E4 Network I/O Stressor Experiment - Jetson Orin Nano Super
BLE connections: 0 / 2 / 4 / 6 simultaneous (nRF52840 DK central)
WiFi: 2.4 GHz co-channel interference via GL-AX1800

Protocol:
  - 10 trials per BLE connection level (0/2/4/6)
  - 500 inferences per trial, cycle-paced to T_NOMINAL=0.100s
  - delta = L-inf norm on consecutive softmax vectors vs nominal baseline
  - STER = fraction of inferences where delta > T_STAR=0.05

Usage (run on Jetson via SSH):
  source ~/e0_env/bin/activate
  python3 e4_jetson.py --conns <0|2|4|6>

Output: ~/e4_experiment/results/e4_jetson_conns<N>.csv
"""

import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import json, time, os, csv

# ── Config ────────────────────────────────────────────────────────────────────
TRT_ENGINE  = os.path.expanduser("~/e0_experiment/models/mobilenetv2_fp16.trt")
MANIFEST    = os.path.expanduser("~/e0_experiment/data/manifest.json")
RESULTS_DIR  = os.path.expanduser("~/e4_experiment/results")
BASELINE_NPY = os.path.expanduser("~/e4_experiment/e4_baseline.npy")

N_INFER    = 500
N_TRIALS   = 10
T_NOMINAL  = 0.100   # seconds — cycle pace target
T_STAR     = 0.05    # STER threshold (ISO 14971 conservative)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="E4 BLE/WiFi stressor inference")
parser.add_argument("--conns", type=int, required=True, choices=[0, 2, 4, 6],
                    help="Number of active BLE connections on nRF52840 DK")
args = parser.parse_args()

N_CONNS     = args.conns
RESULTS_CSV = os.path.join(RESULTS_DIR, f"e4_jetson_conns{N_CONNS}.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n=== E4 Network I/O Stressor — Jetson Orin Nano Super ===")
print(f"BLE connections : {N_CONNS}")
print(f"Trials          : {N_TRIALS}")
print(f"Inferences/trial: {N_INFER}")
print(f"T_NOMINAL       : {T_NOMINAL}s")
print(f"T_STAR          : {T_STAR}")
print(f"Output          : {RESULTS_CSV}")
print(f"=========================================================\n")

# ── Load manifest ─────────────────────────────────────────────────────────────
with open(MANIFEST) as f:
    manifest = json.load(f)
# manifest is a plain list — use manifest[:N_INFER]
images = manifest[:N_INFER]
assert len(images) == N_INFER, f"Need {N_INFER} images, got {len(images)}"

# ── TensorRT engine ───────────────────────────────────────────────────────────
print("Loading TensorRT engine...")
logger = trt.Logger(trt.Logger.WARNING)
with open(TRT_ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
in_shape    = (1, 3, 224, 224)
out_shape   = (1, 1000)
print(f"  Engine loaded: {input_name} → {output_name}")

# Pinned host + device buffers
h_in  = cuda.pagelocked_empty(in_shape,  dtype=np.float16)
h_out = cuda.pagelocked_empty(out_shape, dtype=np.float16)
d_in  = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)
stream = cuda.Stream()

context.set_tensor_address(input_name,  int(d_in))
context.set_tensor_address(output_name, int(d_out))

# ── Preprocessing ─────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1).astype(np.float16)[np.newaxis, :]

# ── Inference ─────────────────────────────────────────────────────────────────
def infer(arr):
    np.copyto(h_in, arr)
    cuda.memcpy_htod_async(d_in, h_in, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()
    return h_out[0].copy()

def softmax(x):
    x = x.astype(np.float32)
    e = np.exp(x - x.max())
    return e / e.sum()

# ── Load fixed pre-captured baseline ─────────────────────────────────────────
assert os.path.exists(BASELINE_NPY), f"Baseline not found: {BASELINE_NPY}\nRun capture_baseline.py first."
nominal_baseline = np.load(BASELINE_NPY).astype(np.float32)
print(f"Fixed baseline loaded. Top-1: {np.argmax(nominal_baseline)}  sum={nominal_baseline.sum():.6f}")

# ── Main experiment loop ──────────────────────────────────────────────────────
print(f"\nStarting E4 trials (BLE conns={N_CONNS})...")

fieldnames = [
    "trial", "ble_conns", "inference_idx",
    "latency_s", "delta", "ster_flag",
    "top1_class", "timestamp"
]

with open(RESULTS_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for trial in range(1, N_TRIALS + 1):
        trial_deltas = []
        trial_ster   = 0
        trial_start  = time.time()

        print(f"  Trial {trial}/{N_TRIALS} ...", end="", flush=True)

        for i, e in enumerate(images):
            t0  = time.time()
            arr = preprocess(e["path"])
            out = infer(arr)
            sv  = softmax(out)
            t1  = time.time()

            latency = t1 - t0
            delta   = float(np.max(np.abs(sv - nominal_baseline)))
            ster_flag = 1 if delta > T_STAR else 0
            trial_deltas.append(delta)
            trial_ster += ster_flag

            writer.writerow({
                "trial"         : trial,
                "ble_conns"     : N_CONNS,
                "inference_idx" : i,
                "latency_s"     : round(latency, 6),
                "delta"         : round(delta, 6),
                "ster_flag"     : ster_flag,
                "top1_class"    : int(np.argmax(sv)),
                "timestamp"     : round(t1, 3),
            })

            # Cycle-pace to T_NOMINAL
            elapsed = time.time() - t0
            sleep_t = T_NOMINAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        trial_duration = time.time() - trial_start
        ster = trial_ster / N_INFER
        d_mean = float(np.mean(trial_deltas))
        d_p99  = float(np.percentile(trial_deltas, 99))

        print(f" STER={ster:.4f}  δ_mean={d_mean:.4f}  δ_P99={d_p99:.4f}"
              f"  t={trial_duration:.1f}s")

print(f"\nE4 complete. Results saved to {RESULTS_CSV}")

# ── Trial-level summary ───────────────────────────────────────────────────────
print("\n--- Per-trial summary ---")
import csv as _csv
with open(RESULTS_CSV) as f:
    rows = list(_csv.DictReader(f))

for t in range(1, N_TRIALS + 1):
    trial_rows = [r for r in rows if int(r["trial"]) == t]
    deltas = [float(r["delta"]) for r in trial_rows]
    ster   = sum(int(r["ster_flag"]) for r in trial_rows) / N_INFER
    print(f"  Trial {t:2d}: STER={ster:.4f}  "
          f"δ_mean={np.mean(deltas):.4f}  "
          f"δ_P99={np.percentile(deltas,99):.4f}")

all_deltas = [float(r["delta"]) for r in rows]
all_ster   = sum(int(r["ster_flag"]) for r in rows) / len(rows)
print(f"\nAGGREGATE (conns={N_CONNS}):")
print(f"  STER    = {all_ster:.4f}")
print(f"  δ_mean  = {float(np.mean(all_deltas)):.4f}")
print(f"  δ_P99   = {float(np.percentile(all_deltas, 99)):.4f}")
