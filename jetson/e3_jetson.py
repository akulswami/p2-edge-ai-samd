"""
E3 GPU Co-Tenancy Experiment - Jetson Orin Nano Super
Parallel TensorRT inference workers at 1/2/3/4 co-tenants
5 trials per level, 500 inferences each
Output: ~/e3_experiment/results/e3_jetson.csv
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import json, time, os, csv, subprocess, sys

# ── Config ────────────────────────────────────────────────────────────────────
TRT_ENGINE  = os.path.expanduser("~/e0_experiment/models/mobilenetv2_fp16.trt")
MANIFEST    = os.path.expanduser("~/e0_experiment/data/manifest.json")
RESULTS_DIR = os.path.expanduser("~/e3_experiment/results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e3_jetson.csv")
WORKER_SCRIPT = os.path.expanduser("~/e3_experiment/e3_worker.py")

N_INFER    = 500
N_TRIALS   = 5
T_NOMINAL  = 0.100    # seconds

# Co-tenant levels: number of parallel inference workers on the same GPU
COTENANT_LEVELS = [0, 1, 2, 3, 4]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── TensorRT engine load ───────────────────────────────────────────────────────
print("Loading TensorRT engine...")
logger = trt.Logger(trt.Logger.WARNING)
with open(TRT_ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
in_shape    = tuple(engine.get_tensor_shape(input_name))
out_shape   = tuple(engine.get_tensor_shape(output_name))
print(f"  input={input_name} {in_shape}  output={output_name} {out_shape}")

# Pinned host + device buffers
h_in  = cuda.pagelocked_empty(in_shape,  dtype=np.float16)
h_out = cuda.pagelocked_empty(out_shape, dtype=np.float16)
d_in  = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)
stream = cuda.Stream()

def infer(arr):
    np.copyto(h_in, arr)
    cuda.memcpy_htod_async(d_in,  h_in,  stream)
    context.set_tensor_address(input_name,  int(d_in))
    context.set_tensor_address(output_name, int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()
    logits = h_out[0].astype(np.float32)
    e = np.exp(logits - logits.max())
    return e / e.sum()

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading manifest and preprocessing images...")
with open(MANIFEST) as f:
    manifest = json.load(f)
manifest = manifest[:N_INFER]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1).astype(np.float16)[np.newaxis, :]

images = [preprocess(e["path"]) for e in manifest]
labels = [e["label"] for e in manifest]
print(f"  {len(images)} images preprocessed.")

# ── Nominal baseline ──────────────────────────────────────────────────────────
print("Computing nominal baseline outputs...")
nominal = np.array([infer(arr) for arr in images])
print("  Baseline done.")

# ── CSV header ────────────────────────────────────────────────────────────────
with open(RESULTS_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "cotenants", "trial",
        "ster", "accuracy",
        "delta_mean", "delta_p99",
        "n_infer", "n_violations"
    ])

# ── Main experiment loop ──────────────────────────────────────────────────────
print(f"\nStarting E3: {len(COTENANT_LEVELS)} levels × {N_TRIALS} trials × {N_INFER} inferences\n")

for n_cotenants in COTENANT_LEVELS:
    print(f"=== Co-tenants: {n_cotenants} ===")

    for trial in range(1, N_TRIALS + 1):
        workers = []
        try:
            # Launch co-tenant worker processes
            for _ in range(n_cotenants):
                p = subprocess.Popen(
                    [sys.executable, WORKER_SCRIPT,
                     TRT_ENGINE, MANIFEST],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                workers.append(p)

            # Give workers 3 s to load engines and start hammering the GPU
            if n_cotenants > 0:
                time.sleep(3.0)

            violations = 0
            correct    = 0
            deltas     = []

            for i, (arr, label) in enumerate(zip(images, labels)):
                t0      = time.perf_counter()
                softmax = infer(arr)
                t_inf   = time.perf_counter() - t0

                t_sleep = max(0.0, T_NOMINAL - t_inf)
                time.sleep(t_sleep)
                cycle = time.perf_counter() - t0
                if cycle > T_NOMINAL * 1.05:
                    violations += 1

                if np.argmax(softmax) == label:
                    correct += 1

                delta = float(np.max(np.abs(softmax - nominal[i])))
                deltas.append(delta)

            ster     = violations / N_INFER
            accuracy = correct    / N_INFER
            d_mean   = float(np.mean(deltas))
            d_p99    = float(np.percentile(deltas, 99))

            print(
                f"  Trial {trial:02d}/{N_TRIALS} | "
                f"cotenants={n_cotenants} | "
                f"STER={ster:.4f} | "
                f"Acc={accuracy:.4f} | "
                f"δ_mean={d_mean:.4f} | "
                f"δ_P99={d_p99:.4f}"
            )

            with open(RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    n_cotenants, trial,
                    f"{ster:.6f}", f"{accuracy:.6f}",
                    f"{d_mean:.6f}", f"{d_p99:.6f}",
                    N_INFER, violations
                ])

        finally:
            for p in workers:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            time.sleep(2.0)   # GPU cooldown between trials

print(f"\nE3 complete. Results → {RESULTS_CSV}")
