"""
E2 Memory Pressure Experiment - Jetson Orin Nano Super
stress-ng --vm at 25/50/75/90% LPDDR5 fill
5 trials per level, 500 inferences each
Output: ~/e2_experiment/results/e2_jetson.csv
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import json, time, os, csv, subprocess, signal

# ── Config ────────────────────────────────────────────────────────────────────
TRT_ENGINE  = os.path.expanduser("~/e0_experiment/models/mobilenetv2_fp16.trt")
MANIFEST    = os.path.expanduser("~/e0_experiment/data/manifest.json")
RESULTS_DIR = os.path.expanduser("~/e2_experiment/results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "e2_jetson.csv")

N_INFER     = 500
N_TRIALS    = 5
T_NOMINAL   = 0.100          # seconds — pace target per inference cycle

# Jetson Orin Nano Super has 8 GB LPDDR5 shared CPU+GPU
# stress-ng --vm uses virtual memory workers; --vm-bytes sets fill size
# We target % of total RAM; 8 GB board → sizes below
TOTAL_RAM_MB   = 8192
VM_LEVELS      = [25, 50, 75, 90]   # percent fill

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── TensorRT engine load ───────────────────────────────────────────────────────
print("Loading TensorRT engine...")
logger = trt.Logger(trt.Logger.WARNING)
with open(TRT_ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
in_shape    = tuple(engine.get_tensor_shape(input_name))   # (1,3,224,224)
out_shape   = tuple(engine.get_tensor_shape(output_name))  # (1,1000)
print(f"  input={input_name} {in_shape}  output={output_name} {out_shape}")

# Pinned host + device buffers
h_in  = cuda.pagelocked_empty(in_shape,  dtype=np.float16)
h_out = cuda.pagelocked_empty(out_shape, dtype=np.float16)
d_in  = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)
stream = cuda.Stream()

def infer(arr):
    """Run one FP16 inference; return softmax vector."""
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
    return arr.transpose(2, 0, 1).astype(np.float16)[np.newaxis, :]   # (1,3,224,224)

images = [preprocess(e["path"]) for e in manifest]
labels = [e["label"] for e in manifest]
print(f"  {len(images)} images preprocessed.")

# ── Nominal baseline (no stressor, used for δ) ────────────────────────────────
print("Computing nominal baseline outputs...")
nominal = []
for arr in images:
    nominal.append(infer(arr))
nominal = np.array(nominal)   # (500, 1000)
print("  Baseline done.")

# ── stress-ng helper ──────────────────────────────────────────────────────────
def start_vm_stress(pct):
    """Launch stress-ng --vm worker filling pct% of total RAM. Returns Popen."""
    mb = int(TOTAL_RAM_MB * pct / 100)
    cmd = [
        "stress-ng", "--vm", "1",
        "--vm-bytes", f"{mb}M",
        "--vm-keep",          # keep pages mapped (sustained pressure)
        "--vm-method", "all", # touch all pages
        "-t", "0",            # run until killed
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_stress(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ── CSV header ────────────────────────────────────────────────────────────────
with open(RESULTS_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "vm_pct", "trial",
        "ster", "accuracy",
        "delta_mean", "delta_p99",
        "n_infer", "n_violations"
    ])

# ── Main experiment loop ──────────────────────────────────────────────────────
print(f"\nStarting E2: {len(VM_LEVELS)} levels × {N_TRIALS} trials × {N_INFER} inferences\n")

for vm_pct in VM_LEVELS:
    print(f"=== VM fill: {vm_pct}% ===")

    for trial in range(1, N_TRIALS + 1):
        stress_proc = None
        try:
            # Start stressor, give it 2 s to fully allocate
            stress_proc = start_vm_stress(vm_pct)
            time.sleep(2.0)

            violations  = 0
            correct     = 0
            deltas      = []

            for i, (arr, label) in enumerate(zip(images, labels)):
                t0    = time.perf_counter()
                softmax = infer(arr)
                t_inf = time.perf_counter() - t0

                # Timing violation: full cycle exceeds T_NOMINAL
                t_sleep = max(0.0, T_NOMINAL - t_inf)
                time.sleep(t_sleep)
                cycle = time.perf_counter() - t0
                if cycle > T_NOMINAL * 1.05:   # 5% tolerance
                    violations += 1

                # Accuracy
                if np.argmax(softmax) == label:
                    correct += 1

                # δ = L-inf norm vs nominal softmax
                delta = float(np.max(np.abs(softmax - nominal[i])))
                deltas.append(delta)

            ster     = violations / N_INFER
            accuracy = correct    / N_INFER
            d_mean   = float(np.mean(deltas))
            d_p99    = float(np.percentile(deltas, 99))

            print(
                f"  Trial {trial:02d}/{N_TRIALS} | "
                f"VM={vm_pct}% | "
                f"STER={ster:.4f} | "
                f"Acc={accuracy:.4f} | "
                f"δ_mean={d_mean:.4f} | "
                f"δ_P99={d_p99:.4f}"
            )

            with open(RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    vm_pct, trial,
                    f"{ster:.6f}", f"{accuracy:.6f}",
                    f"{d_mean:.6f}", f"{d_p99:.6f}",
                    N_INFER, violations
                ])

        finally:
            stop_stress(stress_proc)
            time.sleep(1.0)   # cooldown between trials

print(f"\nE2 complete. Results → {RESULTS_CSV}")
