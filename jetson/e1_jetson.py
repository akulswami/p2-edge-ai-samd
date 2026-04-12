"""
E1 — CPU Contention Experiment, Jetson Orin Nano Super
Protocol: stress-ng --cpu at 0/25/50/75/100% load, 10 trials x 500 inferences per level
Output: e1_jetson.csv with STER, delta_mean, delta_p99 per trial
Fix: cycle time = inference + sleep, violation measured on full 100ms cycle
"""

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import json, time, os, csv, subprocess

# ── Config ────────────────────────────────────────────────────────────────────
TRT_ENGINE  = 'models/mobilenetv2_fp16.trt'
MANIFEST    = 'data/manifest.json'
RESULTS     = 'results/e1_jetson.csv'
N_INFER     = 500
N_TRIALS    = 10
T_NOMINAL   = 0.100   # 10 Hz target cycle time
T_STAR      = 0.05    # ±50ms tolerance band around T_NOMINAL

LOAD_LEVELS = [
    {'label': '0',   'workers': 0},
    {'label': '25',  'workers': 2},
    {'label': '50',  'workers': 3},
    {'label': '75',  'workers': 5},
    {'label': '100', 'workers': 6},
]

# ── TensorRT helpers ──────────────────────────────────────────────────────────
def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, 'rb') as f, trt.Runtime(logger) as rt:
        return rt.deserialize_cuda_engine(f.read())

def build_context(engine):
    context      = engine.create_execution_context()
    input_name   = engine.get_tensor_name(0)
    output_name  = engine.get_tensor_name(1)
    output_shape = tuple(engine.get_tensor_shape(output_name))
    h_input  = cuda.pagelocked_empty((1, 3, 224, 224), dtype=np.float16)
    h_output = cuda.pagelocked_empty(output_shape, dtype=np.float16)
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream   = cuda.Stream()
    return context, input_name, output_name, output_shape, \
           h_input, h_output, d_input, d_output, stream

def softmax(x):
    x = x.astype(np.float32)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def infer(context, input_name, output_name, output_shape,
          h_input, h_output, d_input, d_output, stream, img_array):
    np.copyto(h_input, img_array)
    context.set_tensor_address(input_name,  int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return softmax(h_output.flatten())

# ── Image preprocessing ───────────────────────────────────────────────────────
from PIL import Image

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1).astype(np.float16)[np.newaxis, :]

# ── Stressor control ──────────────────────────────────────────────────────────
def start_stressor(workers):
    if workers == 0:
        return None
    proc = subprocess.Popen(
        ['stress-ng', '--cpu', str(workers), '--cpu-method', 'matrixprod'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(10)
    return proc

def stop_stressor(proc):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(3)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading TensorRT engine...")
    engine   = load_engine(TRT_ENGINE)
    ctx_args = build_context(engine)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    images = manifest[:N_INFER]

    print(f"Preprocessing {N_INFER} images...")
    preloaded = [preprocess(e['path']) for e in images]
    labels    = [e['label'] for e in images]

    os.makedirs('results', exist_ok=True)
    with open(RESULTS, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['load_pct', 'trial', 'ster', 'delta_mean', 'delta_p99',
                         'accuracy', 'n_timing_violations'])

        for level in LOAD_LEVELS:
            label   = level['label']
            workers = level['workers']
            print(f"\n{'='*60}")
            print(f"Load level: {label}% ({workers} stress-ng workers)")
            print(f"{'='*60}")

            stressor = start_stressor(workers)

            for trial in range(1, N_TRIALS + 1):
                deltas    = []
                cycles    = []
                correct   = 0
                prev_prob = None

                for img_arr, lbl in zip(preloaded, labels):
                    t_start = time.perf_counter()

                    prob = infer(*ctx_args, img_arr)

                    t_infer = time.perf_counter()
                    elapsed_infer = t_infer - t_start

                    # Pace to T_NOMINAL
                    remaining = T_NOMINAL - elapsed_infer
                    if remaining > 0:
                        time.sleep(remaining)

                    t_end = time.perf_counter()
                    cycle_time = t_end - t_start   # full cycle: infer + sleep
                    cycles.append(cycle_time)

                    if np.argmax(prob) == lbl:
                        correct += 1

                    if prev_prob is not None:
                        delta = float(np.max(np.abs(prob - prev_prob)))
                        deltas.append(delta)
                    prev_prob = prob.copy()

                cycles_arr  = np.array(cycles)
                violations  = int(np.sum(np.abs(cycles_arr - T_NOMINAL) > T_STAR))
                ster        = violations / N_INFER
                delta_mean  = float(np.mean(deltas)) if deltas else 0.0
                delta_p99   = float(np.percentile(deltas, 99)) if deltas else 0.0
                accuracy    = correct / N_INFER

                print(f"  Trial {trial:2d}: STER={ster:.4f}  "
                      f"δ_mean={delta_mean:.4f}  δ_P99={delta_p99:.4f}  "
                      f"Acc={accuracy:.4f}  violations={violations}")

                writer.writerow([label, trial, f'{ster:.6f}',
                                 f'{delta_mean:.6f}', f'{delta_p99:.6f}',
                                 f'{accuracy:.4f}', violations])
                csvfile.flush()

            stop_stressor(stressor)

    print(f"\nDone. Results saved to {RESULTS}")

if __name__ == '__main__':
    main()
