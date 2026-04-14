#!/usr/bin/env python3
"""
e6_jetson_cpu_v3.py — CPU-only inference stability (correct per-image reference)

δᵢ = ‖σᵢᶜ − σ̄ᵢ‖∞  where σ̄ᵢ = softmax of image i at zero load (Phase 1)
STER = (1/N) Σᵢ 𝟙[δᵢ > T*]

Matches E0/E2/E3 Jetson GPU δ definition exactly.
Same 500 images from ~/e0_experiment/data/manifest.json.
Platform: Jetson Orin Nano Super, ONNX Runtime CPUExecutionProvider, MobileNetV2 FP32.
"""

import os, sys, json, time, subprocess, numpy as np, csv
from pathlib import Path
from datetime import datetime

HOME        = Path.home()
E6_DIR      = HOME / "e6_experiment"
RESULTS_DIR = E6_DIR / "results"
CONFIG_PATH = E6_DIR / "e6_config.json"
MANIFEST    = HOME / "e0_experiment" / "data" / "manifest.json"

N_INFER      = 500
N_TRIALS_E0  = 10   # zero-load reference trials (δ should be ~0 — sanity check)
N_TRIALS_E1  = 10   # trials per CPU load level
N_TRIALS_E5  = 10   # combined load trials
T_STAR       = 0.05
T_STAR_EXTRA = [0.01, 0.02, 0.03]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import onnxruntime as ort
from torchvision import transforms
from PIL import Image

with open(CONFIG_PATH) as f:
    config = json.load(f)
ONNX_PATH  = config["onnx_path"]
INPUT_NAME = config["input_name"]

print(f"Loading ONNX model: {ONNX_PATH}")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print(f"Provider: {sess.get_providers()}")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ── Load and preprocess images ────────────────────────────────────────────────
print("Loading manifest and preprocessing images...")
with open(MANIFEST) as f:
    manifest = json.load(f)
entries = (manifest if isinstance(manifest, list)
           else manifest.get("images", manifest))[:N_INFER]

images = []
for entry in entries:
    img_path = entry if isinstance(entry, str) else entry["path"]
    img = Image.open(img_path).convert("RGB")
    images.append(preprocess(img).unsqueeze(0).numpy().astype(np.float32))
print(f"Loaded {len(images)} images")

# ── Phase 1: build per-image reference at zero load ──────────────────────────
print("\nPhase 1: computing per-image reference softmax at zero load...")
reference = np.zeros((N_INFER, 1000), dtype=np.float32)
for i, inp in enumerate(images):
    logits = sess.run(None, {INPUT_NAME: inp})[0][0]
    reference[i] = softmax(logits)

ref_path = E6_DIR / "e6_cpu_reference.npy"
np.save(str(ref_path), reference)
print(f"Reference saved: {ref_path}")
print(f"  mean top-1 score: {reference.max(axis=1).mean():.4f}")
print(f"  mean softmax max: {reference.max(axis=1).mean():.4f}")

# ── Stressor management ───────────────────────────────────────────────────────
class Stressor:
    def __init__(self): self.procs = []

    def start_cpu(self, pct):
        n = os.cpu_count()
        p = subprocess.Popen(
            ["stress-ng","--cpu",str(n),"--cpu-load",str(pct),"--timeout","0","-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p); time.sleep(1.0)
        print(f"  CPU stressor: {pct}% (PID {p.pid})")

    def start_mem(self, pct):
        import psutil
        fill_mb = int(psutil.virtual_memory().total / 1e6 * pct / 100)
        p = subprocess.Popen(
            ["stress-ng","--vm","1","--vm-bytes",f"{fill_mb}M","--timeout","0","-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p); time.sleep(1.0)
        print(f"  MEM stressor: {pct}% / {fill_mb}MB (PID {p.pid})")

    def start_disk(self):
        p = subprocess.Popen(
            ["fio","--name=e6_disk","--filename=/tmp/e6_fio_v3",
             "--rw=write","--bs=1M","--size=2G","--numjobs=1",
             "--runtime=3600","--time_based","--ioengine=libaio",
             "--direct=1","--group_reporting","--output=/dev/null"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p); time.sleep(1.0)
        print(f"  Disk stressor: fio (PID {p.pid})")

    def stop_all(self):
        for p in self.procs:
            try: p.terminate(); p.wait(timeout=3)
            except:
                try: p.kill()
                except: pass
        self.procs.clear(); time.sleep(0.5)

# ── Core inference trial ──────────────────────────────────────────────────────
def run_trial(trial_id, condition_label):
    """
    Run N_INFER inferences. δᵢ = L-inf(σᵢ − reference[i]).
    Returns list of row dicts.
    """
    rows = []
    for i, inp in enumerate(images):
        t0 = time.perf_counter()
        logits = sess.run(None, {INPUT_NAME: inp})[0][0]
        t1 = time.perf_counter()

        sm    = softmax(logits)
        delta = float(np.max(np.abs(sm - reference[i])))

        rows.append({
            "trial":         trial_id,
            "condition":     condition_label,
            "inference_idx": i,
            "delta":         delta,
            "top1_class":    int(np.argmax(sm)),
            "latency_ms":    round((t1 - t0) * 1000, 3),
        })
    return rows

# ── Metrics helpers ───────────────────────────────────────────────────────────
def ster(rows, t=T_STAR):
    exc = sum(1 for r in rows if r["delta"] > t)
    return exc / len(rows), exc, len(rows)

def print_summary(label, rows):
    deltas = [r["delta"] for r in rows]
    lats   = [r["latency_ms"] for r in rows]
    s, exc, n = ster(rows)
    print(f"    {label}: STER={s:.4f} ({exc}/{n}) | "
          f"δ_mean={np.mean(deltas):.4f} δ_P99={np.percentile(deltas,99):.4f} "
          f"δ_max={np.max(deltas):.4f} | latency={np.mean(lats):.1f}ms")
    for t_ in T_STAR_EXTRA:
        s2, e2, _ = ster(rows, t_)
        print(f"      T*={t_:.2f}: STER={s2:.4f} ({e2}/{n})")

def write_csv(rows, path):
    fields = ["trial","condition","inference_idx","delta","top1_class","latency_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"    Wrote {len(rows)} rows → {path}")

# ── E0: zero-load reference validation ───────────────────────────────────────
print("\n" + "="*60)
print("E0 — zero-load validation (δ should be ~0 — confirms reference)")
print("="*60)
e0_rows = []
for trial in range(N_TRIALS_E0):
    print(f"  Trial {trial+1}/{N_TRIALS_E0}...")
    rows = run_trial(trial, "E0_zero_load")
    e0_rows.extend(rows)
    print_summary(f"  T{trial+1}", rows)
write_csv(e0_rows, RESULTS_DIR / "e6v3_cpu_e0.csv")

# ── E1: CPU stress ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("E1 — CPU stress (25/50/75/100%)")
print("="*60)
e1_rows = []
for pct in [25, 50, 75, 100]:
    print(f"\n  CPU load: {pct}%")
    s = Stressor(); s.start_cpu(pct)
    for trial in range(N_TRIALS_E1):
        print(f"  Trial {trial+1}/{N_TRIALS_E1} @ {pct}%...")
        rows = run_trial(trial, f"E1_cpu{pct}pct")
        e1_rows.extend(rows)
        print_summary(f"  T{trial+1}", rows)
    s.stop_all()
    print(f"  Stressor stopped.")
write_csv(e1_rows, RESULTS_DIR / "e6v3_cpu_e1.csv")

# ── E5: combined load ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("E5 — Combined load (CPU75% + MEM50% + disk)")
print("="*60)
e5_rows = []
s = Stressor(); s.start_cpu(75); s.start_mem(50); s.start_disk()
print("  All stressors active. Waiting 3s for stability...")
time.sleep(3.0)
for trial in range(N_TRIALS_E5):
    print(f"  Trial {trial+1}/{N_TRIALS_E5}...")
    rows = run_trial(trial, "E5_combined")
    e5_rows.extend(rows)
    print_summary(f"  T{trial+1}", rows)
s.stop_all()
write_csv(e5_rows, RESULTS_DIR / "e6v3_cpu_e5.csv")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

def agg(rows, label):
    deltas = [r["delta"] for r in rows]
    lats   = [r["latency_ms"] for r in rows]
    s, exc, n = ster(rows)
    res = {
        "condition":       label,
        "n_inferences":    n,
        "STER_T005":       round(s, 6),
        "exceedances":     exc,
        "delta_mean":      round(float(np.mean(deltas)), 6),
        "delta_p99":       round(float(np.percentile(deltas, 99)), 6),
        "delta_max":       round(float(np.max(deltas)), 6),
        "latency_mean_ms": round(float(np.mean(lats)), 1),
        "latency_p99_ms":  round(float(np.percentile(lats, 99)), 1),
    }
    for t_ in T_STAR_EXTRA:
        s2, e2, _ = ster(rows, t_)
        res[f"STER_T{int(t_*100):02d}"] = round(s2, 6)
    return res

summary = {
    "experiment":  "E6_CPU_v3",
    "timestamp":   datetime.now().isoformat(),
    "platform":    "Jetson Orin Nano Super — ONNX Runtime CPUExecutionProvider",
    "model":       "MobileNetV2 FP32 ONNX",
    "delta_def":   "per-image L-inf: delta_i = max|sigma_i_condition - sigma_i_reference|",
    "reference":   "single zero-load inference per image (Phase 1, no stressor)",
    "n_images":    N_INFER,
    "T_star":      T_STAR,
    "results": [
        agg(e0_rows, "E0_CPU_zero_load"),
        *[agg([r for r in e1_rows if f"cpu{p}pct" in r["condition"]],
              f"E1_CPU_{p}pct_load") for p in [25,50,75,100]],
        agg(e5_rows, "E5_CPU_combined"),
    ]
}

summary_path = RESULTS_DIR / "e6v3_results_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary → {summary_path}\n")
for r in summary["results"]:
    print(f"  {r['condition']}")
    print(f"    STER(T*=0.05)={r['STER_T005']}  exceedances={r['exceedances']}/{r['n_inferences']}")
    print(f"    δ_mean={r['delta_mean']}  δ_P99={r['delta_p99']}  δ_max={r['delta_max']}")
    print(f"    latency_mean={r['latency_mean_ms']}ms  latency_P99={r['latency_p99_ms']}ms\n")

print("✅  E6 v3 complete.")
