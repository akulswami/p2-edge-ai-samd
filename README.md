# P2 — Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**Submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Paper status:** Submission-ready — 4-page IEEEtran LaTeX, all review fixes applied

---

## Paper Summary

This letter demonstrates that accuracy-based pre-market validation can certify systems that violate timing constraints under deployment load, and introduces STER to isolate output stability from temporal safety — two properties current validation does not distinguish.

**The novelty is not timing degradation** — that is known — but the validation blindness to it: a system passes accuracy-based certification while being undeployable at the required clinical activation rate, and no current FDA pre-market method detects this.

**Experimental design:** The same MobileNetV2 model is evaluated under identical adversarial load on two execution paths of the same physical hardware — a dedicated GPU accelerator (TensorRT FP16) and a general-purpose CPU (ONNX Runtime FP32) — on the NVIDIA Jetson Orin Nano Super. All hardware variables are held constant so any observed divergence arises from execution context, isolating the software execution path as the operative factor.

**Central finding:** Both paths maintain STER = 0.0000 (zero output exceedances above T* = 0.05) across ~158,000 inference activations. The paths diverge sharply on timing: the GPU path maintains latency <15 ms under all conditions; the CPU path degrades 7.2× under combined load (14.5 ms → 104.0 ms mean, P99 = 165.1 ms), breaching the 10 Hz clinical cycle budget by 65%. This violation persists undetected under accuracy-based validation (Δacc < 1.0%).

**Regulatory anchor:** FDA Draft Guidance FDA-2024-D-4488 (January 2025) requires robustness assessment for foreseeable conditions of use. Joint STER + latency verification is proposed as a concrete pre-market acceptance protocol operationalizing that requirement, consistent with IEC 62304 and ISO 14971.

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate:**

```
STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]
```

where `δᵢ = ‖σ(yᵢ) − σ̄ᵢ‖∞` is the per-image L-infinity norm on the softmax probability vector vs. the per-image zero-load reference, and `T* = 0.05`.

**Why STER ≠ accuracy:** Accuracy evaluates argmax(σ(y)) — a single class label — and cannot detect bounded distribution drift below the argmax threshold. This is a structural limitation, not a sensitivity issue. STER captures this class of silent deviations; accuracy cannot by construction. STER = 0.0000 on both paths confirms output instability is not the failure mode, isolating timing degradation as the sole safety-relevant divergence.

---

## Hardware Platforms

| Platform | Role | Key Specification |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB (GPU path) | Primary inference host | 1024-core Ampere GPU, TensorRT FP16, shared LPDDR5 |
| NVIDIA Jetson Orin Nano Super 8GB (CPU path) | Negative control — same hardware | ONNX Runtime FP32, CPUExecutionProvider, 6-core Arm Cortex-A78AE |
| nRF52840 DK | BLE network stressor | Cortex-M4, BLE 5.0, multi-connection central |
| TI Bluetooth SensorTag | Second BLE node | Patient-analog wearable stressor |
| GL-AX1800 WiFi 6 Router | RF environment control | 2.4 GHz co-channel interference |
| Insignia Dock + 2× 298GB HDD | Disk I/O stressor | fio sequential writes, USB-attached |

**Note:** Coral Dev Board Micro results preserved in git history (commit 3dcd63f) for companion paper P4.

---

## Experiment Status

| ID | Stressor | GPU Path | CPU Path (E6) | STER Result |
|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ | ✅ | 0.0000 both; latency matched ~14.8 ms |
| **E1** | CPU contention (0–100%) | ✅ | ✅ | 0.0000 both; GPU stable, CPU 14.5→73.7 ms |
| **E2** | Memory pressure (25–90%) | ✅ | — | 0.0000 GPU |
| **E3** | GPU co-tenancy (0–4 workers) | ✅ | — | 0.0000 GPU |
| **E4** | Network I/O — BLE/WiFi | ✅ | — | 0.0000 GPU |
| **E5** | Combined realistic load | ✅ | ✅ | 0.0000 both; GPU <15 ms, CPU 104 ms mean / 165 ms P99 |
| **E6** | CPU-only negative control | — | ✅ | 0.0000 STER; 7.2× latency degradation — timing violation confirmed |

**Total inference activations:** ~158,000 (GPU path) + ~50,000 (CPU path) = ~208,000 collected.

---

## Key Results

### E6 — CPU Negative Control (Primary Contrastive Result)

| Condition | STER | δ_mean | Latency_mean | Latency_P99 |
|---|---|---|---|---|
| Zero load | 0.0000 | 0.0000 | 14.5 ms | 18.6 ms |
| CPU 25% | 0.0000 | 0.0000 | 33.4 ms | 111.4 ms |
| CPU 50% | 0.0000 | 0.0000 | 57.8 ms | 115.8 ms |
| CPU 75% | 0.0000 | 0.0000 | 73.7 ms | 122.1 ms |
| CPU 100% | 0.0000 | 0.0000 | 70.9 ms | 117.0 ms |
| **Combined** | **0.0000** | **0.0000** | **104.0 ms** | **165.1 ms ⚠️** |

⚠️ P99 = 165.1 ms exceeds 10 Hz cycle budget (100 ms) by 65%. STER passes; deployment timing fails. Validation blindness confirmed: Δacc < 1.0% under identical conditions.

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── data/
│   │   └── prepare_dataset.py
│   ├── e0_jetson.py              E0 baseline (TensorRT FP16, GPU path)
│   ├── e1_jetson.py              E1 CPU stress (GPU path)
│   ├── e2_jetson.py              E2 memory pressure (GPU path)
│   ├── e3_jetson.py              E3 GPU co-tenancy
│   ├── e3_worker.py              E3 co-tenant worker process
│   ├── e4_jetson.py              E4 network I/O (GPU path)
│   ├── e5_jetson.py              E5 combined load (GPU path)
│   ├── e6_jetson_cpu.py          E6 CPU-only negative control (ONNX FP32)
│   └── results/
│       ├── e0_jetson.csv
│       ├── e1_jetson.csv
│       ├── e1_run.log
│       ├── e2_jetson.csv
│       ├── e3_jetson.csv
│       ├── e4_jetson_conns0.csv
│       ├── e4_jetson_conns2.csv
│       ├── e4_jetson_conns4.csv
│       ├── e4_jetson_conns6.csv
│       ├── e5_results_summary.json
│       └── e6_results_summary.json
├── coral/                             Preserved for companion paper P4
│   ├── coral_capture.py               Shared serial capture utility
│   ├── e0_infer_baseline/             E0 firmware source (MobileNetV1 int8)
│   │   ├── e0_infer_baseline.cc
│   │   ├── CMakeLists_e0_infer_baseline.txt
│   │   ├── analyze_e0_coral_infer.py
│   │   └── Experiment0_Coral_Final.txt
│   ├── e1_cpu_stress/
│   │   ├── e1_coral.py
│   │   └── results/
│   │       └── e1_coral.csv
│   ├── e2_mem_pressure/
│   │   ├── e2_coral.py
│   │   └── results/
│   │       └── e2_coral.csv
│   ├── e4_coral.py                    E4 BLE/WiFi stressor
│   ├── e5_coral.py                    E5 combined load
│   ├── supporting_timing_baseline/
│   └── results/
│       ├── e0_coral_summary.csv
│       ├── e0_coral_infer_summary.csv
│       ├── e0_infer_log.txt
│       ├── e4_coral_conns0.csv
│       ├── e4_coral_conns2.csv
│       ├── e4_coral_conns4.csv
│       ├── e4_coral_conns6.csv
│       └── e5_coral_results.json
└── paper/
    ├── P2_IEEE_ESL_Draft_E6.docx       Early draft (pre-pivot, archived)
    ├── P2_IEEE_ESL_Draft_Pivot.docx    Word draft (pivot framing)
    ├── p2_paper_submission.tex         Submission-ready LaTeX source ← LATEST
    └── p2_paper_submission.pdf         Compiled submission PDF ← LATEST
```

---

## Jetson Setup

- **IP:** 192.168.8.102 | **User:** akulswami | **Venv:** `~/e0_env`
- **OS:** JetPack 6 (R36.4.4), CUDA 12.6, TensorRT 10.3.0
- **GPU model:** `~/e0_experiment/models/mobilenetv2_fp16.trt`
- **CPU model:** `~/e6_experiment/models/mobilenetv2_cpu.onnx`
- **Dataset:** 500 Tiny ImageNet images, `~/e0_experiment/data/manifest.json`
- **E6 reference:** `~/e6_experiment/e6_cpu_reference.npy` (500×1000 per-image softmax at zero load)

---

## BLE / nRF Setup

- **nRF52840 DK** connected to Ubuntu via USB (E4/E5 stressor)
- **JLink symlink fix:**
  ```bash
  sudo ln -sf /opt/SEGGER/JLink/libjlinkarm.so.7.94.5 /opt/SEGGER/JLink/libjlinkarm.so.7
  ```
- **E4/E5 firmware:** `e4_conns4.hex` — BLE central, 4 simultaneous connections

---

## Paper Status

**Submission files:** `paper/p2_paper_submission.tex` and `paper/p2_paper_submission.pdf`

| Section | Status | Notes |
|---|---|---|
| Title | ✅ Final | Architectural isolation framing |
| Abstract | ✅ Final | Opens with unifying sentence: "A system can satisfy accuracy-based validation... and still violate timing constraints" |
| Introduction | ✅ Final | Paradigm failure stated, kill shot added ("novelty is validation blindness"), contribution collapsed to single claim |
| Related Work | ✅ Final | Problem class separation vs. Martín et al. [4] explicit |
| System Model (Sec. III) | ✅ Final | Formal violation definition + data linkage + operational anchor + downstream consequence (N9→N10→N13→N16 chain) |
| Hardware & Protocol (Sec. IV) | ✅ Final | Thermal monitoring confirmed, determinism declared, stats statement |
| Results (Sec. V) | ✅ Final | E0–E6; obvious result preemption at section end |
| Discussion (Sec. VI) | ✅ Final | Contribution lock opens section; engineering evidence scoped; repetition trimmed |
| Conclusion (Sec. VII) | ✅ Final | Output correctness alone insufficient; temporal validation required |
| References [1]–[15] | ✅ Final | 15 refs, all on page 4 |
| Page count | ✅ 4 pages | IEEEtran two-column, 10pt, letter format |

**Remaining before May 20:**
1. Submit via IEEE Author Portal: ieee.atyponrex.com/journal/les-ieee
2. Simultaneous arXiv upload — account: akulswami, primary category: eess.SY, cross-list: cs.AR
3. Update six_research_papers_v2.docx P2 entry to match final framing

---

## Key Design Decisions

**Why same-hardware design is the methodological contribution:**
Cross-platform comparisons leave hardware differences as alternative explanations for any observed divergence. Running GPU and CPU paths on the same Jetson Orin Nano Super holds all hardware variables constant. Any observed divergence therefore arises from execution context, isolating the software execution path as the operative factor. This is not a performance comparison; it is a controlled experiment exposing a validation failure.

**Why the contribution is validation blindness, not architectural characterization:**
The claim is not "GPU is faster than CPU under load" — that is known. The claim is that current FDA pre-market validation certifies both the GPU-accelerated and CPU-only deployments as equivalent (Δacc < 1.0%), while one violates timing constraints by 65% under foreseeable deployment conditions. The novelty is the validation failure.

**Why STER = 0.0000 on both paths is not a null result:**
It establishes that output instability is not the failure mode under realistic contention. This makes timing degradation the sole safety-relevant divergence — and makes the validation blindness claim clean. Without STER confirming output stability independently, timing failures cannot be isolated from correctness failures by construction.

---

## Dependencies

**Jetson (Python 3.10, venv: `~/e0_env`):**
```
tensorrt==10.3.0
pycuda
onnxruntime
Pillow
numpy
torchvision
psutil
```

**Ubuntu host (Python 3.10):**
```
pyserial
numpy
psutil
stress-ng   (apt)
fio         (apt)
```

---

## Citation

> A. M. Swami, "Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices: Controlled Experimental Evidence on a Shared-Silicon Platform," *IEEE Embedded Systems Letters*, submitted May 2026.

---

## Author

**Akul Mallayya Swami**  
Varian Medical Systems, A Siemens Healthineers Company, Palo Alto, CA, USA  
swami.akul@alumni.uml.edu | ORCID: 0009-0003-9549-5543
