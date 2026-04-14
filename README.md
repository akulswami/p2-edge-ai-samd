# P2 — Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**Submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Paper status:** Pivot complete — controlled isolation experiment framing, Coral removed, LaTeX conversion in progress

---

## Paper Summary

This letter demonstrates, through a controlled same-hardware experiment, that output correctness and timing reliability are independent safety properties of edge AI inference pipelines — and that dedicated inference accelerators provide timing isolation as a distinct architectural safety primitive.

**Experimental design:** The same MobileNetV2 model is evaluated under identical adversarial load on two execution paths of the same physical hardware — a dedicated GPU accelerator (TensorRT FP16) and a general-purpose CPU (ONNX Runtime FP32) — on the NVIDIA Jetson Orin Nano Super. Using the same physical device for both paths eliminates platform confounds and enables causal attribution of timing behavior to the accelerator architecture specifically.

**Central finding:** Both paths maintain STER = 0.0000 (zero output exceedances above T* = 0.05) across ~158,000 inference activations. The paths diverge sharply on timing: the GPU path maintains latency <15 ms under all conditions; the CPU path degrades 7.2× under combined load (14.5 ms → 104.0 ms mean, P99 = 165.1 ms), breaching the 10 Hz clinical cycle budget by 65%.

**Causal claim:** Because same-hardware design eliminates chip-to-chip variation as a confound, timing divergence is causally attributable to the dedicated accelerator architecture. A device can be output-correct and still clinically unsafe.

**Regulatory anchor:** FDA Draft Guidance FDA-2024-D-4488 (January 2025) requires robustness assessment for reasonably foreseeable conditions of use. Joint STER + latency verification under foreseeable stressor conditions is proposed as a concrete pre-market acceptance protocol operationalizing that requirement.

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate:**
STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]

where `δᵢ = ‖σ(yᵢ) − σ̄ᵢ‖∞` is the per-image L-infinity norm on the softmax probability vector vs. the per-image zero-load reference, and `T* = 0.05`.

**Why STER ≠ accuracy:** Accuracy is argmax(σ(y)) — a single label. STER is a function of the full distribution σ(y). These are structurally distinct. STER = 0.0000 on both paths is the experimental control confirming output instability is not the failure mode — isolating timing degradation as the sole safety-relevant divergence.

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
| **E6** | CPU-only negative control | — | ✅ | 0.0000 STER; 7.2× latency degradation |

**Total inference activations:** ~158,000 (GPU path) + ~50,000 (CPU path) = ~208,000 collected.

---

## Key Results

### E6 — CPU Negative Control

| Condition | STER | δ_mean | Latency_mean | Latency_P99 |
|---|---|---|---|---|
| Zero load | 0.0000 | 0.0000 | 14.5 ms | 18.6 ms |
| CPU 25% | 0.0000 | 0.0000 | 33.4 ms | 111.4 ms |
| CPU 50% | 0.0000 | 0.0000 | 57.8 ms | 115.8 ms |
| CPU 75% | 0.0000 | 0.0000 | 73.7 ms | 122.1 ms |
| CPU 100% | 0.0000 | 0.0000 | 70.9 ms | 117.0 ms |
| **Combined** | **0.0000** | **0.0000** | **104.0 ms** | **165.1 ms ⚠️** |

⚠️ P99 = 165.1 ms exceeds 10 Hz cycle budget (100 ms) by 65%. STER passes; deployment fails.

---

## Repository Structure
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py              E0 baseline (TensorRT FP16)
│   ├── e1_jetson.py              E1 CPU stress (GPU path)
│   ├── e2_jetson.py              E2 memory pressure (GPU path)
│   ├── e3_jetson.py              E3 GPU co-tenancy
│   ├── e3_worker.py              E3 co-tenant worker
│   ├── e4_jetson.py              E4 network I/O (GPU path)
│   ├── e5_jetson.py              E5 combined load (GPU path)
│   ├── e6_jetson_cpu.py          E6 CPU-only negative control (ONNX FP32)
│   └── results/
│       ├── e0_jetson.csv
│       ├── e1_jetson.csv
│       ├── e2_jetson.csv
│       ├── e3_jetson.csv
│       ├── e4_jetson_conns{0,2,4,6}.csv
│       ├── e5_results_summary.json
│       └── e6_results_summary.json
├── coral/                        Preserved for P4
└── paper/
└── P2_IEEE_ESL_Draft_Pivot.docx    Current draft ← LATEST

---

## Jetson Setup

- **IP:** 192.168.8.102 | **User:** akulswami | **Venv:** `~/e0_env`
- **GPU model:** `~/e0_experiment/models/mobilenetv2_fp16.trt`
- **CPU model:** `~/e6_experiment/models/mobilenetv2_cpu.onnx`
- **Dataset:** 500 Tiny ImageNet images, `~/e0_experiment/data/manifest.json`
- **E6 reference:** `~/e6_experiment/e6_cpu_reference.npy` (500×1000 per-image softmax)

---

## Paper Status

**Current draft:** `paper/P2_IEEE_ESL_Draft_Pivot.docx`

All sections updated for pivot framing. Remaining: LaTeX conversion (IEEEtran, 4-page limit), submission via IEEE Author Portal, simultaneous arXiv upload.

---

## Key Design Decision: Why Same-Hardware Design

Cross-platform comparisons cannot establish causality because hardware differences confound the result. Running GPU and CPU paths on the same Jetson Orin Nano Super eliminates all hardware-level confounds. The 7.2× timing divergence under identical stressors is therefore causally attributable to the accelerator architecture — the methodological contribution that distinguishes this paper from prior benchmarking work.

---

## Citation

> A. M. Swami, "Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices: Controlled Experimental Evidence on a Shared-Silicon Platform," *IEEE Embedded Systems Letters*, submitted May 2026.

---

## Author

**Akul Mallayya Swami**  
Varian Medical Systems, A Siemens Healthineers Company, Palo Alto, CA, USA  
swami.akul@alumni.uml.edu | ORCID: 0009-0003-9549-5543
