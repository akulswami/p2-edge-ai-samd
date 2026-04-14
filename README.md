# P2 — Runtime Inference Stability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**Submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Paper status:** E0–E6 complete on all three execution paths — narrative reframe in progress for ESL submission

---

## Paper Summary

This letter empirically characterizes inference output stability and timing stability under resource contention across three execution paths: the NVIDIA Jetson Orin Nano Super (TensorRT FP16), the Coral Dev Board Micro (Edge TPU int8), and a general-purpose CPU baseline (ONNX Runtime FP32 on the same Jetson hardware). It introduces the Safety-Threshold Exceedance Rate (STER) as a formally-defined robustness verification metric for FDA-regulated Software as a Medical Device (SaMD).

**Central finding:** All three execution paths achieve STER = 0.0000 with zero output exceedances across approximately 208,000 inference activations under five stressor classes. The CPU-only negative control (E6) is the critical distinguishing result: it preserves output stability (STER = 0.0000, δ = 0.0000) but exhibits 7.2× latency degradation under combined load (14.5 ms → 104.0 ms), with P99 latency reaching 165.1 ms — exceeding the 10 Hz clinical cycle budget by 65%. This establishes that dedicated inference accelerators contribute **timing isolation** as a distinct safety property that output stability metrics alone cannot capture.

**Contribution framing (post-review reframe):** The paper is a robustness validation and characterization paper, not a failure-detection paper. STER = 0.0000 confirms that modern dedicated edge AI accelerators pass the metric under realistic contention. The CPU negative control demonstrates what breaks when the accelerator is removed — timing stability — and establishes the boundary of STER's coverage.

**Regulatory anchor:** FDA Draft Guidance FDA-2024-D-4488 (January 2025) requires robustness assessment for reasonably foreseeable conditions of use. STER is proposed as a candidate empirical method for operationalizing that requirement.

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate:**

```
STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]
```

where `δᵢ = ‖σ(yᵢ) − σ̄ᵢ‖∞` is the per-image L-infinity norm on the softmax probability vector vs. the per-image zero-load reference, and `T* = 0.05` (±5% relative deviation, conservative relative to ISO 14971 Table B.1).

**δ definition note:** δ is computed per-image against a reference softmax captured at zero load for that same image. This is consistent across E0/E2/E3/E6. E1 uses consecutive-frame L-inf (equivalent for deterministic hardware — confirmed empirically: δ_mean identical at all load levels). E4/E5 use a mean baseline (slightly different scale, ~0.0156 vs ~0.0181); STER conclusion identical (zero exceedances) across all definitions.

**Coral platform note:** The Edge TPU produces deterministic int8 outputs (δ = 0.0000 exactly). STER on Coral is characterized via class identity stability: `STER = fraction of inferences where top-1 class ≠ REF_CLASS`. `REF_CLASS = 905`, `REF_SCORE = 0.320312`.

---

## Hardware Platforms

| Platform | Role | Key Specification |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB | GPU inference host | 1024-core Ampere GPU, 67 TOPS, shared LPDDR5, TensorRT FP16 |
| NVIDIA Jetson Orin Nano Super 8GB | CPU negative control | Same hardware, ONNX Runtime FP32, CPUExecutionProvider (no GPU) |
| Coral Dev Board Micro + Env. Sensor Board | Dedicated TPU inference host | Google Edge TPU, 4 TOPS, on-chip SRAM, TFLite int8 |
| nRF52840 DK | BLE peripheral / network stressor | Cortex-M4, BLE 5.0, multi-connection central |
| TI Bluetooth SensorTag | Second BLE data source | Patient-analog wearable stressor node |
| GL-AX1800 WiFi 6 Router | RF environment control | 2.4 GHz / 5 GHz co-channel interference |
| Insignia Dock + 2× 298GB HDD | Disk I/O stressor | fio sequential writes, USB-attached |

---

## Experiment Status

| ID | Stressor | Jetson GPU | Coral | Jetson CPU (E6) | STER Result |
|---|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ Complete | ✅ Complete | ✅ Complete | 0.0000 all paths |
| **E1** | CPU contention (0–100%) | ✅ Complete | ✅ Complete | ✅ Complete | 0.0000 all paths |
| **E2** | Memory pressure (25–90%) | ✅ Complete | ✅ Complete | — | 0.0000 both platforms |
| **E3** | GPU co-tenancy (0–4 workers) | ✅ Complete | N/A (architectural) | N/A | 0.0000 Jetson |
| **E4** | Network I/O — BLE/WiFi | ✅ Complete | ✅ Complete | — | 0.0000 both platforms |
| **E5** | Combined realistic load | ✅ Complete | ✅ Complete | ✅ Complete | 0.0000 all paths |
| **E6** | CPU-only negative control | — | — | ✅ Complete | 0.0000 STER; 7.2× latency degradation |

**Total inference activations:** ~208,000 across all experiments and all execution paths.  
**STER exceedances:** Zero across all conditions on all paths.

---

## Confirmed Results

### E0 — Baseline Calibration

| Platform | STER_nominal | δ_mean | δ_P99 | Latency_mean |
|---|---|---|---|---|
| Jetson GPU (TensorRT FP16) | 0.0000 | 0.0181 | 0.0233 | ~15 ms |
| Coral Edge TPU (int8) | 0.0000 | 0.0000 | 0.0000 | — |
| Jetson CPU (ONNX FP32) | 0.0000 | 0.0000 | 0.0000 | 14.5 ms |

### E1 — CPU Contention

| CPU Load | Jetson GPU STER | Coral STER | Jetson CPU STER | Jetson CPU Latency |
|---|---|---|---|---|
| 25% | 0.0000 | 0.0000 | 0.0000 | 33.4 ms |
| 50% | 0.0000 | 0.0000 | 0.0000 | 57.8 ms |
| 75% | 0.0000 | 0.0000 | 0.0000 | 73.7 ms |
| 100% | 0.0000 | 0.0000 | 0.0000 | 70.9 ms |

### E5 — Combined Realistic Deployment Load

| Platform | STER | δ_mean | Latency_mean | Latency_P99 |
|---|---|---|---|---|
| Jetson GPU | 0.0000 | 0.0156 | ~15 ms | — |
| Coral | 0.0000 | 0.0000 | — | — |
| Jetson CPU | 0.0000 | 0.0000 | 104.0 ms | 165.1 ms ⚠️ |

⚠️ CPU combined latency P99 = 165.1 ms exceeds 10 Hz cycle budget (100 ms) by 65%. STER = 0.0000 but timing fails — the key negative control finding.

### E6 — CPU-Only Full Summary

| Condition | STER | δ_mean | δ_max | Latency_mean | Latency_P99 |
|---|---|---|---|---|---|
| Zero load | 0.0000 | 0.0000 | 0.0000 | 14.5 ms | 18.6 ms |
| CPU 25% | 0.0000 | 0.0000 | 0.0000 | 33.4 ms | 111.4 ms |
| CPU 50% | 0.0000 | 0.0000 | 0.0000 | 57.8 ms | 115.8 ms |
| CPU 75% | 0.0000 | 0.0000 | 0.0000 | 73.7 ms | 122.1 ms |
| CPU 100% | 0.0000 | 0.0000 | 0.0000 | 70.9 ms | 117.0 ms |
| Combined | 0.0000 | 0.0000 | 0.0000 | 104.0 ms | 165.1 ms |

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py              E0 baseline inference (TensorRT FP16)
│   ├── e1_jetson.py              E1 CPU stress (TensorRT FP16)
│   ├── e2_jetson.py              E2 memory pressure (TensorRT FP16)
│   ├── e3_jetson.py              E3 GPU co-tenancy
│   ├── e3_worker.py              E3 co-tenant worker process
│   ├── e4_jetson.py              E4 network I/O (TensorRT FP16)
│   ├── e5_jetson.py              E5 combined load (TensorRT FP16)
│   ├── e6_jetson_cpu.py          E6 CPU-only negative control (ONNX FP32) ← NEW
│   └── results/
│       ├── e0_jetson.csv
│       ├── e1_jetson.csv
│       ├── e2_jetson.csv
│       ├── e3_jetson.csv
│       ├── e4_jetson_conns{0,2,4,6}.csv
│       ├── e5_results_summary.json
│       └── e6_results_summary.json         ← NEW
├── coral/
│   ├── coral_capture.py          Shared serial capture utility
│   ├── e0_infer_baseline/        E0 firmware (MobileNetV1 int8, 2000 inferences)
│   ├── e1_coral.py
│   ├── e2_coral.py
│   ├── e4_coral.py
│   ├── e5_coral.py
│   └── results/
│       ├── e1_coral.csv
│       ├── e2_coral.csv
│       ├── e4_coral_conns{0,2,4,6}.csv
│       └── e5_coral_results.json
└── paper/
    └── P2_IEEE_ESL_Draft_E6.docx    Current draft (E0–E6, reframe in progress) ← NEW
```

---

## Jetson Setup

- **IP:** 192.168.8.102 (GL-AX1800 network)
- **User:** akulswami
- **OS:** JetPack 6 (R36.4.4), CUDA 12.6, TensorRT 10.3.0
- **Venv:** `~/e0_env` → `source ~/e0_env/bin/activate`
- **GPU model:** `~/e0_experiment/models/mobilenetv2_fp16.trt`
- **CPU model:** `~/e6_experiment/models/mobilenetv2_cpu.onnx` (MobileNetV2 FP32)
- **Dataset:** 500 Tiny ImageNet images, manifest at `~/e0_experiment/data/manifest.json`
- **E6 reference:** `~/e6_experiment/e6_cpu_reference.npy` (500×1000 per-image softmax at zero load)

Key script notes:
- Manifest is plain JSON list — use `manifest[:N_INFER]`, not `manifest['images']`
- Image paths are absolute — use `e['path']` directly
- E6 δ uses per-image reference (correct); E4/E5 use mean baseline (different scale, identical STER)

---

## Coral Setup

- **Connection:** Ubuntu via USB-C → `/dev/ttyACM2` (when nRF DK also connected)
- **Firmware:** `e0_infer_baseline` (MobileNetV1 int8, 2000 inferences, synthetic input)
- **Serial capture:** `coral/coral_capture.py`
- **REF_CLASS:** 905, **REF_SCORE:** 0.320312
- **Serial output format:** `infer,<iteration>,<class_id>,<score>`
- **n_captured note:** E4/E5 capture ~1400 inferences per trial; STER = 0.0000 confirmed across all

---

## BLE / nRF Setup

- **nRF52840 DK** connected to Ubuntu via USB
- **JLink symlink fix** (required before each flash):
  ```bash
  sudo ln -sf /opt/SEGGER/JLink/libjlinkarm.so.7.94.5 /opt/SEGGER/JLink/libjlinkarm.so.7
  ```
- **E4/E5 firmware:** `e4_conns4.hex` — BLE central, 4 simultaneous connections
- **Flash command:**
  ```bash
  nrfjprog --program ~/e4_experiment/firmware/hex/e4_conns4.hex --chiperase --verify -f NRF52 && nrfjprog --reset -f NRF52
  ```
- **Confirmation:** Serial monitor should show `STATUS=4/4` before running inference

---

## Paper Status

**Current draft:** `paper/P2_IEEE_ESL_Draft_E6.docx`

| Section | Status |
|---|---|
| Abstract | ✅ Updated — three execution paths, 208k activations, timing finding |
| Introduction + Contributions | ✅ Updated — CPU negative control bullet added |
| Related Work | ✅ Complete |
| System Model (Sec. III) | ✅ Complete |
| Hardware & Protocol (Sec. IV) | ✅ Updated — three execution paths described |
| Results E0–E5 (Sec. V.A–F) | ✅ Complete with real data |
| Results E6 (Sec. V.F new) | ✅ Added — Table VIII-B + full narrative |
| Summary Table IX | ✅ Complete |
| Discussion (Sec. VI) | ✅ New Sec. VI.C — CPU negative control argument |
| Conclusion (Sec. VII) | ✅ Updated — three platforms, timing finding |
| Acknowledgements | ✅ Complete |
| References [1]–[15] | ✅ Complete |

**Remaining before May 20 submission:**
1. Narrative reframe: detection paper → validation/characterization paper (claim alignment)
2. Scope cut for ESL: compress E2/E3, reduce regulatory framing
3. Convert to IEEE two-column LaTeX (IEEEtran) — hard 4-page limit
4. Submit PDF via IEEE Author Portal: ieee.atyponrex.com/journal/les-ieee
5. Simultaneous arXiv upload

---

## Key Design Decisions

**Why STER = 0.0000 is the finding, not a null result:**  
STER provides a formally-defined, reproducible confirmation that output deviation is within the T* = 0.05 bound under all tested conditions. Accuracy-based testing provides no such confirmation (Δaccuracy < 1.0%, insensitive to output distribution). The positive finding is: modern dedicated edge AI accelerators pass STER under realistic contention. The negative control (E6) establishes the boundary: STER passes even without a dedicated accelerator, but timing fails — defining what the accelerator actually contributes to the safety profile.

**Why the CPU negative control (E6) is the strongest result:**  
Running the same model on the same hardware without the GPU accelerator produces STER = 0.0000 but 7.2× latency degradation. This isolates the accelerator's contribution to timing stability (not numerical stability), sharpens the architectural safety claim, and demonstrates that STER alone is insufficient as a sole pre-market safety criterion.

**Why MobileNetV2:**  
Representative of the model complexity class deployed in FDA-authorized diagnostic SaMD (compact CNN, 3.4M parameters, production-proven on embedded hardware).

**Why T* = 0.05:**  
Conservative relative to ISO 14971 Table B.1 risk control examples. Ensures STER captures stability degradation unacceptable before reaching levels flagged by clinical validation protocols.

**Why the Coral uses synthetic input:**  
The Edge TPU's deterministic int8 inference is unconditionally reproducible for fixed inputs. δ = 0.0000 exactly across all trials and all stressor conditions.

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

**Coral firmware:** coralmicro SDK, FreeRTOS, TFLite Micro, Edge TPU runtime  
**nRF firmware:** nRF5 SDK, SoftDevice S140, built with nRF Connect SDK

---

## Citation

> A. M. Swami, "Runtime Inference Stability Under Resource Contention in Edge AI Medical Devices: Characterizing a Pre-Market Validation Gap for FDA SaMD," *IEEE Embedded Systems Letters*, submitted May 2026.

*arXiv preprint to be posted simultaneously with submission (target: May 20, 2026).*

---

## Author

**Akul Mallayya Swami**  
Varian Medical Systems, A Siemens Healthineers Company, Palo Alto, CA, USA  
swami.akul@alumni.uml.edu  
ORCID: 0009-0003-9549-5543
