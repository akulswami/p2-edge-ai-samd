# P2 — Runtime Inference Stability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**arXiv / submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Latest commit:** 918ffe7  
**Paper status:** Draft complete — E0–E5 confirmed, all placeholders filled, IEEE ESL compliance review completed

---

## Paper Summary

This letter empirically characterizes inference output stability under resource contention on two architecturally distinct edge AI platforms — the NVIDIA Jetson Orin Nano Super (TensorRT FP16, shared LPDDR5) and the Coral Dev Board Micro (Edge TPU int8, dedicated on-chip SRAM) — introducing the Safety-Threshold Exceedance Rate (STER) as a candidate pre-market verification metric for FDA-regulated Software as a Medical Device (SaMD).

**Central finding:** Both platforms achieve STER = 0.0000 with zero exceedances across approximately 158,000 inference activations under five stressor classes, including combined realistic deployment load. Accuracy-based metrics are structurally insensitive to these conditions (Δaccuracy < 1.0%), confirming they cannot substitute for STER in stability characterization. These results provide the first empirical evidence that dedicated inference accelerator architectures may inherently satisfy STER-based pre-market criteria.

**Regulatory anchor:** FDA Draft Guidance FDA-2024-D-4488 (January 2025) requires robustness assessment for reasonably foreseeable conditions of use. STER is proposed as a candidate empirical method for operationalizing that requirement.

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate:**

```
STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]
```

where `δᵢ = ‖σ(yᵢ) − σ(ȳ)‖∞` is the L-infinity norm on the softmax probability vector vs. the nominal mean baseline, and `T* = 0.05` (±5% relative deviation, conservative relative to ISO 14971 Table B.1).

**Coral platform note:** The Edge TPU produces deterministic int8 outputs (δ = 0.0000 exactly). STER on Coral is additionally characterized via scalar score deviation `Δscoreᵢ = |scoreᵢ − score_ref|` with `REF_CLASS = 905`, `REF_SCORE = 0.320312`.

---

## Hardware Platforms

| Platform | Role | Key Specification |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB | Primary GPU inference host | 1024-core Ampere GPU, 67 TOPS, shared LPDDR5, TensorRT FP16 |
| Coral Dev Board Micro + Env. Sensor Board | Dedicated TPU inference host | Google Edge TPU, 4 TOPS, on-chip SRAM, TFLite int8 |
| nRF52840 DK | BLE peripheral / network stressor | Cortex-M4, BLE 5.0, multi-connection central |
| TI Bluetooth SensorTag | Second BLE data source | Patient-analog wearable stressor node |
| GL-AX1800 WiFi 6 Router | RF environment control | 2.4 GHz / 5 GHz co-channel interference |
| Insignia Dock + 2× 298GB HDD | Disk I/O stressor | fio sequential writes, USB-attached |

---

## Experiment Status

| ID | Stressor | Jetson | Coral | STER Result |
|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ Complete | ✅ Complete | 0.0000 both platforms |
| **E1** | CPU contention (0–100%) | ✅ Complete | ✅ Complete | 0.0000 both platforms |
| **E2** | Memory pressure (25–90% fill) | ✅ Complete | ✅ Complete | 0.0000 both platforms |
| **E3** | GPU co-tenancy (0–4 workers) | ✅ Complete | N/A (architectural) | 0.0000 Jetson; N/A Coral |
| **E4** | Network I/O — BLE/WiFi (novel) | ✅ Complete | ✅ Complete | 0.0000 both platforms |
| **E5** | Combined realistic deployment load | ✅ Complete | ✅ Complete | 0.0000 both platforms |

**Total inference activations:** ~158,000 across all experiments and both platforms.  
**STER exceedances:** Zero across all conditions.

---

## Confirmed Results

### E0 — Baseline Calibration

| Platform | STER_nominal | Acc_nominal | δ_mean | δ_P99 |
|---|---|---|---|---|
| Jetson Orin Nano Super | 0.0000 | 1.60%* | 0.0181 | 0.0233 |
| Coral Dev Board Micro | 0.0000 | 100.00%† | 0.0000 | 0.0000 |

\* Jetson Acc reflects Tiny ImageNet label remapping, not a benchmark claim.  
† Coral uses deterministic synthetic input — 100% reflects hardware determinism.

### E1 — CPU Contention

| CPU Load | Jetson STER | Jetson Acc | Coral STER | Coral Acc |
|---|---|---|---|---|
| 25% | 0.0000 | 1.42% | 0.0000 | 0.0000 |
| 50% | 0.0000 | 1.70% | 0.0000 | 0.0000 |
| 75% | 0.0000 | 1.54% | 0.0000 | 0.0000 |
| 100% | 0.0000 | 1.40% | 0.0000 | 0.0000 |

Jetson GPU executes entirely on GPU — no architectural path for CPU load to disturb inference. Coral Edge TPU isolated by dedicated on-chip SRAM.

### E2 — Memory Pressure

| RAM Fill | Jetson STER | Jetson Acc | Coral STER | Coral Acc | Sig. |
|---|---|---|---|---|---|
| 25% | 0.0000 | 1.44% | 0.0000 | 1.56% | No |
| 50% | 0.0000 | 1.40% | 0.0000 | 1.64% | No |
| 75% | 0.0000 | 1.54% | 0.0000 | 0.0000 | No |
| 90% | 0.0000 | 1.48% | 0.0000 | 0.0000 | No |

TensorRT allocations via GPU DMA path isolated from host DDR. Edge TPU SRAM architecturally separate from host DDR.

### E3 — GPU Co-tenancy (Jetson Only)

| Co-tenants | Jetson STER | Jetson Acc | STER Ratio | Sig. |
|---|---|---|---|---|
| 1 | 0.0000 | 1.60% | N/A | No |
| 2 | 0.0000 | 1.40% | 1.00× | No |
| 3 | 0.0000 | 1.64% | 1.00× | No |
| 4 | 0.0000 | 1.60% | 1.00× | No |

GPU co-tenancy architecturally impossible on Coral Edge TPU — confirmed absolute isolation property.

### E4 — Network I/O (Novel Stressor Class)

| BLE Connections | Jetson STER | Jetson Acc | Coral STER | Coral Acc | WiFi Compound? |
|---|---|---|---|---|---|
| 0 | 0.0000 | 1.64% | 0.0000 | +0.04% | No |
| 2 | 0.0000 | 1.64% | 0.0000 | 100.00% | No |
| 4 | 0.0000 | 1.64% | 0.0000 | 100.00% | No |
| 6 | 0.0000 | 1.64% | 0.0000 | 100.00% | No |

IRQ rate confirmed elevated (+88% at 4 connections). GPU pipeline isolated from network interrupt pathway. BLE managed by nRF52840 DK (e4_conns4.hex firmware).

### E5 — Combined Realistic Deployment Load

**Stressor protocol:** CPU 75% (stress-ng) + Memory 50% fill (stress-ng --vm) + BLE 4 connections (nRF52840 DK) + Disk fio sequential write (USB HDD)

| Platform | STER_combined | Acc_combined | δ_mean | δ_P99 | STER vs. worst single |
|---|---|---|---|---|---|
| Jetson Orin Nano Super | 0.0000 | 1.52% | 0.0156 | 0.0214 | 1.00× (E1–E4) |
| Coral Dev Board Micro | 0.0000 | 100.00% | 0.0000 | 0.0000 | 1.00× (E4) |

10 trials × 500 inferences (Jetson). 10 trials × ~1452 inferences captured (Coral, mean n_captured from serial). Additive stressor independence confirmed — combined load produces no degradation beyond any individual stressor.

---

## Inference Pipeline

Both platforms execute MobileNetV2 (224×224 RGB, ImageNet 1000-class head) at a nominal rate of 10 Hz (T_n = 100 ms). Jetson runs TensorRT FP16; Coral runs TFLite int8 compiled for Edge TPU. Fixed test set: 500 Tiny ImageNet images (Jetson) / deterministic synthetic input (Coral). Baseline: mean softmax vector across 500 images, captured under E0 nominal conditions.

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py            E0 baseline inference
│   ├── e1_jetson.py            E1 CPU stress experiment
│   ├── e2_jetson.py            E2 memory pressure experiment
│   ├── e3_jetson.py            E3 GPU co-tenancy experiment
│   ├── e3_worker.py            E3 co-tenant worker process
│   ├── e4_jetson.py            E4 network I/O experiment
│   ├── e5_jetson.py            E5 combined load experiment
│   └── results/
│       ├── e0_jetson.csv
│       ├── e1_jetson.csv
│       ├── e2_jetson.csv
│       ├── e3_jetson.csv
│       ├── e4_jetson_conns{0,2,4,6}.csv
│       └── e5_results_summary.json
├── coral/
│   ├── e0_infer_baseline/      E0 firmware (MobileNetV1 int8, 2000 inferences)
│   ├── e1_coral.py             E1 Coral experiment script
│   ├── e2_coral.py             E2 Coral experiment script
│   ├── e4_coral.py             E4 Coral experiment script
│   ├── e5_coral.py             E5 Coral experiment script
│   └── results/
│       ├── e0_coral_summary.csv
│       ├── e1_coral.csv
│       ├── e2_coral.csv
│       ├── e4_coral_conns{0,2,4,6}.csv
│       └── e5_coral_results.json
└── paper/
    └── P2_IEEE_ESL_Draft_E5.docx   Current paper draft (all E0–E5 filled)
```

---

## Jetson Setup

- **IP:** 192.168.8.102 (GL-AX1800 network)
- **User:** akulswami
- **OS:** JetPack 6 (R36.4.4), CUDA 12.6, TensorRT 10.3.0
- **Venv:** `~/e0_env` → `source ~/e0_env/bin/activate`
- **Model:** `~/e0_experiment/models/mobilenetv2_fp16.trt`
- **Dataset:** 500 Tiny ImageNet images, manifest at `~/e0_experiment/data/manifest.json`
- **E4 baseline:** `~/e4_experiment/e4_baseline.npy` (mean softmax, shape (1000,))
- **Inference pattern:** pycuda + TensorRT FP16, cycle-paced to T_NOMINAL=0.100s

Key script notes:
- Manifest is plain JSON list — use `manifest[:N_INFER]`, not `manifest['images']`
- Image paths are absolute — use `e['path']` directly
- Baseline shape is `(1000,)` — global mean softmax vector, not per-image

---

## Coral Setup

- **Connection:** Ubuntu via USB-C → `/dev/ttyACM2` (when nRF DK also connected)
- **Firmware:** `e0_infer_baseline` (MobileNetV1 int8, 2000 inferences, synthetic input)
- **Serial capture:** `~/e1_coral/coral_capture.py`
- **REF_CLASS:** 905, **REF_SCORE:** 0.320312
- **Serial output format:** `infer,<iteration>,<class_id>,<score>`
- **n_captured note:** E4/E5 capture ~1400 inferences per trial (firmware completes ~2000, serial buffer timing limits capture); STER=0.0000 confirmed across all captured inferences

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

**Draft:** `paper/P2_IEEE_ESL_Draft_E5.docx`

| Section | Status |
|---|---|
| Abstract | ✅ Complete (~157 words, tight) |
| Introduction | ✅ Complete — clinical anchor updated to Lee et al. JAMA 2025 |
| Related Work | ✅ Complete |
| System Model (Sec. III) | ✅ Complete |
| Hardware & Protocol (Sec. IV) | ✅ Complete |
| Results E0–E4 (Sec. V.A–E) | ✅ Complete with real data |
| Results E5 (Sec. V.F) | ✅ Complete with real data |
| Summary Table IX | ✅ Complete — all rows filled |
| Discussion (Sec. VI) | ✅ Complete — reframed for positive isolation finding |
| Conclusion (Sec. VII) | ✅ Complete — all [X] placeholders removed |
| Acknowledgements | ✅ Complete — AI disclosure + ORCID |
| References [1]–[15] | ✅ Complete — all 15 references real and formatted |

**Remaining before May 20 submission:**
1. Convert to IEEE two-column LaTeX format (IEEEtran) — hard 4-page limit requires ~40–50% content cut
2. Submit PDF via IEEE Author Portal: ieee.atyponrex.com/journal/les-ieee
3. Upload simultaneously to arXiv
4. Cover letter arguing scope fit ("profiling/measurement of embedded inference accelerators")

---

## Key Design Decisions

**Why STER = 0.0000 is the finding, not a null result:**  
Accuracy-based metrics also show no change (Δaccuracy < 1.0%), meaning accuracy-based pre-market testing would pass these devices — but cannot confirm that output deviation is within bounds. STER provides that confirmation. The finding establishes a positive safety baseline: current dedicated inference accelerator architectures may inherently satisfy STER-based pre-market criteria across all tested stressor classes.

**Why MobileNetV2:**  
Representative of the model complexity class deployed in FDA-authorized diagnostic SaMD (compact CNN, 3.4M parameters, production-proven on embedded hardware, used in multiple cleared devices).

**Why T* = 0.05:**  
Conservative relative to ISO 14971 Table B.1 risk control examples (typical clinical tolerance bands 10–20% for diagnostic classifiers). Ensures STER captures stability degradation unacceptable before reaching levels flagged by clinical validation protocols.

**Why the Coral uses synthetic input:**  
The Edge TPU's deterministic int8 inference is unconditionally reproducible for fixed inputs. Using synthetic input isolates the hardware determinism property cleanly. δ = 0.0000 exactly across all trials and all stressor conditions.

---

## Dependencies

**Jetson (Python 3.10, venv: `~/e0_env`):**
```
tensorrt==10.3.0
pycuda
Pillow
numpy
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
