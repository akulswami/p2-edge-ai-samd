# P2 — Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**Submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Paper status:** Submission-ready — final revision complete, all peer-review fixes applied

---

## Paper Summary

This letter demonstrates a challenge in operationalizing Software as a Medical Device (SaMD) robustness requirements: while FDA Draft Guidance FDA-2024-D-4488 requires assessment under foreseeable conditions of use, accuracy-based pre-market certification protocols do not jointly verify output stability and timing reliability at the inference layer under deployment load.

**The novelty is not timing degradation** — that is known — but that accuracy-based validation cannot detect it: both execution stacks pass accuracy certification while one violates the clinical timing budget by 65% under foreseeable load.

**Experimental design:** The same MobileNetV2 model is evaluated under identical adversarial load on two execution paths of the same physical hardware — a GPU accelerator (TensorRT FP16) and a general-purpose CPU (ONNX Runtime FP32) — on the NVIDIA Jetson Orin Nano Super. Both paths run on the same device, eliminating chip-to-chip variation and thermal confounds. The comparison reflects two complete execution stacks; the observed latency difference reflects the combined effect of architecture, runtime, and precision.

**Central finding:** Both paths maintain STER = 0.0000 (zero output exceedances above T\* = 0.05) across 107,500 verified inference activations. The GPU path maintains latency below 11 ms (mean 10.6 ms, P99 10.9 ms) under all conditions. The CPU path degrades 7.2× under combined load (14.5 ms → 104.0 ms mean, P99 = 165.1 ms), breaching the 10 Hz clinical cycle budget by 65%. STER = 0 on both paths confirms output stability, making timing degradation the sole identifiable safety-relevant divergence. This violation is undetectable by accuracy-based validation (Δacc < 1.0%).

**Regulatory anchor:** Joint STER + latency verification is proposed as a candidate method for operationalizing FDA-2024-D-4488 robustness requirements at the inference layer, consistent with IEC 62304 and ISO 14971, and subject to regulatory review and clinical validation for specific device classes.

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate:**

```
STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]
```

where `δᵢ = ‖σ(yᵢ) − σ̄ᵢ‖∞` is the per-image L-infinity norm on the softmax probability vector vs. the per-image zero-load reference, and `T* = 0.05`.

**Why STER = 0 is not a null result:** Without a confirmed output-stability metric, an observed latency increase could reflect output-level degradation (numerical instability) rather than pure scheduling pressure. STER = 0 rules this out: scheduling pressure delays computation without perturbing arithmetic. This isolates timing as the failure mode and makes the validation blindness claim attributable.

**Why STER ≠ accuracy:** Accuracy evaluates argmax(σ(y)) — a single class label — and cannot detect subthreshold distribution drift. STER captures this class of silent deviations by measuring the full softmax vector deviation per inference.

---

## Hardware Platforms

| Platform | Role | Key Specification |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB (GPU path) | Primary inference host | 1024-core Ampere GPU, TensorRT FP16, shared LPDDR5 |
| NVIDIA Jetson Orin Nano Super 8GB (CPU path) | Contention path — same hardware | ONNX Runtime FP32, CPUExecutionProvider, 6-core Arm Cortex-A78AE |
| nRF52840 DK | BLE network stressor | Cortex-M4, BLE 5.0, multi-connection central |
| TI Bluetooth SensorTag | Second BLE node | Patient-analog wearable stressor |
| GL-AX1800 WiFi 6 Router | RF environment control | 2.4 GHz co-channel interference |
| Insignia Dock + 2× 298GB HDD | Disk I/O stressor | fio sequential writes, USB-attached |

**Note:** Coral Dev Board Micro results preserved in git history (commit 3dcd63f) for companion paper P4.

---

## Experiment Summary

| ID | Stressor | GPU Path | CPU Path | STER | Key Result |
|---|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ 5,000 inf. | ✅ 5,000 inf. | 0.0000 both | GPU: 10.6 ms mean, 10.9 ms P99; CPU: 14.5 ms mean, 18.6 ms P99 |
| **E1** | CPU contention (0/25/50/75/100%) | ✅ 25,000 inf. | ✅ (via E6) | 0.0000 both | GPU stable ≈10.6 ms; CPU reaches 73.7 ms at 75% load |
| **E2** | Memory pressure (25–90% LPDDR5) | ✅ 10,000 inf. | — | 0.0000 GPU | No latency or output effect on GPU path |
| **E3** | GPU co-tenancy (0–4 workers) | ✅ 12,500 inf. | — | 0.0000 GPU | No effect even with 4 concurrent TensorRT workers |
| **E4** | Network I/O (BLE 0–6 + WiFi) | ✅ 20,000 inf. | — | 0.0000 GPU | +88% interrupt rate at BLE=4 had no effect; authoritative GPU latency source |
| **E5** | Combined realistic load | ✅ 5,000 inf. | ✅ 5,000 inf. | 0.0000 both | GPU <11 ms; CPU 104.0 ms mean, 165.1 ms P99 — 65% budget breach |
| **E6** | CPU contention characterization | — | ✅ 30,000 inf. | 0.0000 CPU | Full CPU-only characterization across all E0+E1+E5 conditions |

**Total verified inference activations: 107,500**

| Experiment | Count | Source |
|---|---|---|
| E0 (GPU) | 5,000 | `jetson/results/e0_jetson.csv` |
| E1 (GPU, 5 load levels) | 25,000 | `jetson/results/e1_jetson.csv` |
| E2 (GPU, 4 memory levels) | 10,000 | `jetson/results/e2_jetson.csv` |
| E3 (GPU, 5 co-tenant levels) | 12,500 | `jetson/results/e3_jetson.csv` |
| E4 (GPU, 4 BLE conn levels) | 20,000 | `jetson/results/e4_jetson_conns{0,2,4,6}.csv` |
| E5 (GPU) | 5,000 | `jetson/results/e5_results_summary.json` |
| E6 (CPU, 6 conditions) | 30,000 | `jetson/results/e6_results_summary.json` |

---

## Key Results

### E0 Baseline

| Path | STER | δ_mean | Lat. mean | Lat. SD | P99 |
|---|---|---|---|---|---|
| GPU (TensorRT FP16)† | 0.0000 | 0.0182 | 10.6 ms | 0.2 ms | 10.9 ms |
| CPU (ONNX FP32) | 0.0000 | 0.0000 | 14.5 ms | 0.2 ms | 18.6 ms |

†GPU latency measured end-to-end (preprocess + infer + softmax), N=5,000 from E4 conns=0.

### E6 — CPU Contention Characterization

| Condition | STER | δ_mean | Lat. mean | Lat. SD | P99 |
|---|---|---|---|---|---|
| Zero load | 0.0000 | 0.0000 | 14.5 ms | 0.2 ms | 18.6 ms |
| CPU 25% | 0.0000 | 0.0000 | 33.4 ms | 1.8 ms | 111.4 ms |
| CPU 50% | 0.0000 | 0.0000 | 57.8 ms | 3.2 ms | 115.8 ms |
| CPU 75% | 0.0000 | 0.0000 | 73.7 ms | 4.1 ms | 122.1 ms |
| CPU 100% | 0.0000 | 0.0000 | 70.9 ms | 3.8 ms | 117.0 ms |
| **Combined** | **0.0000** | **0.0000** | **104.0 ms** | **5.7 ms** | **165.1 ms ⚠️** |

⚠️ P99 = 165.1 ms exceeds the 10 Hz cycle budget (100 ms) by 65%. STER = 0 passes; deployment timing fails. Δacc < 1.0% under identical conditions — accuracy-based validation cannot detect this failure.

---

## Paper Files

```
paper/
├── p2_ieee_esl_final.tex          Final submission LaTeX source  ← LATEST
├── p2_ieee_esl_final.pdf          Compiled submission PDF         ← LATEST
├── p2_arxiv.tex                   arXiv preprint version (article class)
├── p2_cover_letter.docx           IEEE ESL submission cover letter
└── figures/
    ├── fig1_arch.png              Architecture diagram (887×592px, 300 DPI)
    ├── fig_2.png                  CDF of inference latency (1053×783px, 300 DPI)
    └── fig_3.png                  STER vs. latency scatter (1053×723px, 300 DPI)
```

**Archived drafts** (pre-final, kept for version history):
```
paper/
├── P2_IEEE_ESL_Draft_E6.docx      Early draft (pre-pivot)
├── P2_IEEE_ESL_Draft_Pivot.docx   Word draft (pivot framing)
├── p2_paper.tex / .pdf            Earlier version
└── p2_paper_submission.tex / .pdf Previous submission candidate
```

### Building the PDF

```bash
cd paper
mkdir -p figures  # figures must be in paper/figures/ subfolder
pdflatex p2_ieee_esl_final.tex
pdflatex p2_ieee_esl_final.tex   # second pass for cross-references
evince p2_ieee_esl_final.pdf
```

Requires: `texlive-full` or `texlive-latex-extra texlive-fonts-recommended texlive-science`

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── data/
│   │   └── prepare_dataset.py        Dataset preparation (Tiny ImageNet, 500 images)
│   ├── e0_jetson.py                  E0 baseline (TensorRT FP16, GPU path)
│   ├── e1_jetson.py                  E1 CPU stress (both paths)
│   ├── e2_jetson.py                  E2 memory pressure (GPU path)
│   ├── e3_jetson.py                  E3 GPU co-tenancy
│   ├── e3_worker.py                  E3 co-tenant worker process
│   ├── e4_jetson.py                  E4 network I/O / BLE (GPU path)
│   ├── e5_jetson.py                  E5 combined load (both paths)
│   ├── e6_jetson_cpu.py              E6 CPU contention characterization (ONNX FP32)
│   └── results/
│       ├── e0_jetson.csv             E0 GPU baseline (10 trials × 500 inf.)
│       ├── e1_jetson.csv             E1 CPU stress both paths (50 trials × 500 inf.)
│       ├── e1_run.log                E1 run log
│       ├── e2_jetson.csv             E2 memory pressure (20 trials × 500 inf.)
│       ├── e3_jetson.csv             E3 co-tenancy (25 trials × 500 inf.)
│       ├── e4_jetson_conns0.csv      E4 BLE=0 (10 trials × 500 inf., N=5000)
│       ├── e4_jetson_conns2.csv      E4 BLE=2 (10 trials × 500 inf., N=5000)
│       ├── e4_jetson_conns4.csv      E4 BLE=4 (10 trials × 500 inf., N=5000)
│       ├── e4_jetson_conns6.csv      E4 BLE=6 (10 trials × 500 inf., N=5000)
│       ├── e5_results_summary.json   E5 combined load summary
│       └── e6_results_summary.json   E6 CPU contention (6 conditions × 10 trials × 500 inf.)
├── coral/                            Preserved for companion paper P4
│   └── ...                           (see git history, commit 3dcd63f)
└── paper/
    ├── p2_ieee_esl_final.tex         ← Submission LaTeX
    ├── p2_ieee_esl_final.pdf         ← Submission PDF
    ├── p2_arxiv.tex                  ← arXiv preprint
    ├── p2_cover_letter.docx          ← Cover letter
    └── figures/
        ├── fig1_arch.png
        ├── fig_2.png
        └── fig_3.png
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

| Section | Status | Notes |
|---|---|---|
| Title | ✅ Final | |
| Abstract | ✅ Final | Operationalizing framing; runtime labels in latency comparison |
| Introduction | ✅ Final | Operationalizing challenge framing; FDA requirement acknowledged |
| Related Work | ✅ Final | Explicit gap sentence: no prior same-hardware controlled design for orthogonality |
| System Model (Sec. III) | ✅ Final | ISO 14971 mischaracterization corrected; T*=0.05 as proposed threshold |
| Hardware & Protocol (Sec. IV) | ✅ Final | Statistics sentence reworded; variability context added |
| Results (Sec. V) | ✅ Final | E6 count corrected to 30,000 |
| Discussion (Sec. VI) | ✅ Final | VI.B: STER=0 as enabling result; VI.C heading updated; VI.D: two-hypothesis framing |
| Limitations (Sec. VI.E) | ✅ Final | Explicit confound disclosure: "cannot be attributed solely to architectural isolation" |
| Conclusion (Sec. VII) | ✅ Final | Execution stack framing; regulatory claim moderated |
| References [1]–[14] | ✅ Final | 14 refs; J. Hao, Martín, B. Lee all corrected |
| Data Availability | ✅ Final | Public repo URL included |
| Page count | ✅ 4 pages | IEEEtran two-column, 10pt |

**Remaining before May 20, 2026:**
1. Submit via IEEE Author Portal: https://ieee.atyponrex.com/journal/les-ieee
2. Upload arXiv preprint (`paper/p2_arxiv.tex`) — primary: eess.SY, cross-list: cs.AR
3. Submit cover letter (`paper/p2_cover_letter.docx`) alongside manuscript

---

## Key Design Decisions

**Why same-hardware design is the methodological contribution:**  
Cross-platform comparisons leave hardware differences as alternative explanations. Running both paths on the same Jetson Orin Nano Super holds all hardware variables constant. Any observed divergence arises from execution context alone.

**Why the comparison is framed as execution stack, not pure architectural isolation:**  
The GPU and CPU paths differ in runtime (TensorRT vs. ONNX), precision (FP16 vs. FP32), and measurement scope (end-to-end vs. runtime-only). The 9.8× latency difference reflects the combined effect of these factors. Controlled ablation studies isolating individual factors are identified as future work.

**Why STER = 0.0000 on both paths is not a null result:**  
Without confirmed output stability, a CPU-path latency increase could reflect output-level degradation rather than scheduling pressure. STER = 0 rules this out, making timing degradation the sole identifiable safety-relevant divergence and the validation blindness claim clean and attributable.

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

**LaTeX (for building paper):**
```
texlive-latex-base
texlive-latex-recommended
texlive-latex-extra
texlive-fonts-recommended
texlive-science
```

---

## Citation

> A. M. Swami, "Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices: Controlled Experimental Evidence on a Shared-Silicon Platform," *IEEE Embedded Systems Letters*, submitted May 2026.

---

## Author

**Akul Mallayya Swami**  
Varian Medical Systems, A Siemens Healthineers Company, Palo Alto, CA, USA  
swami.akul@alumni.uml.edu | ORCID: 0009-0003-9549-5543
