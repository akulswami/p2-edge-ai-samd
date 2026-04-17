# Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices

This repository contains the experiment code, result artifacts, and manuscript sources for the study:

“Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices: Controlled Experimental Evidence on a Shared-Silicon Platform”

---

## Scope

This work evaluates whether output stability and timing reliability remain coupled under deployment load in edge AI inference systems.

Two execution paths are evaluated on the same hardware:

- GPU path: TensorRT FP16
- CPU path: ONNX Runtime FP32

The study demonstrates that output correctness can be preserved while timing constraints are violated under realistic load.

---

## Key Result

Under combined system load:

- Both execution paths maintain:
  - STER = 0.0000 (no output distribution deviation beyond threshold)
- GPU path:
  - Mean latency ≈ 10.6 ms
  - P99 latency ≈ 10.9 ms
- CPU path:
  - Mean latency ≈ 104.0 ms
  - P99 latency ≈ 165.1 ms
  - Exceeds 10 Hz (100 ms) cycle budget by 65%

This establishes that output stability and timing reliability are independent properties and must be verified jointly.

Failure to detect this divergence can result in deployment of systems that are functionally correct but temporally unsafe.

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py
│   ├── e1_jetson.py
│   ├── e2_jetson.py
│   ├── e3_jetson.py
│   ├── e3_worker.py
│   ├── e4_jetson.py
│   ├── e5_jetson.py
│   ├── e6_jetson_cpu.py
│   └── results/
├── coral/
├── paper/
│   ├── p2_arxiv.tex
│   ├── p2_ieee_esl_final.tex
│   ├── figures/
│   └── ...
└── README.md
```

---

## Mapping to Manuscript

| Paper Section | Experiment | Script |
|--------------|------------|--------|
| Section 5.1 | E0 | jetson/e0_jetson.py |
| Section 5.2 | E1 | jetson/e1_jetson.py |
| Section 5.3 | E2–E4 | jetson/e2_jetson.py, e3_jetson.py, e4_jetson.py |
| Section 5.4 | E5 | jetson/e5_jetson.py |
| Section 5.5 | E6 | jetson/e6_jetson_cpu.py |

---

## Reproducing the Key Result (E5)

The central claim is that output stability is preserved while timing degrades under combined load.

### Step 1 — Run GPU path

Run from repository root:

python3 jetson/e5_jetson.py --mode gpu

### Step 2 — Run CPU path

Run from repository root:

python3 jetson/e6_jetson_cpu.py

### Step 3 — Apply combined load

- CPU: stress-ng at 75%
- Memory: 50% allocation
- BLE: 4 connections
- Disk: fio write workload

### Step 4 — Expected outcome

- GPU latency remains ≈ 10.6 ms
- CPU latency degrades to ≈ 104 ms mean, ≈ 165 ms P99
- STER = 0 for both paths

---

## Experiments

| ID | Description |
|----|------------|
| E0 | Baseline zero-load characterization |
| E1 | CPU contention sweep |
| E2 | Memory pressure (GPU path) |
| E3 | GPU co-tenancy |
| E4 | Network I/O stress (BLE + WiFi) |
| E5 | Combined realistic load |
| E6 | CPU-only contention characterization |

---

## Key Metric

STER (Safety Threshold Exceedance Rate):

- Measures deviation in output distribution relative to baseline
- Detects subthreshold changes not captured by accuracy metrics
- Defined in manuscript Section 3

---

## Environment

- Hardware: NVIDIA Jetson Orin Nano Super
- OS: JetPack 6
- Python: 3.10+
- TensorRT (GPU path)
- ONNX Runtime (CPU path)

---

## Version

This repository version corresponds to IEEE ESL submission (April 2026).

Tag:
v1.0-esl-submission

---

## Building the Manuscript

cd paper
pdflatex -interaction=nonstopmode p2_arxiv.tex
pdflatex -interaction=nonstopmode p2_arxiv.tex

---

## Citation

If using this work:

A. M. Swami,
“Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices,”
arXiv preprint, 2026.

---

## Notes

This repository supports experimental transparency and reproducibility.
No clinical or deployment claims are made beyond the scope of the controlled experiments described.
