# Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices

This repository contains code, experiment artifacts, and manuscript sources for a study on timing safety in edge AI inference pipelines for medical device relevant workloads.

## Overview

This work evaluates whether output stability and timing reliability remain coupled under deployment load. Using the same NVIDIA Jetson Orin Nano Super hardware, the study compares two execution paths for the same MobileNetV2 model:

- GPU path using TensorRT FP16
- CPU path using ONNX Runtime FP32

The central result is that both paths preserve output stability under the tested stress conditions, while only one path preserves timing within the target cycle budget. This demonstrates that accuracy style validation alone can miss timing unsafe behavior under foreseeable deployment load.

## Main finding

Across 107,500 verified inference activations:

- Both execution paths maintained `STER = 0.0000`
- The GPU path remained below 11 ms latency under all tested conditions
- The CPU path degraded to 104.0 ms mean latency and 165.1 ms P99 under combined load
- The CPU path exceeded the nominal 10 Hz cycle budget of 100 ms by 65 percent under combined load

This supports the paper’s core claim that output correctness and timing reliability are independent safety properties that should be evaluated jointly.

## Repository contents

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
│   └── ...
├── paper/
│   ├── p2_arxiv.tex
│   ├── p2_ieee_esl_final.tex
│   ├── p2_ieee_esl_final.pdf
│   └── ...
└── README.md

## Experiments

| ID | Description | Scope |
|---|---|---|
| E0 | Baseline zero load characterization | GPU and CPU |
| E1 | CPU contention sweep | GPU and CPU |
| E2 | Memory pressure | GPU |
| E3 | GPU co tenancy | GPU |
| E4 | Network I/O and BLE stress | GPU |
| E5 | Combined realistic load | GPU and CPU |
| E6 | CPU only contention characterization | CPU |

## Key metric

STER, or Safety Threshold Exceedance Rate, is defined in the manuscript as a per inference output stability metric relative to the zero load reference output.

It is used here to distinguish timing degradation from output distribution instability.

## Reproducibility

This repository preserves the experiment scripts and result artifacts used in the manuscript. Reproducing the full study requires:

- NVIDIA Jetson Orin Nano Super class hardware
- TensorRT and ONNX Runtime configured for the tested execution paths
- The same or equivalent image dataset preparation flow
- Stress generation for CPU, memory, network I/O, and storage I/O

Some scripts in this repository contain machine specific paths from the original experiment environment. These should be adapted to the local setup before rerunning experiments.

## Building the manuscript

cd paper
pdflatex -interaction=nonstopmode p2_arxiv.tex
pdflatex -interaction=nonstopmode p2_arxiv.tex

## Citation

Until the preprint is publicly posted, cite this work as:

A. M. Swami, “Architectural Isolation as a Timing Safety Primitive for Edge AI Medical Devices: Controlled Experimental Evidence on a Shared-Silicon Platform,” unpublished manuscript, 2026.

After the arXiv version is live, replace the citation above with the arXiv identifier.

## Author

Akul Mallayya Swami  
ORCID: 0009-0003-9549-5543

## Notes

This repository is intended to support manuscript transparency and reproducibility. It does not make any clinical claim or deployment recommendation beyond the experimental scope described in the paper.
