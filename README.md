# P2: Runtime Inference Instability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter
**arXiv / Submit target:** May 20, 2026
**Authors:** Akul Mallayya Swami — Varian Medical Systems, A Siemens Healthineers Company

## Overview

This repository contains all experimental code, firmware, results, and analysis scripts for the paper:

> Runtime Inference Instability Under Resource Contention in Edge AI Medical Devices: Characterizing a Pre-Market Validation Gap for FDA SaMD

The paper empirically characterizes inference output instability under resource contention on two edge AI platforms, introducing the Safety-Threshold Exceedance Rate (STER) as a candidate FDA pre-market verification metric.

## Hardware Platforms

| Platform | Role | Key Spec |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB | GPU inference host | TensorRT FP16, shared LPDDR5 |
| Coral Dev Board Micro | Dedicated TPU inference host | Edge TPU, TFLite int8, on-chip SRAM |

## Experiments

| ID | Stressor | Status |
|---|---|---|
| E0 | Baseline (zero load) | Complete |
| E1 | CPU contention | Pending |
| E2 | Memory pressure | Pending |
| E3 | GPU co-tenancy (Jetson only) | Pending |
| E4 | Network I/O BLE/WiFi | Pending |
| E5 | Combined realistic deployment load | Pending |

## Repository Structure

    p2-edge-ai-samd/
    jetson/                  Jetson Orin Nano Super experiment code
        e0_jetson.py         E0 baseline inference script
        data/                Dataset preparation and manifest
        results/             CSV results per experiment
    coral/                   Coral Dev Board Micro firmware and analysis
        e0_infer_baseline/   E0 inference determinism firmware (primary)
        supporting_timing_baseline/  E0 timing stability firmware
        results/             Logged output and summary CSVs
    paper/                   Paper draft (IEEE ESL format)

## Key Metric: STER

Safety-Threshold Exceedance Rate (STER) = proportion of inferences where output deviation exceeds tolerance band T* = 0.05 (plus or minus 5% relative deviation, conservative relative to ISO 14971 Table B.1).

E0 baseline target: STER = 0.0 on both platforms. Confirmed.

## E0 Baseline Results

| Platform | STER_nominal | delta_mean | delta_P99 |
|---|---|---|---|
| Jetson Orin Nano Super | 0.0000 | 0.0181 | 0.0233 |
| Coral Dev Board Micro | 0.0000 | 0.0000 | 0.0000 |

## Dependencies

Jetson: JetPack 6.x (R36.4.4), CUDA 12.6, TensorRT 10.3.0, cuDNN 9.3.0, Python 3.10, pycuda, Pillow, datasets, tqdm

Coral: coralmicro SDK, FreeRTOS, TFLite Micro, Edge TPU runtime. Host: Ubuntu 22.04, Python 3.10

## Citation

To be added upon arXiv publication (target: May 20, 2026)
