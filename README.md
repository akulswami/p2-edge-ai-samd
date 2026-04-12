# P2 — Runtime Inference Instability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**arXiv / submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd

---

## Paper Summary

This letter empirically characterizes inference output instability under resource contention on two architecturally distinct edge AI platforms, introducing the Safety-Threshold Exceedance Rate (STER) as a candidate pre-market verification metric for FDA-regulated SaMD. The central claim is an **accuracy/stability dissociation**: model accuracy remains statistically unchanged under stressor conditions while STER rises above baseline, confirming that accuracy-centered pre-market validation cannot detect this class of deployment instability.

---

## Hardware Platforms

| Platform | Role | Key Spec |
|---|---|---|
| NVIDIA Jetson Orin Nano Super 8GB | GPU inference host | TensorRT FP16, 6-core Arm, shared LPDDR5 |
| Coral Dev Board Micro + Env. Sensor Board | Dedicated TPU inference host | Edge TPU, TFLite int8, on-chip SRAM |
| nRF52840 DK | BLE peripheral / network stressor | Cortex-M4, BLE 5.0 |
| TI Bluetooth SensorTag | Second BLE data source | Patient-analog wearable stressor node |
| GL-AX1800 WiFi 6 Router | RF environment control | 2.4 GHz / 5 GHz co-channel interference |
| 1TB SSD + Insignia Docking Station | Disk I/O stressor | fio benchmark, sustained sequential writes |

---

## Experiment Status

| ID | Stressor | Jetson | Coral | Key Finding |
|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ Complete | ✅ Complete | STER=0.0000 both platforms |
| **E1** | CPU contention | ✅ Complete | ⏳ Pending | GPU isolation confirmed — STER=0.0000 at all load levels |
| **E2** | Memory pressure | 🔲 Pending | 🔲 Pending | — |
| **E3** | GPU co-tenancy (Jetson only) | 🔲 Pending | N/A | — |
| **E4** | Network I/O — BLE/WiFi (novel) | 🔲 Pending | 🔲 Pending | — |
| **E5** | Combined realistic deployment load | 🔲 Pending | 🔲 Pending | — |

---

## Confirmed Results

### E0 — Baseline Calibration

| Platform | STER_nominal | Acc_nominal | δ_mean | δ_P99 |
|---|---|---|---|---|
| Jetson Orin Nano Super | 0.0000 | 1.60%* | 0.0181 | 0.0233 |
| Coral Dev Board Micro | 0.0000 | 100.00%† | 0.0000 | 0.0000 |

*Jetson Acc reflects Tiny ImageNet label remapping (not a benchmark claim).  
†Coral uses deterministic synthetic input — accuracy is a stability baseline, not a benchmark.

δ defined as L-infinity norm on consecutive softmax probability vectors:  
`δᵢ = ‖σ(yᵢ) − σ(ȳ)‖∞ = max_k |σ(yᵢ)_k − σ(ȳ)_k|`

### E1 — CPU Contention (Jetson Confirmed)

| CPU Load | Jetson STER | Jetson Acc | δ_mean | δ_P99 |
|---|---|---|---|---|
| 0% | 0.0000 | 1.60% | 0.0181 | 0.0233 |
| 25% | 0.0000 | 1.42% | 0.0181 | 0.0232 |
| 50% | 0.0000 | 1.70% | 0.0179 | 0.0233 |
| 75% | 0.0000 | 1.54% | 0.0181 | 0.0235 |
| 100% | 0.0000 | 1.40% | 0.0181 | 0.0234 |

**Finding:** TensorRT FP16 inference executes entirely on the Jetson GPU. The 6 Arm CPU cores saturated by `stress-ng --cpu matrixprod` have no architectural path to disturb the GPU inference pipeline. STER = 0.0000 and δ flat across all load levels. This is a **GPU architectural isolation** result — the CPU contention pathway is absent by design.

Coral E1 architectural isolation test pending (Edge TPU on dedicated SRAM expected to show identical isolation).

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py          E0 baseline inference script (TensorRT FP16)
│   ├── e1_jetson.py          E1 CPU contention experiment script
│   ├── data/
│   │   └── prepare_dataset.py
│   └── results/
│       ├── e0_jetson.csv     E0 results (10 trials × 500 inferences)
│       ├── e1_jetson.csv     E1 results (10 trials × 500 inferences × 5 load levels)
│       └── e1_run.log        Full E1 run log
├── coral/
│   ├── e0_infer_baseline/    E0 Edge TPU firmware (MobileNetV1 int8)
│   ├── supporting_timing_baseline/
│   └── results/
│       ├── e0_coral_infer_summary.csv
│       └── e0_infer_log.txt
└── paper/
    └── P2_IEEE_ESL_Draft.docx
```

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate (STER)** = proportion of inferences where cycle time deviates from nominal T_n = 100 ms beyond tolerance band T* = 0.05 (±50 ms).

`STER = (1/N) · Σᵢ 𝟙[|t_cycle_i − T_n| > T*]`

E0 target: STER = 0.0 on both platforms. ✅ Confirmed.

---

## Setup

### Jetson Orin Nano Super
- JetPack 6.x (R36.4.4), CUDA 12.6, TensorRT 10.3.0
- Python venv: `~/e0_env` — activate: `source ~/e0_env/bin/activate`
- IP: 192.168.8.102

### Coral Dev Board Micro
- coralmicro SDK in `~/coralmicro/`
- Connected via USB-C → `/dev/ttyACM0`

### Stress tooling
- `stress-ng` 0.13.12 installed on Jetson
- E1 worker mapping (6-core Orin): 25%→2 workers, 50%→3, 75%→5, 100%→6

---

## Dependencies

**Jetson:** JetPack 6.x, CUDA 12.6, TensorRT 10.3.0, cuDNN 9.3.0, Python 3.10, pycuda, Pillow, numpy, stress-ng

**Coral:** coralmicro SDK, FreeRTOS, TFLite Micro, Edge TPU runtime. Host: Ubuntu 22.04, Python 3.10

---

## Citation

*To be added upon arXiv publication (target: May 20, 2026)*
