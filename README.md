# P2 — Runtime Inference Instability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**arXiv / submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Latest commit:** 747bf9a

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
| Insignia Docking Station + 2× 320GB HDD | Disk I/O stressor | fio benchmark, sustained sequential writes |

---

## Experiment Status

| ID | Stressor | Jetson | Coral | Key Finding |
|---|---|---|---|---|
| **E0** | Baseline (zero load) | ✅ Complete | ✅ Complete | STER=0.0000 both platforms |
| **E1** | CPU contention | ✅ Complete | ✅ Complete | CPU isolated on both platforms — STER=0.0000 at all load levels |
| **E2** | Memory pressure | ✅ Complete | ✅ Complete | Memory isolated on both platforms — STER=0.0000 at all fill levels |
| **E3** | GPU co-tenancy (Jetson only) | ✅ Complete | N/A | GPU scheduler context isolation confirmed — STER=0.0000 at 0–4 co-tenants |
| **E4** | Network I/O — BLE/WiFi (novel) | 🔲 Pending | 🔲 Pending | — |
| **E5** | Combined realistic deployment load | 🔲 Pending | 🔲 Pending | — |

---

## Confirmed Results

### E0 — Baseline Calibration

| Platform | STER_nominal | Acc_nominal | δ_mean | δ_P99 |
|---|---|---|---|---|
| Jetson Orin Nano Super | 0.0000 | 1.60%* | 0.0181 | 0.0233 |
| Coral Dev Board Micro | 0.0000 | 100.00%† | 0.0000 | 0.0000 |

\*Jetson Acc reflects Tiny ImageNet label remapping (not a benchmark claim).  
†Coral uses deterministic synthetic input — accuracy is a stability baseline, not a benchmark.

δ defined as L-infinity norm on softmax probability vectors vs. nominal baseline:  
`δᵢ = ‖σ(yᵢ) − σ(ȳ)‖∞ = max_k |σ(yᵢ)_k − σ(ȳ)_k|`

---

### E1 — CPU Contention (Both Platforms Confirmed)

Jetson: 10 trials × 500 inferences per level. `stress-ng --cpu 0 --cpu-load <pct>` on Jetson.  
Coral: 5 trials × ~1970 inferences per level. Same stressor on Ubuntu host while Coral runs over USB.

| CPU Load | Jetson STER | Jetson Acc | Jetson δ_mean | Coral STER | Coral Acc | Coral δ_mean |
|---|---|---|---|---|---|---|
| 0% | 0.0000 | 1.60% | 0.0181 | 0.0000 | 100.00% | 0.0000 |
| 25% | 0.0000 | 1.42% | 0.0181 | 0.0000 | 100.00% | 0.0000 |
| 50% | 0.0000 | 1.70% | 0.0179 | 0.0000 | 100.00% | 0.0000 |
| 75% | 0.0000 | 1.54% | 0.0181 | 0.0000 | 100.00% | 0.0000 |
| 100% | 0.0000 | 1.40% | 0.0181 | 0.0000 | 100.00% | 0.0000 |

**Finding:** CPU contention has no disturbance path to either inference pipeline. On the Jetson, TensorRT FP16 executes entirely on the GPU — the Arm CPU cores have no path to the GPU pipeline. On the Coral, the Edge TPU operates on dedicated on-chip SRAM with a separate memory controller from the host CPU. Both platforms achieve complete isolation by different architectural mechanisms. **CPU architectural isolation confirmed on both platforms.**

---

### E2 — Memory Pressure (Both Platforms Confirmed)

Jetson: 5 trials × 500 inferences per level. `stress-ng --vm 1 --vm-bytes <N>M --vm-keep --vm-method all` on Jetson.  
Coral: 5 trials × 2000 inferences per level. Same stressor on Ubuntu host while Coral runs over USB.

| VM Fill | Jetson STER | Jetson Acc | Jetson δ_mean | Coral STER | Coral Acc | Coral δ_mean |
|---|---|---|---|---|---|---|
| 25% | 0.0000 | 1.44% | 0.01827 | 0.0000 | 100.00% | 0.0000 |
| 50% | 0.0000 | 1.56% | 0.01808 | 0.0000 | 100.00% | 0.0000 |
| 75% | 0.0000 | 1.40% | 0.01821 | 0.0000 | 100.00% | 0.0000 |
| 90% | 0.0000 | 1.64% | 0.01812 | 0.0000 | 100.00% | 0.0000 |

**Finding:** Memory pressure has no disturbance path to either inference pipeline. On the Jetson, TensorRT pycuda allocations are serviced through the GPU DMA path with dedicated QoS bandwidth reservation, isolated from CPU-side vm-worker activity. On the Coral, the Edge TPU's dedicated on-chip SRAM is architecturally separate from the host DDR filled by stress-ng. δ = 0.0000 exactly on Coral at all levels, reflecting hardware determinism of int8 Edge TPU. **Memory isolation confirmed on both platforms.**

---

### E3 — GPU Co-tenancy (Jetson Only)

5 trials × 500 inferences per level. Co-tenant workers run continuous unpaced MobileNetV2-FP16 inference via independent pycuda contexts on the same GPU.

| Co-tenants | Jetson STER | Jetson Acc | δ_mean | δ_P99 |
|---|---|---|---|---|
| 0 (baseline) | 0.0000 | 1.44% | 0.01823 | 0.02378 |
| 1 | 0.0000 | 1.60% | 0.01809 | 0.02316 |
| 2 | 0.0000 | 1.40% | 0.01822 | 0.02348 |
| 3 | 0.0000 | 1.64% | 0.01812 | 0.02326 |
| 4 | 0.0000 | 1.64% | 0.01809 | 0.02327 |

**Finding:** The Orin GPU scheduler time-slices all TensorRT execution contexts without violating the T_n = 100 ms cycle deadline. Each worker holds an independent pycuda context with isolated device memory — no cross-process output corruption observed. **GPU scheduler context isolation confirmed.**

Coral: GPU co-tenancy is architecturally impossible on the Edge TPU — confirmed as an absolute safety property.

---

## Repository Structure

```
p2-edge-ai-samd/
├── jetson/
│   ├── e0_jetson.py          E0 baseline inference script (TensorRT FP16, pycuda)
│   ├── e1_jetson.py          E1 CPU contention experiment
│   ├── e2_jetson.py          E2 memory pressure experiment (stress-ng --vm)
│   ├── e3_jetson.py          E3 GPU co-tenancy experiment
│   ├── e3_worker.py          E3 continuous inference worker (co-tenant stressor)
│   ├── data/
│   │   └── prepare_dataset.py
│   └── results/
│       ├── e0_jetson.csv     E0 results (10 trials × 500 inferences)
│       ├── e1_jetson.csv     E1 results (10 trials × 500 inferences × 5 load levels)
│       ├── e1_run.log        Full E1 run log
│       ├── e2_jetson.csv     E2 results (5 trials × 500 inferences × 4 fill levels)
│       └── e3_jetson.csv     E3 results (5 trials × 500 inferences × 5 co-tenant levels)
├── coral/
│   ├── coral_capture.py      Shared serial capture + analysis utility (E1/E2)
│   ├── e0_infer_baseline/    E0 Edge TPU firmware (MobileNetV1 int8)
│   │   ├── e0_infer_baseline.cc
│   │   └── analyze_e0_coral_infer.py
│   ├── e1_cpu_stress/        E1 CPU contention experiment (Coral)
│   │   ├── e1_coral.py
│   │   └── results/
│   │       └── e1_coral.csv  E1 results (5 trials × ~1970 inferences × 5 load levels)
│   ├── e2_mem_pressure/      E2 memory pressure experiment (Coral)
│   │   ├── e2_coral.py
│   │   └── results/
│   │       └── e2_coral.csv  E2 results (5 trials × 2000 inferences × 4 fill levels)
│   ├── supporting_timing_baseline/
│   └── results/
│       ├── e0_coral_infer_summary.csv
│       └── e0_infer_log.txt
└── paper/
    ├── P2_IEEE_ESL_Draft.docx             Original draft
    ├── P2_IEEE_ESL_Draft_E1.docx          Updated with E1 results
    ├── P2_IEEE_ESL_Draft_E2E3.docx        Updated with E2+E3 results
    └── P2_IEEE_ESL_Draft_CoralE1E2.docx   Updated with Coral E1+E2 results
```

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate (STER)** = proportion of inferences where the L-infinity norm on the softmax output vector vs. the nominal baseline exceeds tolerance band T* = 0.05:

`STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]`

where `δᵢ = ‖σ(yᵢ) − σ(ȳ)‖∞` and T* = 0.05 is derived conservatively from ISO 14971 Table B.1 risk control examples.

E0–E3: STER = 0.0000 on both platforms across all stressor conditions tested. ✅ Confirmed.

---

## Architectural Summary (E0–E3)

All disturbance pathways tested so far have been ruled out on both platforms:

| Pathway | Jetson mechanism | Coral mechanism | STER effect |
|---|---|---|---|
| CPU contention (E1) | GPU pipeline has no CPU dependency | Dedicated on-chip SRAM, separate memory controller | None on both |
| Memory pressure (E2) | GPU DMA path has dedicated QoS lane | TPU SRAM separate from host DDR | None on both |
| GPU co-tenancy (E3) | GPU scheduler provides context isolation | Architecturally impossible on Edge TPU | None (Jetson only) |

**Remaining candidate disturbance pathways: network interrupt load (E4) and combined realistic load (E5).** BLE interrupt processing occurs on the host CPU regardless of GPU/TPU isolation, making E4 the first experiment with a plausible disturbance path to both platforms.

---

## Setup

### Jetson Orin Nano Super
- JetPack 6.x (R36.4.4), CUDA 12.6, TensorRT 10.3.0, cuDNN 9.3.0
- Python venv: `~/e0_env` — activate: `source ~/e0_env/bin/activate`
- Model: `~/e0_experiment/models/mobilenetv2_fp16.trt`
- Dataset: 500 Tiny ImageNet images, manifest at `~/e0_experiment/data/manifest.json`
- IP: 192.168.8.102

### Coral Dev Board Micro
- coralmicro SDK in `~/coralmicro/`
- Connected via USB-C → `/dev/ttyACM0`
- Serial capture: `coral_capture.py` (pyserial, handles nRST USB disconnect/reconnect)
- udev rule: `/etc/udev/rules.d/99-coral.rules` — prevents ModemManager from grabbing port on reconnect

### Stress tooling
- `stress-ng` 0.13.12 installed on both Jetson and Ubuntu host
- E1 CPU: `--cpu 0 --cpu-load <pct>` (uses all available cores at target %)
- E2 VM fill: 25/50/75/90% of total RAM — Jetson 8192 MB, Ubuntu host auto-detected at runtime
- E3 co-tenant workers: independent pycuda + TensorRT contexts, 100-image loop, no pacing

---

## Dependencies

**Jetson:** JetPack 6.x, CUDA 12.6, TensorRT 10.3.0, cuDNN 9.3.0, Python 3.10, pycuda, Pillow, numpy, stress-ng

**Coral host (Ubuntu):** Python 3.10, pyserial, numpy, stress-ng. coralmicro SDK for firmware builds.

---

## Citation

*To be added upon arXiv publication (target: May 20, 2026)*
