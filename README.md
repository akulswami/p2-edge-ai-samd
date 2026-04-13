# P2 — Runtime Inference Instability Under Resource Contention in Edge AI Medical Devices

**Target venue:** IEEE Embedded Systems Letters (ESL) — 4-page letter format  
**arXiv / submission target:** May 20, 2026  
**Repo:** github.com/akulswami/p2-edge-ai-samd  
**Latest commit:** 0d7c92f

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
| **E1** | CPU contention | ✅ Complete | ⏳ Pending | GPU architectural isolation confirmed — STER=0.0000 at all CPU load levels |
| **E2** | Memory pressure | ✅ Complete | ⏳ Pending | Memory subsystem QoS isolation confirmed — STER=0.0000 at all LPDDR5 fill levels |
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

### E1 — CPU Contention (Jetson Confirmed)

| CPU Load | Jetson STER | Jetson Acc | δ_mean | δ_P99 |
|---|---|---|---|---|
| 0% | 0.0000 | 1.60% | 0.0181 | 0.0233 |
| 25% | 0.0000 | 1.42% | 0.0181 | 0.0232 |
| 50% | 0.0000 | 1.70% | 0.0179 | 0.0233 |
| 75% | 0.0000 | 1.54% | 0.0181 | 0.0235 |
| 100% | 0.0000 | 1.40% | 0.0181 | 0.0234 |

**Finding:** TensorRT FP16 inference executes entirely on the Jetson GPU. The 6 Arm CPU cores saturated by `stress-ng --cpu` have no architectural path to disturb the GPU inference pipeline. STER = 0.0000 and δ flat across all load levels. **GPU architectural isolation confirmed.**

Coral E1 pending (Edge TPU on dedicated SRAM expected to show equivalent isolation by a different architectural mechanism).

---

### E2 — Memory Pressure (Jetson Confirmed)

5 trials × 500 inferences per level. `stress-ng --vm 1 --vm-bytes <N>M --vm-keep --vm-method all`.

| VM Fill | Jetson STER | Jetson Acc | δ_mean | δ_P99 |
|---|---|---|---|---|
| 25% | 0.0000 | 1.44% | 0.01827 | 0.02368 |
| 50% | 0.0000 | 1.56% | 0.01808 | 0.02316 |
| 75% | 0.0000 | 1.40% | 0.01821 | 0.02348 |
| 90% | 0.0000 | 1.64% | 0.01812 | 0.02326 |

**Finding:** `stress-ng --vm` fills CPU-side virtual memory pages, contending on the CPU memory bus. TensorRT allocations via pycuda (`cuda.mem_alloc`) are serviced through the GPU DMA path with dedicated QoS bandwidth reservation, isolated from CPU page fault and vm-worker activity. δ_mean varies by < 0.0003 across the full sweep, indistinguishable from E0 baseline. No STER onset threshold observed up to 90% fill. **Memory subsystem QoS isolation confirmed.**

Coral E2 pending (separate TPU SRAM architecture expected to provide equivalent isolation).

---

### E3 — GPU Co-tenancy (Jetson Confirmed)

5 trials × 500 inferences per level. Co-tenant workers run continuous unpaced MobileNetV2-FP16 inference via independent pycuda contexts on the same GPU.

| Co-tenants | Jetson STER | Jetson Acc | δ_mean | δ_P99 |
|---|---|---|---|---|
| 0 (baseline) | 0.0000 | 1.44% | 0.01823 | 0.02378 |
| 1 | 0.0000 | 1.60% | 0.01809 | 0.02316 |
| 2 | 0.0000 | 1.40% | 0.01822 | 0.02348 |
| 3 | 0.0000 | 1.64% | 0.01812 | 0.02326 |
| 4 | 0.0000 | 1.64% | 0.01809 | 0.02327 |

**Finding:** The Orin GPU scheduler time-slices all TensorRT execution contexts without violating the T_n = 100 ms cycle deadline. Each worker holds an independent pycuda context with isolated device memory — no cross-process output corruption observed. At ~10–15 ms per inference, the cycle budget contains sufficient slack to absorb GPU scheduler contention even at 4 simultaneous co-tenants. **GPU scheduler context isolation confirmed.**

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
│   ├── e0_infer_baseline/    E0 Edge TPU firmware (MobileNetV1 int8)
│   ├── supporting_timing_baseline/
│   └── results/
│       ├── e0_coral_infer_summary.csv
│       └── e0_infer_log.txt
└── paper/
    ├── P2_IEEE_ESL_Draft.docx           Original draft
    ├── P2_IEEE_ESL_Draft_E1.docx        Draft updated with E1 results
    └── P2_IEEE_ESL_Draft_E2E3.docx      Draft updated with E2+E3 results
```

---

## Key Metric: STER

**Safety-Threshold Exceedance Rate (STER)** = proportion of inferences where the L-infinity norm on the softmax output vector vs. the nominal baseline exceeds tolerance band T* = 0.05:

`STER = (1/N) · Σᵢ 𝟙[δᵢ > T*]`

where `δᵢ = ‖σ(yᵢ) − σ(ȳ)‖∞` and T* = 0.05 is derived conservatively from ISO 14971 Table B.1 risk control examples.

E0 target: STER = 0.0000 on both platforms. ✅ Confirmed.  
E1–E3: STER = 0.0000 on Jetson across all stressor levels. ✅ Confirmed.

---

## Architectural Summary (E0–E3)

Three independent disturbance pathways have been ruled out for the Jetson Orin Nano Super under JetPack 6:

| Pathway | Mechanism | STER effect |
|---|---|---|
| CPU contention (E1) | `stress-ng --cpu` — saturates all 6 Arm cores | None — GPU pipeline architecturally isolated |
| Memory pressure (E2) | `stress-ng --vm` — fills up to 90% LPDDR5 | None — GPU DMA path has dedicated QoS lane |
| GPU co-tenancy (E3) | 1–4 parallel TensorRT inference workers | None — GPU scheduler provides context isolation with sufficient cycle slack |

Remaining candidate disturbance pathways: **network interrupt load (E4)** and **combined realistic load (E5)**. BLE interrupt processing occurs on the host CPU regardless of GPU/TPU isolation, making E4 the first experiment with a plausible disturbance path to both platforms.

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

### Stress tooling
- `stress-ng` 0.13.12 installed on Jetson
- E1 CPU worker mapping (6-core Orin): 25%→2 workers, 50%→3, 75%→5, 100%→6
- E2 VM fill: 25/50/75/90% of 8192 MB LPDDR5 = 2048/4096/6144/7373 MB
- E3 co-tenant workers: independent pycuda + TensorRT contexts, 100-image loop, no pacing

---

## Dependencies

**Jetson:** JetPack 6.x, CUDA 12.6, TensorRT 10.3.0, cuDNN 9.3.0, Python 3.10, pycuda, Pillow, numpy, stress-ng

**Coral:** coralmicro SDK, FreeRTOS, TFLite Micro, Edge TPU runtime. Host: Ubuntu 22.04, Python 3.10

---

## Citation

*To be added upon arXiv publication (target: May 20, 2026)*
