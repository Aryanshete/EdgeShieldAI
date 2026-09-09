# EdgeShield AI — Verified Performance Measurements

**Measured Timestamp:** `2026-09-09 15:28:49`  
**Hardware Backend:** `CPU` (`cpu`)  
**Detection Model:** `yolov8n.pt`  
**Dataset Evaluated:** `intrusion_demo.mp4` (`480` frames)  

---

## 1. Verified Component Latency Breakdown

| Pipeline Component | Metric Type | Measured Latency | Target Threshold |
|:---|:---:|:---:|:---:|
| **YOLOv8 Detection** | Mean / P95 | **54.64 ms** / 45.84 ms | < 50 ms |
| **ByteTrack Tracking** | Mean / P95 | **40.36 ms** / 45.29 ms | < 30 ms |
| **Zone & Event Engine** | Mean / P95 | **0.11 ms** / 0.14 ms | < 5 ms |
| **Temporal Context Synthesis** | Single Batch | **0.37 ms** | < 10 ms |
| **Deterministic Risk Engine** | Single Context | **0.03 ms** | < 2 ms |
| **AI Reasoning Interpretation** | Full Explanation | **0.05 ms** | < 100 ms |

---

## 2. System Throughput & Resource Utilization

| Metric | Result | Target Criteria |
|:---|:---:|:---:|
| **Video Processing Throughput** | **10.18 FPS** | ≥ 15 FPS (Real-time capability) |
| **Total Pipeline Decision Time** | **47167.56 ms** | Complete pipeline cycle |
| **Peak Resident Memory Footprint** | **82.7 MB** | Lightweight edge deployment |

---
*Note: Per Rule 3 of the EdgeShield specification, all metrics are recorded from direct hardware measurements. No synthetic or placeholder numbers are used.*
