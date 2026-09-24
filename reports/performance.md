# EdgeShield AI — Verified Performance Measurements

**Measured Timestamp:** `2026-09-09 16:30:12`  
**Hardware Backend:** `CPU` (`cpu`)  
**Detection Model:** `yolov8n.pt`  
**Dataset Evaluated:** `intrusion_demo.mp4` (`30` frames)  

---

## 1. Verified Component Latency Breakdown

| Pipeline Component | Metric Type | Measured Latency | Target Threshold |
|:---|:---:|:---:|:---:|
| **YOLOv8 Detection** | Mean / P95 | **284.72 ms** / 69.42 ms | < 50 ms |
| **ByteTrack Tracking** | Mean / P95 | **45.29 ms** / 61.03 ms | < 30 ms |
| **Zone & Event Engine** | Mean / P95 | **0.15 ms** / 0.18 ms | < 5 ms |
| **Temporal Context Synthesis** | Single Batch | **0.2 ms** | < 10 ms |
| **Deterministic Risk Engine** | Single Context | **0.05 ms** | < 2 ms |
| **AI Reasoning Interpretation** | Full Explanation | **0.05 ms** | < 100 ms |

---

## 2. System Throughput & Resource Utilization

| Metric | Result | Target Criteria |
|:---|:---:|:---:|
| **Video Processing Throughput** | **2.99 FPS** | ≥ 15 FPS (Real-time capability) |
| **Total Pipeline Decision Time** | **10032.04 ms** | Complete pipeline cycle |
| **Peak Resident Memory Footprint** | **81.95 MB** | Lightweight edge deployment |

---
*Note: All latency and throughput metrics are measured directly on local hardware without synthetic interpolation.*

