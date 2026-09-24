# EdgeShield AI

> **See. Understand. Act.**

EdgeShield AI is an intelligent, privacy-aware edge security analytics platform. It combines edge computer vision, anonymous multi-object tracking, spatial-temporal event generation, deterministic heuristic risk assessment, and evidence-grounded AI reasoning to deliver actionable physical security intelligence in real time.

---

## Overview

Modern physical security systems suffer from alarm fatigue, heavy cloud bandwidth costs, privacy vulnerabilities, and rigid rule engines that fail to provide actionable context. **EdgeShield AI** solves these challenges by evaluating surveillance video directly at the edge, converting raw pixels into chronological structured events, assessing risk deterministically, and using compact large language models (Llama 3.2) to synthesize plain-English incident briefings and operator recommendations—without sending continuous raw video streams to the cloud.

---

## Problem

- **Alert Fatigue & False Positives**: Traditional video analytics flood security operations centers (SOCs) with disjointed motion alerts lacking contextual understanding.
- **Bandwidth & Latency Bottlenecks**: Continuously streaming high-definition surveillance video to the cloud incurs immense network bandwidth and recurring infrastructure costs.
- **Privacy & Compliance Hazards**: Centralized storage of raw footage and biometric facial recognition poses severe GDPR, CCPA, and civil liberties compliance liabilities.
- **Unreliable "Black-Box" Alarms**: End-to-end AI models frequently hallucinate facts or fail silently without deterministic, verifiable audit trails.

---

## Solution

EdgeShield AI enforces a strict multi-stage **Sense → Structure → Score → Reason** architecture:
1. **Sense**: Ingests video streams locally using lightweight object detection (YOLOv8) and anonymous multi-object tracking (ByteTrack).
2. **Structure**: Converts continuous coordinates into discrete, verifiable security events (`zone_approach`, `restricted_zone_entry`, `restricted_zone_dwell`, `after_hours_activity`, `loitering_detected`, `abandoned_object_detected`).
3. **Score**: Calculates a transparent, deterministic baseline risk score (0–100) using audited security heuristics.
4. **Reason**: Employs Llama 3.2 to interpret the structured chronological timeline, explaining *why* the sequence matters and recommending concrete operator actions grounded strictly in observed evidence.

---

## Key Features

- **🛡️ Privacy-Aware Edge Processing**: Pure anonymous tracking (`Person #07`). No facial recognition, biometric profiling, or identity databases. Downstream reasoning operates strictly on structured metadata.
- **📐 Interactive Spatial Zones**: Define polygonal restricted, monitored, or access perimeters with zone-specific operational windows (e.g. 08:00–18:00) using OpenCV vector geometry (`cv2.pointPolygonTest`).
- **⏱️ Stateful Security Event Engine**: Detects approaches, boundary crossings, dwell duration, stationary object interactions, loitering, and unusual movements.
- **⚖️ Deterministic Risk Engine**: Reproducible baseline scoring (0–29 LOW, 30–59 MEDIUM, 60–79 HIGH, 80–100 CRITICAL) guarantees operational reliability even during network or LLM offline states.
- **🧠 Grounded Llama 3.2 Reasoning**: Zero hallucination policy. Directives strictly prohibit inventing unobserved facts or claiming proven criminal intent, using probabilistic security risk language.
- **📋 Lifecycle Incident Management**: Auto-generates formal incident audits (`INC-2026-001`) with timeline tracking, status workflows (`OPEN`, `INVESTIGATING`, `RESOLVED`, `DISMISSED`), and one-click JSON/Markdown audit report exports.
- **⚡ Configurable Security Scenarios**: Five operational scenario profiles:
  - *Scenario 1*: Restricted Area Intrusion
  - *Scenario 2*: Sensitive Area Loitering
  - *Scenario 3*: Abandoned / Unattended Object
  - *Scenario 4*: After-Hours Facility Activity
  - *Scenario 5*: Unusual High-Velocity Movement
- **🖥️ Dark Enterprise Operations Dashboard**: Streamlit SOC interface with real-time video overlays, risk gauges, live chronological timeline, and architecture verification tabs.

---

## Architecture

```text
[Surveillance Camera / RTSP Feed]
              │
              ▼
  [OpenCV Frame Ingestion]
              │
              ▼
  [YOLOv8 Object Detection]  ─── 54.64 ms (Edge Inference)
              │
              ▼
  [ByteTrack Anonymous MOT]  ─── 40.36 ms (Track ID Association)
              │
              ▼
  [Spatial Zone Engine]      ─── cv2.pointPolygonTest
              │
              ▼
  [Security Event Engine]    ─── 0.11 ms (Stateful Event Sequencing)
              │
              ▼
  [Temporal Context Engine]  ─── Chronological Synthesis
              │
              ▼
  [Deterministic Risk Engine] ─── 0-100 Heuristic Baseline
              │
              ▼
  [Llama 3.2 Reasoning Agent] ── Evidence-Grounded Context & Actions
              │
              ▼
  [Incident Manager & Audit] ─── Persistent Records (data/incidents.json)
              │
              ▼
  [Edge Operations Dashboard] ── http://localhost:8501
```


## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Camera or video files (MP4, AVI)

### Setup
```powershell
# Clone the repository
git clone https://github.com/Aryanshete/EdgeShieldAI
cd "EdgeShield AI"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### 1. Operations Dashboard
```powershell
.\.venv\Scripts\streamlit run app.py
```
Access the dark enterprise SOC dashboard at `http://localhost:8501`.

### 2. Automated Test Suite
Run the comprehensive 94-test suite covering zones, events, context, risk, reasoning, incidents, benchmarks, scenarios, privacy, failure handling, and AMD ROCm/Ryzen AI execution:
```powershell
.\.venv\Scripts\pytest -v
```

### 3. AMD Hardware Profiler & Benchmark Suite
Switch AMD GPU hardware profiles, export environment variables, and run multi-backend benchmarks:
```powershell
# Inspect and configure AMD GPU hardware target (Instinct MI300/MI250, Radeon RX 7000/6000, Ryzen AI NPU)
python scripts/configure_amd_gpu.py --list
python scripts/configure_amd_gpu.py --set instinct_mi300

# Benchmark across PyTorch and ONNX Runtime engines
python scripts/benchmark_amd.py --frames 60
```

---

## Configuration

Security zones and authorized operating windows are defined in `config/zones.json`:
```json
{
  "zones": [
    {
      "id": "server_room_perimeter",
      "name": "Server Room",
      "type": "restricted",
      "polygon": [[580, 140], [1160, 140], [1160, 680], [580, 680]],
      "authorized_hours": {
        "start": "08:00",
        "end": "18:00"
      }
    }
  ]
}
```

AMD GPU execution profiles and ROCm environment overrides are configured in `config/amd_gpu.yaml`.

---

## Project Structure

```text
EdgeShield AI/
├── app.py                      # Main entrypoint launching Streamlit SOC dashboard
├── config/
│   ├── amd_gpu.yaml            # AMD GPU hardware profiles & ROCm environment overrides
│   ├── bytetrack.yaml          # ByteTrack tracking hyper-parameters
│   └── zones.json              # Vector coordinates & hours for restricted zones
├── data/
│   ├── events.json             # Generated chronological security events
│   ├── context.json            # Synthesized temporal contexts
│   └── incidents.json          # Persistent incident registry
├── docs/                       # Architectural specs and AMD deployment guides
│   ├── amd_deployment_guide.md # End-to-end AMD deployment documentation
│   └── amd-rocm.md             # ROCm & Ryzen AI architecture reference
├── models/
│   ├── yolov8n.pt              # YOLOv8 nano edge detection PyTorch weights
│   └── yolov8n.onnx            # Exported ONNX model optimized for AMD inference
├── reports/
│   ├── performance.json        # Verified hardware benchmark metrics
│   ├── performance.md          # Markdown performance audit table
│   ├── performance_amd.json    # Multi-backend AMD performance metrics
│   └── performance_amd.md      # AMD hardware acceleration comparison report
├── scripts/
│   ├── benchmark_amd.py        # Automated multi-backend AMD profiler
│   ├── configure_amd_gpu.py    # AMD GPU profile manager & environment exporter
│   ├── deploy_amd.sh           # Automated Docker container deployment for AMD
│   ├── export_amd_model.py     # YOLOv8 ONNX model export for AMD targets
│   ├── run_benchmark.py        # Pipeline latency measurement script
│   ├── setup_rocm.sh           # Linux ROCm host initialization & driver setup
│   ├── verify_environment.py   # Runtime and dependency verifier
│   └── verify_rocm.py          # AMD ROCm hardware probe
├── src/
│   ├── amd_backend.py          # AMD hardware telemetry & execution provider engine
│   ├── amd_config.py           # AMD GPU YAML profile loader & environment manager
│   ├── benchmark.py            # Latency and memory profiling engine
│   ├── context.py              # Temporal Event Analysis & context synthesis
│   ├── detector.py             # YOLOv8 PyTorch & ONNX object detection wrapper
│   ├── events.py               # Security Event Engine & chronological sequencer
│   ├── incidents.py            # Incident lifecycle manager & audit reports
│   ├── privacy.py              # Privacy-aware architecture & audit compliance
│   ├── reasoning.py            # Llama 3.2 reasoning agent & deterministic fallback
│   ├── risk.py                 # Deterministic heuristic scoring engine
│   ├── runtime.py              # Truthful ML backend & ROCm detection
│   ├── scenarios.py            # Secondary scenario analyzers (loitering, abandoned)
│   ├── tracker.py              # ByteTrack anonymous multi-object tracking
│   └── zones.py                # Polygon spatial analysis & OpenCV point tests
├── tests/                      # 94 unit tests covering 100% of pipeline modules
├── ui/
│   ├── components.py           # Enterprise cybersecurity dark CSS & AMD telemetry panel
│   └── dashboard.py            # Multi-tab operational SOC interface with AMD Hub
├── videos/demo/                # Evaluation surveillance footage
├── Dockerfile.rocm             # Container configuration for AMD ROCm cloud deployment
├── docker-compose.rocm.yml     # Compose with /dev/kfd and /dev/dri hardware passthrough
├── requirements.txt            # Python dependencies (PyTorch, ONNX, OpenCV, Streamlit)
└── README.md                   # System documentation
```

---

## Performance

The following results represent **actual measured performance** from the local edge verification suite (`reports/performance.json`), evaluated across 480 surveillance frames (1280x720):

| Component | Mean Latency | Min Latency | Max Latency | P95 Latency | Operational Role |
|:---|:---:|:---:|:---:|:---:|:---|
| **YOLOv8n Detection** | **54.64 ms** | 43.51 ms | 83.05 ms | 61.35 ms | Person & object proposal |
| **ByteTrack Tracking** | **40.36 ms** | 32.55 ms | 65.40 ms | 48.20 ms | Anonymous spatial-temporal association |
| **Event Engine** | **0.11 ms** | 0.00 ms | 1.00 ms | 1.00 ms | Zone containment & state transitions |
| **End-to-End Frame** | **95.11 ms** | 77.06 ms | 134.45 ms | 108.75 ms | Complete perception cycle |
| **Overall Throughput** | **10.18 FPS** | — | — | — | Real-time video processing |
| **Peak Memory RSS** | **82.7 MB** | — | — | — | Edge-compatible memory footprint |

*Tested on multi-core host CPU architecture, PyTorch runtime.*

---

## Limitations

- **Camera Cuts**: Multi-object tracking (ByteTrack) relies on spatial continuity; hard camera cuts will reset anonymous track IDs (e.g. Person #01 becoming Person #02).
- **Extreme Occlusion**: Severe physical occlusions or low-light conditions may degrade detection confidence.
- **Edge LLM Hardware**: Running local 3B+ parameter LLMs at low latency requires dedicated GPU acceleration (such as AMD ROCm) or an edge AI coprocessor.

---

## Future Work

1. **AMD ROCm Cloud Deployment**: Execute distributed inference on AMD Instinct™ MI300/MI250 hardware accelerators.
2. **Re-Identification Across Disjoint Cameras**: Privacy-preserving feature embedding re-identification without facial recognition.
3. **ONNX Runtime / TensorRT / ROCm MIOpen Acceleration**: Int8 model quantization for ultra-low-power edge nodes (sub-15ms inference).
4. **Automated PTZ Hand-off**: Automated pan-tilt-zoom camera tracking driven by zone breach coordinates.

---

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
