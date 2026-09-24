# AMD ROCm™ & Ryzen™ AI Runtime Architecture

## 1. Truthful Runtime Detection

EdgeShield AI enforces strict hardware honesty:
- **ROCm Active**: Requires both `torch.cuda.is_available() == True` and a non-empty `torch.version.hip`.
- **Ryzen AI Active**: Requires ONNX Runtime with `VitisAIExecutionProvider` (AMD XDNA NPU).
- **DirectML Active**: Requires `DmlExecutionProvider` targeting an AMD GPU/APU via DirectX 12.
- **CUDA (Non-AMD)**: Identifies NVIDIA accelerators without fabricating AMD claims.
- **CPU Reference**: Used whenever no compatible accelerator is present.

Run the verification probe in any environment:
```bash
python scripts/verify_rocm.py
```

---

## 2. AMD GPU Configuration & Profiles (`config/amd_gpu.yaml`)

EdgeShield AI manages AMD GPU hardware targeting through `config/amd_gpu.yaml` and the `scripts/configure_amd_gpu.py` utility.

### Pre-Tuned Hardware Profiles

| Profile Key | Target Hardware | Architecture | HSA Override GFX | FP16 Precision |
|:---|:---|:---|:---:|:---:|
| `instinct_mi300` | AMD Instinct™ MI300X / MI300A | CDNA™ 3 (gfx942) | `9.4.2` | Enabled |
| `instinct_mi250` | AMD Instinct™ MI250X / MI250 / MI210 | CDNA™ 2 (gfx90a) | `9.0.a` | Enabled |
| `radeon_rx7900` | AMD Radeon™ RX 7900 XTX / 7800 XT | RDNA™ 3 (gfx1100) | `11.0.0` | Enabled |
| `radeon_rx6000` | AMD Radeon™ RX 6900 XT / 6800 XT | RDNA™ 2 (gfx1030) | `10.3.0` | Enabled |
| `ryzen_ai_npu` | AMD Ryzen™ AI 300 / 8040 / 7040 | XDNA™ NPU | Native | NPU INT8/FP16 |
| `auto` | Auto-detect optimal host settings | Auto | Auto | Enabled if GPU |

### Switching Hardware Profiles
```bash
# List all profiles
python scripts/configure_amd_gpu.py --list

# Switch to AMD Instinct MI300
python scripts/configure_amd_gpu.py --set instinct_mi300

# Switch to AMD Radeon RX 7900
python scripts/configure_amd_gpu.py --set radeon_rx7900

# Export environment variables for shell
# Linux / Bash:
eval $(python3 scripts/configure_amd_gpu.py --export-bash)
# Windows PowerShell:
python scripts/configure_amd_gpu.py --export-ps1 | Invoke-Expression
```

---

## 3. Container & Cloud Deployment

### Docker Compose with AMD GPU Passthrough
```bash
# Export optimized ONNX model
python scripts/export_amd_model.py

# Launch container with /dev/kfd and /dev/dri device access
docker compose -f docker-compose.rocm.yml up --build -d
```

### Automated Setup Scripts
- `scripts/setup_rocm.sh`: Initializes AMD ROCm host, checks `/dev/kfd`, and installs drivers.
- `scripts/deploy_amd.sh`: Builds container, probes hardware, and starts the SOC dashboard.
- `scripts/benchmark_amd.py`: Benchmarks PyTorch and ONNX inference, generating `reports/performance_amd.md`.

For in-depth deployment instructions, refer to [AMD Deployment Guide](file:///docs/amd_deployment_guide.md).
