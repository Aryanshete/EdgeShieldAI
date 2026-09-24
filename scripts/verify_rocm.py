"""Fail clearly unless EdgeShield is running with active AMD ROCm or Ryzen AI acceleration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow `python scripts/verify_rocm.py` from a fresh clone without requiring a package install first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amd_backend import probe_amd_telemetry
from src.runtime import get_runtime_status


def main() -> int:
    """Print truthful runtime facts and make ROCm / AMD verification scriptable."""
    parser = argparse.ArgumentParser(description="Verify AMD ROCm and Ryzen AI hardware environment.")
    parser.add_argument(
        "--allow-edge",
        action="store_true",
        help="Allow AMD Ryzen AI NPU or DirectML acceleration to pass verification if ROCm HIP is inactive.",
    )
    args = parser.parse_args()

    status = get_runtime_status()
    telemetry = probe_amd_telemetry()

    print("=" * 65)
    print("EDGESHIELD AI — AMD HARDWARE VERIFICATION PROBE")
    print("=" * 65)
    print(f"Hardware Tier:         {status.hardware_tier}")
    print(f"Device Name:           {status.device_name or 'Host CPU'}")
    print(f"ROCm HIP Active:       {status.amd_rocm_active} (HIP: {status.rocm_version or 'N/A'})")
    print(f"Inference Device:      {status.device} (Backend: {status.backend})")
    print(f"ONNX Providers:        {', '.join(status.onnx_providers) or 'None'}")
    print(f"Active ONNX Target:    {status.active_onnx_provider or 'None'}")
    print(f"VRAM Allocated / Total: {status.vram_used_mb or 0} MB / {status.vram_total_mb or 0} MB")
    print(f"PyTorch Version:       {status.pytorch_version}")
    print("=" * 65)

    print("\nDetailed Telemetry Dict:")
    print(json.dumps(status.as_dict(), indent=2))

    if status.amd_rocm_active:
        print("\n[+] ROCm verification passed: AMD ROCm HIP acceleration is active.")
        return 0

    if args.allow_edge and status.is_amd_hardware:
        print(f"\n[+] AMD Edge verification passed: Hardware accelerated via {status.active_onnx_provider}.")
        return 0

    print("\n[-] ROCm verification failed: an AMD ROCm-enabled PyTorch accelerator is not active.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
