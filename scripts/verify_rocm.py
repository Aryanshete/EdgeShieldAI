"""Fail clearly unless EdgeShield is running with active AMD ROCm PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/verify_rocm.py` from a fresh clone without requiring a
# package install first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import get_runtime_status


def main() -> int:
    """Print runtime facts and make ROCm verification scriptable."""
    status = get_runtime_status()
    print(status.as_dict())
    if not status.amd_rocm_active:
        print("ROCm verification failed: an AMD ROCm-enabled PyTorch accelerator is not active.")
        return 1

    print("ROCm verification passed: AMD acceleration is active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
