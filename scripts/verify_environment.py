"""Validate the Python runtime and dependencies without loading model weights."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


# Ultralytics creates a settings file during its first import. Keep that
# runtime-only file inside this project rather than relying on a user-profile
# path, which is often unavailable in isolated deployments.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))


REQUIRED_IMPORTS = {
    "torch": "torch",
    "OpenCV": "cv2",
    "Streamlit": "streamlit",
    "Ultralytics YOLO": "ultralytics",
    "NumPy": "numpy",
    "Pandas": "pandas",
    "Pillow": "PIL",
    "PyYAML": "yaml",
}


def main() -> int:
    """Print package and PyTorch device information; return nonzero on failure."""
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    failures: list[str] = []

    for label, module_name in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {label}: {version}")
        except (ImportError, OSError) as error:
            failures.append(label)
            print(f"[UNAVAILABLE] {label}: {error}")

    if "torch" not in failures:
        torch = importlib.import_module("torch")
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA/ROCm available: {cuda_available}")
        if cuda_available:
            print(f"PyTorch device: {torch.cuda.get_device_name(0)}")
        else:
            print("PyTorch device: CPU (expected for local development if no GPU runtime is configured)")

    if failures:
        print("\nInstall missing dependencies with: python -m pip install -r requirements.txt")
        return 1

    print("\nEnvironment verification passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
