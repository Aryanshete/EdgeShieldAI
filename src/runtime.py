"""Truthful ML runtime detection for local and AMD ROCm deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeStatus:
    """Runtime facts derived from PyTorch, not inferred from a device label."""

    backend: str
    device: str
    device_name: str | None
    accelerator_available: bool
    amd_rocm_active: bool
    rocm_version: str | None
    pytorch_version: str

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return JSON-ready runtime information for the UI and reports."""
        return asdict(self)


def get_runtime_status(torch_module: Any = torch) -> RuntimeStatus:
    """Detect whether the active PyTorch build is ROCm, CUDA, or CPU.

    PyTorch exposes ROCm devices through the ``torch.cuda`` interface. A
    non-empty ``torch.version.hip`` together with an available accelerator is
    therefore required before EdgeShield reports AMD ROCm acceleration.
    """
    hip_version = getattr(torch_module.version, "hip", None)
    accelerator_available = bool(torch_module.cuda.is_available())
    device_name: str | None = None

    if accelerator_available:
        try:
            device_name = str(torch_module.cuda.get_device_name(0))
        except Exception:  # Device metadata must not stop the application.
            device_name = None

    if accelerator_available and hip_version:
        backend = "ROCm"
        device = "cuda:0"
        amd_rocm_active = True
    elif accelerator_available:
        backend = "CUDA"
        device = "cuda:0"
        amd_rocm_active = False
    else:
        backend = "CPU"
        device = "cpu"
        amd_rocm_active = False

    return RuntimeStatus(
        backend=backend,
        device=device,
        device_name=device_name,
        accelerator_available=accelerator_available,
        amd_rocm_active=amd_rocm_active,
        rocm_version=str(hip_version) if hip_version else None,
        pytorch_version=str(torch_module.__version__),
    )
