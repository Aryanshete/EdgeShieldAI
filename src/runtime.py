"""Truthful ML runtime detection for local and AMD ROCm/Ryzen deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeStatus:
    """Runtime facts derived from PyTorch and ONNX, not inferred from a device label."""

    backend: str
    device: str
    device_name: str | None
    accelerator_available: bool
    amd_rocm_active: bool
    rocm_version: str | None
    pytorch_version: str
    onnx_providers: list[str] = field(default_factory=list)
    active_onnx_provider: str | None = None
    is_amd_hardware: bool = False
    hardware_tier: str = "CPU Reference Baseline (Deterministic Fallback)"
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready runtime information for the UI and reports."""
        return asdict(self)


def get_runtime_status(torch_module: Any = torch, ort_module: Any = None) -> RuntimeStatus:
    """Detect whether the active ML build is ROCm, CUDA, or CPU, and inspect AMD tiers.

    PyTorch exposes ROCm devices through the ``torch.cuda`` interface. A
    non-empty ``torch.version.hip`` together with an available accelerator is
    therefore required before EdgeShield reports AMD ROCm acceleration.
    """
    hip_version: str | None = None
    if hasattr(torch_module, "version"):
        raw_hip = getattr(torch_module.version, "hip", None)
        if raw_hip:
            hip_version = str(raw_hip)

    accelerator_available = False
    if hasattr(torch_module, "cuda"):
        try:
            accelerator_available = bool(torch_module.cuda.is_available())
        except Exception:
            accelerator_available = False

    device_name: str | None = None
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None

    if accelerator_available:
        try:
            device_name = str(torch_module.cuda.get_device_name(0))
        except Exception:  # Device metadata must not stop the application.
            device_name = None

        try:
            if hasattr(torch_module.cuda, "get_device_properties"):
                props = torch_module.cuda.get_device_properties(0)
                vram_total_mb = round(props.total_memory / (1024 * 1024), 2)
            if hasattr(torch_module.cuda, "memory_allocated"):
                vram_used_mb = round(torch_module.cuda.memory_allocated(0) / (1024 * 1024), 2)
        except Exception:
            pass

    # Inspect ONNX Runtime providers if available
    onnx_providers: list[str] = []
    if ort_module is not None:
        try:
            onnx_providers = list(ort_module.get_available_providers())
        except Exception:
            pass
    else:
        try:
            import onnxruntime as ort

            onnx_providers = list(ort.get_available_providers())
        except Exception:
            pass

    # Classify execution backend
    if accelerator_available and hip_version:
        backend = "ROCm"
        device = "cuda:0"
        amd_rocm_active = True
    elif accelerator_available:
        backend = "CUDA"
        device = "cuda:0"
        amd_rocm_active = False
    elif "VitisAIExecutionProvider" in onnx_providers:
        backend = "Ryzen AI NPU"
        device = "npu:0"
        amd_rocm_active = False
    elif "DmlExecutionProvider" in onnx_providers:
        backend = "DirectML"
        device = "dml:0"
        amd_rocm_active = False
    else:
        backend = "CPU"
        device = "cpu"
        amd_rocm_active = False

    # Determine hardware tier and AMD hardware status
    from src.amd_backend import classify_hardware_tier

    tier, is_amd = classify_hardware_tier(
        device_name=device_name,
        hip_version=hip_version,
        onnx_providers=onnx_providers,
        cuda_available=accelerator_available,
    )

    active_provider = None
    if "VitisAIExecutionProvider" in onnx_providers:
        active_provider = "VitisAIExecutionProvider"
    elif "ROCMExecutionProvider" in onnx_providers:
        active_provider = "ROCMExecutionProvider"
    elif "MIGraphXExecutionProvider" in onnx_providers:
        active_provider = "MIGraphXExecutionProvider"
    elif "DmlExecutionProvider" in onnx_providers:
        active_provider = "DmlExecutionProvider"
    elif "CPUExecutionProvider" in onnx_providers:
        active_provider = "CPUExecutionProvider"

    pytorch_ver = str(getattr(torch_module, "__version__", "unknown"))

    return RuntimeStatus(
        backend=backend,
        device=device,
        device_name=device_name,
        accelerator_available=accelerator_available,
        amd_rocm_active=amd_rocm_active,
        rocm_version=hip_version,
        pytorch_version=pytorch_ver,
        onnx_providers=onnx_providers,
        active_onnx_provider=active_provider,
        is_amd_hardware=is_amd,
        hardware_tier=tier,
        vram_total_mb=vram_total_mb,
        vram_used_mb=vram_used_mb,
    )
