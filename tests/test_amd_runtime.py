"""Unit tests for AMD ROCm, Ryzen AI NPU, and ONNX Runtime execution stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.amd_backend import (
    AMDHardwareTier,
    classify_hardware_tier,
    get_amd_reference_benchmarks,
    probe_amd_telemetry,
)
from src.detector import DEFAULT_MODEL_PATH, DEFAULT_ONNX_PATH, YOLODetector
from src.runtime import RuntimeStatus, get_runtime_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _MockCudaROCm:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        return "AMD Instinct MI300X Accelerator"


class _MockTorchROCm:
    __version__ = "2.3.0+rocm6.2"
    cuda = _MockCudaROCm()

    class version:
        hip = "6.2.0"


class _MockOrtVitisAI:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["VitisAIExecutionProvider", "CPUExecutionProvider"]


class _MockOrtDML:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_classify_hardware_tier_rocm_instinct() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name="AMD Instinct MI250",
        hip_version="6.2.0",
        onnx_providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
        cuda_available=True,
    )
    assert tier == AMDHardwareTier.AMD_INSTINCT.value
    assert is_amd is True


def test_classify_hardware_tier_rocm_radeon() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name="AMD Radeon RX 7900 XTX",
        hip_version="6.2.0",
        onnx_providers=[],
        cuda_available=True,
    )
    assert tier == AMDHardwareTier.AMD_RADEON.value
    assert is_amd is True


def test_classify_hardware_tier_ryzen_ai_npu() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name=None,
        hip_version=None,
        onnx_providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        cuda_available=False,
    )
    assert tier == AMDHardwareTier.AMD_RYZEN_AI.value
    assert is_amd is True


def test_classify_hardware_tier_directml_amd_apu() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name="AMD Radeon 780M Graphics",
        hip_version=None,
        onnx_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        cuda_available=False,
    )
    assert tier == AMDHardwareTier.AMD_APU.value
    assert is_amd is True


def test_classify_hardware_tier_cuda_non_amd() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name="NVIDIA GeForce RTX 3050 Laptop GPU",
        hip_version=None,
        onnx_providers=["CPUExecutionProvider"],
        cuda_available=True,
    )
    assert tier == AMDHardwareTier.CUDA_NON_AMD.value
    assert is_amd is False


def test_classify_hardware_tier_cpu_reference() -> None:
    tier, is_amd = classify_hardware_tier(
        device_name=None,
        hip_version=None,
        onnx_providers=["CPUExecutionProvider"],
        cuda_available=False,
    )
    assert tier == AMDHardwareTier.CPU_BASELINE.value
    assert is_amd is False


def test_probe_amd_telemetry_with_rocm_mock() -> None:
    telemetry = probe_amd_telemetry(torch_module=_MockTorchROCm())
    assert telemetry.rocm_active is True
    assert telemetry.hip_version == "6.2.0"
    assert telemetry.is_amd_hardware is True
    assert "Instinct" in telemetry.gpu_device_name


def test_probe_amd_telemetry_with_vitis_ai_mock() -> None:
    telemetry = probe_amd_telemetry(torch_module=None, ort_module=_MockOrtVitisAI())
    assert "VitisAIExecutionProvider" in telemetry.onnx_providers
    assert telemetry.active_onnx_provider == "VitisAIExecutionProvider"


def test_runtime_status_contains_amd_fields() -> None:
    status = get_runtime_status()
    assert isinstance(status, RuntimeStatus)
    assert hasattr(status, "onnx_providers")
    assert hasattr(status, "active_onnx_provider")
    assert hasattr(status, "is_amd_hardware")
    assert hasattr(status, "hardware_tier")
    assert isinstance(status.onnx_providers, list)
    status_dict = status.as_dict()
    assert "onnx_providers" in status_dict
    assert "hardware_tier" in status_dict


def test_onnx_detector_initialization_and_inference() -> None:
    if not DEFAULT_ONNX_PATH.is_file():
        pytest.skip("models/yolov8n.onnx not present for testing")

    detector = YOLODetector(model_path=DEFAULT_ONNX_PATH, backend="onnx")
    info = detector.runtime_info()
    assert info["model_format"] == "ONNX"
    assert info["is_onnx_runtime"] is True

    # Test dummy frame inference
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)


def test_amd_reference_benchmarks_structure() -> None:
    ref = get_amd_reference_benchmarks()
    assert "architectures" in ref
    assert len(ref["architectures"]) >= 5
    tiers = [a["tier"] for a in ref["architectures"]]
    assert any("MI300X" in t for t in tiers)
    assert any("Ryzen" in t for t in tiers)
    assert any("CPU" in t for t in tiers)


def test_load_amd_gpu_config() -> None:
    from src.amd_config import AMDGPUConfig, load_amd_gpu_config

    config = load_amd_gpu_config()
    assert isinstance(config, AMDGPUConfig)
    assert "instinct_mi300" in config.profiles
    assert "radeon_rx7900" in config.profiles
    assert config.profiles["instinct_mi300"].hsa_override_gfx_version == "9.4.2"
    assert config.profiles["radeon_rx7900"].hsa_override_gfx_version == "11.0.0"


def test_apply_amd_gpu_environment_profile() -> None:
    from src.amd_config import apply_amd_gpu_environment

    applied = apply_amd_gpu_environment(profile_name="radeon_rx7900")
    assert applied.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert applied.get("PYTORCH_ROCM_ARCH") == "gfx1100"
    assert applied.get("MIOPEN_FIND_MODE") == "FAST"


def test_set_active_profile_and_reset(tmp_path: Path) -> None:
    import shutil
    from src.amd_config import DEFAULT_CONFIG_PATH, load_amd_gpu_config, set_active_profile

    temp_config = tmp_path / "amd_gpu.yaml"
    shutil.copy2(DEFAULT_CONFIG_PATH, temp_config)

    set_active_profile("instinct_mi300", config_path=temp_config)
    reloaded = load_amd_gpu_config(temp_config)
    assert reloaded.active_profile == "instinct_mi300"

    with pytest.raises(ValueError, match="Unknown profile"):
        set_active_profile("non_existent_profile_xyz", config_path=temp_config)

