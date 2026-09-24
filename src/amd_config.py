"""AMD GPU Configuration Manager and ROCm Environment Controller for EdgeShield AI.

Parses config/amd_gpu.yaml, manages hardware profiles (Instinct MI300/MI250, Radeon RX 7000/6000,
Ryzen AI NPU), and applies necessary ROCm/HIP environment variables (HSA_OVERRIDE_GFX_VERSION,
HIP_VISIBLE_DEVICES, MIOPEN_FIND_MODE, etc.) prior to model initialization.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "amd_gpu.yaml"


@dataclass
class AMDGPUProfile:
    """Settings for a specific AMD hardware target."""

    name: str
    architecture: str
    hsa_override_gfx_version: str = ""
    pytorch_rocm_arch: str = ""
    half_precision: bool = True
    batch_size: int = 1
    vram_target_gb: int = 0
    miopen_find_mode: str = "FAST"
    description: str = ""
    onnx_execution_provider: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AMDGPUConfig:
    """Complete AMD GPU runtime configuration."""

    active_profile: str
    environment: dict[str, str] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, AMDGPUProfile] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_profile": self.active_profile,
            "environment": self.environment,
            "inference": self.inference,
            "profiles": {k: v.as_dict() for k, v in self.profiles.items()},
        }


def load_amd_gpu_config(config_path: str | Path | None = None) -> AMDGPUConfig:
    """Load AMD GPU configuration from YAML file or return robust defaults."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        logger.warning("AMD GPU config not found at %s. Using default profile.", path)
        return AMDGPUConfig(
            active_profile="auto",
            environment={"HIP_VISIBLE_DEVICES": "0", "MIOPEN_FIND_MODE": "FAST"},
            inference={"device": "cuda:0", "half_precision": True},
            profiles={},
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_profiles = data.get("profiles", {})
    profiles = {}
    for k, v in raw_profiles.items():
        if isinstance(v, dict):
            profiles[k] = AMDGPUProfile(
                name=v.get("name", k),
                architecture=v.get("architecture", "Unknown"),
                hsa_override_gfx_version=str(v.get("hsa_override_gfx_version", "")),
                pytorch_rocm_arch=str(v.get("pytorch_rocm_arch", "")),
                half_precision=bool(v.get("half_precision", True)),
                batch_size=int(v.get("batch_size", 1)),
                vram_target_gb=int(v.get("vram_target_gb", 0)),
                miopen_find_mode=str(v.get("miopen_find_mode", "FAST")),
                description=str(v.get("description", "")),
                onnx_execution_provider=v.get("onnx_execution_provider"),
            )

    return AMDGPUConfig(
        active_profile=str(data.get("active_profile", "auto")),
        environment={str(k): str(v) for k, v in data.get("environment", {}).items()},
        inference=dict(data.get("inference", {})),
        profiles=profiles,
    )


def apply_amd_gpu_environment(
    profile_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """Apply ROCm and HIP environment variables for the selected AMD profile to os.environ.

    Returns:
        dict of environment variables that were set.
    """
    config = load_amd_gpu_config(config_path)
    target_profile_key = profile_name or config.active_profile

    applied_vars: dict[str, str] = {}

    # 1. Apply global environment defaults
    for key, val in config.environment.items():
        if val != "":
            os.environ[key] = str(val)
            applied_vars[key] = str(val)

    # 2. Apply profile-specific overrides if a known profile is selected
    if target_profile_key in config.profiles:
        profile = config.profiles[target_profile_key]
        if profile.hsa_override_gfx_version:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = profile.hsa_override_gfx_version
            applied_vars["HSA_OVERRIDE_GFX_VERSION"] = profile.hsa_override_gfx_version

        if profile.pytorch_rocm_arch:
            os.environ["PYTORCH_ROCM_ARCH"] = profile.pytorch_rocm_arch
            applied_vars["PYTORCH_ROCM_ARCH"] = profile.pytorch_rocm_arch

        if profile.miopen_find_mode:
            os.environ["MIOPEN_FIND_MODE"] = profile.miopen_find_mode
            applied_vars["MIOPEN_FIND_MODE"] = profile.miopen_find_mode

    logger.debug("Applied AMD GPU environment variables: %s", applied_vars)
    return applied_vars


def set_active_profile(
    profile_name: str,
    config_path: str | Path | None = None,
) -> AMDGPUConfig:
    """Set the active hardware profile in config/amd_gpu.yaml and apply its environment."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_amd_gpu_config(path)

    if profile_name != "auto" and profile_name not in config.profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available: {list(config.profiles.keys()) + ['auto']}"
        )

    config.active_profile = profile_name

    # Save updated YAML file
    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    raw_data["active_profile"] = profile_name

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_data, f, sort_keys=False)

    apply_amd_gpu_environment(profile_name, config_path=path)
    return config
