"""Unit tests for Phase 1 detection utilities without downloading model weights."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import YOLODetector
from src.runtime import get_runtime_status


def test_draw_detections_returns_annotated_copy() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    detections = [
        {"class_name": "person", "confidence": 0.91, "bbox": [20, 20, 80, 90]}
    ]

    annotated = YOLODetector.draw_detections(frame, detections)

    assert annotated.shape == frame.shape
    assert np.array_equal(frame, np.zeros((100, 160, 3), dtype=np.uint8))
    assert not np.array_equal(annotated, frame)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_detector_rejects_invalid_confidence_before_loading_model(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        YOLODetector(confidence_threshold=threshold)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "AMD Instinct test GPU"


class _FakeTorchRocm:
    __version__ = "test"
    cuda = _FakeCuda()

    class version:
        hip = "7.2.1"


def test_runtime_requires_hip_to_report_rocm() -> None:
    status = get_runtime_status(_FakeTorchRocm())

    assert status.backend == "ROCm"
    assert status.amd_rocm_active is True
    assert status.rocm_version == "7.2.1"
