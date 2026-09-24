"""Unit tests for performance measurement and benchmark profiling."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.benchmark import PerformanceMetrics, PipelineBenchmark, calculate_percentile


def test_calculate_percentile() -> None:
    data = list(range(1, 101))  # 1 to 100
    assert calculate_percentile(data, 50) == 51.0
    assert calculate_percentile(data, 95) == 96.0
    assert calculate_percentile([], 95) == 0.0


def test_performance_metrics_as_dict() -> None:
    metrics = PerformanceMetrics(
        frames_processed=100,
        total_elapsed_seconds=5.0,
        processing_fps=20.0,
        yolo_mean_ms=30.5,
        yolo_p95_ms=45.0,
        tracking_mean_ms=12.0,
        tracking_p95_ms=18.0,
        event_processing_mean_ms=1.2,
        event_processing_p95_ms=2.0,
        context_synthesis_ms=0.8,
        risk_evaluation_ms=0.3,
        reasoning_ms=15.0,
        end_to_end_decision_ms=5000.0,
        memory_peak_mb=150.0,
        backend="CPU",
        device="cpu",
        model="yolov8n.pt",
        timestamp="2026-09-09 12:00:00",
    )
    d = metrics.as_dict()
    assert d["frames_processed"] == 100
    assert d["processing_fps"] == 20.0
    assert d["yolo_mean_ms"] == 30.5
    assert d["backend"] == "CPU"


def test_generate_markdown_report() -> None:
    metrics = PerformanceMetrics(
        frames_processed=50,
        total_elapsed_seconds=2.5,
        processing_fps=20.0,
        yolo_mean_ms=28.4,
        yolo_p95_ms=35.1,
        tracking_mean_ms=10.2,
        tracking_p95_ms=14.0,
        event_processing_mean_ms=0.9,
        event_processing_p95_ms=1.5,
        context_synthesis_ms=0.5,
        risk_evaluation_ms=0.2,
        reasoning_ms=10.0,
        end_to_end_decision_ms=2500.0,
        memory_peak_mb=120.5,
        backend="CPU",
        device="cpu",
        model="yolov8n.pt",
        timestamp="2026-09-09 12:00:00",
    )
    benchmark = PipelineBenchmark()
    report = benchmark.generate_markdown_report(metrics)

    assert "# EdgeShield AI — Verified Performance Measurements" in report
    assert "YOLOv8 Detection" in report
    assert "28.4 ms" in report
    assert "20.0 FPS" in report


def test_save_reports(tmp_path: Path) -> None:
    metrics = PerformanceMetrics(
        frames_processed=10,
        total_elapsed_seconds=0.5,
        processing_fps=20.0,
        yolo_mean_ms=25.0,
        yolo_p95_ms=30.0,
        tracking_mean_ms=10.0,
        tracking_p95_ms=12.0,
        event_processing_mean_ms=1.0,
        event_processing_p95_ms=1.5,
        context_synthesis_ms=0.4,
        risk_evaluation_ms=0.2,
        reasoning_ms=5.0,
        end_to_end_decision_ms=500.0,
        memory_peak_mb=100.0,
        backend="CPU",
        device="cpu",
        model="yolov8n.pt",
        timestamp="2026-09-09 12:00:00",
    )
    benchmark = PipelineBenchmark()
    json_p, md_p = benchmark.save_reports(metrics, output_dir=tmp_path)

    assert json_p.is_file()
    assert md_p.is_file()
