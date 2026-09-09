"""CLI script to run performance benchmarking across the EdgeShield AI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import PipelineBenchmark
DEFAULT_VIDEO = PROJECT_ROOT / "videos" / "demo" / "intrusion_demo.mp4"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark EdgeShield AI video security pipeline.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Path to input surveillance video")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to benchmark (default: all)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS, help="Directory to save audit reports")
    args = parser.parse_args()

    print(f"Starting EdgeShield AI performance benchmark...")
    print(f"Input video: {args.video}")
    print(f"Max frames: {args.frames or 'All'}")

    benchmark = PipelineBenchmark()
    metrics = benchmark.run(args.video, max_frames=args.frames)

    json_path, md_path = benchmark.save_reports(metrics, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED — {metrics.frames_processed} frames in {metrics.total_elapsed_seconds}s")
    print("=" * 60)
    print(f"  • Video Processing:      {metrics.processing_fps} FPS")
    print(f"  • YOLOv8 Inference:       Mean: {metrics.yolo_mean_ms} ms | P95: {metrics.yolo_p95_ms} ms")
    print(f"  • ByteTrack Tracking:     Mean: {metrics.tracking_mean_ms} ms | P95: {metrics.tracking_p95_ms} ms")
    print(f"  • Zone & Event Engine:    Mean: {metrics.event_processing_mean_ms} ms | P95: {metrics.event_processing_p95_ms} ms")
    print(f"  • Context Synthesis:      {metrics.context_synthesis_ms} ms")
    print(f"  • Risk Calculation:       {metrics.risk_evaluation_ms} ms")
    print(f"  • Reasoning Agent:        {metrics.reasoning_ms} ms")
    print(f"  • Total Pipeline Latency: {metrics.end_to_end_decision_ms} ms")
    print(f"  • Peak Memory RSS:        {metrics.memory_peak_mb} MB")
    print(f"  • Active Device:          {metrics.backend} ({metrics.device})")
    print("=" * 60)
    print(f"Saved audit reports to:\n  - {json_path}\n  - {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
