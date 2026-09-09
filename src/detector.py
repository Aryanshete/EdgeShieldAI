"""YOLOv8 object detection and annotated-video generation for EdgeShield."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Ultralytics writes settings on import. Keep those runtime files in the
# workspace rather than in a user-profile location that may be unavailable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

from src.runtime import RuntimeStatus, get_runtime_status


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
PERSON_CLASS_NAME = "person"


@dataclass(frozen=True)
class Detection:
    """A single model prediction in pixel coordinates."""

    class_name: str
    confidence: float
    bbox: list[int]

    def as_dict(self) -> dict[str, str | float | list[int]]:
        """Return the public detection structure used by later pipeline stages."""
        return asdict(self)


@dataclass(frozen=True)
class VideoDetectionSummary:
    """Measured outcome of a completed video detection run."""

    input_path: str
    output_path: str
    frames_processed: int
    frames_with_detections: int
    total_detections: int
    elapsed_seconds: float
    processing_fps: float
    model: str
    device: str

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-ready summary."""
        return asdict(self)


class YOLODetector:
    """Run YOLOv8 predictions and render their results onto OpenCV frames."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.runtime: RuntimeStatus = get_runtime_status()
        self.device = device or self.runtime.device
        self.model = YOLO(str(self.model_path))
        self.class_names: dict[int, str] = {
            int(index): str(name) for index, name in self.model.names.items()
        }

    def runtime_info(self) -> dict[str, str | bool | None]:
        """Return runtime facts for the dashboard and reproducibility records."""
        return self.runtime.as_dict() | {
            "model": self.model_path.name,
            "model_path": str(self.model_path.resolve()),
            "inference_device": self.device,
        }

    def detect(self, frame: Any) -> list[dict[str, str | float | list[int]]]:
        """Detect COCO objects in one BGR OpenCV frame.

        The returned schema deliberately contains only visual model output;
        interpretation belongs to later event and context phases.
        """
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("frame must be a valid OpenCV image")

        result = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        detections: list[dict[str, str | float | list[int]]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detection = Detection(
                class_name=self.class_names.get(class_id, f"class_{class_id}"),
                confidence=float(box.conf.item()),
                bbox=[int(value) for value in box.xyxy[0].tolist()],
            )
            detections.append(detection.as_dict())
        return detections

    @staticmethod
    def draw_detections(
        frame: Any,
        detections: Iterable[dict[str, str | float | list[int]]],
    ) -> Any:
        """Draw clearly labelled boxes on a copy of a BGR frame."""
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]  # type: ignore[misc]
            class_name = str(detection["class_name"])
            confidence = float(detection["confidence"])
            color = (50, 205, 50) if class_name == PERSON_CLASS_NAME else (0, 165, 255)
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_top = max(y1 - label_height - baseline - 6, 0)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + label_width + 8, label_top + label_height + baseline + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_top + label_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> VideoDetectionSummary:
        """Detect objects frame-by-frame and save an annotated MP4 video."""
        input_video = Path(input_path)
        output_video = Path(output_path)
        if not input_video.is_file():
            raise FileNotFoundError(f"Video file not found: {input_video}")
        if input_video.resolve() == output_video.resolve():
            raise ValueError("output_path must be different from input_path")

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {input_video}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Video has invalid dimensions: {input_video}")

            output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Unable to create output video: {output_video}")

            started = perf_counter()
            frames_processed = 0
            frames_with_detections = 0
            total_detections = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    detections = self.detect(frame)
                    writer.write(self.draw_detections(frame, detections))
                    frames_processed += 1
                    total_detections += len(detections)
                    frames_with_detections += bool(detections)
            finally:
                writer.release()

            elapsed_seconds = perf_counter() - started
            return VideoDetectionSummary(
                input_path=str(input_video.resolve()),
                output_path=str(output_video.resolve()),
                frames_processed=frames_processed,
                frames_with_detections=frames_with_detections,
                total_detections=total_detections,
                elapsed_seconds=elapsed_seconds,
                processing_fps=(frames_processed / elapsed_seconds if elapsed_seconds else 0.0),
                model=self.model_path.name,
                device=self.device,
            )
        finally:
            capture.release()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line interface."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on a video.")
    parser.add_argument("input", type=Path, help="Path to an input video")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase1_annotated.mp4",
        help="Destination for the annotated MP4 video",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="YOLOv8 model weights path (downloads yolov8n on first run)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum model confidence from 0.0 to 1.0",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu or cuda:0",
    )
    return parser


def main() -> int:
    """Run the command-line detector and print reproducible result metadata."""
    arguments = build_argument_parser().parse_args()
    detector = YOLODetector(
        model_path=arguments.model,
        confidence_threshold=arguments.confidence,
        device=arguments.device,
    )
    print("Runtime:", detector.runtime_info())
    summary = detector.process_video(arguments.input, arguments.output)
    print("Detection completed:", summary.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
