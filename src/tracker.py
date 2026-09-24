"""Persistent person tracking with Ultralytics ByteTrack for EdgeShield."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from src.detector import DEFAULT_MODEL_PATH, PROJECT_ROOT, YOLODetector
from src.zones import ZoneManager, draw_zones
from src.events import EventEngine


TrackHistory = dict[int, dict[str, Any]]


@dataclass(frozen=True)
class TrackedDetection:
    """One tracked visual entity, including its anonymous persistent ID."""

    track_id: int
    class_name: str
    confidence: float
    bbox: list[int]

    def as_dict(self) -> dict[str, int | str | float | list[int]]:
        """Return the public tracked-detection schema."""
        return asdict(self)


@dataclass(frozen=True)
class VideoTrackingSummary:
    """Measured result of one completed video tracking run."""

    input_path: str
    output_path: str
    frames_processed: int
    frames_with_tracks: int
    total_tracked_detections: int
    unique_track_ids: list[int]
    elapsed_seconds: float
    processing_fps: float
    model: str
    device: str
    tracker: str
    events_count: int = 0

    def as_dict(self) -> dict[str, str | int | float | list[int]]:
        """Return a JSON-ready summary."""
        return asdict(self)


def bottom_center(bbox: list[int]) -> list[int]:
    """Return the ground-contact reference point used by spatial zone evaluation."""
    x1, _y1, x2, y2 = bbox
    return [(x1 + x2) // 2, y2]


def update_track_history(
    track_history: TrackHistory,
    tracked_detection: dict[str, int | str | float | list[int]],
    timestamp_seconds: float,
    zone_manager: ZoneManager | None = None,
) -> None:
    """Add a visual observation while retaining the planned history schema."""
    track_id = int(tracked_detection["track_id"])
    bbox = list(tracked_detection["bbox"])  # type: ignore[arg-type]
    pos = bottom_center(bbox)
    observation = {
        "timestamp_seconds": timestamp_seconds,
        "position": pos,
        "bbox": bbox,
    }
    if track_id not in track_history:
        track_history[track_id] = {
            "first_seen": timestamp_seconds,
            "last_seen": timestamp_seconds,
            "positions": [observation],
            "zones": [],
            "events": [],
        }
    else:
        track_history[track_id]["last_seen"] = timestamp_seconds
        track_history[track_id]["positions"].append(observation)

    if zone_manager is not None:
        occupied = zone_manager.check_point(pos)
        zone_manager.update_track_history_zones(
            track_history, track_id, occupied, timestamp_seconds
        )


class ObjectTracker:
    """Track anonymous people over successive frames with ByteTrack."""

    def __init__(
        self,
        detector: YOLODetector | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        tracker_config: str | Path = PROJECT_ROOT / "config" / "bytetrack.yaml",
        person_only: bool = True,
        classes: list[int] | None = None,
        zone_manager: ZoneManager | None = None,
        event_engine: EventEngine | None = None,
    ) -> None:
        if not tracker_config:
            raise ValueError("tracker_config must not be empty")
        self.detector = detector or YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self.tracker_config = str(tracker_config)
        self.person_only = person_only
        self.classes = classes
        self.zone_manager = zone_manager
        self.event_engine = event_engine
        self.track_history: TrackHistory = {}

    def track(
        self,
        frame: Any,
        timestamp_seconds: float,
    ) -> list[dict[str, int | str | float | list[int]]]:
        """Track people in the next BGR frame and update ``track_history``."""
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("frame must be a valid OpenCV image")
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must not be negative")

        arguments: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.detector.confidence_threshold,
            "device": self.detector.device,
            "verbose": False,
        }
        if self.classes is not None:
            arguments["classes"] = self.classes
        elif self.person_only:
            arguments["classes"] = [0]

        result = self.detector.model.track(frame, **arguments)[0]
        if result.boxes is None or result.boxes.id is None:
            if self.event_engine is not None:
                self.event_engine.process_tracks([], timestamp_seconds)
            return []

        tracked_detections: list[dict[str, int | str | float | list[int]]] = []
        for box in result.boxes:
            if box.id is None:
                continue
            class_id = int(box.cls.item())
            tracked = TrackedDetection(
                track_id=int(box.id.item()),
                class_name=self.detector.class_names.get(class_id, f"class_{class_id}"),
                confidence=float(box.conf.item()),
                bbox=[int(value) for value in box.xyxy[0].tolist()],
            ).as_dict()
            update_track_history(
                self.track_history,
                tracked,
                timestamp_seconds,
                zone_manager=self.zone_manager,
            )
            tracked_detections.append(tracked)

        if self.event_engine is not None:
            new_events = self.event_engine.process_tracks(tracked_detections, timestamp_seconds)
            for ev in new_events:
                if ev.track_id in self.track_history:
                    self.track_history[ev.track_id].setdefault("events", []).append(ev.as_dict())

        return tracked_detections

    @staticmethod
    def draw_tracks(
        frame: Any,
        tracked_detections: Iterable[dict[str, int | str | float | list[int]]],
    ) -> Any:
        """Render anonymous IDs and confidence labels on a copy of a frame."""
        annotated = frame.copy()
        for tracked in tracked_detections:
            x1, y1, x2, y2 = tracked["bbox"]  # type: ignore[misc]
            track_id = int(tracked["track_id"])
            class_name = str(tracked["class_name"])
            confidence = float(tracked["confidence"])
            label = f"{class_name} #{track_id:02d} {confidence:.2f}"
            color = (0, 191, 255)

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
                (16, 27, 39),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> VideoTrackingSummary:
        """Track people throughout a video and save an annotated MP4 copy."""
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
            frames_with_tracks = 0
            total_tracked_detections = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    timestamp_seconds = frames_processed / fps
                    tracks = self.track(frame, timestamp_seconds)

                    if self.zone_manager is not None:
                        active_ids = {
                            z.id
                            for t in tracks
                            for z in self.zone_manager.evaluate_track(t)
                        }
                        annotated = draw_zones(
                            frame, self.zone_manager.zones, active_zone_ids=active_ids
                        )
                        annotated = self.draw_tracks(annotated, tracks)
                    else:
                        annotated = self.draw_tracks(frame, tracks)

                    writer.write(annotated)
                    frames_processed += 1
                    frames_with_tracks += bool(tracks)
                    total_tracked_detections += len(tracks)
            finally:
                writer.release()

            elapsed_seconds = perf_counter() - started
            events_count = len(self.event_engine.events) if self.event_engine else 0
            return VideoTrackingSummary(
                input_path=str(input_video.resolve()),
                output_path=str(output_video.resolve()),
                frames_processed=frames_processed,
                frames_with_tracks=frames_with_tracks,
                total_tracked_detections=total_tracked_detections,
                unique_track_ids=sorted(self.track_history),
                elapsed_seconds=elapsed_seconds,
                processing_fps=(frames_processed / elapsed_seconds if elapsed_seconds else 0.0),
                model=self.detector.model_path.name,
                device=self.detector.device,
                tracker=self.tracker_config,
                events_count=events_count,
            )
        finally:
            capture.release()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the object tracking command-line interface."""
    parser = argparse.ArgumentParser(description="Track people across a video with ByteTrack.")
    parser.add_argument("input", type=Path, help="Path to an input video")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tracked_output.mp4",
        help="Destination for the annotated MP4 video",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tracker-config",
        default=PROJECT_ROOT / "config" / "bytetrack.yaml",
        help="Ultralytics tracker configuration, such as bytetrack.yaml or botsort.yaml",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Track all detected classes instead of people only",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=None,
        help="Path to zones JSON configuration file (e.g. config/zones.json)",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Enable security event engine processing",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "events.json",
        help="Destination for generated events JSON file",
    )
    return parser


def main() -> int:
    """Run video tracking and print the resulting track summary."""
    arguments = build_argument_parser().parse_args()
    zone_mgr = ZoneManager(config_path=arguments.zones) if arguments.zones else None
    event_engine = EventEngine(zone_manager=zone_mgr) if (arguments.events or arguments.zones) else None
    tracker = ObjectTracker(
        model_path=arguments.model,
        confidence_threshold=arguments.confidence,
        device=arguments.device,
        tracker_config=arguments.tracker_config,
        person_only=not arguments.all_classes,
        zone_manager=zone_mgr,
        event_engine=event_engine,
    )
    print("Runtime:", tracker.detector.runtime_info())
    summary = tracker.process_video(arguments.input, arguments.output)
    if event_engine is not None:
        event_engine.export_events(arguments.events_out)
        print(f"Exported {len(event_engine.events)} events to {arguments.events_out}")
    print("Tracking completed:", summary.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
