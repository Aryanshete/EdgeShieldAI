"""Security event generation, chronological sequencing, and event models for EdgeShield."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.zones import Zone, ZoneManager, is_restricted_zone
from src.scenarios import SCENARIOS, ScenarioConfig, SecondaryScenarioAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_FILE = PROJECT_ROOT / "data" / "events.json"

# Standard EdgeShield event types
EVENT_PERSON_DETECTED = "person_detected"
EVENT_ZONE_APPROACH = "zone_approach"
EVENT_RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
EVENT_RESTRICTED_ZONE_DWELL = "restricted_zone_dwell"
EVENT_RESTRICTED_ZONE_EXIT = "restricted_zone_exit"
EVENT_OBJECT_INTERACTION = "object_interaction"
EVENT_AFTER_HOURS_ACTIVITY = "after_hours_activity"
EVENT_LOITERING_DETECTED = "loitering_detected"
EVENT_ABANDONED_OBJECT = "abandoned_object_detected"
EVENT_UNUSUAL_MOVEMENT = "unusual_movement"


def parse_time_string(time_str: str) -> dt_time:
    """Parse HH:MM or HH:MM:SS string into a datetime.time object."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return dt_time(hour=int(parts[0]), minute=int(parts[1]))
    if len(parts) == 3:
        return dt_time(hour=int(parts[0]), minute=int(parts[1]), second=int(float(parts[2])))
    raise ValueError(f"Invalid time format: {time_str}")


def format_timestamp(timestamp_seconds: float, base_time: str | None = "22:17:00") -> str:
    """Convert a video offset in seconds into an HH:MM:SS timestamp string."""
    if base_time:
        base_t = parse_time_string(base_time)
        base_dt = datetime(2026, 1, 1, base_t.hour, base_t.minute, base_t.second)
        actual_dt = base_dt + timedelta(seconds=timestamp_seconds)
        return actual_dt.strftime("%H:%M:%S")

    total_secs = max(0, int(timestamp_seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_time_after_hours(time_str: str, authorized_hours: dict[str, str] | None) -> bool:
    """Return True if time_str falls outside configured authorized operating hours."""
    if not authorized_hours:
        return False

    start_str = authorized_hours.get("start")
    end_str = authorized_hours.get("end")
    if not start_str or not end_str:
        return False

    current_t = parse_time_string(time_str)
    start_t = parse_time_string(start_str)
    end_t = parse_time_string(end_str)

    if start_t <= end_t:
        # Standard day shift, e.g. 08:00 to 18:00
        return current_t < start_t or current_t > end_t
    else:
        # Overnight window, e.g. 22:00 to 06:00
        return not (current_t >= start_t or current_t <= end_t)


def distance_point_to_polygon(
    point: Sequence[int] | tuple[int, int],
    polygon: Sequence[Sequence[int]],
) -> float:
    """Calculate Euclidean distance in pixels from a point to a polygon boundary.

    Returns 0.0 if the point is strictly inside or on the boundary.
    """
    if len(polygon) < 3:
        return float("inf")
    pts = np.array(polygon, dtype=np.int32)
    dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=True)
    if dist >= 0.0:
        return 0.0
    return abs(float(dist))


@dataclass(frozen=True)
class SecurityEvent:
    """A single structured security event produced by EdgeShield's event engine."""

    event_type: str
    track_id: int
    zone: str | None
    timestamp: str
    timestamp_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the exact schema expected by downstream reasoning and incidents."""
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "zone": self.zone,
            "timestamp": self.timestamp,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "metadata": dict(self.metadata),
        }


@dataclass
class _TrackState:
    """Internal state maintained for a tracked subject."""

    first_seen: float
    last_seen: float
    current_zones: set[str] = field(default_factory=set)
    zone_entry_times: dict[str, float] = field(default_factory=dict)
    zone_dwell_fired: set[str] = field(default_factory=set)
    zone_approach_fired: set[str] = field(default_factory=set)
    after_hours_fired: set[str] = field(default_factory=set)
    last_position: list[int] | None = None
    positions_history: list[tuple[float, list[int]]] = field(default_factory=list)
    interaction_fired: bool = False


class EventEngine:
    """Converts frame-by-frame tracked detections into high-level security events."""

    def __init__(
        self,
        zone_manager: ZoneManager | None = None,
        base_time: str | None = "22:17:00",
        dwell_threshold_seconds: float = 5.0,
        approach_distance_pixels: float = 80.0,
        interaction_speed_threshold: float = 15.0,
        scenario_config: ScenarioConfig | None = None,
    ) -> None:
        self.zone_manager = zone_manager or ZoneManager()
        self.base_time = base_time
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.approach_distance_pixels = approach_distance_pixels
        self.interaction_speed_threshold = interaction_speed_threshold
        self.scenario_config = scenario_config or SCENARIOS["restricted_intrusion"]
        self.scenario_analyzer = SecondaryScenarioAnalyzer(config=self.scenario_config)

        self.events: list[SecurityEvent] = []
        self._tracks: dict[int, _TrackState] = {}
        self._active_track_ids_last_frame: set[int] = set()

    def get_timestamp(self, timestamp_seconds: float) -> str:
        """Format the given offset into an HH:MM:SS string."""
        return format_timestamp(timestamp_seconds, self.base_time)

    def _record_event(
        self,
        event_type: str,
        track_id: int,
        zone: str | None,
        timestamp_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> SecurityEvent:
        event = SecurityEvent(
            event_type=event_type,
            track_id=track_id,
            zone=zone,
            timestamp=self.get_timestamp(timestamp_seconds),
            timestamp_seconds=timestamp_seconds,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    def process_tracks(
        self,
        tracked_detections: Iterable[dict[str, Any]],
        timestamp_seconds: float,
        object_detections: Sequence[dict[str, Any]] | None = None,
    ) -> list[SecurityEvent]:
        """Process one frame of tracked detections and return any newly triggered events."""
        new_events: list[SecurityEvent] = []
        raw_tracks = list(tracked_detections)
        person_tracks = [t for t in raw_tracks if t.get("class_name", "person") == "person"]

        # Check for abandoned/unattended objects
        raw_objects = list(object_detections) if object_detections is not None else [
            t for t in raw_tracks if t.get("class_name", "person") != "person"
        ]
        if raw_objects:
            abandoned_alerts = self.scenario_analyzer.check_abandoned_object(
                object_detections=raw_objects,
                person_tracks=person_tracks,
                timestamp_seconds=timestamp_seconds,
            )
            for item in abandoned_alerts:
                bbox = item.get("bbox", [0, 0, 0, 0])
                obj_pos = [(bbox[0] + bbox[2]) // 2, bbox[3]]
                occ = self.zone_manager.check_point(obj_pos)
                zone_name = occ[0].name if occ else None
                new_events.append(
                    self._record_event(
                        event_type=EVENT_ABANDONED_OBJECT,
                        track_id=item["object_id"],
                        zone=zone_name,
                        timestamp_seconds=timestamp_seconds,
                        metadata=item,
                    )
                )

        current_frame_track_ids: set[int] = set()

        for track in person_tracks:
            track_id = int(track["track_id"])
            current_frame_track_ids.add(track_id)
            bbox = track.get("bbox", [0, 0, 0, 0])
            x1, _y1, x2, y2 = bbox
            pos = [(x1 + x2) // 2, y2]

            # 1. PERSON_DETECTED (first time this track ID is observed)
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackState(
                    first_seen=timestamp_seconds,
                    last_seen=timestamp_seconds,
                    last_position=pos,
                )
                new_events.append(
                    self._record_event(
                        event_type=EVENT_PERSON_DETECTED,
                        track_id=track_id,
                        zone=None,
                        timestamp_seconds=timestamp_seconds,
                        metadata={"class_name": track.get("class_name", "person"), "initial_bbox": bbox},
                    )
                )

            state = self._tracks[track_id]
            state.last_seen = timestamp_seconds
            state.positions_history.append((timestamp_seconds, pos))

            # Determine currently occupied zones
            occupied_zones = self.zone_manager.check_point(pos)
            occupied_zone_ids = {z.id for z in occupied_zones}
            zone_map = {z.id: z for z in occupied_zones}

            # 2. ZONE_APPROACH (within buffer of restricted zone before entry)
            for zone in self.zone_manager.zones:
                if not is_restricted_zone(zone):
                    continue
                if zone.id in occupied_zone_ids:
                    continue  # Already inside
                if zone.id in state.zone_approach_fired:
                    continue  # Already signaled approach

                dist = distance_point_to_polygon(pos, zone.polygon)
                if 0 < dist <= self.approach_distance_pixels:
                    state.zone_approach_fired.add(zone.id)
                    new_events.append(
                        self._record_event(
                            event_type=EVENT_ZONE_APPROACH,
                            track_id=track_id,
                            zone=zone.name,
                            timestamp_seconds=timestamp_seconds,
                            metadata={"distance_pixels": round(dist, 1), "zone_id": zone.id},
                        )
                    )

            # 3. RESTRICTED_ZONE_ENTRY & AFTER_HOURS_ACTIVITY
            for zone in occupied_zones:
                if zone.id not in state.current_zones:
                    # Fresh entry into this zone
                    state.current_zones.add(zone.id)
                    state.zone_entry_times[zone.id] = timestamp_seconds

                    if is_restricted_zone(zone):
                        new_events.append(
                            self._record_event(
                                event_type=EVENT_RESTRICTED_ZONE_ENTRY,
                                track_id=track_id,
                                zone=zone.name,
                                timestamp_seconds=timestamp_seconds,
                                metadata={"zone_id": zone.id},
                            )
                        )

                        # Check AFTER_HOURS_ACTIVITY
                        time_str = self.get_timestamp(timestamp_seconds)
                        if zone.authorized_hours and is_time_after_hours(time_str, zone.authorized_hours):
                            if zone.id not in state.after_hours_fired:
                                state.after_hours_fired.add(zone.id)
                                new_events.append(
                                    self._record_event(
                                        event_type=EVENT_AFTER_HOURS_ACTIVITY,
                                        track_id=track_id,
                                        zone=zone.name,
                                        timestamp_seconds=timestamp_seconds,
                                        metadata={
                                            "zone_id": zone.id,
                                            "authorized_hours": zone.authorized_hours,
                                            "current_time": time_str,
                                        },
                                    )
                                )

            # 4. RESTRICTED_ZONE_DWELL
            for zone_id in list(state.current_zones):
                zone = self.zone_manager.get_zone_by_id(zone_id)
                if zone and is_restricted_zone(zone):
                    entry_time = state.zone_entry_times.get(zone_id, timestamp_seconds)
                    dwell_duration = timestamp_seconds - entry_time
                    if (
                        dwell_duration >= self.dwell_threshold_seconds
                        and zone_id not in state.zone_dwell_fired
                    ):
                        state.zone_dwell_fired.add(zone_id)
                        new_events.append(
                            self._record_event(
                                event_type=EVENT_RESTRICTED_ZONE_DWELL,
                                track_id=track_id,
                                zone=zone.name,
                                timestamp_seconds=timestamp_seconds,
                                metadata={
                                    "zone_id": zone.id,
                                    "dwell_seconds": round(dwell_duration, 1),
                                },
                            )
                        )

            # 5. OBJECT_INTERACTION
            # If inside a restricted zone, has dwelt for a while, and speed becomes low/stationary
            if occupied_zones and any(is_restricted_zone(z) for z in occupied_zones):
                if not state.interaction_fired and len(state.positions_history) >= 15:
                    recent = state.positions_history[-15:]
                    dt = recent[-1][0] - recent[0][0]
                    if dt >= 1.5:
                        dx = recent[-1][1][0] - recent[0][1][0]
                        dy = recent[-1][1][1] - recent[0][1][1]
                        speed = (dx * dx + dy * dy) ** 0.5 / dt
                        if speed <= self.interaction_speed_threshold:
                            primary_restricted = next(z for z in occupied_zones if is_restricted_zone(z))
                            state.interaction_fired = True
                            new_events.append(
                                self._record_event(
                                    event_type=EVENT_OBJECT_INTERACTION,
                                    track_id=track_id,
                                    zone=primary_restricted.name,
                                    timestamp_seconds=timestamp_seconds,
                                    metadata={
                                        "zone_id": primary_restricted.id,
                                        "stationary_speed": round(speed, 2),
                                    },
                                )
                            )

            # 6. RESTRICTED_ZONE_EXIT (track moved out of a zone it previously occupied)
            exited_zone_ids = set(state.current_zones) - occupied_zone_ids
            for exited_id in exited_zone_ids:
                state.current_zones.remove(exited_id)
                zone = self.zone_manager.get_zone_by_id(exited_id)
                if zone and is_restricted_zone(zone):
                    entry_time = state.zone_entry_times.pop(exited_id, timestamp_seconds)
                    total_dwell = timestamp_seconds - entry_time
                    state.zone_dwell_fired.discard(exited_id)
                    state.zone_approach_fired.discard(exited_id)
                    state.after_hours_fired.discard(exited_id)
                    new_events.append(
                        self._record_event(
                            event_type=EVENT_RESTRICTED_ZONE_EXIT,
                            track_id=track_id,
                            zone=zone.name,
                            timestamp_seconds=timestamp_seconds,
                            metadata={
                                "zone_id": zone.id,
                                "total_dwell_seconds": round(total_dwell, 1),
                            },
                        )
                    )

            # Secondary scenario behavioral evaluations: Loitering & Unusual Movement
            current_zone_name = occupied_zones[0].name if occupied_zones else None
            if self.scenario_analyzer.check_loitering(
                track_id, state.positions_history, timestamp_seconds, current_zone_name
            ):
                new_events.append(
                    self._record_event(
                        event_type=EVENT_LOITERING_DETECTED,
                        track_id=track_id,
                        zone=current_zone_name,
                        timestamp_seconds=timestamp_seconds,
                        metadata={"loitering_duration": round(timestamp_seconds - state.first_seen, 1)},
                    )
                )

            unusual = self.scenario_analyzer.check_unusual_movement(
                track_id, state.positions_history, timestamp_seconds
            )
            if unusual:
                new_events.append(
                    self._record_event(
                        event_type=EVENT_UNUSUAL_MOVEMENT,
                        track_id=track_id,
                        zone=current_zone_name,
                        timestamp_seconds=timestamp_seconds,
                        metadata=unusual,
                    )
                )

            state.last_position = pos

        # Check for tracks that disappeared from the frame completely while inside a zone
        disappeared_tracks = self._active_track_ids_last_frame - current_frame_track_ids
        for tid in disappeared_tracks:
            if tid in self._tracks:
                state = self._tracks[tid]
                for zone_id in list(state.current_zones):
                    zone = self.zone_manager.get_zone_by_id(zone_id)
                    if zone and is_restricted_zone(zone):
                        entry_time = state.zone_entry_times.pop(zone_id, timestamp_seconds)
                        total_dwell = timestamp_seconds - entry_time
                        new_events.append(
                            self._record_event(
                                event_type=EVENT_RESTRICTED_ZONE_EXIT,
                                track_id=tid,
                                zone=zone.name,
                                timestamp_seconds=timestamp_seconds,
                                metadata={
                                    "zone_id": zone.id,
                                    "total_dwell_seconds": round(total_dwell, 1),
                                    "disappeared": True,
                                },
                            )
                        )
                state.current_zones.clear()

        self._active_track_ids_last_frame = current_frame_track_ids
        return new_events

    def export_events(self, output_path: str | Path = DEFAULT_EVENTS_FILE) -> None:
        """Serialize all generated events to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [event.as_dict() for event in self.events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    @classmethod
    def load_events(cls, path: str | Path = DEFAULT_EVENTS_FILE) -> list[SecurityEvent]:
        """Load events from a JSON file into SecurityEvent objects."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Events file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        events: list[SecurityEvent] = []
        for item in data:
            t_sec = float(item.get("timestamp_seconds", 0.0))
            if t_sec == 0.0 and "timestamp" in item and ":" in item["timestamp"]:
                try:
                    t_obj = parse_time_string(item["timestamp"])
                    t_sec = float(t_obj.hour * 3600 + t_obj.minute * 60 + t_obj.second)
                except Exception:
                    pass
            events.append(
                SecurityEvent(
                    event_type=item["event_type"],
                    track_id=item["track_id"],
                    zone=item.get("zone"),
                    timestamp=item["timestamp"],
                    timestamp_seconds=t_sec,
                    metadata=item.get("metadata", {}),
                )
            )
        return events

    def clear(self) -> None:
        """Reset internal event history and track state."""
        self.events.clear()
        self._tracks.clear()
        self._active_track_ids_last_frame.clear()
        self.scenario_analyzer.reset()
