"""Secondary security scenarios for EdgeShield AI: Loitering, Abandoned Object, and Unusual Movement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.zones import Zone, is_restricted_zone


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration parameters for a security monitoring scenario."""

    scenario_id: str
    title: str
    description: str
    loitering_threshold_seconds: float = 6.0
    abandoned_threshold_seconds: float = 4.0
    speed_anomaly_threshold: float = 120.0  # pixels / second
    pacing_reversals_threshold: int = 3
    restricted_entry_active: bool = True
    after_hours_active: bool = True


# Standard presets defined in the EdgeShield development specification
SCENARIOS: dict[str, ScenarioConfig] = {
    "restricted_intrusion": ScenarioConfig(
        scenario_id="restricted_intrusion",
        title="Scenario 1 — Restricted Area Intrusion (Primary MVP)",
        description="Detects after-hours entry into a secure zone with dwell and evidence-grounded risk escalation.",
        loitering_threshold_seconds=8.0,
        abandoned_threshold_seconds=5.0,
        speed_anomaly_threshold=140.0,
    ),
    "loitering": ScenarioConfig(
        scenario_id="loitering",
        title="Scenario 2 — Sensitive Area Loitering",
        description="Identifies individuals lingering in monitored access perimeters without clear transit.",
        loitering_threshold_seconds=5.0,
        abandoned_threshold_seconds=5.0,
        speed_anomaly_threshold=140.0,
    ),
    "abandoned_object": ScenarioConfig(
        scenario_id="abandoned_object",
        title="Scenario 3 — Abandoned / Unattended Object",
        description="Flags stationary bags, packages, or luggage left unattended after the subject departs.",
        loitering_threshold_seconds=8.0,
        abandoned_threshold_seconds=3.0,
        speed_anomaly_threshold=140.0,
    ),
    "after_hours": ScenarioConfig(
        scenario_id="after_hours",
        title="Scenario 4 — After-Hours Facility Activity",
        description="Highlights perimeter and interior access occurring outside authorized shift hours (08:00 - 18:00).",
        loitering_threshold_seconds=6.0,
        abandoned_threshold_seconds=5.0,
        speed_anomaly_threshold=140.0,
    ),
    "unusual_movement": ScenarioConfig(
        scenario_id="unusual_movement",
        title="Scenario 5 — Unusual High-Velocity Movement",
        description="Detects erratic direction reversals, sudden running, or pacing in restricted corridors.",
        loitering_threshold_seconds=8.0,
        abandoned_threshold_seconds=5.0,
        speed_anomaly_threshold=90.0,
        pacing_reversals_threshold=2,
    ),
}


@dataclass
class _TrackedObjectState:
    """Internal state tracking stationary inanimate objects."""

    class_name: str
    first_seen: float
    last_seen: float
    position: list[int]
    bbox: list[int]
    fired: bool = False


class SecondaryScenarioAnalyzer:
    """Evaluates behavioral anomalies (loitering, abandoned objects, unusual movement)."""

    def __init__(self, config: ScenarioConfig | None = None) -> None:
        self.config = config or SCENARIOS["restricted_intrusion"]
        self._tracked_objects: dict[int, _TrackedObjectState] = {}
        self._loitering_fired: set[int] = set()
        self._unusual_movement_fired: set[int] = set()

    def check_loitering(
        self,
        track_id: int,
        positions_history: list[tuple[float, list[int]]],
        timestamp_seconds: float,
        current_zone_name: str | None = None,
    ) -> bool:
        """Check if an individual has lingered in a monitored area beyond the threshold."""
        if track_id in self._loitering_fired:
            return False

        if len(positions_history) < 10:
            return False

        first_time = positions_history[0][0]
        duration = timestamp_seconds - first_time
        if duration >= self.config.loitering_threshold_seconds:
            # Verify the person has stayed within a confined perimeter (not moving fast)
            recent_positions = [pos for _, pos in positions_history]
            xs = [p[0] for p in recent_positions]
            ys = [p[1] for p in recent_positions]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            # If the subject stayed within an ~200px bounding radius for the duration
            if span_x < 220 and span_y < 220:
                self._loitering_fired.add(track_id)
                return True
        return False

    def check_unusual_movement(
        self,
        track_id: int,
        positions_history: list[tuple[float, list[int]]],
        timestamp_seconds: float,
    ) -> dict[str, Any] | None:
        """Check for sprinting velocity or erratic direction reversals."""
        if track_id in self._unusual_movement_fired:
            return None

        if len(positions_history) < 8:
            return None

        recent = positions_history[-8:]
        dt = recent[-1][0] - recent[0][0]
        if dt <= 0.2:
            return None

        dx = recent[-1][1][0] - recent[0][1][0]
        dy = recent[-1][1][1] - recent[0][1][1]
        speed = ((dx * dx + dy * dy) ** 0.5) / dt

        # 1. High-velocity sprinting check
        if speed >= self.config.speed_anomaly_threshold:
            self._unusual_movement_fired.add(track_id)
            return {"type": "high_velocity", "speed": round(speed, 1)}

        # 2. Direction reversal / erratic pacing check
        if len(positions_history) >= 14:
            long_recent = positions_history[-14:]
            x_diffs = [
                long_recent[i][1][0] - long_recent[i - 1][1][0]
                for i in range(1, len(long_recent))
            ]
            # Count sign flips in x movement
            reversals = 0
            for i in range(1, len(x_diffs)):
                if abs(x_diffs[i]) > 8 and abs(x_diffs[i - 1]) > 8:
                    if (x_diffs[i] > 0 and x_diffs[i - 1] < 0) or (x_diffs[i] < 0 and x_diffs[i - 1] > 0):
                        reversals += 1

            if reversals >= self.config.pacing_reversals_threshold:
                self._unusual_movement_fired.add(track_id)
                return {"type": "erratic_pacing", "reversals": reversals}

        return None

    def check_abandoned_object(
        self,
        object_detections: Sequence[dict[str, Any]],
        person_tracks: Sequence[dict[str, Any]],
        timestamp_seconds: float,
    ) -> list[dict[str, Any]]:
        """Identify stationary unattended items (backpacks, suitcases) separated from people."""
        triggered: list[dict[str, Any]] = []
        target_classes = {"backpack", "handbag", "suitcase", "laptop", "box"}

        person_centers = []
        for p in person_tracks:
            bbox = p.get("bbox", [0, 0, 0, 0])
            person_centers.append([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])

        for obj in object_detections:
            cname = str(obj.get("class_name", "")).lower()
            if cname not in target_classes and not any(t in cname for t in target_classes):
                continue

            obj_id = int(obj.get("track_id", hash(str(obj.get("bbox")))))
            bbox = obj.get("bbox", [0, 0, 0, 0])
            obj_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

            if obj_id not in self._tracked_objects:
                self._tracked_objects[obj_id] = _TrackedObjectState(
                    class_name=cname,
                    first_seen=timestamp_seconds,
                    last_seen=timestamp_seconds,
                    position=obj_center,
                    bbox=bbox,
                )
            else:
                state = self._tracked_objects[obj_id]
                state.last_seen = timestamp_seconds
                duration = timestamp_seconds - state.first_seen

                # Check if any person is within proximity radius (e.g. 150px)
                is_attended = False
                for pc in person_centers:
                    dist = ((pc[0] - obj_center[0]) ** 2 + (pc[1] - obj_center[1]) ** 2) ** 0.5
                    if dist <= 150.0:
                        is_attended = True
                        break

                if not is_attended and duration >= self.config.abandoned_threshold_seconds and not state.fired:
                    state.fired = True
                    triggered.append({
                        "object_id": obj_id,
                        "class_name": cname,
                        "bbox": bbox,
                        "duration": round(duration, 1),
                    })

        return triggered

    def reset(self) -> None:
        """Reset scenario state."""
        self._tracked_objects.clear()
        self._loitering_fired.clear()
        self._unusual_movement_fired.clear()
