"""Restricted zone definition, spatial point-in-polygon queries, and visualization for EdgeShield."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONES_CONFIG = PROJECT_ROOT / "config" / "zones.json"


@dataclass(frozen=True)
class Zone:
    """A spatial polygon area with security attributes."""

    id: str
    name: str
    type: str
    polygon: list[list[int]]
    color: tuple[int, int, int] = (0, 0, 220)  # BGR
    authorized_hours: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        """Instantiate a Zone from a configuration dictionary."""
        polygon = [[int(x), int(y)] for x, y in data["polygon"]]
        raw_color = data.get("color", [0, 0, 220])
        color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
        return cls(
            id=str(data.get("id", data["name"].lower().replace(" ", "_"))),
            name=str(data["name"]),
            type=str(data.get("type", "restricted")),
            polygon=polygon,
            color=color,
            authorized_hours=dict(data.get("authorized_hours", {})),
            description=str(data.get("description", "")),
        )


ZONE_PRESETS: dict[str, dict[str, Any]] = {
    "server_room_demo": {
        "title": "Indoor Server Room Facility (Demo Video)",
        "description": "Configured for indoor demonstration video (Server Room & Entry Corridor).",
        "zones": [
            Zone(
                id="server_room",
                name="Server Room",
                type="restricted",
                polygon=[[450, 300], [850, 300], [850, 720], [450, 720]],
                color=(0, 0, 220),
                authorized_hours={"start": "08:00", "end": "18:00"},
                description="High-security server rack enclosure containing sensitive communications equipment.",
            ),
            Zone(
                id="entry_corridor",
                name="Entry Corridor",
                type="monitored",
                polygon=[[100, 300], [450, 300], [450, 720], [100, 720]],
                color=(0, 180, 255),
                authorized_hours={"start": "00:00", "end": "23:59"},
                description="General access hallway connecting the outer corridor to the secure facility entry.",
            ),
        ],
    },
    "outdoor_perimeter": {
        "title": "Outdoor Perimeter & Access Walkway (Night/Construction)",
        "description": "Secures upper & central perimeter walkways, barrier fences, and access gates.",
        "zones": [
            Zone(
                id="perimeter_walkway",
                name="Secured Perimeter & Walkway",
                type="restricted",
                polygon=[[0, 50], [1280, 50], [1280, 580], [0, 580]],
                color=(0, 0, 220),
                authorized_hours={"start": "08:00", "end": "18:00"},
                description="Outdoor perimeter boundary, construction fence, and access walkway.",
            ),
            Zone(
                id="outer_staging",
                name="Outer Transit Roadway",
                type="monitored",
                polygon=[[0, 580], [1280, 580], [1280, 720], [0, 720]],
                color=(0, 180, 255),
                authorized_hours={"start": "00:00", "end": "23:59"},
                description="Outer transit staging and public access corridor.",
            ),
        ],
    },
    "full_frame_secure": {
        "title": "Full-Scene Secure Perimeter (Universal — All Feeds)",
        "description": "Monitors the entire camera view as a secured restricted perimeter.",
        "zones": [
            Zone(
                id="full_facility_perimeter",
                name="Full Secured Perimeter",
                type="restricted",
                polygon=[[0, 0], [1280, 0], [1280, 720], [0, 720]],
                color=(0, 0, 220),
                authorized_hours={"start": "08:00", "end": "18:00"},
                description="Entire visual field monitored as an active high-security zone.",
            ),
        ],
    },
    "dual_facility_split": {
        "title": "Dual Facility Split (Left Restricted / Right Monitored)",
        "description": "Divides camera into secured access bay (left) and staging area (right).",
        "zones": [
            Zone(
                id="secure_zone_left",
                name="Restricted Bay (Left)",
                type="restricted",
                polygon=[[0, 0], [640, 0], [640, 720], [0, 720]],
                color=(0, 0, 220),
                authorized_hours={"start": "08:00", "end": "18:00"},
                description="Secured left quadrant of the monitored facility.",
            ),
            Zone(
                id="monitored_zone_right",
                name="Monitored Staging (Right)",
                type="monitored",
                polygon=[[640, 0], [1280, 0], [1280, 720], [640, 720]],
                color=(0, 180, 255),
                authorized_hours={"start": "00:00", "end": "23:59"},
                description="Monitored staging and transit area.",
            ),
        ],
    },
}


def scale_zones_to_frame(
    zones: Sequence[Zone],
    frame_width: int,
    frame_height: int,
    ref_width: int = 1280,
    ref_height: int = 720,
) -> list[Zone]:
    """Scale zone polygon coordinates to match actual video frame dimensions."""
    if frame_width <= 0 or frame_height <= 0:
        return list(zones)
    if frame_width == ref_width and frame_height == ref_height:
        return list(zones)

    sx = frame_width / float(ref_width)
    sy = frame_height / float(ref_height)
    scaled: list[Zone] = []
    for z in zones:
        new_poly = [[int(pt[0] * sx), int(pt[1] * sy)] for pt in z.polygon]
        scaled.append(
            Zone(
                id=z.id,
                name=z.name,
                type=z.type,
                polygon=new_poly,
                color=z.color,
                authorized_hours=dict(z.authorized_hours),
                description=z.description,
            )
        )
    return scaled


def point_inside_polygon(
    point: Sequence[int] | tuple[int, int],
    polygon: Sequence[Sequence[int]],
) -> bool:
    """Determine whether a 2D coordinate lies within or on the boundary of a polygon.

    Uses OpenCV's cv2.pointPolygonTest with a ray-casting fallback for robustness.
    Points exactly on the boundary are considered inside (measureDist=False returns >= 0).
    """
    if len(polygon) < 3:
        return False

    px, py = float(point[0]), float(point[1])
    try:
        pts = np.array(polygon, dtype=np.int32)
        dist = cv2.pointPolygonTest(pts, (px, py), measureDist=False)
        return dist >= 0.0
    except Exception:
        # Pure Python ray-casting algorithm fallback
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


def is_restricted_zone(zone: Zone | dict[str, Any] | str) -> bool:
    """Check if a zone or zone type is designated as restricted."""
    if hasattr(zone, "type"):
        return str(zone.type).lower() == "restricted"
    if isinstance(zone, dict):
        return str(zone.get("type", "")).lower() == "restricted"
    if isinstance(zone, str):
        return zone.lower() == "restricted"

    return False


def get_zone(
    point: Sequence[int] | tuple[int, int],
    zones: Iterable[Zone],
) -> Zone | None:
    """Return the primary zone enclosing a given point, prioritizing restricted zones."""
    candidate_zones: list[Zone] = []
    for zone in zones:
        if point_inside_polygon(point, zone.polygon):
            if is_restricted_zone(zone):
                return zone
            candidate_zones.append(zone)
    return candidate_zones[0] if candidate_zones else None


def get_all_zones(
    point: Sequence[int] | tuple[int, int],
    zones: Iterable[Zone],
) -> list[Zone]:
    """Return all zones enclosing a given coordinate."""
    return [zone for zone in zones if point_inside_polygon(point, zone.polygon)]


def load_zones(config_path: str | Path = DEFAULT_ZONES_CONFIG) -> list[Zone]:
    """Load and validate spatial zones from a JSON configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Zone configuration not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "zones" not in data or not isinstance(data["zones"], list):
        raise ValueError("Invalid format: 'zones' key must contain a list of zone definitions")

    raw_zones = data["zones"]

    zones: list[Zone] = []
    for idx, item in enumerate(raw_zones):
        if not isinstance(item, dict) or "name" not in item or "polygon" not in item:
            raise ValueError(f"Zone at index {idx} is missing required 'name' or 'polygon' fields")
        zones.append(Zone.from_dict(item))
    return zones


def draw_zones(
    frame: Any,
    zones: Iterable[Zone],
    active_zone_ids: set[str] | None = None,
    alpha: float = 0.25,
) -> Any:
    """Render semi-transparent filled polygons and labeled borders on a frame.

    Returns a new copy of the frame leaving the original unchanged.
    """
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("frame must be a valid OpenCV image")

    annotated = frame.copy()
    overlay = frame.copy()
    active_ids = active_zone_ids or set()

    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        is_active = zone.id in active_ids or zone.name in active_ids

        # Fill color: red/amber if restricted or active, custom zone color otherwise
        if is_active:
            fill_color = (0, 0, 255) if is_restricted_zone(zone) else (0, 220, 255)
            border_thickness = 3
        else:
            fill_color = zone.color
            border_thickness = 2

        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.polylines(annotated, [pts], isClosed=True, color=fill_color, thickness=border_thickness)

        # Label tag
        label_text = f"{zone.name} [{zone.type.upper()}]"
        if is_active:
            label_text += " [OCCUPIED]"

        x_coords = [p[0] for p in zone.polygon]
        y_coords = [p[1] for p in zone.polygon]
        top_left_x = max(min(x_coords), 5)
        top_left_y = max(min(y_coords), 25)

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        tag_bg_p1 = (top_left_x, top_left_y - text_h - 6)
        tag_bg_p2 = (top_left_x + text_w + 8, top_left_y + baseline)
        cv2.rectangle(annotated, tag_bg_p1, tag_bg_p2, fill_color, -1)
        cv2.putText(
            annotated,
            label_text,
            (top_left_x + 4, top_left_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Blend overlay with original frame for glassmorphism / semi-transparency
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated


class ZoneManager:
    """High-level manager for zone spatial indexing and track association."""

    def __init__(self, zones: list[Zone] | None = None, config_path: str | Path | None = None) -> None:
        if zones is not None:
            self.zones = list(zones)
        elif config_path is not None:
            self.zones = load_zones(config_path)
        elif DEFAULT_ZONES_CONFIG.is_file():
            self.zones = load_zones(DEFAULT_ZONES_CONFIG)
        else:
            self.zones = []

    def get_zone_by_id(self, zone_id: str) -> Zone | None:
        """Find zone by its unique identifier."""
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        return None

    def check_point(self, point: Sequence[int] | tuple[int, int]) -> list[Zone]:
        """Return all zones containing the specified 2D point."""
        return get_all_zones(point, self.zones)

    def check_primary_zone(self, point: Sequence[int] | tuple[int, int]) -> Zone | None:
        """Return the primary zone for a coordinate, prioritizing restricted areas."""
        return get_zone(point, self.zones)

    def evaluate_track(
        self,
        tracked_detection: dict[str, Any],
    ) -> list[Zone]:
        """Determine which zones are occupied by a tracked detection using its ground-contact point."""
        bbox = tracked_detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return []
        x1, _y1, x2, y2 = bbox
        ground_point = [(x1 + x2) // 2, y2]
        return self.check_point(ground_point)

    def update_track_history_zones(
        self,
        track_history: dict[int, dict[str, Any]],
        track_id: int,
        occupied_zones: list[Zone],
        timestamp_seconds: float,
    ) -> None:
        """Update track history with current zone occupancy if newly entered."""
        if track_id not in track_history:
            return

        zones_entry = track_history[track_id].setdefault("zones", [])
        current_zone_names = [z.name for z in occupied_zones]

        # Check if the track already recorded this zone continuously or if this is a new entry
        for zone in occupied_zones:
            # Check if this zone is already recorded as the latest zone
            if not zones_entry or zones_entry[-1].get("zone_name") != zone.name:
                zones_entry.append({
                    "zone_id": zone.id,
                    "zone_name": zone.name,
                    "zone_type": zone.type,
                    "timestamp_seconds": timestamp_seconds,
                    "is_restricted": is_restricted_zone(zone),
                })
