"""Unit tests for restricted zones and spatial geometry analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tracker import update_track_history
from src.zones import (
    ZONE_PRESETS,
    Zone,
    ZoneManager,
    draw_zones,
    get_all_zones,
    get_zone,
    is_restricted_zone,
    load_zones,
    point_inside_polygon,
    scale_zones_to_frame,
)


@pytest.fixture
def sample_zone() -> Zone:
    return Zone(
        id="server_room",
        name="Server Room",
        type="restricted",
        polygon=[[10, 10], [100, 10], [100, 100], [10, 100]],
        color=(0, 0, 220),
        authorized_hours={"start": "08:00", "end": "18:00"},
        description="Secure server area",
    )


@pytest.fixture
def sample_corridor_zone() -> Zone:
    return Zone(
        id="entry_corridor",
        name="Entry Corridor",
        type="monitored",
        polygon=[[100, 10], [200, 10], [200, 100], [100, 100]],
        color=(0, 180, 255),
    )


def test_point_inside_polygon_interior() -> None:
    rect = [[10, 10], [100, 10], [100, 100], [10, 100]]
    assert point_inside_polygon((50, 50), rect) is True
    assert point_inside_polygon([11, 11], rect) is True
    assert point_inside_polygon([99, 99], rect) is True


def test_point_inside_polygon_exterior() -> None:
    rect = [[10, 10], [100, 10], [100, 100], [10, 100]]
    assert point_inside_polygon((5, 5), rect) is False
    assert point_inside_polygon([105, 50], rect) is False
    assert point_inside_polygon([50, 150], rect) is False


def test_point_inside_polygon_boundary_conditions() -> None:
    rect = [[10, 10], [100, 10], [100, 100], [10, 100]]
    # Points on edges and vertices are considered inside
    assert point_inside_polygon((10, 10), rect) is True
    assert point_inside_polygon((100, 50), rect) is True
    assert point_inside_polygon((50, 10), rect) is True

    # Degenerate polygons with fewer than 3 points
    assert point_inside_polygon((10, 10), [[10, 10], [20, 20]]) is False
    assert point_inside_polygon((10, 10), []) is False


def test_is_restricted_zone() -> None:
    restricted = Zone(id="1", name="R", type="restricted", polygon=[[0, 0], [1, 0], [1, 1]])
    monitored = Zone(id="2", name="M", type="monitored", polygon=[[0, 0], [1, 0], [1, 1]])

    assert is_restricted_zone(restricted) is True
    assert is_restricted_zone(monitored) is False
    assert is_restricted_zone({"type": "restricted"}) is True
    assert is_restricted_zone({"type": "monitored"}) is False
    assert is_restricted_zone("restricted") is True
    assert is_restricted_zone("monitored") is False


def test_get_zone_prioritizes_restricted(sample_zone: Zone, sample_corridor_zone: Zone) -> None:
    zones = [sample_corridor_zone, sample_zone]
    # Point inside server room
    found = get_zone((50, 50), zones)
    assert found is not None
    assert found.id == "server_room"

    # Point inside corridor
    found_corridor = get_zone((150, 50), zones)
    assert found_corridor is not None
    assert found_corridor.id == "entry_corridor"

    # Point outside all zones
    assert get_zone((500, 500), zones) is None


def test_get_all_zones(sample_zone: Zone, sample_corridor_zone: Zone) -> None:
    zones = [sample_zone, sample_corridor_zone]
    # Point on border between the two
    border_point = (100, 50)
    all_found = get_all_zones(border_point, zones)
    zone_ids = {z.id for z in all_found}
    assert "server_room" in zone_ids
    assert "entry_corridor" in zone_ids


def test_load_zones_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "zones.json"
    data = {
        "zones": [
            {
                "id": "vault",
                "name": "Vault",
                "type": "restricted",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "color": [0, 0, 255],
                "authorized_hours": {"start": "09:00", "end": "17:00"},
            }
        ]
    }
    config_file.write_text(json.dumps(data), encoding="utf-8")

    zones = load_zones(config_file)
    assert len(zones) == 1
    assert zones[0].id == "vault"
    assert zones[0].name == "Vault"
    assert zones[0].type == "restricted"
    assert zones[0].polygon == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert zones[0].color == (0, 0, 255)


def test_load_zones_invalid_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_zones(non_existent)

    invalid_format = tmp_path / "invalid.json"
    invalid_format.write_text(json.dumps({"invalid_key": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="'zones' key"):
        load_zones(invalid_format)


def test_draw_zones_returns_annotated_copy(sample_zone: Zone) -> None:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    annotated = draw_zones(frame, [sample_zone], active_zone_ids={"server_room"})

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
    # Original frame must remain untouched
    assert np.array_equal(frame, np.zeros((120, 120, 3), dtype=np.uint8))


def test_track_history_records_zone_occupancy(sample_zone: Zone) -> None:
    manager = ZoneManager(zones=[sample_zone])
    history: dict[int, dict] = {}
    track = {
        "track_id": 7,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [40, 40, 60, 80],  # bottom_center = [50, 80], inside sample_zone
    }

    update_track_history(history, track, timestamp_seconds=3.5, zone_manager=manager)

    assert 7 in history
    assert len(history[7]["zones"]) == 1
    zone_rec = history[7]["zones"][0]
    assert zone_rec["zone_id"] == "server_room"
    assert zone_rec["zone_name"] == "Server Room"
    assert zone_rec["is_restricted"] is True
    assert zone_rec["timestamp_seconds"] == 3.5


def test_zone_presets_validity() -> None:
    expected_keys = {"server_room_demo", "outdoor_perimeter", "full_frame_secure", "dual_facility_split"}
    assert expected_keys.issubset(ZONE_PRESETS.keys())

    for key, preset in ZONE_PRESETS.items():
        assert "title" in preset
        assert "description" in preset
        assert "zones" in preset
        assert len(preset["zones"]) >= 1
        for z in preset["zones"]:
            assert isinstance(z, Zone)
            assert len(z.polygon) >= 3
            assert z.type in ("restricted", "monitored")


def test_scale_zones_to_frame() -> None:
    original = [
        Zone(
            id="test_zone",
            name="Test",
            type="restricted",
            polygon=[[100, 200], [500, 200], [500, 600], [100, 600]],
        )
    ]

    # Scaling 1280x720 to 1920x1080 (1.5x)
    scaled = scale_zones_to_frame(original, frame_width=1920, frame_height=1080, ref_width=1280, ref_height=720)
    assert len(scaled) == 1
    assert scaled[0].polygon == [[150, 300], [750, 300], [750, 900], [150, 900]]

    # Identity scaling
    identity = scale_zones_to_frame(original, frame_width=1280, frame_height=720, ref_width=1280, ref_height=720)
    assert identity[0].polygon == original[0].polygon

    # Invalid dimension fallback
    fallback = scale_zones_to_frame(original, frame_width=0, frame_height=0)
    assert fallback[0].polygon == original[0].polygon

