"""Unit tests for event engine and security event generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.events import (
    EVENT_AFTER_HOURS_ACTIVITY,
    EVENT_PERSON_DETECTED,
    EVENT_RESTRICTED_ZONE_DWELL,
    EVENT_RESTRICTED_ZONE_ENTRY,
    EVENT_RESTRICTED_ZONE_EXIT,
    EVENT_ZONE_APPROACH,
    EventEngine,
    SecurityEvent,
    format_timestamp,
    is_time_after_hours,
)
from src.zones import Zone, ZoneManager


@pytest.fixture
def test_zone_manager() -> ZoneManager:
    server_room = Zone(
        id="server_room",
        name="Server Room",
        type="restricted",
        polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
        authorized_hours={"start": "08:00", "end": "18:00"},
    )
    corridor = Zone(
        id="corridor",
        name="Corridor",
        type="monitored",
        polygon=[[300, 100], [500, 100], [500, 300], [300, 300]],
        authorized_hours={"start": "00:00", "end": "23:59"},
    )
    return ZoneManager(zones=[server_room, corridor])


def test_format_timestamp() -> None:
    # 22:17:00 + 11.5s -> 22:17:11
    assert format_timestamp(11.5, base_time="22:17:00") == "22:17:11"
    # Relative fallback when base_time is None
    assert format_timestamp(65.0, base_time=None) == "00:01:05"


def test_is_time_after_hours() -> None:
    hours = {"start": "08:00", "end": "18:00"}
    # Daytime inside
    assert is_time_after_hours("12:00:00", hours) is False
    assert is_time_after_hours("08:00:00", hours) is False
    assert is_time_after_hours("18:00:00", hours) is False
    # Nighttime outside
    assert is_time_after_hours("22:17:11", hours) is True
    assert is_time_after_hours("04:30:00", hours) is True

    # Overnight shift (e.g. 20:00 to 06:00)
    overnight = {"start": "20:00", "end": "06:00"}
    assert is_time_after_hours("22:00:00", overnight) is False
    assert is_time_after_hours("02:00:00", overnight) is False
    assert is_time_after_hours("12:00:00", overnight) is True

    # Empty hours -> False
    assert is_time_after_hours("22:00:00", None) is False


def test_person_detected_triggers_once(test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(zone_manager=test_zone_manager, base_time="22:17:00")
    track = {"track_id": 7, "class_name": "person", "bbox": [10, 10, 30, 50]}

    # Frame 1
    events1 = engine.process_tracks([track], timestamp_seconds=0.0)
    assert len(events1) == 1
    assert events1[0].event_type == EVENT_PERSON_DETECTED
    assert events1[0].track_id == 7
    assert events1[0].timestamp == "22:17:00"

    # Frame 2 (same track) - should NOT repeat person_detected
    events2 = engine.process_tracks([track], timestamp_seconds=0.5)
    person_detected_events = [e for e in events2 if e.event_type == EVENT_PERSON_DETECTED]
    assert len(person_detected_events) == 0


def test_restricted_zone_entry_and_after_hours(test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(zone_manager=test_zone_manager, base_time="22:17:00")
    # Bottom center is at (200, 250), inside Server Room
    track_inside = {"track_id": 7, "class_name": "person", "bbox": [180, 150, 220, 250]}

    events = engine.process_tracks([track_inside], timestamp_seconds=1.0)
    types = [e.event_type for e in events]
    assert EVENT_PERSON_DETECTED in types
    assert EVENT_RESTRICTED_ZONE_ENTRY in types
    assert EVENT_AFTER_HOURS_ACTIVITY in types

    entry_event = next(e for e in events if e.event_type == EVENT_RESTRICTED_ZONE_ENTRY)
    assert entry_event.zone == "Server Room"
    assert entry_event.track_id == 7
    assert entry_event.timestamp == "22:17:01"


def test_restricted_zone_dwell(test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(
        zone_manager=test_zone_manager,
        base_time="22:17:00",
        dwell_threshold_seconds=3.0,
    )
    track_inside = {"track_id": 7, "class_name": "person", "bbox": [180, 150, 220, 250]}

    # Entry at t = 1.0s
    engine.process_tracks([track_inside], timestamp_seconds=1.0)
    # Still inside at t = 2.5s (dwell duration = 1.5s < 3.0s threshold)
    events_mid = engine.process_tracks([track_inside], timestamp_seconds=2.5)
    assert not any(e.event_type == EVENT_RESTRICTED_ZONE_DWELL for e in events_mid)

    # At t = 4.2s (dwell duration = 3.2s >= 3.0s threshold)
    events_dwell = engine.process_tracks([track_inside], timestamp_seconds=4.2)
    dwell_event = next(e for e in events_dwell if e.event_type == EVENT_RESTRICTED_ZONE_DWELL)
    assert dwell_event.zone == "Server Room"
    assert dwell_event.metadata["dwell_seconds"] >= 3.0

    # At t = 5.0s, dwell event should not fire again for the same dwell session
    events_later = engine.process_tracks([track_inside], timestamp_seconds=5.0)
    assert not any(e.event_type == EVENT_RESTRICTED_ZONE_DWELL for e in events_later)


def test_restricted_zone_exit(test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(zone_manager=test_zone_manager, base_time="22:17:00")
    # Step 1: inside Server Room
    track_inside = {"track_id": 7, "class_name": "person", "bbox": [180, 150, 220, 250]}
    engine.process_tracks([track_inside], timestamp_seconds=1.0)

    # Step 2: moved out to (20, 20), outside all zones
    track_outside = {"track_id": 7, "class_name": "person", "bbox": [10, 10, 30, 20]}
    events_exit = engine.process_tracks([track_outside], timestamp_seconds=5.0)

    exit_event = next(e for e in events_exit if e.event_type == EVENT_RESTRICTED_ZONE_EXIT)
    assert exit_event.zone == "Server Room"
    assert exit_event.metadata["total_dwell_seconds"] == 4.0


def test_zone_approach(test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(
        zone_manager=test_zone_manager,
        base_time="22:17:00",
        approach_distance_pixels=50.0,
    )
    # Server room is at x: 100-300, y: 100-300.
    # Point at (70, 200) is 30px away from the left boundary (100, y).
    track_near = {"track_id": 7, "class_name": "person", "bbox": [60, 150, 80, 200]}
    events = engine.process_tracks([track_near], timestamp_seconds=0.5)

    approach_event = next((e for e in events if e.event_type == EVENT_ZONE_APPROACH), None)
    assert approach_event is not None
    assert approach_event.zone == "Server Room"
    assert approach_event.metadata["distance_pixels"] == 30.0


def test_export_and_load_events(tmp_path: Path, test_zone_manager: ZoneManager) -> None:
    engine = EventEngine(zone_manager=test_zone_manager, base_time="22:17:00")
    track = {"track_id": 7, "class_name": "person", "bbox": [180, 150, 220, 250]}
    engine.process_tracks([track], timestamp_seconds=1.0)

    out_file = tmp_path / "events.json"
    engine.export_events(out_file)
    assert out_file.is_file()

    loaded = EventEngine.load_events(out_file)
    assert len(loaded) == len(engine.events)
    assert loaded[0].event_type == engine.events[0].event_type
    assert loaded[0].timestamp == engine.events[0].timestamp
