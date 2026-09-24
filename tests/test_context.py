"""Unit tests for temporal event analysis and context building."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context import ContextEngine, SecurityContext
from src.events import (
    EVENT_AFTER_HOURS_ACTIVITY,
    EVENT_PERSON_DETECTED,
    EVENT_RESTRICTED_ZONE_DWELL,
    EVENT_RESTRICTED_ZONE_ENTRY,
    EVENT_RESTRICTED_ZONE_EXIT,
    EVENT_ZONE_APPROACH,
    SecurityEvent,
)
from src.zones import Zone, ZoneManager


@pytest.fixture
def sample_zone_manager() -> ZoneManager:
    server_room = Zone(
        id="server_room",
        name="Server Room",
        type="restricted",
        polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
        authorized_hours={"start": "08:00", "end": "18:00"},
    )
    return ZoneManager(zones=[server_room])


@pytest.fixture
def sequence_events() -> list[SecurityEvent]:
    return [
        SecurityEvent(
            event_type=EVENT_PERSON_DETECTED,
            track_id=7,
            zone=None,
            timestamp="22:17:01",
            timestamp_seconds=1.0,
        ),
        SecurityEvent(
            event_type=EVENT_ZONE_APPROACH,
            track_id=7,
            zone="Server Room",
            timestamp="22:17:08",
            timestamp_seconds=8.0,
            metadata={"distance_pixels": 45.0, "zone_id": "server_room"},
        ),
        SecurityEvent(
            event_type=EVENT_RESTRICTED_ZONE_ENTRY,
            track_id=7,
            zone="Server Room",
            timestamp="22:17:11",
            timestamp_seconds=11.0,
            metadata={"zone_id": "server_room"},
        ),
        SecurityEvent(
            event_type=EVENT_AFTER_HOURS_ACTIVITY,
            track_id=7,
            zone="Server Room",
            timestamp="22:17:11",
            timestamp_seconds=11.0,
            metadata={"zone_id": "server_room"},
        ),
        SecurityEvent(
            event_type=EVENT_RESTRICTED_ZONE_DWELL,
            track_id=7,
            zone="Server Room",
            timestamp="22:17:20",
            timestamp_seconds=20.0,
            metadata={"zone_id": "server_room", "dwell_seconds": 9.0},
        ),
        SecurityEvent(
            event_type=EVENT_RESTRICTED_ZONE_EXIT,
            track_id=7,
            zone="Server Room",
            timestamp="22:17:34",
            timestamp_seconds=34.0,
            metadata={"zone_id": "server_room", "total_dwell_seconds": 23.0},
        ),
    ]


def test_build_context_for_track(
    sample_zone_manager: ZoneManager, sequence_events: list[SecurityEvent]
) -> None:
    engine = ContextEngine(zone_manager=sample_zone_manager)
    ctx = engine.build_context_for_track(7, sequence_events)

    assert ctx is not None
    assert ctx.track_id == 7
    assert ctx.location == "Server Room"
    assert ctx.time == "22:17"
    assert ctx.authorized_hours == "08:00-18:00"
    assert "restricted_zone_entry" in ctx.events
    assert "after_hours_activity" in ctx.events
    assert "extended_dwell" in ctx.events
    assert "restricted_zone_exit" in ctx.events
    assert ctx.start_timestamp == "22:17:01"
    assert ctx.end_timestamp == "22:17:34"
    assert ctx.duration_seconds == 33.0


def test_context_as_dict_matches_spec_schema(
    sample_zone_manager: ZoneManager, sequence_events: list[SecurityEvent]
) -> None:
    engine = ContextEngine(zone_manager=sample_zone_manager)
    ctx = engine.build_context_for_track(7, sequence_events)
    assert ctx is not None

    d = ctx.as_dict()
    # Check exact keys from PDF Page 15
    assert d["location"] == "Server Room"
    assert d["time"] == "22:17"
    assert d["authorized_hours"] == "08:00-18:00"
    assert d["track_id"] == 7
    assert isinstance(d["events"], list)
    assert isinstance(d["event_descriptions"], list)


def test_event_descriptions_contain_grounded_facts(
    sample_zone_manager: ZoneManager, sequence_events: list[SecurityEvent]
) -> None:
    engine = ContextEngine(zone_manager=sample_zone_manager)
    ctx = engine.build_context_for_track(7, sequence_events)
    assert ctx is not None

    descriptions = ctx.event_descriptions
    # Verify grounded factual statements
    assert any("entered restricted zone" in desc for desc in descriptions)
    assert any("outside authorized hours" in desc for desc in descriptions)
    assert any("exited restricted zone" in desc for desc in descriptions)


def test_build_all_contexts_groups_by_track(sample_zone_manager: ZoneManager) -> None:
    events = [
        SecurityEvent(
            event_type=EVENT_PERSON_DETECTED,
            track_id=1,
            zone="Server Room",
            timestamp="22:17:01",
            timestamp_seconds=1.0,
        ),
        SecurityEvent(
            event_type=EVENT_PERSON_DETECTED,
            track_id=2,
            zone="Server Room",
            timestamp="22:17:05",
            timestamp_seconds=5.0,
        ),
    ]
    engine = ContextEngine(zone_manager=sample_zone_manager)
    contexts = engine.build_all_contexts(events)

    assert len(contexts) == 2
    assert {c.track_id for c in contexts} == {1, 2}


def test_empty_events_returns_empty_context(sample_zone_manager: ZoneManager) -> None:
    engine = ContextEngine(zone_manager=sample_zone_manager)
    assert engine.build_all_contexts([]) == []
    assert engine.build_context_for_track(99, []) is None


def test_export_and_load_contexts(tmp_path: Path, sample_zone_manager: ZoneManager, sequence_events: list[SecurityEvent]) -> None:
    engine = ContextEngine(zone_manager=sample_zone_manager)
    contexts = engine.build_all_contexts(sequence_events)

    out_file = tmp_path / "context.json"
    engine.export_contexts(contexts, out_file)
    assert out_file.is_file()

    loaded = ContextEngine.load_contexts(out_file)
    assert len(loaded) == 1
    assert loaded[0].track_id == 7
    assert loaded[0].location == "Server Room"
    assert loaded[0].events == contexts[0].events
