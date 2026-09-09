"""Unit tests for Phase 13: Secondary Security Scenarios (Loitering, Abandoned Object, Unusual Movement)."""

import pytest
from src.context import ContextEngine, SecurityContext
from src.events import (
    EVENT_ABANDONED_OBJECT,
    EVENT_LOITERING_DETECTED,
    EVENT_PERSON_DETECTED,
    EVENT_UNUSUAL_MOVEMENT,
    EventEngine,
    SecurityEvent,
)
from src.risk import RiskEngine, RISK_CRITICAL, RISK_HIGH, RISK_LOW, RISK_MEDIUM
from src.scenarios import SCENARIOS, ScenarioConfig, SecondaryScenarioAnalyzer
from src.zones import Zone, ZoneManager


def test_scenario_presets_exist():
    """Verify all 5 standard scenario presets are properly configured."""
    expected_scenarios = {
        "restricted_intrusion",
        "loitering",
        "abandoned_object",
        "after_hours",
        "unusual_movement",
    }
    assert expected_scenarios.issubset(set(SCENARIOS.keys()))
    for key, sc in SCENARIOS.items():
        assert isinstance(sc, ScenarioConfig)
        assert sc.loitering_threshold_seconds > 0
        assert sc.abandoned_threshold_seconds > 0
        assert sc.speed_anomaly_threshold > 0


def test_loitering_detection_triggers_event():
    """Verify that a subject lingering in one spot triggers loitering_detected."""
    config = ScenarioConfig(
        scenario_id="test_loiter",
        title="Test Loiter",
        description="Testing loitering trigger",
        loitering_threshold_seconds=2.0,
    )
    engine = EventEngine(scenario_config=config)

    # Subject remains nearly stationary at [200, 300] for 2.5 seconds
    events = []
    for i in range(15):
        t = i * 0.2
        # Slight jitter within 10px
        jitter = (i % 3) * 2
        track = {
            "track_id": 1,
            "class_name": "person",
            "bbox": [190 + jitter, 250, 210 + jitter, 310],
            "confidence": 0.9,
        }
        evs = engine.process_tracks([track], timestamp_seconds=t)
        events.extend(evs)

    loiter_events = [e for e in events if e.event_type == EVENT_LOITERING_DETECTED]
    assert len(loiter_events) == 1
    assert loiter_events[0].track_id == 1
    assert "loitering_duration" in loiter_events[0].metadata


def test_moving_subject_does_not_trigger_loitering():
    """Verify that a transit subject moving across the screen does not trigger loitering."""
    config = ScenarioConfig(
        scenario_id="test_transit",
        title="Test Transit",
        description="Testing fast transit does not loiter",
        loitering_threshold_seconds=2.0,
    )
    engine = EventEngine(scenario_config=config)

    events = []
    for i in range(15):
        t = i * 0.2
        # Significant movement in x
        x = 50 + i * 40
        track = {
            "track_id": 2,
            "class_name": "person",
            "bbox": [x - 10, 200, x + 10, 260],
            "confidence": 0.9,
        }
        evs = engine.process_tracks([track], timestamp_seconds=t)
        events.extend(evs)

    loiter_events = [e for e in events if e.event_type == EVENT_LOITERING_DETECTED]
    assert len(loiter_events) == 0


def test_abandoned_object_detection():
    """Verify unattended stationary object triggers abandoned_object_detected."""
    config = ScenarioConfig(
        scenario_id="test_abandoned",
        title="Test Abandoned",
        description="Testing abandoned object trigger",
        abandoned_threshold_seconds=1.5,
    )
    engine = EventEngine(scenario_config=config)

    backpack = {
        "track_id": 101,
        "class_name": "backpack",
        "bbox": [300, 400, 340, 450],
        "confidence": 0.85,
    }
    # Person is far away at [800, 400]
    person = {
        "track_id": 1,
        "class_name": "person",
        "bbox": [780, 300, 820, 450],
        "confidence": 0.9,
    }

    events = []
    for i in range(10):
        t = i * 0.2
        evs = engine.process_tracks([person], timestamp_seconds=t, object_detections=[backpack])
        events.extend(evs)

    abandoned_events = [e for e in events if e.event_type == EVENT_ABANDONED_OBJECT]
    assert len(abandoned_events) == 1
    assert abandoned_events[0].track_id == 101
    assert abandoned_events[0].metadata["class_name"] == "backpack"


def test_attended_object_does_not_trigger_abandoned():
    """Verify object with person in immediate proximity is NOT flagged as abandoned."""
    config = ScenarioConfig(
        scenario_id="test_attended",
        title="Test Attended",
        description="Testing attended object",
        abandoned_threshold_seconds=1.5,
    )
    engine = EventEngine(scenario_config=config)

    backpack = {
        "track_id": 102,
        "class_name": "backpack",
        "bbox": [300, 400, 340, 450],
        "confidence": 0.85,
    }
    # Person is right next to backpack at [310, 400]
    person = {
        "track_id": 1,
        "class_name": "person",
        "bbox": [300, 350, 330, 450],
        "confidence": 0.9,
    }

    events = []
    for i in range(10):
        t = i * 0.2
        evs = engine.process_tracks([person], timestamp_seconds=t, object_detections=[backpack])
        events.extend(evs)

    abandoned_events = [e for e in events if e.event_type == EVENT_ABANDONED_OBJECT]
    assert len(abandoned_events) == 0


def test_unusual_movement_high_velocity():
    """Verify sudden sprint triggers unusual_movement high velocity alert."""
    config = ScenarioConfig(
        scenario_id="test_sprint",
        title="Test Sprint",
        description="Testing sprint trigger",
        speed_anomaly_threshold=80.0,
    )
    engine = EventEngine(scenario_config=config)

    events = []
    # 10 steps of high velocity sprint: moving 30px every 0.1s => 300px/s > 80.0
    for i in range(10):
        t = i * 0.1
        x = 100 + i * 30
        track = {
            "track_id": 3,
            "class_name": "person",
            "bbox": [x - 10, 200, x + 10, 260],
            "confidence": 0.9,
        }
        evs = engine.process_tracks([track], timestamp_seconds=t)
        events.extend(evs)

    unusual = [e for e in events if e.event_type == EVENT_UNUSUAL_MOVEMENT]
    assert len(unusual) == 1
    assert unusual[0].metadata["type"] == "high_velocity"


def test_risk_scoring_with_secondary_events():
    """Verify risk engine evaluates abandoned object (+25) and loitering (+15)."""
    risk_engine = RiskEngine()

    # Context with loitering only
    loiter_ctx = {
        "events": ["person_detected", "loitering_detected"],
        "location": "Perimeter Gate",
    }
    assessment_loiter = risk_engine.evaluate_context(loiter_ctx)
    assert assessment_loiter.score == 15
    assert assessment_loiter.level == RISK_LOW
    assert "loitering" in assessment_loiter.factors

    # Context with abandoned object + after hours
    abandoned_ctx = {
        "events": ["abandoned_object_detected", "after_hours_activity"],
        "location": "Main Lobby",
    }
    assessment_abandoned = risk_engine.evaluate_context(abandoned_ctx)
    # 25 (abandoned) + 20 (after_hours) = 45 -> MEDIUM
    assert assessment_abandoned.score == 45
    assert assessment_abandoned.level == RISK_MEDIUM
    assert "abandoned_object" in assessment_abandoned.factors


def test_context_engine_formatting_secondary_events():
    """Verify ContextEngine produces factual descriptions for secondary events."""
    engine = ContextEngine()
    events = [
        SecurityEvent(
            event_type=EVENT_PERSON_DETECTED,
            track_id=7,
            zone="Corridor",
            timestamp="22:15:00",
            timestamp_seconds=0.0,
        ),
        SecurityEvent(
            event_type=EVENT_LOITERING_DETECTED,
            track_id=7,
            zone="Corridor",
            timestamp="22:15:08",
            timestamp_seconds=8.0,
            metadata={"loitering_duration": 8.0},
        ),
        SecurityEvent(
            event_type=EVENT_UNUSUAL_MOVEMENT,
            track_id=7,
            zone="Corridor",
            timestamp="22:15:10",
            timestamp_seconds=10.0,
            metadata={"type": "high_velocity", "speed": 165.2},
        ),
    ]

    ctx = engine.build_context_for_track(7, events)
    assert ctx is not None
    assert "loitering_detected" in ctx.events
    assert "unusual_movement" in ctx.events
    descs = " ".join(ctx.event_descriptions)
    assert "lingered in Corridor for 8 seconds" in descs
    assert "high velocity" in descs
