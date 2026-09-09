"""Unit tests for Phase 6 deterministic risk engine and heuristic scoring."""

from __future__ import annotations

import pytest

from src.context import SecurityContext
from src.events import (
    EVENT_AFTER_HOURS_ACTIVITY,
    EVENT_OBJECT_INTERACTION,
    EVENT_PERSON_DETECTED,
    EVENT_RESTRICTED_ZONE_DWELL,
    EVENT_RESTRICTED_ZONE_ENTRY,
    SecurityEvent,
)
from src.risk import (
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RiskEngine,
    get_risk_level,
)


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine()


def test_get_risk_level_boundaries() -> None:
    assert get_risk_level(0) == RISK_LOW
    assert get_risk_level(29) == RISK_LOW
    assert get_risk_level(30) == RISK_MEDIUM
    assert get_risk_level(59) == RISK_MEDIUM
    assert get_risk_level(60) == RISK_HIGH
    assert get_risk_level(79) == RISK_HIGH
    assert get_risk_level(80) == RISK_CRITICAL
    assert get_risk_level(100) == RISK_CRITICAL


def test_low_risk_baseline(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Entry Corridor",
        "time": "14:00",
        "authorized_hours": "08:00-18:00",
        "track_id": 1,
        "events": ["person_detected"],
    }
    assessment = risk_engine.evaluate_context(ctx)

    assert assessment.score == 0
    assert assessment.level == RISK_LOW
    assert len(assessment.factors) == 0
    assert len(assessment.evidence) == 0


def test_medium_risk_restricted_entry_daytime(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Server Room",
        "time": "14:00",
        "authorized_hours": "08:00-18:00",
        "track_id": 2,
        "events": ["person_detected", "restricted_zone_entry"],
    }
    assessment = risk_engine.evaluate_context(ctx)

    assert assessment.score == 40
    assert assessment.level == RISK_MEDIUM
    assert assessment.factors["restricted_entry"] == 40
    assert "Restricted-area entry" in assessment.evidence


def test_high_risk_after_hours_entry(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Server Room",
        "time": "22:17",
        "authorized_hours": "08:00-18:00",
        "track_id": 1,
        "events": ["person_detected", "restricted_zone_entry", "after_hours_activity"],
    }
    assessment = risk_engine.evaluate_context(ctx)

    # 40 (restricted entry) + 20 (after hours) = 60
    assert assessment.score == 60
    assert assessment.level == RISK_HIGH
    assert assessment.factors["restricted_entry"] == 40
    assert assessment.factors["after_hours"] == 20
    assert "Restricted-area entry" in assessment.evidence
    assert "After-hours activity" in assessment.evidence


def test_critical_risk_full_intrusion(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Server Room",
        "time": "22:17",
        "authorized_hours": "08:00-18:00",
        "track_id": 7,
        "events": [
            "person_detected",
            "restricted_zone_entry",
            "after_hours_activity",
            "extended_dwell",
            "object_interaction",
        ],
    }
    assessment = risk_engine.evaluate_context(ctx)

    # 40 + 20 + 15 + 20 = 95
    assert assessment.score == 95
    assert assessment.level == RISK_CRITICAL
    assert len(assessment.evidence) == 4
    assert "Critical security alert" in assessment.summary_label


def test_score_clamping_to_100(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Server Room",
        "time": "22:17",
        "authorized_hours": "08:00-18:00",
        "track_id": 7,
        "events": [
            "restricted_zone_entry",  # 40
            "after_hours_activity",   # 20
            "extended_dwell",         # 15
            "object_interaction",     # 20
            "unusual_movement",       # 10
        ],
    }
    assessment = risk_engine.evaluate_context(ctx)

    # Sum is 105, clamped to 100
    assert assessment.score == 100
    assert assessment.level == RISK_CRITICAL


def test_strict_determinism(risk_engine: RiskEngine) -> None:
    ctx = {
        "location": "Server Room",
        "time": "22:17",
        "authorized_hours": "08:00-18:00",
        "track_id": 1,
        "events": ["restricted_zone_entry", "after_hours_activity"],
    }
    first_result = risk_engine.evaluate_context(ctx)
    for _ in range(50):
        subsequent = risk_engine.evaluate_context(ctx)
        assert subsequent.score == first_result.score
        assert subsequent.level == first_result.level
        assert subsequent.factors == first_result.factors
        assert subsequent.evidence == first_result.evidence


def test_evaluate_security_context_object(risk_engine: RiskEngine) -> None:
    sec_ctx = SecurityContext(
        location="Server Room",
        time="22:17",
        authorized_hours="08:00-18:00",
        track_id=7,
        events=["restricted_zone_entry", "extended_dwell"],
        event_descriptions=["Person entered", "Person dwelt"],
        start_timestamp="22:17:00",
        end_timestamp="22:17:25",
        duration_seconds=25.0,
    )
    assessment = risk_engine.evaluate_context(sec_ctx)

    # 40 (restricted) + 15 (dwell) = 55 -> MEDIUM
    assert assessment.score == 55
    assert assessment.level == RISK_MEDIUM


def test_evaluate_raw_events(risk_engine: RiskEngine) -> None:
    events = [
        SecurityEvent(event_type=EVENT_PERSON_DETECTED, track_id=1, zone=None, timestamp="22:17:00", timestamp_seconds=0.0),
        SecurityEvent(event_type=EVENT_RESTRICTED_ZONE_ENTRY, track_id=1, zone="Server Room", timestamp="22:17:01", timestamp_seconds=1.0),
        SecurityEvent(event_type=EVENT_AFTER_HOURS_ACTIVITY, track_id=1, zone="Server Room", timestamp="22:17:01", timestamp_seconds=1.0),
        SecurityEvent(event_type=EVENT_RESTRICTED_ZONE_DWELL, track_id=1, zone="Server Room", timestamp="22:17:15", timestamp_seconds=15.0),
    ]
    assessment = risk_engine.evaluate_events(events)

    # 40 + 20 + 15 = 75 -> HIGH
    assert assessment.score == 75
    assert assessment.level == RISK_HIGH
