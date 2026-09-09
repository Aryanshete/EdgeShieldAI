"""Unit tests for Phase 8 Incident Manager, persistent storage, and report generation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.context import SecurityContext
from src.incidents import (
    Incident,
    IncidentManager,
    generate_next_incident_id,
    generate_report,
)
from src.reasoning import ReasoningOutput
from src.risk import RiskAssessment


@pytest.fixture
def sample_pipeline_data() -> tuple[SecurityContext, RiskAssessment, ReasoningOutput]:
    ctx = SecurityContext(
        location="Server Room",
        time="22:17",
        authorized_hours="08:00-18:00",
        track_id=7,
        events=["restricted_zone_entry", "after_hours_activity", "extended_dwell"],
        event_descriptions=["Person entered", "After hours", "Dwelt 23s"],
        start_timestamp="22:17:11",
        end_timestamp="22:17:35",
        duration_seconds=24.0,
    )
    assessment = RiskAssessment(
        score=85,
        level="HIGH",
        factors={"restricted_entry": 40, "after_hours": 20, "extended_dwell": 15},
        evidence=["Restricted-area entry", "After-hours activity", "Extended dwell"],
        summary_label="Potentially unauthorized activity detected",
    )
    reasoning = ReasoningOutput(
        risk_level="HIGH",
        evidence=["Restricted-area entry", "After-hours activity", "Extended dwell"],
        explanation="The sequence indicates potentially unauthorized activity in Server Room outside normal operating hours.",
        recommended_action="Alert security personnel and review associated camera footage.",
        summary="Potentially unauthorized activity detected in the restricted server room.",
    )
    return ctx, assessment, reasoning


def test_generate_next_incident_id() -> None:
    assert generate_next_incident_id([]) == "INC-2026-001"
    assert generate_next_incident_id(["INC-2026-001"]) == "INC-2026-002"
    assert generate_next_incident_id(["INC-2026-001", "INC-2026-005"]) == "INC-2026-006"


def test_create_incident_matches_schema(
    tmp_path: Path, sample_pipeline_data: tuple[SecurityContext, RiskAssessment, ReasoningOutput]
) -> None:
    ctx, assessment, reasoning = sample_pipeline_data
    mgr = IncidentManager(storage_path=tmp_path / "incidents.json")

    incident = mgr.create_incident(ctx, assessment, reasoning)
    d = incident.as_dict()

    # Exact fields required by PDF Page 20
    assert d["incident_id"] == "INC-2026-001"
    assert d["timestamp"] == "22:17:35"
    assert d["location"] == "Server Room"
    assert d["risk_level"] == "HIGH"
    assert d["risk_score"] == 85
    assert d["events"] == ["restricted_zone_entry", "after_hours_activity", "extended_dwell"]
    assert d["evidence"] == ["Restricted-area entry", "After-hours activity", "Extended dwell"]
    assert "potentially unauthorized" in d["explanation"]
    assert "Alert security" in d["recommended_action"]
    assert d["status"] == "OPEN"


def test_save_and_load_incidents(
    tmp_path: Path, sample_pipeline_data: tuple[SecurityContext, RiskAssessment, ReasoningOutput]
) -> None:
    ctx, assessment, reasoning = sample_pipeline_data
    file_path = tmp_path / "incidents.json"
    mgr = IncidentManager(storage_path=file_path)

    inc1 = mgr.create_incident(ctx, assessment, reasoning)
    assert file_path.is_file()

    # Create a fresh manager pointing to the same file
    mgr2 = IncidentManager(storage_path=file_path)
    assert len(mgr2.incidents) == 1
    loaded = mgr2.get_incident("INC-2026-001")
    assert loaded is not None
    assert loaded.incident_id == inc1.incident_id
    assert loaded.risk_score == 85


def test_update_incident_status(
    tmp_path: Path, sample_pipeline_data: tuple[SecurityContext, RiskAssessment, ReasoningOutput]
) -> None:
    ctx, assessment, reasoning = sample_pipeline_data
    mgr = IncidentManager(storage_path=tmp_path / "incidents.json")
    mgr.create_incident(ctx, assessment, reasoning)

    updated = mgr.update_incident_status("INC-2026-001", "INVESTIGATING")
    assert updated is not None
    assert updated.status == "INVESTIGATING"

    resolved = mgr.update_incident_status("INC-2026-001", "RESOLVED")
    assert resolved is not None
    assert resolved.status == "RESOLVED"

    # Reload from disk to verify persistence
    mgr_reloaded = IncidentManager(storage_path=tmp_path / "incidents.json")
    assert mgr_reloaded.get_incident("INC-2026-001").status == "RESOLVED"

    # Invalid status should raise ValueError
    with pytest.raises(ValueError, match="Invalid status"):
        mgr.update_incident_status("INC-2026-001", "INVALID_STATUS")


def test_generate_report(tmp_path: Path, sample_pipeline_data: tuple[SecurityContext, RiskAssessment, ReasoningOutput]) -> None:
    ctx, assessment, reasoning = sample_pipeline_data
    mgr = IncidentManager(storage_path=tmp_path / "dummy.json")
    incident = mgr.create_incident(ctx, assessment, reasoning)

    report = generate_report(incident)
    assert "# EdgeShield AI - Security Incident Report" in report
    assert "`INC-2026-001`" in report
    assert "Server Room" in report
    assert "HIGH" in report
    assert "85/100" in report
    assert "Alert security personnel" in report


def test_generate_html_and_text_reports(tmp_path: Path, sample_pipeline_data: tuple[SecurityContext, RiskAssessment, ReasoningOutput]) -> None:
    ctx, assessment, reasoning = sample_pipeline_data
    mgr = IncidentManager(storage_path=tmp_path / "dummy_html.json")
    incident = mgr.create_incident(ctx, assessment, reasoning)

    html = mgr.generate_html_report(incident)
    assert "<!DOCTYPE html>" in html
    assert "INC-2026-001" in html
    assert "Server Room" in html
    assert "Print / Save as PDF" in html

    text = mgr.generate_text_report(incident)
    assert "EDGESHIELD AI - SECURITY INCIDENT AUDIT REPORT" in text
    assert "INCIDENT ID:      INC-2026-001" in text
    assert "Server Room" in text
