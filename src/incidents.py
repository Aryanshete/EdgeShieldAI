"""Persistent security incident management, audit reporting, and lifecycle tracking for EdgeShield AI."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.context import SecurityContext
from src.reasoning import ReasoningOutput
from src.risk import RiskAssessment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCIDENTS_FILE = PROJECT_ROOT / "data" / "incidents.json"

VALID_STATUSES = ("OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED")


@dataclass(frozen=True)
class Incident:
    """Persistent security incident record matching EdgeShield specification schema."""

    incident_id: str
    timestamp: str
    location: str
    risk_level: str
    risk_score: int
    events: list[str]
    evidence: list[str]
    explanation: str
    recommended_action: str
    status: str = "OPEN"
    summary: str = ""
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a structured JSON dictionary matching the specification."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "location": self.location,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "events": list(self.events),
            "evidence": list(self.evidence),
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "status": self.status,
            "summary": self.summary,
            "track_id": self.track_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Incident:
        """Construct an Incident from a dictionary."""
        return cls(
            incident_id=str(data["incident_id"]),
            timestamp=str(data["timestamp"]),
            location=str(data["location"]),
            risk_level=str(data["risk_level"]),
            risk_score=int(data["risk_score"]),
            events=list(data.get("events", [])),
            evidence=list(data.get("evidence", [])),
            explanation=str(data.get("explanation", "")),
            recommended_action=str(data.get("recommended_action", "")),
            status=str(data.get("status", "OPEN")),
            summary=str(data.get("summary", "")),
            track_id=data.get("track_id"),
            metadata=dict(data.get("metadata", {})),
        )


def generate_next_incident_id(existing_ids: Sequence[str], year: int = 2026) -> str:
    """Generate the next sequential incident identifier (e.g. INC-2026-001)."""
    max_num = 0
    pattern = re.compile(rf"^INC-{year}-(\d+)$")
    for item in existing_ids:
        match = pattern.match(item.strip())
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"INC-{year}-{max_num + 1:03d}"


def generate_report(incident: Incident) -> str:
    """Generate an audit-ready, formatted Markdown security incident report."""
    evidence_lines = "\n".join(f"- {item}" for item in incident.evidence) if incident.evidence else "- None recorded"
    events_lines = "\n".join(f"1. `{ev}`" for ev in incident.events) if incident.events else "None recorded"
    subject_str = f"Anonymous Track #{incident.track_id:02d}" if incident.track_id is not None else "Unassigned"

    return f"""# EdgeShield AI — Security Incident Report

**Incident ID:** `{incident.incident_id}`  
**Status:** `{incident.status}`  
**Severity:** `{incident.risk_level}` (Deterministic Risk Score: `{incident.risk_score}/100`)  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Overview
- **Location:** {incident.location}
- **Timestamp:** {incident.timestamp}
- **Subject:** {subject_str}
- **Summary:** {incident.summary or incident.explanation}

---

## 2. Physical & Spatial Evidence Grounding
The following objective observations were verified by the EdgeShield vision pipeline:
{evidence_lines}

---

## 3. Verified Event Sequence
{events_lines}

---

## 4. Contextual AI Interpretation
{incident.explanation}

---

## 5. Recommended Operator Response
> [!IMPORTANT]
> **Action Required:**
> {incident.recommended_action}

---
*Report generated automatically by EdgeShield AI Edge Security Platform. All observations are based on deterministic event verification without facial recognition.*
"""


class IncidentManager:
    """Manages creation, persistent storage, status updates, and reports for incidents."""

    def __init__(self, storage_path: str | Path = DEFAULT_INCIDENTS_FILE) -> None:
        self.storage_path = Path(storage_path)
        self.incidents: list[Incident] = []
        if self.storage_path.is_file():
            self.load_incidents()

    def load_incidents(self) -> list[Incident]:
        """Load incidents from the configured JSON file."""
        if not self.storage_path.is_file():
            self.incidents = []
            return []

        with open(self.storage_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

        self.incidents = [Incident.from_dict(item) for item in data if isinstance(item, dict)]
        return list(self.incidents)

    def _persist(self) -> None:
        """Write current incidents to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [inc.as_dict() for inc in self.incidents]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def get_incident(self, incident_id: str) -> Incident | None:
        """Find an incident by its unique identifier."""
        for inc in self.incidents:
            if inc.incident_id == incident_id:
                return inc
        return None

    def save_incident(self, incident: Incident) -> None:
        """Save a new incident or update an existing one if ID matches."""
        for i, existing in enumerate(self.incidents):
            if existing.incident_id == incident.incident_id:
                self.incidents[i] = incident
                self._persist()
                return

        self.incidents.append(incident)
        self._persist()

    def create_incident(
        self,
        context: SecurityContext | dict[str, Any],
        assessment: RiskAssessment | dict[str, Any],
        reasoning: ReasoningOutput | dict[str, Any],
        status: str = "OPEN",
        incident_id: str | None = None,
    ) -> Incident:
        """Construct, register, and persist a new incident from pipeline outputs."""
        ctx_dict = context.as_dict() if isinstance(context, SecurityContext) else dict(context)
        risk_dict = assessment.as_dict() if isinstance(assessment, RiskAssessment) else dict(assessment)
        reas_dict = reasoning.as_dict() if isinstance(reasoning, ReasoningOutput) else dict(reasoning)

        if not incident_id:
            existing_ids = [inc.incident_id for inc in self.incidents]
            incident_id = generate_next_incident_id(existing_ids)

        timestamp = ctx_dict.get("end_timestamp") or ctx_dict.get("start_timestamp") or ctx_dict.get("time", "00:00")
        if len(timestamp) == 5:  # HH:MM -> HH:MM:00
            timestamp += ":00"

        incident = Incident(
            incident_id=incident_id,
            timestamp=timestamp,
            location=ctx_dict.get("location", "Monitored Area"),
            risk_level=reas_dict.get("risk_level", risk_dict.get("level", "LOW")),
            risk_score=int(risk_dict.get("score", 0)),
            events=list(ctx_dict.get("events", [])),
            evidence=list(reas_dict.get("evidence", risk_dict.get("evidence", []))),
            explanation=str(reas_dict.get("explanation", "")),
            recommended_action=str(reas_dict.get("recommended_action", "")),
            status=status if status in VALID_STATUSES else "OPEN",
            summary=str(reas_dict.get("summary", "")),
            track_id=ctx_dict.get("track_id"),
            metadata={
                "duration_seconds": ctx_dict.get("duration_seconds", 0.0),
                "is_fallback": reas_dict.get("is_fallback", False),
            },
        )

        self.save_incident(incident)
        return incident

    def update_incident_status(self, incident_id: str, new_status: str) -> Incident | None:
        """Update lifecycle status of an existing incident."""
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}")

        incident = self.get_incident(incident_id)
        if not incident:
            return None

        updated = Incident(
            incident_id=incident.incident_id,
            timestamp=incident.timestamp,
            location=incident.location,
            risk_level=incident.risk_level,
            risk_score=incident.risk_score,
            events=incident.events,
            evidence=incident.evidence,
            explanation=incident.explanation,
            recommended_action=incident.recommended_action,
            status=new_status,
            summary=incident.summary,
            track_id=incident.track_id,
            metadata=dict(incident.metadata),
        )
        self.save_incident(updated)
        return updated

    def generate_report(self, incident: Incident) -> str:
        """Generate formatted Markdown audit report for the given incident."""
        return generate_report(incident)
