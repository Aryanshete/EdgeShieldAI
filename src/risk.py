"""Deterministic risk engine and heuristic scoring for EdgeShield AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from src.context import SecurityContext
from src.events import (
    EVENT_ABANDONED_OBJECT,
    EVENT_AFTER_HOURS_ACTIVITY,
    EVENT_LOITERING_DETECTED,
    EVENT_OBJECT_INTERACTION,
    EVENT_RESTRICTED_ZONE_DWELL,
    EVENT_RESTRICTED_ZONE_ENTRY,
    EVENT_UNUSUAL_MOVEMENT,
    SecurityEvent,
)

# Risk Level Classification Constants
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"

# Prototype heuristic factor weights (as defined in project specification)
DEFAULT_WEIGHT_RESTRICTED_ENTRY = 40
DEFAULT_WEIGHT_AFTER_HOURS = 20
DEFAULT_WEIGHT_EXTENDED_DWELL = 15
DEFAULT_WEIGHT_OBJECT_INTERACTION = 20
DEFAULT_WEIGHT_UNUSUAL_MOVEMENT = 10
DEFAULT_WEIGHT_ABANDONED_OBJECT = 25
DEFAULT_WEIGHT_LOITERING = 15


@dataclass(frozen=True)
class RiskAssessment:
    """Objective baseline risk evaluation derived deterministically from event sequences."""

    score: int
    level: str
    factors: dict[str, int]
    evidence: list[str]
    summary_label: str

    def as_dict(self) -> dict[str, Any]:
        """Return a structured dictionary representation."""
        return {
            "score": self.score,
            "level": self.level,
            "factors": dict(self.factors),
            "evidence": list(self.evidence),
            "summary_label": self.summary_label,
        }


def get_risk_level(score: int) -> str:
    """Map a numerical risk score (0-100) to its discrete security severity level.

    0-29:   LOW
    30-59:  MEDIUM
    60-79:  HIGH
    80-100: CRITICAL
    """
    if score < 30:
        return RISK_LOW
    if score < 60:
        return RISK_MEDIUM
    if score < 80:
        return RISK_HIGH
    return RISK_CRITICAL


def get_summary_label(level: str) -> str:
    """Return an explainable, professional summary label for the assessed risk level."""
    if level == RISK_CRITICAL:
        return "Critical security alert: High-priority unauthorized sequence"
    if level == RISK_HIGH:
        return "Potentially unauthorized activity detected"
    if level == RISK_MEDIUM:
        return "Elevated security notice: Monitored activity"
    return "Routine observation: Normal operating parameters"


class RiskEngine:
    """Calculates deterministic baseline risk scores and severity levels from security events.

    Provides stable, reproducible evaluation before involving LLM reasoning, ensuring
    alert reliability even during offline or fallback operation.
    """

    def __init__(
        self,
        weight_restricted_entry: int = DEFAULT_WEIGHT_RESTRICTED_ENTRY,
        weight_after_hours: int = DEFAULT_WEIGHT_AFTER_HOURS,
        weight_extended_dwell: int = DEFAULT_WEIGHT_EXTENDED_DWELL,
        weight_object_interaction: int = DEFAULT_WEIGHT_OBJECT_INTERACTION,
        weight_unusual_movement: int = DEFAULT_WEIGHT_UNUSUAL_MOVEMENT,
        weight_abandoned_object: int = DEFAULT_WEIGHT_ABANDONED_OBJECT,
        weight_loitering: int = DEFAULT_WEIGHT_LOITERING,
    ) -> None:
        self.weight_restricted_entry = weight_restricted_entry
        self.weight_after_hours = weight_after_hours
        self.weight_extended_dwell = weight_extended_dwell
        self.weight_object_interaction = weight_object_interaction
        self.weight_unusual_movement = weight_unusual_movement
        self.weight_abandoned_object = weight_abandoned_object
        self.weight_loitering = weight_loitering

    def evaluate_context(self, context: SecurityContext | dict[str, Any]) -> RiskAssessment:
        """Evaluate a SecurityContext instance or dictionary and return a RiskAssessment."""
        if isinstance(context, SecurityContext):
            event_tokens = set(context.events)
        elif isinstance(context, dict):
            event_tokens = set(context.get("events", []))
        else:
            raise TypeError(f"Expected SecurityContext or dict, got {type(context)}")

        score = 0
        factors: dict[str, int] = {}
        evidence: list[str] = []

        # 1. Restricted-area entry (+40)
        if "restricted_zone_entry" in event_tokens:
            score += self.weight_restricted_entry
            factors["restricted_entry"] = self.weight_restricted_entry
            evidence.append("Restricted-area entry")

        # 2. Activity outside authorized hours (+20)
        if "after_hours_activity" in event_tokens:
            score += self.weight_after_hours
            factors["after_hours"] = self.weight_after_hours
            evidence.append("After-hours activity")

        # 3. Extended dwell time (+15)
        if "extended_dwell" in event_tokens or "restricted_zone_dwell" in event_tokens:
            score += self.weight_extended_dwell
            factors["extended_dwell"] = self.weight_extended_dwell
            evidence.append("Extended dwell")

        # 4. Object interaction / stationary manipulation (+20)
        if "object_interaction" in event_tokens:
            score += self.weight_object_interaction
            factors["object_interaction"] = self.weight_object_interaction
            evidence.append("Object interaction")

        # 5. Unusual movement pattern (+10)
        if "unusual_movement" in event_tokens:
            score += self.weight_unusual_movement
            factors["unusual_movement"] = self.weight_unusual_movement
            evidence.append("Unusual movement pattern")

        # 6. Unattended / abandoned object (+25)
        if "abandoned_object_detected" in event_tokens:
            score += self.weight_abandoned_object
            factors["abandoned_object"] = self.weight_abandoned_object
            evidence.append("Unattended / abandoned object detected")

        # 7. Loitering in monitored area (+15)
        if "loitering_detected" in event_tokens:
            score += self.weight_loitering
            factors["loitering"] = self.weight_loitering
            evidence.append("Persistent loitering in monitored area")

        # Clamp score between 0 and 100
        clamped_score = min(max(score, 0), 100)
        level = get_risk_level(clamped_score)
        summary = get_summary_label(level)

        return RiskAssessment(
            score=clamped_score,
            level=level,
            factors=factors,
            evidence=evidence,
            summary_label=summary,
        )

    def evaluate_events(self, events: Sequence[SecurityEvent]) -> RiskAssessment:
        """Evaluate a raw sequence of SecurityEvent objects directly."""
        event_types = {e.event_type for e in events}
        # Adapt raw event types into canonical tokens for evaluation
        tokens: set[str] = set()
        for et in event_types:
            if et == EVENT_RESTRICTED_ZONE_ENTRY:
                tokens.add("restricted_zone_entry")
            elif et == EVENT_AFTER_HOURS_ACTIVITY:
                tokens.add("after_hours_activity")
            elif et == EVENT_RESTRICTED_ZONE_DWELL:
                tokens.add("extended_dwell")
            elif et == EVENT_OBJECT_INTERACTION:
                tokens.add("object_interaction")
            elif et == EVENT_UNUSUAL_MOVEMENT or et == "unusual_movement":
                tokens.add("unusual_movement")
            elif et == EVENT_ABANDONED_OBJECT:
                tokens.add("abandoned_object_detected")
            elif et == EVENT_LOITERING_DETECTED:
                tokens.add("loitering_detected")

        dummy_ctx = {"events": list(tokens)}
        return self.evaluate_context(dummy_ctx)
