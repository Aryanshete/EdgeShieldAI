"""Llama 3.2 reasoning agent and contextual interpretation for EdgeShield AI."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.context import SecurityContext
from src.risk import RiskAssessment, get_risk_level

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are EdgeShield AI, an intelligent contextual edge security reasoning assistant.
You analyze structured chronological security events produced by a computer vision pipeline.

OPERATIONAL DIRECTIVES:
1. Ground every claim strictly in the supplied event sequence. Never invent, extrapolate, or imagine facts.
2. Never claim proven criminal intent, guilt, or premeditation. Use objective, probabilistic risk assessment language (e.g., "potentially unauthorized activity", "unverified presence").
3. Explain the security significance of the evidence clearly and concisely.
4. Recommend appropriate, pragmatic operator actions (e.g., "Alert security personnel to verify subject authorization", "Review associated camera footage").
5. Return ONLY a valid JSON object matching the following schema:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "evidence": ["<evidence 1>", "<evidence 2>"],
  "explanation": "<contextual explanation grounded only in supplied events>",
  "recommended_action": "<actionable security operator response>",
  "summary": "<concise one-line assessment summary>"
}"""


@dataclass(frozen=True)
class ReasoningOutput:
    """Structured security reasoning output produced by EdgeShield's reasoning agent."""

    risk_level: str
    evidence: list[str]
    explanation: str
    recommended_action: str
    summary: str
    is_fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a structured JSON-ready dictionary representation."""
        return {
            "risk_level": self.risk_level,
            "evidence": list(self.evidence),
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "summary": self.summary,
            "is_fallback": self.is_fallback,
        }


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from text, handling markdown fences or leading/trailing words."""
    cleaned = text.strip()
    # Match markdown json codeblock: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        # Match outermost curly braces
        match_braces = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match_braces:
            cleaned = match_braces.group(1)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def generate_deterministic_fallback(
    context: SecurityContext | dict[str, Any],
    assessment: RiskAssessment | dict[str, Any],
) -> ReasoningOutput:
    """Generate a grounded, specification-compliant reasoning explanation without calling an external LLM.

    Implements Phase 16 Failure Handling: ensures complete operational continuity
    if Llama 3.2 is offline, unconfigured, or returns malformed output.
    """
    ctx_dict = context.as_dict() if isinstance(context, SecurityContext) else dict(context)
    risk_dict = assessment.as_dict() if isinstance(assessment, RiskAssessment) else dict(assessment)

    location = ctx_dict.get("location", "Monitored Area")
    authorized_hours = ctx_dict.get("authorized_hours", "08:00-18:00")
    level = risk_dict.get("level", get_risk_level(risk_dict.get("score", 0)))
    evidence = list(risk_dict.get("evidence", []))
    if not evidence and "events" in ctx_dict:
        # Fallback evidence from events
        evidence = [ev.replace("_", " ").title() for ev in ctx_dict["events"] if ev != "person_detected"]

    if level in ("HIGH", "CRITICAL"):
        explanation = (
            f"The sequence indicates potentially unauthorized activity in {location} "
            f"outside normal operating hours ({authorized_hours})."
        )
        recommended_action = "Alert security personnel and review associated camera footage."
        summary = f"Potentially unauthorized activity detected in the restricted {location.lower()}."
    elif level == "MEDIUM":
        explanation = (
            f"Activity was detected in {location}. While restricted entry occurred, "
            f"the event is within or near authorized parameters."
        )
        recommended_action = "Log observation for routine operator review and monitor zone continuity."
        summary = f"Elevated activity noted in {location}."
    else:
        explanation = f"Routine visual observation in {location} within normal operating parameters."
        recommended_action = "No immediate operational response required."
        summary = f"Routine activity in {location}."

    return ReasoningOutput(
        risk_level=level,
        evidence=evidence,
        explanation=explanation,
        recommended_action=recommended_action,
        summary=summary,
        is_fallback=True,
    )


class LlamaReasoningAgent:
    """Edge reasoning agent that interprets structured event contexts with Llama 3.2."""

    def __init__(
        self,
        backend_callable: Callable[[str, str], str] | None = None,
        model_name: str = "Llama-3.2",
    ) -> None:
        """Initialize the reasoning agent.

        Args:
            backend_callable: An optional function `f(system_prompt, user_prompt) -> response_str`.
                              If None, the agent uses the deterministic fallback reasoner.
            model_name: Identifier of the reasoning model.
        """
        self.backend_callable = backend_callable
        self.model_name = model_name

    def format_prompt(
        self,
        context: SecurityContext | dict[str, Any],
        assessment: RiskAssessment | dict[str, Any],
    ) -> str:
        """Format the input prompt matching the specification schema (PDF Page 18)."""
        ctx_dict = context.as_dict() if isinstance(context, SecurityContext) else dict(context)
        risk_dict = assessment.as_dict() if isinstance(assessment, RiskAssessment) else dict(assessment)

        # Prioritize factual event descriptions over raw tokens
        events_payload = ctx_dict.get("event_descriptions") or ctx_dict.get("events", [])

        input_payload = {
            "location": ctx_dict.get("location", "Monitored Area"),
            "current_time": ctx_dict.get("time", "00:00"),
            "authorized_hours": ctx_dict.get("authorized_hours", "08:00-18:00"),
            "risk_score": risk_dict.get("score", 0),
            "events": events_payload,
        }
        return json.dumps(input_payload, indent=2)

    def parse_response(
        self,
        raw_text: str,
        context: SecurityContext | dict[str, Any],
        assessment: RiskAssessment | dict[str, Any],
    ) -> ReasoningOutput:
        """Parse raw LLM response, validate schema fields, and repair missing elements."""
        parsed_data = extract_json_from_text(raw_text)
        if not parsed_data:
            # Malformed output -> fallback
            return generate_deterministic_fallback(context, assessment)

        # Ensure all required fields exist
        fallback = generate_deterministic_fallback(context, assessment)
        risk_level = str(parsed_data.get("risk_level", fallback.risk_level)).upper()
        if risk_level not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            risk_level = fallback.risk_level

        evidence = parsed_data.get("evidence")
        if not isinstance(evidence, list):
            evidence = fallback.evidence
        else:
            evidence = [str(e) for e in evidence]

        explanation = str(parsed_data.get("explanation", fallback.explanation))
        recommended_action = str(parsed_data.get("recommended_action", fallback.recommended_action))
        summary = str(parsed_data.get("summary", fallback.summary))

        return ReasoningOutput(
            risk_level=risk_level,
            evidence=evidence,
            explanation=explanation,
            recommended_action=recommended_action,
            summary=summary,
            is_fallback=False,
        )

    def analyze(
        self,
        context: SecurityContext | dict[str, Any],
        assessment: RiskAssessment | dict[str, Any],
    ) -> ReasoningOutput:
        """Synthesize contextual explanation and operational recommendations."""
        if self.backend_callable is None:
            # Deterministic fallback when no LLM runtime is attached
            return generate_deterministic_fallback(context, assessment)

        user_prompt = self.format_prompt(context, assessment)
        try:
            raw_response = self.backend_callable(SYSTEM_PROMPT, user_prompt)
            return self.parse_response(raw_response, context, assessment)
        except Exception:
            # In case of API failure, timeout, or model crash
            return generate_deterministic_fallback(context, assessment)

    # Alias for convenience
    reason = analyze
