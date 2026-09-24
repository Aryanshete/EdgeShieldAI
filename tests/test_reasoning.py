"""Unit tests for Llama 3.2 reasoning agent and fallback handling."""

from __future__ import annotations

import json
import pytest

from src.context import SecurityContext
from src.reasoning import (
    LlamaReasoningAgent,
    ReasoningOutput,
    extract_json_from_text,
    generate_deterministic_fallback,
)
from src.risk import RiskAssessment


@pytest.fixture
def sample_context() -> SecurityContext:
    return SecurityContext(
        location="Server Room",
        time="22:17",
        authorized_hours="08:00-18:00",
        track_id=7,
        events=[
            "restricted_zone_entry",
            "after_hours_activity",
            "extended_dwell",
            "object_interaction",
        ],
        event_descriptions=[
            "Person #07 entered restricted zone",
            "Activity occurred outside authorized hours",
            "Person remained inside for 23 seconds",
            "Person approached an unattended object",
        ],
        start_timestamp="22:17:00",
        end_timestamp="22:17:23",
        duration_seconds=23.0,
    )


@pytest.fixture
def sample_assessment() -> RiskAssessment:
    return RiskAssessment(
        score=85,
        level="HIGH",
        factors={"restricted_entry": 40, "after_hours": 20, "extended_dwell": 15, "object_interaction": 20},
        evidence=["Restricted-area entry", "After-hours activity", "Extended dwell", "Object interaction"],
        summary_label="Potentially unauthorized activity detected",
    )


def test_format_prompt(sample_context: SecurityContext, sample_assessment: RiskAssessment) -> None:
    agent = LlamaReasoningAgent()
    prompt_str = agent.format_prompt(sample_context, sample_assessment)
    data = json.loads(prompt_str)

    # Must match PDF Page 18 schema
    assert data["location"] == "Server Room"
    assert data["current_time"] == "22:17"
    assert data["authorized_hours"] == "08:00-18:00"
    assert data["risk_score"] == 85
    assert len(data["events"]) == 4
    assert "Person #07 entered restricted zone" in data["events"]


def test_parse_valid_json(sample_context: SecurityContext, sample_assessment: RiskAssessment) -> None:
    agent = LlamaReasoningAgent()
    raw_llm_json = json.dumps({
        "risk_level": "HIGH",
        "evidence": ["Restricted-area entry", "After-hours activity"],
        "explanation": "Subject was detected inside the server room outside normal working hours.",
        "recommended_action": "Alert on-duty security guard to inspect the server room.",
        "summary": "Potentially unauthorized server room intrusion.",
    })

    result = agent.parse_response(raw_llm_json, sample_context, sample_assessment)
    assert isinstance(result, ReasoningOutput)
    assert result.risk_level == "HIGH"
    assert result.evidence == ["Restricted-area entry", "After-hours activity"]
    assert "security guard" in result.recommended_action
    assert result.is_fallback is False


def test_parse_markdown_fenced_json(sample_context: SecurityContext, sample_assessment: RiskAssessment) -> None:
    agent = LlamaReasoningAgent()
    markdown_wrapped = """Here is my assessment of the event:
```json
{
  "risk_level": "CRITICAL",
  "evidence": ["Restricted-area entry", "Extended dwell"],
  "explanation": "Prolonged dwell time inside server enclosure.",
  "recommended_action": "Immediate dispatch.",
  "summary": "Critical intrusion."
}
```
Please let me know if further review is required."""

    result = agent.parse_response(markdown_wrapped, sample_context, sample_assessment)
    assert result.risk_level == "CRITICAL"
    assert "Immediate dispatch" in result.recommended_action
    assert result.is_fallback is False


def test_parse_invalid_json_triggers_fallback(
    sample_context: SecurityContext, sample_assessment: RiskAssessment
) -> None:
    agent = LlamaReasoningAgent()
    malformed = "I am an LLM and I think this is bad { unclosed json... "

    result = agent.parse_response(malformed, sample_context, sample_assessment)
    assert isinstance(result, ReasoningOutput)
    assert result.risk_level == "HIGH"
    assert result.is_fallback is True
    assert "Server Room" in result.explanation


def test_missing_fields_repaired_from_fallback(
    sample_context: SecurityContext, sample_assessment: RiskAssessment
) -> None:
    agent = LlamaReasoningAgent()
    # Missing 'explanation' and 'recommended_action'
    partial_json = json.dumps({
        "risk_level": "HIGH",
        "evidence": ["Restricted-area entry"],
    })

    result = agent.parse_response(partial_json, sample_context, sample_assessment)
    assert result.risk_level == "HIGH"
    assert result.evidence == ["Restricted-area entry"]
    # Repaired from fallback
    assert len(result.explanation) > 0
    assert len(result.recommended_action) > 0


def test_deterministic_fallback_generation(
    sample_context: SecurityContext, sample_assessment: RiskAssessment
) -> None:
    fallback = generate_deterministic_fallback(sample_context, sample_assessment)
    assert fallback.risk_level == "HIGH"
    assert fallback.is_fallback is True
    assert "potentially unauthorized" in fallback.explanation.lower()
    assert "Server Room" in fallback.explanation
    assert "08:00-18:00" in fallback.explanation
    assert "alert security personnel" in fallback.recommended_action.lower()


def test_backend_exception_handled_gracefully(
    sample_context: SecurityContext, sample_assessment: RiskAssessment
) -> None:
    def failing_backend(system_prompt: str, user_prompt: str) -> str:
        raise ConnectionError("LLM API endpoint unavailable")

    agent = LlamaReasoningAgent(backend_callable=failing_backend)
    result = agent.analyze(sample_context, sample_assessment)

    # Must NOT crash! Must return deterministic fallback
    assert result.is_fallback is True
    assert result.risk_level == "HIGH"


def test_default_agent_without_backend_uses_fallback(
    sample_context: SecurityContext, sample_assessment: RiskAssessment
) -> None:
    agent = LlamaReasoningAgent()
    result = agent.analyze(sample_context, sample_assessment)

    assert result.is_fallback is True
    assert result.risk_level == "HIGH"
    assert result.evidence == sample_assessment.evidence
