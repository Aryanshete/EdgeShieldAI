"""Unit tests for failure handling and operational continuity."""

import pytest
from src.context import SecurityContext
from src.reasoning import (
    LlamaReasoningAgent,
    ReasoningOutput,
    extract_json_from_text,
    generate_deterministic_fallback,
)
from src.risk import RiskAssessment, RiskEngine
from src.runtime import RuntimeStatus, get_runtime_status


def test_cpu_fallback_status_is_truthful():
    """Verify runtime status truthfully reports CPU mode without pretending AMD is active."""
    status = get_runtime_status()
    assert isinstance(status, RuntimeStatus)
    # On this development machine without AMD ROCm active, it must not fabricate AMD acceleration
    if status.device == "cpu":
        assert status.backend.lower() == "cpu"
        assert status.amd_rocm_active is False


def test_llama_unavailable_deterministic_fallback():
    """Verify system falls back deterministically when Llama is unavailable."""
    ctx = SecurityContext(
        location="Server Room",
        time="22:17",
        authorized_hours="08:00-18:00",
        track_id=1,
        events=["restricted_zone_entry", "after_hours_activity", "extended_dwell"],
        event_descriptions=["Person #01 entered restricted zone Server Room"],
        start_timestamp="22:17:00",
        end_timestamp="22:17:15",
        duration_seconds=15.0,
    )
    assessment = RiskEngine().evaluate_context(ctx)

    # LlamaReasoningAgent initialized without backend callable -> simulates LLM offline
    agent = LlamaReasoningAgent(backend_callable=None)
    output = agent.reason(ctx, assessment)

    assert output.is_fallback is True
    assert output.risk_level == assessment.level
    assert "potentially unauthorized" in output.explanation.lower() or "server room" in output.explanation.lower()
    assert output.recommended_action != ""


def test_llama_malformed_json_fallback():
    """Verify agent handles malformed LLM response without crashing."""
    ctx = SecurityContext(
        location="Server Room",
        time="22:17",
        authorized_hours="08:00-18:00",
        track_id=1,
        events=["restricted_zone_entry"],
        event_descriptions=["Person entered restricted zone"],
        start_timestamp="22:17:00",
        end_timestamp="22:17:05",
        duration_seconds=5.0,
    )
    assessment = RiskAssessment(
        score=65,
        level="HIGH",
        factors={"restricted_entry": 40},
        evidence=["Restricted-area entry"],
        summary_label="Potentially unauthorized activity detected",
    )

    # Mock backend returning broken or non-JSON text
    def broken_llm_backend(system_prompt: str, user_prompt: str) -> str:
        return "I am an LLM but I forgot to return JSON! There is an intruder in the room."

    agent = LlamaReasoningAgent(backend_callable=broken_llm_backend)
    output = agent.reason(ctx, assessment)

    # Must fall back gracefully rather than throwing JSONDecodeError
    assert output.is_fallback is True
    assert output.risk_level == "HIGH"
    assert len(output.evidence) > 0


def test_llama_missing_keys_fallback():
    """Verify agent handles JSON with missing schema fields without crashing."""
    ctx = SecurityContext(
        location="Vault",
        time="23:00",
        authorized_hours="08:00-18:00",
        track_id=2,
        events=["restricted_zone_entry"],
        event_descriptions=["Entry detected"],
        start_timestamp="23:00:00",
        end_timestamp="23:00:02",
        duration_seconds=2.0,
    )
    assessment = RiskAssessment(
        score=40,
        level="MEDIUM",
        factors={"restricted_entry": 40},
        evidence=["Restricted-area entry"],
        summary_label="Elevated security notice",
    )

    def incomplete_llm_backend(system_prompt: str, user_prompt: str) -> str:
        # Missing 'recommended_action' and 'summary'
        return '{"risk_level": "MEDIUM", "evidence": ["entry"], "explanation": "Some explanation"}'

    agent = LlamaReasoningAgent(backend_callable=incomplete_llm_backend)
    output = agent.reason(ctx, assessment)

    # Either completed with defaults or fallback without crashing
    assert isinstance(output, ReasoningOutput)
    assert output.risk_level == "MEDIUM"
    assert output.recommended_action != ""


def test_empty_events_risk_evaluation():
    """Verify risk engine and fallback reasoner handle zero events safely."""
    engine = RiskEngine()
    empty_ctx = {
        "location": "Lobby",
        "events": [],
    }
    assessment = engine.evaluate_context(empty_ctx)
    assert assessment.score == 0
    assert assessment.level == "LOW"

    fallback = generate_deterministic_fallback(empty_ctx, assessment)
    assert fallback.risk_level == "LOW"
    assert fallback.is_fallback is True
