"""Unit tests for Phase 14: Privacy-Aware Edge Architecture."""

import pytest
from src.events import SecurityEvent
from src.privacy import PrivacyAuditor, PrivacyPolicy


def test_privacy_policy_defaults():
    """Verify default privacy policy settings enforce anonymous edge processing."""
    policy = PrivacyPolicy()
    assert policy.edge_local_processing is True
    assert policy.biometrics_disabled is True
    assert policy.anonymous_tracking is True
    assert policy.metadata_only_transmission is True


def test_anonymous_label_format():
    """Verify track ID formatting uses Person #07 style representation."""
    auditor = PrivacyAuditor()
    assert auditor.format_anonymous_label(7) == "Person #07"
    assert auditor.format_anonymous_label(1) == "Person #01"
    assert auditor.format_anonymous_label(23) == "Person #23"


def test_audit_event_compliant():
    """Verify a standard security event passes privacy audit."""
    auditor = PrivacyAuditor()
    event = SecurityEvent(
        event_type="restricted_zone_entry",
        track_id=4,
        zone="Server Room",
        timestamp="22:17:05",
        timestamp_seconds=5.0,
        metadata={"zone_id": "server_room_perimeter"},
    )
    result = auditor.audit_event(event)
    assert result["compliant"] is True
    assert len(result["violations"]) == 0
    assert result["anonymous_label"] == "Person #04"


def test_audit_event_detects_prohibited_biometrics():
    """Verify that prohibited fields (face_id, name) trigger compliance violations."""
    auditor = PrivacyAuditor()
    violating_event = {
        "event_type": "person_detected",
        "track_id": 9,
        "metadata": {
            "name": "John Doe",
            "facial_features": [0.12, 0.45],
        },
    }
    result = auditor.audit_event(violating_event)
    assert result["compliant"] is False
    assert len(result["violations"]) == 2
    assert any("name" in v for v in result["violations"])
    assert any("facial_features" in v for v in result["violations"])


def test_sanitize_payload_removes_pixel_boxes_for_cloud():
    """Verify sanitization strips detailed bounding boxes while retaining semantic metadata."""
    auditor = PrivacyAuditor()
    payload = {
        "location": "Vault",
        "track_id": 5,
        "metadata": {
            "bbox": [100, 100, 200, 300],
            "initial_bbox": [100, 100, 200, 300],
            "dwell_seconds": 12.0,
        },
    }
    sanitized = auditor.sanitize_payload_for_reasoning(payload)
    assert "bbox" not in sanitized["metadata"]
    assert "initial_bbox" not in sanitized["metadata"]
    assert sanitized["metadata"]["dwell_seconds"] == 12.0


def test_compliance_report():
    """Verify enterprise compliance report contents."""
    auditor = PrivacyAuditor()
    report = auditor.get_compliance_report()
    assert report["status"] == "COMPLIANT"
    assert "Biometric Safeguard" in report["principles"]
    assert "Anonymous Representation" in report["principles"]
