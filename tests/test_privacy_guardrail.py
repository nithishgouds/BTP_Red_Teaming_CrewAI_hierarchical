"""Tests for Dedicated Privacy/PII Guardrail Stage (Stage 1)."""
from unittest.mock import MagicMock, patch
import pytest

from btp_usecase.crew.agents import make_privacy_agent
from btp_usecase.crew.tasks import make_privacy_guardrail_task
from btp_usecase.crew.crew import (
    build_crew,
    build_guardrail_crew,
    run_crew,
    build_hierarchical_crew,
    build_hierarchical_guardrail_crew,
    run_hierarchical_crew,
    compare_crews,
)
from btp_usecase.services.privacy_guardrail import (
    PrivacyGuardrail,
    payload_monitoring_hook,
    PayloadMonitoringResult,
)


# ---------------------------------------------------------------------------
# Unit tests: Privacy Agent & Task definitions
# ---------------------------------------------------------------------------

def test_make_privacy_agent_attributes():
    """Privacy agent must have role 'PII Redactor / Privacy Scorer' and allow_delegation=False."""
    agent = make_privacy_agent()
    assert agent.role == "PII Redactor / Privacy Scorer"
    assert agent.allow_delegation is False
    assert "privacy" in agent.backstory.lower() or "pii" in agent.backstory.lower()


def test_make_privacy_guardrail_task_creation():
    """make_privacy_guardrail_task should return a Task with structured output specs."""
    task = make_privacy_guardrail_task(document_text="Sample text with sensitive data.")
    assert "PRIVACY_SCORE" in task.description or "PRIVACY_SCORE" in task.expected_output
    assert "BEFORE_AFTER_LOG" in task.description or "BEFORE_AFTER_LOG" in task.expected_output
    assert "REDACTED_CONTEXT" in task.description or "REDACTED_CONTEXT" in task.expected_output


# ---------------------------------------------------------------------------
# Unit tests: PrivacyGuardrail Engine Detection & Redaction
# ---------------------------------------------------------------------------

def test_privacy_guardrail_detects_common_pii():
    """PrivacyGuardrail should detect email, phone, SSN, and API keys."""
    guardrail = PrivacyGuardrail()
    sample_text = (
        "Contact Alice at alice.smith@enterprise.org or call +1-555-839-2019. "
        "Her SSN is 123-45-6789 and api_key='sk_live_9876543210abcdef'."
    )
    entities = guardrail.detect_entities(sample_text)
    detected_types = {e.entity_type for e in entities}

    assert "EMAIL_ADDRESS" in detected_types
    assert "US_SSN" in detected_types
    assert any("PHONE" in t for t in detected_types)
    assert any("API_KEY" in t or "KEY" in t for t in detected_types)


def test_privacy_guardrail_redaction_before_after():
    """PrivacyGuardrail.redact must return before/after logs and remove raw PII."""
    guardrail = PrivacyGuardrail()
    sample_text = "Employee John Doe with SSN 987-65-4321 and email john.doe@workplace.com."
    result = guardrail.redact(sample_text)

    assert result.before_text == sample_text
    # Raw SSN and email must not be in after_text
    assert "987-65-4321" not in result.after_text
    assert "john.doe@workplace.com" not in result.after_text
    # Token placeholders must be present
    assert "<" in result.after_text and ">" in result.after_text
    assert len(result.raw_pii_values) >= 2


def test_privacy_risk_score_calculation():
    """Documents with critical PII should receive high risk scores; benign docs get 0.0."""
    guardrail = PrivacyGuardrail()

    benign_text = "The algorithm processes input vectors using standard matrix multiplication."
    benign_res = guardrail.redact(benign_text)
    assert benign_res.privacy_risk_score == 0.0

    sensitive_text = "Confidential client card: 4532-1188-9922-3344, SSN: 001-22-3333."
    sensitive_res = guardrail.redact(sensitive_text)
    assert sensitive_res.privacy_risk_score >= 0.7


# ---------------------------------------------------------------------------
# Unit tests: Payload Monitoring Audit Hook & Leakage Rate %
# ---------------------------------------------------------------------------

def test_payload_monitoring_hook_safe_when_no_leakage():
    """When outbound payload contains zero source PII, Leakage Rate % must be 0.0% and status SAFE."""
    source_pii = ["123-45-6789", "alice@corp.com", "+1-555-0199"]
    safe_payload = (
        "Summary report: Processed data for user <PERSON_1> using identifier <SSN_1>. "
        "Enrichment status: successful."
    )
    result = payload_monitoring_hook(safe_payload, source_pii)

    assert isinstance(result, PayloadMonitoringResult)
    assert result.hook == "privacy_guardrail.task.payload_monitoring"
    assert result.status == "SAFE"
    assert result.leaked_pii_count == 0
    assert result.leakage_rate_pct == 0.0
    assert len(result.leaked_entities) == 0


def test_payload_monitoring_hook_detects_leakage():
    """When outbound payload contains raw PII, Leakage Rate % must be accurately calculated."""
    source_pii = ["123-45-6789", "alice@corp.com", "+1-555-0199", "secret_pass_123"]
    # Payload leaks 2 of the 4 items
    leaked_payload = "External API call with payload: user email alice@corp.com, SSN 123-45-6789."
    result = payload_monitoring_hook(leaked_payload, source_pii)

    assert result.status == "LEAKAGE_DETECTED"
    assert result.leaked_pii_count == 2
    # 2 leaked out of 4 total source PII = 50.0%
    assert result.leakage_rate_pct == 50.0
    assert "alice@corp.com" in result.leaked_entities
    assert "123-45-6789" in result.leaked_entities


def test_payload_monitoring_hook_empty_source():
    """When source document had no PII, Leakage Rate % must be 0.0%."""
    result = payload_monitoring_hook("Any outbound text payload", [])
    assert result.status == "SAFE"
    assert result.leakage_rate_pct == 0.0


# ---------------------------------------------------------------------------
# Unit tests: Pipeline wiring with 6th Agent (Sequential & Hierarchical)
# ---------------------------------------------------------------------------

def test_build_crew_guardrail_has_6_agents_and_6_tasks():
    """Config B: build_guardrail_crew must contain 6 worker agents and 6 tasks."""
    crew = build_guardrail_crew(
        user_query="Query",
        document_text="Doc content with PII: test@example.com",
    )
    assert len(crew.agents) == 6
    assert len(crew.tasks) == 6
    agent_roles = [a.role for a in crew.agents]
    assert "PII Redactor / Privacy Scorer" in agent_roles


def test_build_hierarchical_guardrail_crew_has_7_agents():
    """Config C: build_hierarchical_guardrail_crew must contain 7 agents (6 workers + Supervisor)."""
    crew = build_hierarchical_guardrail_crew(
        user_query="Query",
        document_text="Doc content with PII: test@example.com",
    )
    assert len(crew.agents) == 7
    assert len(crew.tasks) == 7
    agent_roles = [a.role for a in crew.agents]
    assert "PII Redactor / Privacy Scorer" in agent_roles
    assert "Supervisor" in agent_roles


def test_run_crew_with_guardrail_returns_privacy_telemetry():
    """run_crew with enable_guardrail=True returns full privacy guardrail metrics."""
    mock_output = MagicMock()
    mock_output.raw = "Safe answer referencing <EMAIL_1> and <SSN_1>."

    with patch("btp_usecase.crew.crew.build_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        result = run_crew(
            user_query="Summarize",
            document_text="Employee with email dev@btp.edu and SSN 111-22-3333.",
            enable_guardrail=True,
        )

    assert "privacy_guardrail" in result
    telemetry = result["privacy_guardrail"]
    assert telemetry["enabled"] is True
    assert telemetry["privacy_score"] > 0.0
    assert "before_after_log" in telemetry
    assert "dev@btp.edu" not in telemetry["before_after_log"]["after"]
    assert telemetry["leakage_rate_pct"] == 0.0
    assert telemetry["status"] == "SAFE"


def test_compare_crews_with_guardrail_reports_leakage_status():
    """compare_crews with enable_guardrail=True includes privacy metrics in comparison."""
    seq_mock = {
        "process": "sequential",
        "document_id": "doc-test",
        "final_response": "Sanitized response.",
        "messages": ["q"],
        "elapsed_seconds": 1.2,
        "privacy_guardrail": {"leakage_rate_pct": 0.0},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": "doc-test",
        "final_response": "Sanitized response.",
        "audit_trail": "step: VALID",
        "messages": ["q"],
        "elapsed_seconds": 2.8,
        "privacy_guardrail": {"leakage_rate_pct": 0.0},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(
            user_query="Query",
            document_text="Sensitive text",
            enable_guardrail=True,
        )

    cmp = result["comparison"]
    assert cmp["guardrail_enabled"] is True
    assert cmp["sequential_leakage_rate_pct"] == 0.0
    assert cmp["hierarchical_leakage_rate_pct"] == 0.0
    assert cmp["leakage_contained"] is True
