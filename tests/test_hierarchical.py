"""Tests for the hierarchical CrewAI crew pipeline (Section 6 Experiment)."""
from unittest.mock import MagicMock, patch

import pytest

from btp_usecase.crew.crew import (
    build_hierarchical_crew,
    run_hierarchical_crew,
    compare_crews,
    _response_digest,
)
from btp_usecase.crew.agents import make_supervisor_agent
from btp_usecase.crew.tasks import make_supervisor_audit_task


# ---------------------------------------------------------------------------
# Unit tests: Supervisor agent
# ---------------------------------------------------------------------------

def test_supervisor_agent_has_zero_trust_config():
    """The Supervisor agent must have allow_delegation=False (zero-trust)."""
    agent = make_supervisor_agent()
    assert agent.allow_delegation is False, (
        "Supervisor must enforce zero-trust: allow_delegation must be False"
    )


def test_supervisor_agent_role():
    """The Supervisor agent must carry the 'Supervisor' role."""
    agent = make_supervisor_agent()
    assert agent.role == "Supervisor"


def test_supervisor_agent_verbose():
    """The Supervisor agent must be verbose so audit steps appear in logs."""
    agent = make_supervisor_agent()
    assert agent.verbose is True


# ---------------------------------------------------------------------------
# Unit tests: Supervisor audit task
# ---------------------------------------------------------------------------

def test_supervisor_audit_task_is_created():
    """make_supervisor_audit_task should return a Task with a non-empty description."""
    from crewai import Task
    task = make_supervisor_audit_task()
    assert isinstance(task, Task)
    assert len(task.description) > 50


def test_supervisor_audit_task_forbids_delegation():
    """The audit task description must explicitly forbid delegation."""
    task = make_supervisor_audit_task()
    assert "NOT delegate" in task.description or "not delegate" in task.description.lower()


# ---------------------------------------------------------------------------
# Unit tests: build_hierarchical_crew
# ---------------------------------------------------------------------------

def test_build_hierarchical_crew_returns_crew_instance():
    """build_hierarchical_crew should return a Crew with 6 agents and 6 tasks."""
    from crewai import Crew
    crew = build_hierarchical_crew(
        user_query="Test query",
        document_id="doc-test",
        document_text="Sample document text for testing.",
    )
    assert isinstance(crew, Crew)
    assert len(crew.agents) == 6, f"Expected 6 agents, got {len(crew.agents)}"
    assert len(crew.tasks) == 6, f"Expected 6 tasks, got {len(crew.tasks)}"


def test_build_hierarchical_crew_uses_hierarchical_process():
    """The hierarchical crew must use Process.hierarchical."""
    from crewai import Process
    crew = build_hierarchical_crew(
        user_query="Test query",
        document_text="Some document text.",
    )
    assert crew.process == Process.hierarchical


def test_build_hierarchical_crew_has_manager_llm():
    """Process.hierarchical requires manager_llm to be set on the Crew."""
    crew = build_hierarchical_crew(
        user_query="Test query",
        document_text="Some document text.",
    )
    assert crew.manager_llm is not None, "manager_llm must be set for hierarchical mode"


def test_hierarchical_crew_all_agents_no_delegation():
    """Every agent in the hierarchical crew must have allow_delegation=False."""
    crew = build_hierarchical_crew(
        user_query="Test query",
        document_text="Some document text.",
    )
    for agent in crew.agents:
        assert agent.allow_delegation is False, (
            f"Agent '{agent.role}' must have allow_delegation=False (zero-trust)"
        )


# ---------------------------------------------------------------------------
# Unit tests: run_hierarchical_crew
# ---------------------------------------------------------------------------

def test_run_hierarchical_crew_returns_expected_keys():
    """
    run_hierarchical_crew should return a dict with all required keys including
    the new 'process', 'audit_trail', and 'elapsed_seconds' fields.
    """
    mock_output = MagicMock()
    mock_output.raw = (
        "AUDIT_TRAIL:\n"
        '{\"step\": \"normalize_query\", \"status\": \"VALID\"}\n'
        "FINAL_ANSWER:\n"
        "This is the synthesised answer."
    )

    with patch("btp_usecase.crew.crew.build_hierarchical_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        result = run_hierarchical_crew(
            user_query="Summarise the document",
            document_id="doc-hier-001",
            document_text="Document text for hierarchical test.",
        )

    # Required base keys
    assert result["document_id"] == "doc-hier-001"
    assert result["final_response"] == "This is the synthesised answer."
    assert "messages" in result
    assert "intermediate_results" in result
    # Hierarchical-specific keys
    assert result["process"] == "hierarchical"
    assert "audit_trail" in result
    assert "elapsed_seconds" in result


def test_run_hierarchical_crew_parses_audit_trail():
    """run_hierarchical_crew should split AUDIT_TRAIL / FINAL_ANSWER correctly."""
    mock_output = MagicMock()
    mock_output.raw = (
        "AUDIT_TRAIL:\n"
        "step1: VALID\n"
        "FINAL_ANSWER:\n"
        "The answer is 42."
    )

    with patch("btp_usecase.crew.crew.build_hierarchical_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        result = run_hierarchical_crew(user_query="What is the answer?",
                                        document_text="The answer is 42.")

    assert result["final_response"] == "The answer is 42."
    assert "step1: VALID" in result["audit_trail"]


def test_run_hierarchical_crew_fallback_response():
    """If the crew produces empty output, fallback text should be returned."""
    mock_output = MagicMock()
    mock_output.raw = ""

    with patch("btp_usecase.crew.crew.build_hierarchical_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        result = run_hierarchical_crew(user_query="Empty test", document_text="text")

    assert result["final_response"] == "I could not generate a response."


# ---------------------------------------------------------------------------
# Unit tests: compare_crews
# ---------------------------------------------------------------------------

def test_compare_crews_returns_expected_structure():
    """
    compare_crews must return a dict with 'sequential', 'hierarchical', and
    'comparison' keys, where 'comparison' contains all metric fields.
    """
    seq_mock = {
        "process": "sequential",
        "document_id": "doc-cmp",
        "final_response": "Sequential answer.",
        "messages": ["Query"],
        "elapsed_seconds": 1.0,
        "intermediate_results": {},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": "doc-cmp",
        "final_response": "Hierarchical answer.",
        "audit_trail": "step1: VALID\nstep2: VALID",
        "messages": ["Query"],
        "elapsed_seconds": 2.5,
        "intermediate_results": {},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(
            user_query="Compare query",
            document_id="doc-cmp",
            document_text="Compare document text.",
        )

    assert "sequential" in result
    assert "hierarchical" in result
    assert "comparison" in result

    cmp = result["comparison"]
    assert "sequential_digest" in cmp
    assert "hierarchical_digest" in cmp
    assert "responses_match" in cmp
    assert "reproducibility" in cmp
    assert "auditability" in cmp
    assert "elapsed_delta_seconds" in cmp
    assert "timestamp_utc" in cmp


def test_compare_crews_reproducibility_pass_when_responses_match():
    """
    compare_crews should report PASS for reproducibility when both crews
    return the same final response (identical digests).
    """
    common_response = "Both crews agreed on this answer."
    seq_mock = {
        "process": "sequential",
        "document_id": None,
        "final_response": common_response,
        "messages": ["q"],
        "elapsed_seconds": 1.0,
        "intermediate_results": {},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": None,
        "final_response": common_response,
        "audit_trail": "audit data",
        "messages": ["q"],
        "elapsed_seconds": 2.0,
        "intermediate_results": {},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(user_query="q", document_text="doc")

    assert result["comparison"]["reproducibility"] == "PASS"
    assert result["comparison"]["responses_match"] is True


def test_compare_crews_reproducibility_variance_when_responses_differ():
    """
    compare_crews should report VARIANCE_DETECTED when the two crews
    return different final responses.
    """
    seq_mock = {
        "process": "sequential",
        "document_id": None,
        "final_response": "Answer A.",
        "messages": ["q"],
        "elapsed_seconds": 1.0,
        "intermediate_results": {},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": None,
        "final_response": "Answer B.",
        "audit_trail": "steps here",
        "messages": ["q"],
        "elapsed_seconds": 2.0,
        "intermediate_results": {},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(user_query="q", document_text="doc")

    assert result["comparison"]["reproducibility"] == "VARIANCE_DETECTED"
    assert result["comparison"]["responses_match"] is False


def test_compare_crews_auditability_pass_when_audit_trail_present():
    """Auditability should be PASS when the hierarchical crew has an audit trail."""
    seq_mock = {
        "process": "sequential",
        "document_id": None,
        "final_response": "Answer.",
        "messages": ["q"],
        "elapsed_seconds": 1.0,
        "intermediate_results": {},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": None,
        "final_response": "Answer.",
        "audit_trail": "Non-empty audit trail content.",
        "messages": ["q"],
        "elapsed_seconds": 2.0,
        "intermediate_results": {},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(user_query="q", document_text="doc")

    assert result["comparison"]["auditability"] == "PASS"


def test_compare_crews_auditability_fail_when_no_audit_trail():
    """Auditability should be FAIL when the hierarchical crew has no audit trail."""
    seq_mock = {
        "process": "sequential",
        "document_id": None,
        "final_response": "Answer.",
        "messages": ["q"],
        "elapsed_seconds": 1.0,
        "intermediate_results": {},
    }
    hier_mock = {
        "process": "hierarchical",
        "document_id": None,
        "final_response": "Answer.",
        "audit_trail": "",  # empty audit trail
        "messages": ["q"],
        "elapsed_seconds": 2.0,
        "intermediate_results": {},
    }

    with patch("btp_usecase.crew.crew.run_crew", return_value=seq_mock), \
         patch("btp_usecase.crew.crew.run_hierarchical_crew", return_value=hier_mock):
        result = compare_crews(user_query="q", document_text="doc")

    assert result["comparison"]["auditability"] == "FAIL"


# ---------------------------------------------------------------------------
# Unit tests: _response_digest helper
# ---------------------------------------------------------------------------

def test_response_digest_is_deterministic():
    """_response_digest must return the same hex string for the same input."""
    text = "Deterministic response text."
    assert _response_digest(text) == _response_digest(text)


def test_response_digest_differs_for_different_inputs():
    """_response_digest must produce different hashes for different inputs."""
    assert _response_digest("text A") != _response_digest("text B")


def test_response_digest_length():
    """_response_digest must return a 16-character hex string."""
    digest = _response_digest("any input")
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# Integration-level: AgentService hierarchical methods
# ---------------------------------------------------------------------------

def test_agent_service_run_hierarchical_agent(tmp_path):
    """AgentService.run_hierarchical_agent should delegate to run_hierarchical_crew."""
    from btp_usecase.services.agent_service import AgentService
    from btp_usecase.storage.local import LocalDocumentStorage

    storage = LocalDocumentStorage(base_path=tmp_path / "documents")
    storage.save("doc-h1", b"Hierarchical test document content.")

    mock_result = {
        "process": "hierarchical",
        "document_id": "doc-h1",
        "final_response": "Hierarchical summary.",
        "audit_trail": "step1: VALID\nstep2: VALID",
        "messages": ["Summarise"],
        "elapsed_seconds": 3.2,
        "intermediate_results": {},
    }

    with patch("btp_usecase.services.agent_service.run_hierarchical_crew",
                return_value=mock_result):
        service = AgentService(storage=storage)
        result = service.run_hierarchical_agent(
            user_query="Summarise",
            document_id="doc-h1",
        )

    assert result["process"] == "hierarchical"
    assert result["document_id"] == "doc-h1"
    assert result["final_response"] == "Hierarchical summary."
    assert "audit_trail" in result


def test_agent_service_compare_agent_runs(tmp_path):
    """AgentService.compare_agent_runs should delegate to compare_crews."""
    from btp_usecase.services.agent_service import AgentService
    from btp_usecase.storage.local import LocalDocumentStorage

    storage = LocalDocumentStorage(base_path=tmp_path / "documents")

    mock_comparison = {
        "sequential": {"final_response": "A"},
        "hierarchical": {"final_response": "B", "audit_trail": "trail"},
        "comparison": {
            "sequential_digest": "abc123",
            "hierarchical_digest": "def456",
            "responses_match": False,
            "reproducibility": "VARIANCE_DETECTED",
            "auditability": "PASS",
            "elapsed_delta_seconds": 1.5,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
        },
    }

    with patch("btp_usecase.services.agent_service.compare_crews",
                return_value=mock_comparison):
        service = AgentService(storage=storage)
        result = service.compare_agent_runs(
            user_query="Compare",
            document_text="Some document.",
        )

    assert "comparison" in result
    assert "sequential" in result
    assert "hierarchical" in result
    assert result["comparison"]["reproducibility"] == "VARIANCE_DETECTED"
    assert result["comparison"]["auditability"] == "PASS"
