"""Tests for the CrewAI crew pipeline (replaces the former LangGraph graph tests)."""
from unittest.mock import MagicMock, patch

from btp_usecase.crew.crew import build_crew, run_crew


def test_build_crew_returns_crew_instance():
    """build_crew should return a CrewAI Crew with 5 agents and 5 tasks."""
    from crewai import Crew

    crew = build_crew(
        user_query="Summarize the uploaded document",
        document_id="doc-123",
        document_text="This is a test document.",
    )
    assert isinstance(crew, Crew)
    assert len(crew.agents) == 5
    assert len(crew.tasks) == 5


def test_run_crew_returns_expected_keys():
    """run_crew should return a dict with document_id, final_response, messages,
    and intermediate_results keys, matching the old LangGraph output contract."""
    mock_output = MagicMock()
    mock_output.raw = "Test final response"

    with patch("btp_usecase.crew.crew.build_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        result = run_crew(
            user_query="Summarize the uploaded document",
            document_id="doc-123",
            document_text="This is a test document.",
        )

    assert result["final_response"] == "Test final response"
    assert result["document_id"] == "doc-123"
    assert "messages" in result
    assert "intermediate_results" in result
