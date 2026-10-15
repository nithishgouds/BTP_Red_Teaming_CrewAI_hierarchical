from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services.agent_service import AgentService
from ...storage.local import LocalDocumentStorage

router = APIRouter()


class AgentRequest(BaseModel):
    user_query: str
    document_id: str | None = None
    document_text: str | None = None


def _get_service() -> AgentService:
    storage = LocalDocumentStorage(base_path="./data/documents")
    return AgentService(storage=storage)


# ---------------------------------------------------------------------------
# Original sequential endpoint (unchanged)
# ---------------------------------------------------------------------------

@router.post("/agent/run")
async def run_agent(request: AgentRequest):
    """
    Run the five-agent SEQUENTIAL crew (Process.sequential).

    Each task receives all prior tasks as context via CrewAI's sequential
    process.  No supervisor; no audit trail.
    """
    service = _get_service()
    try:
        result = service.run_agent(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Hierarchical endpoint (Sec 6 - Hierarchical Process Experiment)
# ---------------------------------------------------------------------------

@router.post("/agent/run/hierarchical")
async def run_agent_hierarchical(request: AgentRequest):
    """
    Run the five-agent + Supervisor HIERARCHICAL crew (Process.hierarchical).

    The Supervisor agent operates under zero-trust:
      - allow_delegation=False on every agent.
      - Explicit context passing only; no dynamic re-routing.
      - Returns a structured audit trail alongside the final answer.

    Extra response fields:
      - process: "hierarchical"
      - audit_trail: supervisor validation log (AUDIT_TRAIL section)
      - elapsed_seconds: wall-clock time for the full run
    """
    service = _get_service()
    try:
        result = service.run_hierarchical_agent(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Comparison endpoint (Sec 6 - Sequential vs Hierarchical)
# ---------------------------------------------------------------------------

@router.post("/agent/compare")
async def compare_agents(request: AgentRequest):
    """
    Run BOTH sequential and hierarchical crews on the same input and
    return a structured side-by-side comparison (Section 6).

    Comparison dimensions:
      - reproducibility: whether both crews produce the same response digest.
      - auditability:    whether the hierarchical crew produced an audit trail.
      - elapsed_delta_seconds: overhead of the hierarchical mode.

    Response shape:
    {
        "sequential":   { ...sequential result... },
        "hierarchical": { ...hierarchical result... },
        "comparison": {
            "sequential_digest": "<hex16>",
            "hierarchical_digest": "<hex16>",
            "responses_match": true | false,
            "reproducibility": "PASS" | "VARIANCE_DETECTED",
            "auditability": "PASS" | "FAIL",
            "elapsed_delta_seconds": <float>,
            "timestamp_utc": "<iso8601>"
        }
    }
    """
    service = _get_service()
    try:
        result = service.compare_agent_runs(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
