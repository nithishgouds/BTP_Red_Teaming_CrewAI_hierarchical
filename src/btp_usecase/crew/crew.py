from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

from crewai import Crew, Process

from .agents import (
    make_api_agent,
    make_document_agent,
    make_input_agent,
    make_llm_agent,
    make_response_agent,
    make_supervisor_agent,
)
from .tasks import (
    make_external_api_task,
    make_finalize_response_task,
    make_llm_process_task,
    make_normalize_query_task,
    make_retrieve_document_task,
    make_supervisor_audit_task,
)


# ===========================================================================
# SEQUENTIAL CREW  (original, Process.sequential)
# ===========================================================================

def build_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
) -> Crew:
    """
    Construct and return a CrewAI Crew that replicates the five-step LangGraph
    pipeline:

        input -> document -> llm -> api -> response

    The crew runs sequentially (Process.sequential) so each task has access to
    the outputs of all preceding tasks through the context list.
    """
    input_agent = make_input_agent()
    document_agent = make_document_agent()
    llm_agent = make_llm_agent()
    api_agent = make_api_agent()
    response_agent = make_response_agent()

    t_input = make_normalize_query_task(agent=input_agent, user_query=user_query)

    t_document = make_retrieve_document_task(
        agent=document_agent,
        document_id=document_id,
        document_text=document_text,
        context=[t_input],
    )

    t_llm = make_llm_process_task(
        agent=llm_agent,
        user_query=user_query,
        document_text=document_text,
        context=[t_input, t_document],
    )

    t_api = make_external_api_task(
        agent=api_agent,
        context=[t_input, t_document, t_llm],
    )

    t_response = make_finalize_response_task(
        agent=response_agent,
        context=[t_input, t_document, t_llm, t_api],
    )

    return Crew(
        agents=[input_agent, document_agent, llm_agent, api_agent, response_agent],
        tasks=[t_input, t_document, t_llm, t_api, t_response],
        process=Process.sequential,
        verbose=False,
    )


def run_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
) -> dict:
    """
    Build and kick-off the sequential crew, then return a result dict that
    matches the shape previously returned by the LangGraph AgentService.run_agent.
    """
    start_ts = time.monotonic()
    crew = build_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )
    crew_output = crew.kickoff()
    elapsed = time.monotonic() - start_ts

    final_response: str = (
        getattr(crew_output, "raw", str(crew_output))
        or "I could not generate a response."
    )

    return {
        "process": "sequential",
        "document_id": document_id,
        "final_response": final_response,
        "messages": [user_query],
        "elapsed_seconds": round(elapsed, 3),
        "intermediate_results": {
            "llm": {
                "provider": "ollama",
                "model": None,
                "response": final_response,
            }
        },
    }


# ===========================================================================
# HIERARCHICAL CREW  (Sec 6 - Hierarchical Process Experiment)
# ===========================================================================

def build_hierarchical_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
) -> Crew:
    """
    Construct a hierarchical CrewAI Crew (Process.hierarchical) with a
    zero-trust Supervisor agent (Section 6 of the BTP report).

    Architecture
    ------------
    The five worker agents are identical to the sequential crew.  The
    Supervisor is added as a sixth agent and assigned a dedicated audit task
    that receives ALL five worker tasks as context.  CrewAI's hierarchical
    process makes the supervisor the effective manager:

        Supervisor
           |---> Query Normalizer     (t_input)
           |---> Document Retriever   (t_document)
           |---> LLM Analyst          (t_llm)
           |---> External API Integr. (t_api)
           |---> Response Formatter   (t_response)
           L---> Supervisor Audit     (t_audit) <- collects all 5 outputs

    Zero-trust enforcement
    ----------------------
    * Every agent (worker AND supervisor) has allow_delegation=False.
    * The supervisor receives worker outputs exclusively through the
      task context list - no dynamic re-routing is possible.
    * The audit task description explicitly forbids re-delegation.

    manager_llm
    -----------
    Process.hierarchical requires a manager_llm to be specified on the Crew.
    We reuse the same Ollama string so the manager reasoning stays on the
    same local model as the workers.
    """
    from ..core.config import get_settings
    settings = get_settings()
    manager_llm_str = f"ollama/{settings.llm_model}"

    # Worker agents
    input_agent    = make_input_agent()
    document_agent = make_document_agent()
    llm_agent      = make_llm_agent()
    api_agent      = make_api_agent()
    response_agent = make_response_agent()
    supervisor     = make_supervisor_agent()

    # Worker tasks - same descriptions as the sequential crew
    t_input = make_normalize_query_task(agent=input_agent, user_query=user_query)
    t_document = make_retrieve_document_task(
        agent=document_agent,
        document_id=document_id,
        document_text=document_text,
        context=[t_input],
    )
    t_llm = make_llm_process_task(
        agent=llm_agent,
        user_query=user_query,
        document_text=document_text,
        context=[t_input, t_document],
    )
    t_api = make_external_api_task(
        agent=api_agent,
        context=[t_input, t_document, t_llm],
    )
    t_response = make_finalize_response_task(
        agent=response_agent,
        context=[t_input, t_document, t_llm, t_api],
    )

    # Supervisor audit task - receives ALL worker outputs as context
    t_audit = make_supervisor_audit_task(
        agent=supervisor,
        context=[t_input, t_document, t_llm, t_api, t_response],
    )

    return Crew(
        agents=[
            input_agent, document_agent, llm_agent,
            api_agent, response_agent, supervisor,
        ],
        tasks=[t_input, t_document, t_llm, t_api, t_response, t_audit],
        process=Process.hierarchical,
        manager_llm=manager_llm_str,  # required by CrewAI for hierarchical mode
        verbose=True,
    )


def run_hierarchical_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
) -> dict:
    """
    Build and kick-off the hierarchical crew, returning a result dict that
    includes the full supervisor audit trail alongside the final response.

    Return shape
    ------------
    {
        "process": "hierarchical",
        "document_id": ...,
        "final_response": <synthesised answer>,
        "audit_trail": <raw supervisor output>,
        "messages": [...],
        "elapsed_seconds": ...,
        "intermediate_results": { ... }
    }
    """
    start_ts = time.monotonic()
    crew = build_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )
    crew_output = crew.kickoff()
    elapsed = time.monotonic() - start_ts

    raw: str = getattr(crew_output, "raw", str(crew_output)) or ""

    # Extract FINAL_ANSWER section from supervisor output if present
    final_response = raw
    audit_trail = ""
    if "FINAL_ANSWER:" in raw:
        parts = raw.split("FINAL_ANSWER:", 1)
        audit_trail = parts[0].replace("AUDIT_TRAIL:", "").strip()
        final_response = parts[1].strip()
    elif "AUDIT_TRAIL:" in raw:
        audit_trail = raw

    if not final_response:
        final_response = "I could not generate a response."

    return {
        "process": "hierarchical",
        "document_id": document_id,
        "final_response": final_response,
        "audit_trail": audit_trail,
        "messages": [user_query],
        "elapsed_seconds": round(elapsed, 3),
        "intermediate_results": {
            "llm": {
                "provider": "ollama",
                "model": None,
                "response": final_response,
            }
        },
    }


# ===========================================================================
# COMPARISON UTILITY  (Sec 6 - Sequential vs Hierarchical)
# ===========================================================================

def _response_digest(text: str) -> str:
    """Return the first 16 hex characters of the sha256 digest of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compare_crews(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
) -> dict:
    """
    Run BOTH the sequential and hierarchical crews on the same input and
    return a structured comparison report (Section 6 of the BTP report).

    Comparison dimensions
    ---------------------
    reproducibility
        Whether both crews produced the same response digest.  Identical
        digests indicate full reproducibility; different digests signal
        non-deterministic variance between the two execution modes.

    auditability
        Whether the hierarchical crew produced a non-empty audit trail.
        The sequential crew has no audit trail by design.

    elapsed_seconds
        Wall-clock time for each crew.  Hierarchical crews incur extra
        overhead from supervisor reasoning and validation steps.

    Return shape
    ------------
    {
        "sequential":    { ...run_crew result... },
        "hierarchical":  { ...run_hierarchical_crew result... },
        "comparison": {
            "sequential_digest":    "<hex16>",
            "hierarchical_digest":  "<hex16>",
            "responses_match":      true | false,
            "reproducibility":      "PASS" | "VARIANCE_DETECTED",
            "auditability":         "PASS" | "FAIL",
            "elapsed_delta_seconds": <float>,
            "timestamp_utc":        "<iso8601>",
        }
    }
    """
    seq_result  = run_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )
    hier_result = run_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )

    seq_digest  = _response_digest(seq_result["final_response"])
    hier_digest = _response_digest(hier_result["final_response"])
    responses_match = seq_digest == hier_digest

    has_audit_trail = bool(hier_result.get("audit_trail", "").strip())

    elapsed_delta = (
        hier_result["elapsed_seconds"] - seq_result["elapsed_seconds"]
    )

    return {
        "sequential": seq_result,
        "hierarchical": hier_result,
        "comparison": {
            "sequential_digest":     seq_digest,
            "hierarchical_digest":   hier_digest,
            "responses_match":       responses_match,
            "reproducibility":       "PASS" if responses_match else "VARIANCE_DETECTED",
            "auditability":          "PASS" if has_audit_trail else "FAIL",
            "elapsed_delta_seconds": round(elapsed_delta, 3),
            "timestamp_utc":         datetime.now(timezone.utc).isoformat(),
        },
    }
