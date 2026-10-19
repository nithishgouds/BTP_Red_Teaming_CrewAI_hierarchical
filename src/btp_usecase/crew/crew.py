from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

from crewai import Crew, Process

from .agents import (
    make_api_agent,
    make_document_agent,
    make_input_agent,
    make_llm_agent,
    make_privacy_agent,
    make_response_agent,
    make_supervisor_agent,
)
from .tasks import (
    make_external_api_task,
    make_finalize_response_task,
    make_llm_process_task,
    make_normalize_query_task,
    make_privacy_guardrail_task,
    make_retrieve_document_task,
    make_supervisor_audit_task,
)
from ..services.privacy_guardrail import PrivacyGuardrail, payload_monitoring_hook
from ..services.chunk_audit import ChunkFlowAuditor
from ..storage.vector_store import BaseVectorStore, ChromaVectorStore, ChunkRecord


# ===========================================================================
# SEQUENTIAL CREW (Process.sequential)
# ===========================================================================

def build_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
    enable_guardrail: bool = False,
    chunk_ids: list[str] | None = None,
) -> Crew:
    """
    Construct and return a CrewAI Crew running sequentially (Process.sequential).

    Configurations:
      - enable_guardrail=False (Config A - Baseline): 5 agents
        input -> document -> llm -> api -> response
      - enable_guardrail=True  (Config B - Guardrail Active): 6 agents (Stage 1)
        input -> document -> privacy -> llm -> api -> response
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
        chunk_ids=chunk_ids,
        context=[t_input],
    )

    if enable_guardrail:
        # Dedicated Privacy Guardrail Agent (6th agent)
        privacy_agent = make_privacy_agent()
        t_privacy = make_privacy_guardrail_task(
            agent=privacy_agent,
            document_text=document_text,
            chunk_ids=chunk_ids,
            context=[t_document],
        )

        t_llm = make_llm_process_task(
            agent=llm_agent,
            user_query=user_query,
            document_text=document_text,
            context=[t_input, t_document, t_privacy],
        )

        t_api = make_external_api_task(
            agent=api_agent,
            context=[t_input, t_document, t_privacy, t_llm],
        )

        t_response = make_finalize_response_task(
            agent=response_agent,
            context=[t_input, t_document, t_privacy, t_llm, t_api],
        )

        agents = [input_agent, document_agent, privacy_agent, llm_agent, api_agent, response_agent]
        tasks = [t_input, t_document, t_privacy, t_llm, t_api, t_response]
    else:
        # Baseline 5-agent pipeline
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

        agents = [input_agent, document_agent, llm_agent, api_agent, response_agent]
        tasks = [t_input, t_document, t_llm, t_api, t_response]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )


def build_guardrail_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
    chunk_ids: list[str] | None = None,
) -> Crew:
    """Convenience factory for the 6-agent sequential crew with active PII Guardrail."""
    return build_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=True,
        chunk_ids=chunk_ids,
    )


def run_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
    enable_guardrail: bool = False,
    use_vector_store: bool = False,
    vector_store: BaseVectorStore | None = None,
    top_k_chunks: int = 3,
) -> dict[str, Any]:
    """
    Build and execute the sequential crew, executing chunk retrieval,
    privacy guardrail, and payload monitoring audit hook.
    """
    start_ts = time.monotonic()
    guardrail_engine = PrivacyGuardrail()
    chunk_auditor = ChunkFlowAuditor()

    effective_text = document_text
    retrieved_chunk_ids: list[str] = []

    # Optional Chunk-level Semantic Retrieval (Step 2)
    if use_vector_store and document_id and document_text:
        vstore = vector_store or ChromaVectorStore()
        # Ensure document is indexed
        existing_chunks = vstore.get_document_chunks(document_id)
        if not existing_chunks:
            vstore.index_document(document_id, document_text)
        
        # Search for semantically relevant chunks
        chunks = vstore.search(query=user_query, top_k=top_k_chunks, document_id=document_id)
        if chunks:
            retrieved_chunk_ids = [c.chunk_id for c in chunks]
            effective_text = "\n\n".join(f"[CHUNK_ID: {c.chunk_id}]\n{c.text}" for c in chunks)
            chunk_auditor.record("Document Retriever", "retrieve_document", retrieved_chunk_ids, "retrieved")

    privacy_telemetry: dict[str, Any] = {
        "enabled": enable_guardrail,
        "privacy_score": 0.0,
        "entities_detected": [],
        "entity_counts": {},
        "raw_pii_count": 0,
        "before_after_log": {"before": effective_text or "", "after": effective_text or ""},
        "payload_monitoring": {},
        "leakage_rate_pct": 0.0,
        "status": "SAFE",
    }

    raw_pii_values: list[str] = []

    if effective_text:
        redaction_res = guardrail_engine.redact(effective_text)
        raw_pii_values = redaction_res.raw_pii_values
        privacy_telemetry["privacy_score"] = redaction_res.privacy_risk_score
        privacy_telemetry["entities_detected"] = [e.entity_type for e in redaction_res.detected_entities]
        privacy_telemetry["entity_counts"] = redaction_res.entity_counts
        privacy_telemetry["raw_pii_count"] = len(raw_pii_values)

        if enable_guardrail:
            # Pass sanitized text to downstream agents
            effective_text = redaction_res.after_text
            privacy_telemetry["before_after_log"] = {
                "before": redaction_res.before_text,
                "after": redaction_res.after_text,
            }
            if retrieved_chunk_ids:
                chunk_auditor.record(
                    "PII Redactor / Privacy Scorer",
                    "privacy_guardrail",
                    retrieved_chunk_ids,
                    "redacted",
                    {"privacy_score": redaction_res.privacy_risk_score},
                )

    if retrieved_chunk_ids:
        chunk_auditor.record("LLM Analyst", "llm_process", retrieved_chunk_ids, "analyzed")

    crew = build_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=effective_text,
        enable_guardrail=enable_guardrail,
        chunk_ids=retrieved_chunk_ids,
    )
    crew_output = crew.kickoff()
    elapsed = time.monotonic() - start_ts

    final_response: str = (
        getattr(crew_output, "raw", str(crew_output))
        or "I could not generate a response."
    )

    # Execute payload monitoring audit hook
    monitoring_res = payload_monitoring_hook(final_response, raw_pii_values)
    privacy_telemetry["payload_monitoring"] = monitoring_res.to_dict()
    privacy_telemetry["leakage_rate_pct"] = monitoring_res.leakage_rate_pct
    privacy_telemetry["status"] = monitoring_res.status

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
        "privacy_guardrail": privacy_telemetry,
        "chunk_flow_audit": chunk_auditor.to_dict(),
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


def run_guardrail_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
    use_vector_store: bool = False,
    vector_store: BaseVectorStore | None = None,
) -> dict[str, Any]:
    """Execute the sequential crew with PII Guardrail enabled (Config B)."""
    return run_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=True,
        use_vector_store=use_vector_store,
        vector_store=vector_store,
    )


# ===========================================================================
# HIERARCHICAL CREW (Sec 6 + Stage 1 Privacy Guardrail + Stage 2 Vector)
# ===========================================================================

def build_hierarchical_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
    enable_guardrail: bool = False,
    chunk_ids: list[str] | None = None,
) -> Crew:
    """
    Construct a hierarchical CrewAI Crew (Process.hierarchical) with a
    zero-trust Supervisor agent.

    Configurations:
      - enable_guardrail=False (Sec 6 baseline): 6 agents (5 workers + Supervisor)
      - enable_guardrail=True  (Config C): 7 agents (6 workers incl. PII Redactor + Supervisor)
    """
    from ..core.config import get_settings
    settings = get_settings()
    manager_llm_str = f"ollama/{settings.llm_model}"

    # Worker agents
    input_agent = make_input_agent()
    document_agent = make_document_agent()
    llm_agent = make_llm_agent()
    api_agent = make_api_agent()
    response_agent = make_response_agent()
    supervisor = make_supervisor_agent()

    t_input = make_normalize_query_task(agent=input_agent, user_query=user_query)
    t_document = make_retrieve_document_task(
        agent=document_agent,
        document_id=document_id,
        document_text=document_text,
        chunk_ids=chunk_ids,
        context=[t_input],
    )

    if enable_guardrail:
        privacy_agent = make_privacy_agent()
        t_privacy = make_privacy_guardrail_task(
            agent=privacy_agent,
            document_text=document_text,
            chunk_ids=chunk_ids,
            context=[t_document],
        )

        t_llm = make_llm_process_task(
            agent=llm_agent,
            user_query=user_query,
            document_text=document_text,
            context=[t_input, t_document, t_privacy],
        )

        t_api = make_external_api_task(
            agent=api_agent,
            context=[t_input, t_document, t_privacy, t_llm],
        )

        t_response = make_finalize_response_task(
            agent=response_agent,
            context=[t_input, t_document, t_privacy, t_llm, t_api],
        )

        # Supervisor audit task receives all 6 worker tasks as explicit context
        t_audit = make_supervisor_audit_task(
            agent=supervisor,
            context=[t_input, t_document, t_privacy, t_llm, t_api, t_response],
        )

        agents = [
            input_agent, document_agent, privacy_agent,
            llm_agent, api_agent, response_agent, supervisor,
        ]
        tasks = [t_input, t_document, t_privacy, t_llm, t_api, t_response, t_audit]
    else:
        # Baseline 6 agents (5 workers + supervisor)
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

        t_audit = make_supervisor_audit_task(
            agent=supervisor,
            context=[t_input, t_document, t_llm, t_api, t_response],
        )

        agents = [
            input_agent, document_agent, llm_agent,
            api_agent, response_agent, supervisor,
        ]
        tasks = [t_input, t_document, t_llm, t_api, t_response, t_audit]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm=manager_llm_str,
        verbose=True,
    )


def build_hierarchical_guardrail_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
    chunk_ids: list[str] | None = None,
) -> Crew:
    """Convenience factory for the 7-agent hierarchical crew with active PII Guardrail."""
    return build_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=True,
        chunk_ids=chunk_ids,
    )


def run_hierarchical_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
    enable_guardrail: bool = False,
    use_vector_store: bool = False,
    vector_store: BaseVectorStore | None = None,
    top_k_chunks: int = 3,
) -> dict[str, Any]:
    """
    Build and execute the hierarchical crew, returning the final answer,
    supervisor audit trail, chunk flow audit, and privacy metrics.
    """
    start_ts = time.monotonic()
    guardrail_engine = PrivacyGuardrail()
    chunk_auditor = ChunkFlowAuditor()

    effective_text = document_text
    retrieved_chunk_ids: list[str] = []

    if use_vector_store and document_id and document_text:
        vstore = vector_store or ChromaVectorStore()
        existing_chunks = vstore.get_document_chunks(document_id)
        if not existing_chunks:
            vstore.index_document(document_id, document_text)
        
        chunks = vstore.search(query=user_query, top_k=top_k_chunks, document_id=document_id)
        if chunks:
            retrieved_chunk_ids = [c.chunk_id for c in chunks]
            effective_text = "\n\n".join(f"[CHUNK_ID: {c.chunk_id}]\n{c.text}" for c in chunks)
            chunk_auditor.record("Document Retriever", "retrieve_document", retrieved_chunk_ids, "retrieved")

    privacy_telemetry: dict[str, Any] = {
        "enabled": enable_guardrail,
        "privacy_score": 0.0,
        "entities_detected": [],
        "entity_counts": {},
        "raw_pii_count": 0,
        "before_after_log": {"before": effective_text or "", "after": effective_text or ""},
        "payload_monitoring": {},
        "leakage_rate_pct": 0.0,
        "status": "SAFE",
    }

    raw_pii_values: list[str] = []

    if effective_text:
        redaction_res = guardrail_engine.redact(effective_text)
        raw_pii_values = redaction_res.raw_pii_values
        privacy_telemetry["privacy_score"] = redaction_res.privacy_risk_score
        privacy_telemetry["entities_detected"] = [e.entity_type for e in redaction_res.detected_entities]
        privacy_telemetry["entity_counts"] = redaction_res.entity_counts
        privacy_telemetry["raw_pii_count"] = len(raw_pii_values)

        if enable_guardrail:
            effective_text = redaction_res.after_text
            privacy_telemetry["before_after_log"] = {
                "before": redaction_res.before_text,
                "after": redaction_res.after_text,
            }
            if retrieved_chunk_ids:
                chunk_auditor.record(
                    "PII Redactor / Privacy Scorer",
                    "privacy_guardrail",
                    retrieved_chunk_ids,
                    "redacted",
                    {"privacy_score": redaction_res.privacy_risk_score},
                )

    if retrieved_chunk_ids:
        chunk_auditor.record("LLM Analyst", "llm_process", retrieved_chunk_ids, "analyzed")

    crew = build_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=effective_text,
        enable_guardrail=enable_guardrail,
        chunk_ids=retrieved_chunk_ids,
    )
    crew_output = crew.kickoff()
    elapsed = time.monotonic() - start_ts

    raw: str = getattr(crew_output, "raw", str(crew_output)) or ""

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

    # Payload monitoring audit hook
    monitoring_res = payload_monitoring_hook(final_response, raw_pii_values)
    privacy_telemetry["payload_monitoring"] = monitoring_res.to_dict()
    privacy_telemetry["leakage_rate_pct"] = monitoring_res.leakage_rate_pct
    privacy_telemetry["status"] = monitoring_res.status

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
        "privacy_guardrail": privacy_telemetry,
        "chunk_flow_audit": chunk_auditor.to_dict(),
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


def run_hierarchical_guardrail_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
    use_vector_store: bool = False,
    vector_store: BaseVectorStore | None = None,
) -> dict[str, Any]:
    """Execute the hierarchical crew with active PII Guardrail (Config C)."""
    return run_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=True,
        use_vector_store=use_vector_store,
        vector_store=vector_store,
    )


# ===========================================================================
# COMPARISON UTILITY
# ===========================================================================

def _response_digest(text: str) -> str:
    """Return the first 16 hex characters of the sha256 digest of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compare_crews(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
    enable_guardrail: bool = False,
    use_vector_store: bool = False,
    vector_store: BaseVectorStore | None = None,
) -> dict[str, Any]:
    """
    Run BOTH sequential and hierarchical crews on identical input and
    return a structured side-by-side comparison report.
    """
    seq_result = run_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=enable_guardrail,
        use_vector_store=use_vector_store,
        vector_store=vector_store,
    )
    hier_result = run_hierarchical_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
        enable_guardrail=enable_guardrail,
        use_vector_store=use_vector_store,
        vector_store=vector_store,
    )

    seq_digest = _response_digest(seq_result["final_response"])
    hier_digest = _response_digest(hier_result["final_response"])
    responses_match = seq_digest == hier_digest

    has_audit_trail = bool(hier_result.get("audit_trail", "").strip())
    elapsed_delta = hier_result["elapsed_seconds"] - seq_result["elapsed_seconds"]

    seq_leakage = seq_result.get("privacy_guardrail", {}).get("leakage_rate_pct", 0.0)
    hier_leakage = hier_result.get("privacy_guardrail", {}).get("leakage_rate_pct", 0.0)

    return {
        "sequential": seq_result,
        "hierarchical": hier_result,
        "comparison": {
            "sequential_digest": seq_digest,
            "hierarchical_digest": hier_digest,
            "responses_match": responses_match,
            "reproducibility": "PASS" if responses_match else "VARIANCE_DETECTED",
            "auditability": "PASS" if has_audit_trail else "FAIL",
            "elapsed_delta_seconds": round(elapsed_delta, 3),
            "guardrail_enabled": enable_guardrail,
            "use_vector_store": use_vector_store,
            "sequential_leakage_rate_pct": seq_leakage,
            "hierarchical_leakage_rate_pct": hier_leakage,
            "leakage_contained": (seq_leakage == 0.0 and hier_leakage == 0.0),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
