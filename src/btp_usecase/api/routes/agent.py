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
    enable_guardrail: bool = False
    use_vector_store: bool = False


class DirectRedactRequest(BaseModel):
    text: str


class PayloadAuditRequest(BaseModel):
    payload: str | dict
    source_raw_pii: list[str]


class ChunkSearchRequest(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int = 3


class RetrievalBenchmarkRequest(BaseModel):
    document_id: str
    query: str
    top_k: int = 3


def _get_service() -> AgentService:
    storage = LocalDocumentStorage(base_path="./data/documents")
    return AgentService(storage=storage)


# ---------------------------------------------------------------------------
# Sequential endpoint (Config A / Config B)
# ---------------------------------------------------------------------------

@router.post("/agent/run")
async def run_agent(request: AgentRequest):
    """
    Run the SEQUENTIAL crew (Process.sequential).
    Set enable_guardrail=True for Config B (6 agents with PII Redactor).
    Set use_vector_store=True for ChromaDB chunk-level semantic retrieval.
    """
    service = _get_service()
    try:
        result = service.run_agent(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
            enable_guardrail=request.enable_guardrail,
            use_vector_store=request.use_vector_store,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/agent/run/guardrail")
async def run_agent_guardrail(request: AgentRequest):
    """
    Run the 6-agent SEQUENTIAL crew with active PII Guardrail (Config B).
    Guarantees before/after PII redaction and payload monitoring.
    """
    service = _get_service()
    try:
        result = service.run_guardrail_agent(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
            use_vector_store=request.use_vector_store,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Hierarchical endpoint (Sec 6 / Config C)
# ---------------------------------------------------------------------------

@router.post("/agent/run/hierarchical")
async def run_agent_hierarchical(request: AgentRequest):
    """
    Run the HIERARCHICAL crew (Process.hierarchical) with zero-trust Supervisor.
    Set enable_guardrail=True for Config C (+Guardrail +Supervisor).
    Set use_vector_store=True for ChromaDB chunk semantic search & data-flow audit.
    """
    service = _get_service()
    try:
        result = service.run_hierarchical_agent(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
            enable_guardrail=request.enable_guardrail,
            use_vector_store=request.use_vector_store,
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
    return a structured side-by-side comparison (Section 6 + Stage 1 Privacy metrics + Stage 2 Vector audit).
    """
    service = _get_service()
    try:
        result = service.compare_agent_runs(
            user_query=request.user_query,
            document_id=request.document_id,
            document_text=request.document_text,
            enable_guardrail=request.enable_guardrail,
            use_vector_store=request.use_vector_store,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Dedicated Stage 1 Privacy Guardrail Telemetry Endpoints
# ---------------------------------------------------------------------------

@router.post("/privacy/redact")
async def redact_text(request: DirectRedactRequest):
    """
    Direct inspection endpoint for PII detection, scoring, and before/after text redaction.
    """
    from ...services.privacy_guardrail import PrivacyGuardrail
    engine = PrivacyGuardrail()
    res = engine.redact(request.text)
    return {
        "before_text": res.before_text,
        "after_text": res.after_text,
        "privacy_risk_score": res.privacy_risk_score,
        "detected_entities": [
            {
                "entity_type": e.entity_type,
                "text": e.text,
                "score": e.score,
                "detector": e.detector,
            }
            for e in res.detected_entities
        ],
        "entity_counts": res.entity_counts,
        "raw_pii_count": len(res.raw_pii_values),
    }


@router.post("/privacy/audit-payload")
async def audit_payload(request: PayloadAuditRequest):
    """
    Direct inspection endpoint for audit hook: privacy_guardrail.task.payload_monitoring.
    Computes Leakage Rate % for outbound payloads.
    """
    from ...services.privacy_guardrail import payload_monitoring_hook
    res = payload_monitoring_hook(request.payload, request.source_raw_pii)
    return res.to_dict()


# ---------------------------------------------------------------------------
# Dedicated Stage 2 Vector Store & Retrieval Benchmark Endpoints
# ---------------------------------------------------------------------------

@router.post("/documents/index/{document_id}")
async def index_document_chunks(document_id: str):
    """Index an uploaded document into ChromaDB chunks."""
    service = _get_service()
    try:
        chunk_ids = service.index_document(document_id)
        return {
            "document_id": document_id,
            "indexed_chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "status": "SUCCESS",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/documents/search")
async def search_chunks(request: ChunkSearchRequest):
    """Semantic search across indexed document chunks with distance metrics."""
    service = _get_service()
    chunks = service.search_chunks(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    return {
        "query": request.query,
        "document_id": request.document_id,
        "result_count": len(chunks),
        "chunks": chunks,
    }


@router.post("/benchmarks/retrieval")
async def benchmark_retrieval(request: RetrievalBenchmarkRequest):
    """
    Benchmark retrieval recall vs latency vs leakage surface reduction %.
    Demonstrates quantitative reduction in exposed PII when moving from full doc to chunked retrieval.
    """
    service = _get_service()
    try:
        benchmark_res = service.benchmark_retrieval(
            document_id=request.document_id,
            query=request.query,
            top_k=request.top_k,
        )
        return benchmark_res
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

