"""Unit and integration tests for Persistent Vector Store & Chunk-Level Auditing (Step 2)."""
from unittest.mock import MagicMock, patch
import pytest

from btp_usecase.storage.vector_store import (
    TextChunker,
    ChromaVectorStore,
    ChunkRecord,
)
from btp_usecase.services.chunk_audit import ChunkFlowAuditor
from btp_usecase.services.retrieval_benchmark import RetrievalBenchmarkEngine
from btp_usecase.crew.crew import run_crew, run_hierarchical_crew


# ---------------------------------------------------------------------------
# Unit tests: Text Chunker
# ---------------------------------------------------------------------------

def test_text_chunker_deterministic_splitting():
    """TextChunker should produce deterministic chunks with structured IDs and metadata."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "This is paragraph one of the test file. It contains technical details about systems.\n\n"
        "This is paragraph two which discusses security protocols and encryption standards.\n\n"
        "Finally paragraph three covers database administration and cloud architectures."
    )
    chunks = chunker.chunk_text("doc_101", text, {"category": "test_doc"})

    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.document_id == "doc_101"
        assert c.chunk_id == f"doc_101_chunk_{i}"
        assert c.chunk_index == i
        assert c.metadata["category"] == "test_doc"
        assert len(c.text) > 0


def test_text_chunker_rejects_invalid_overlap():
    """TextChunker should raise ValueError if overlap is greater than or equal to size."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, chunk_overlap=50)


# ---------------------------------------------------------------------------
# Unit tests: Chroma Vector Store Operations
# ---------------------------------------------------------------------------

def test_chroma_vector_store_index_and_search():
    """ChromaVectorStore in in-memory mode should index chunks and return nearest matches."""
    store = ChromaVectorStore(in_memory=True, collection_name="test_search")

    doc_text = (
        "Deep learning and transformer architectures power modern language models. "
        "Attention mechanisms allow capturing long-range semantic dependencies in text."
    )
    chunk_ids = store.index_document("doc_ml", doc_text, {"topic": "ai"})
    assert len(chunk_ids) >= 1

    # Search semantically
    results = store.search(query="transformers and attention", top_k=1)
    assert len(results) == 1
    assert "transformer" in results[0].text.lower()
    assert results[0].document_id == "doc_ml"


def test_chroma_vector_store_metadata_filtering():
    """Search should filter by document_id when requested."""
    store = ChromaVectorStore(in_memory=True, collection_name="test_filter")

    store.index_document("doc_A", "Alpha document contains sensitive medical records for patient 101.")
    store.index_document("doc_B", "Beta document contains financial earnings report for quarter 3.")

    # Search query targeting doc_B specifically
    results_a = store.search(query="financial earnings", document_id="doc_A", top_k=2)
    results_b = store.search(query="financial earnings", document_id="doc_B", top_k=2)

    # Filtering to doc_A should only yield doc_A chunks
    for r in results_a:
        assert r.document_id == "doc_A"

    for r in results_b:
        assert r.document_id == "doc_B"


def test_chroma_vector_store_get_and_delete():
    """get_document_chunks and delete_document should properly manage chunk lifecycle."""
    store = ChromaVectorStore(in_memory=True, collection_name="test_lifecycle")
    store.index_document("doc_del", "Content to be deleted from the vector database.")

    chunks = store.get_document_chunks("doc_del")
    assert len(chunks) >= 1

    store.delete_document("doc_del")
    chunks_after = store.get_document_chunks("doc_del")
    assert len(chunks_after) == 0


# ---------------------------------------------------------------------------
# Unit tests: Chunk Flow Auditor (Provenance Verification)
# ---------------------------------------------------------------------------

def test_chunk_flow_auditor_provenance_verified():
    """When chunks flow through retriever -> redactor -> analyst, provenance is VERIFIED."""
    auditor = ChunkFlowAuditor()
    chunks = ["doc1_chunk_0", "doc1_chunk_1"]

    auditor.record("Document Retriever", "retrieve_document", chunks, "retrieved")
    auditor.record("PII Redactor / Privacy Scorer", "privacy_guardrail", chunks, "redacted")
    auditor.record("LLM Analyst", "llm_process", chunks, "analyzed")

    audit_res = auditor.verify_provenance()
    assert audit_res["status"] == "PROVENANCE_VERIFIED"
    assert audit_res["total_chunks_tracked"] == 2
    assert len(audit_res["unguarded_chunks"]) == 0


def test_chunk_flow_auditor_detects_unguarded_flow():
    """When a chunk flows directly to LLM Analyst without PII Redactor, flag violation."""
    auditor = ChunkFlowAuditor()
    chunks = ["doc1_chunk_unguarded"]

    auditor.record("Document Retriever", "retrieve_document", chunks, "retrieved")
    # Skips PII Redactor!
    auditor.record("LLM Analyst", "llm_process", chunks, "analyzed")

    audit_res = auditor.verify_provenance()
    assert audit_res["status"] == "UNGUARDED_FLOW_DETECTED"
    assert "doc1_chunk_unguarded" in audit_res["unguarded_chunks"]


# ---------------------------------------------------------------------------
# Unit tests: Retrieval Benchmark (Recall vs Latency vs Leakage Surface)
# ---------------------------------------------------------------------------

def test_retrieval_benchmark_leakage_surface_reduction():
    """
    Retrieving only relevant chunks should reduce the PII exposure surface compared
    to ingesting the entire document into agent context.
    """
    store = ChromaVectorStore(in_memory=True, collection_name="test_bench")
    engine = RetrievalBenchmarkEngine()

    # Document with 3 distinct paragraphs:
    # Para 1 has SSN and email
    # Para 2 is neutral technical description
    # Para 3 has credit card
    doc_text = (
        "CONFIDENTIAL IDENTIFIERS: Employee SSN is 111-22-3333 and contact email is sec@corp.org.\n\n"
        "SYSTEM ARCHITECTURE: The distributed backend utilizes PostgreSQL and Redis caching for low latency.\n\n"
        "BILLING DATA: Account credit card on file is 4532-1188-9922-3344 for recurring subscription."
    )
    store.index_document("doc_bench", doc_text, chunk_size=150, chunk_overlap=20)

    # Query targeting ONLY the technical architecture (neutral chunk)
    result = engine.benchmark(
        vector_store=store,
        document_id="doc_bench",
        full_text=doc_text,
        query="PostgreSQL and Redis distributed caching architecture",
        top_k=1,
    )

    assert result.baseline_pii_surface >= 2  # SSN and credit card in full doc
    assert result.retrieval_latency_ms >= 0.0
    assert result.recall_score > 0.0
    # Leakage surface reduction should be positive (top-1 chunk contains fewer PII than whole document)
    assert result.leakage_surface_reduction_pct > 0.0


# ---------------------------------------------------------------------------
# Integration tests: Crew execution with vector store & chunk auditing
# ---------------------------------------------------------------------------

def test_crew_with_vector_store_and_chunk_auditing():
    """run_crew with use_vector_store=True includes chunk_flow_audit and retrieved_chunk_ids."""
    store = ChromaVectorStore(in_memory=True, collection_name="test_crew_vstore")
    mock_output = MagicMock()
    mock_output.raw = "Synthesized response based on retrieved chunks."

    with patch("btp_usecase.crew.crew.build_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        res = run_crew(
            user_query="Database architecture",
            document_id="doc_int",
            document_text="Database architecture uses PostgreSQL and read replicas.",
            enable_guardrail=True,
            use_vector_store=True,
            vector_store=store,
        )

    assert "chunk_flow_audit" in res
    assert "retrieved_chunk_ids" in res
    assert len(res["retrieved_chunk_ids"]) > 0
    flow = res["chunk_flow_audit"]
    assert flow["provenance"]["status"] == "PROVENANCE_VERIFIED"


def test_hierarchical_crew_with_vector_store_and_chunk_auditing():
    """run_hierarchical_crew with use_vector_store=True audits chunk data flow."""
    store = ChromaVectorStore(in_memory=True, collection_name="test_hier_vstore")
    mock_output = MagicMock()
    mock_output.raw = (
        "AUDIT_TRAIL:\n"
        '{"step": "document_retrieval", "status": "VALID"}\n'
        "FINAL_ANSWER:\n"
        "Hierarchical response."
    )

    with patch("btp_usecase.crew.crew.build_hierarchical_crew") as mock_build:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = mock_output
        mock_build.return_value = mock_crew

        res = run_hierarchical_crew(
            user_query="Microservices",
            document_id="doc_hier_v",
            document_text="Microservice services communicate via gRPC protocols.",
            enable_guardrail=True,
            use_vector_store=True,
            vector_store=store,
        )

    assert res["process"] == "hierarchical"
    assert "chunk_flow_audit" in res
    assert len(res["retrieved_chunk_ids"]) > 0
