from __future__ import annotations

from typing import Any

from ..crew.crew import run_crew, run_hierarchical_crew, compare_crews
from ..services.extraction_service import ExtractionService
from ..services.retrieval_benchmark import RetrievalBenchmarkEngine
from ..storage.base import DocumentStorage
from ..storage.vector_store import BaseVectorStore, ChromaVectorStore


class AgentService:
    def __init__(
        self,
        storage: DocumentStorage,
        extraction_service: ExtractionService | None = None,
        vector_store: BaseVectorStore | None = None,
    ):
        self.storage = storage
        self.extraction_service = extraction_service or ExtractionService()
        self.vector_store = vector_store or ChromaVectorStore()
        self.benchmark_engine = RetrievalBenchmarkEngine()

    def _detect_file_type(self, document_id: str, raw: bytes | None = None) -> str:
        if raw is not None:
            if raw.startswith(b"%PDF"):
                return "pdf"
            if raw.startswith(b"PK"):
                return "docx"
            return "txt"

        candidates = [
            (self.storage.base_path / f"{document_id}.txt"),
            (self.storage.base_path / f"{document_id}.pdf"),
            (self.storage.base_path / f"{document_id}.docx"),
            (self.storage.base_path / f"{document_id}.bin"),
        ]
        for path in candidates:
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".pdf":
                    return "pdf"
                if suffix == ".docx":
                    return "docx"
                if suffix == ".txt":
                    return "txt"
                return "txt"
        return "txt"

    def _resolve_document_text(
        self,
        document_id: str | None,
        document_text: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve document text from storage if document_id provided."""
        if document_id:
            raw = self.storage.get(document_id)
            if raw is None:
                raise FileNotFoundError(f"Document not found: {document_id}")
            file_type = self._detect_file_type(document_id, raw)
            extracted = self.extraction_service.extract(file_type, raw)
            document_text = extracted.text or ""
        return document_id, document_text

    # ------------------------------------------------------------------
    # Vector Store Indexing & Semantic Search Operations (Step 2)
    # ------------------------------------------------------------------

    def index_document(self, document_id: str) -> list[str]:
        """Index a document's extracted text into ChromaDB as chunks."""
        _, text = self._resolve_document_text(document_id, None)
        if not text:
            return []
        return self.vector_store.index_document(document_id=document_id, text=text)

    def search_chunks(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search against indexed chunks with optional metadata filtering."""
        chunks = self.vector_store.search(query=query, top_k=top_k, document_id=document_id)
        return [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "distance": c.distance,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    def benchmark_retrieval(
        self,
        document_id: str,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Benchmark retrieval latency, recall, and PII leakage surface reduction."""
        _, text = self._resolve_document_text(document_id, None)
        if not text:
            raise FileNotFoundError(f"Document not found: {document_id}")
        self.index_document(document_id)
        res = self.benchmark_engine.benchmark(
            vector_store=self.vector_store,
            document_id=document_id,
            full_text=text,
            query=query,
            top_k=top_k,
        )
        return res.to_dict()

    # ------------------------------------------------------------------
    # Sequential (Config A & Config B)
    # ------------------------------------------------------------------

    def run_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
        enable_guardrail: bool = False,
        use_vector_store: bool = False,
    ) -> dict[str, Any]:
        """Run the SEQUENTIAL crew pipeline (Process.sequential)."""
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return run_crew(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
            enable_guardrail=enable_guardrail,
            use_vector_store=use_vector_store,
            vector_store=self.vector_store,
        )

    def run_guardrail_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
        use_vector_store: bool = False,
    ) -> dict[str, Any]:
        """Run the SEQUENTIAL crew pipeline with dedicated PII Guardrail active (Config B)."""
        return self.run_agent(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
            enable_guardrail=True,
            use_vector_store=use_vector_store,
        )

    # ------------------------------------------------------------------
    # Hierarchical (Sec 6 & Config C)
    # ------------------------------------------------------------------

    def run_hierarchical_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
        enable_guardrail: bool = False,
        use_vector_store: bool = False,
    ) -> dict[str, Any]:
        """
        Run the HIERARCHICAL crew pipeline (Process.hierarchical) with
        zero-trust Supervisor agent, chunk auditing, and privacy guardrails.
        """
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return run_hierarchical_crew(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
            enable_guardrail=enable_guardrail,
            use_vector_store=use_vector_store,
            vector_store=self.vector_store,
        )

    def run_hierarchical_guardrail_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
        use_vector_store: bool = False,
    ) -> dict[str, Any]:
        """Run the HIERARCHICAL crew with dedicated PII Guardrail active (Config C)."""
        return self.run_hierarchical_agent(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
            enable_guardrail=True,
            use_vector_store=use_vector_store,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_agent_runs(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
        enable_guardrail: bool = False,
        use_vector_store: bool = False,
    ) -> dict[str, Any]:
        """
        Run BOTH sequential and hierarchical crews on the same input and
        return a structured comparison.
        """
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return compare_crews(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
            enable_guardrail=enable_guardrail,
            use_vector_store=use_vector_store,
            vector_store=self.vector_store,
        )
