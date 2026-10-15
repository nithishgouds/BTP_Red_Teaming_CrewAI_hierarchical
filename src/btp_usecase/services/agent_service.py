from __future__ import annotations

from ..crew.crew import run_crew, run_hierarchical_crew, compare_crews
from ..services.extraction_service import ExtractionService
from ..storage.base import DocumentStorage


class AgentService:
    def __init__(self, storage: DocumentStorage, extraction_service: ExtractionService | None = None):
        self.storage = storage
        self.extraction_service = extraction_service or ExtractionService()

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
        """Resolve document text from storage if document_id provided.

        Returns (document_id, resolved_document_text).
        """
        if document_id:
            raw = self.storage.get(document_id)
            if raw is None:
                raise FileNotFoundError(f"Document not found: {document_id}")
            file_type = self._detect_file_type(document_id, raw)
            extracted = self.extraction_service.extract(file_type, raw)
            document_text = extracted.text or ""
        return document_id, document_text

    # ------------------------------------------------------------------
    # Sequential (original)
    # ------------------------------------------------------------------

    def run_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
    ) -> dict:
        """Run the SEQUENTIAL crew pipeline (Process.sequential)."""
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return run_crew(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
        )

    # ------------------------------------------------------------------
    # Hierarchical (Sec 6 - Hierarchical Process Experiment)
    # ------------------------------------------------------------------

    def run_hierarchical_agent(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
    ) -> dict:
        """
        Run the HIERARCHICAL crew pipeline (Process.hierarchical) with
        the zero-trust Supervisor agent.

        Returns the same base shape as run_agent, plus:
          - process: 'hierarchical'
          - audit_trail: the supervisor's structured validation log
          - elapsed_seconds: wall-clock time for the full run
        """
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return run_hierarchical_crew(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
        )

    # ------------------------------------------------------------------
    # Comparison (Sec 6 - Sequential vs Hierarchical)
    # ------------------------------------------------------------------

    def compare_agent_runs(
        self,
        user_query: str,
        document_id: str | None = None,
        document_text: str | None = None,
    ) -> dict:
        """
        Run BOTH sequential and hierarchical crews on the same input and
        return a structured comparison including reproducibility and
        auditability metrics.

        See crew.compare_crews() for the full return-shape specification.
        """
        document_id, document_text = self._resolve_document_text(document_id, document_text)
        return compare_crews(
            user_query=user_query,
            document_id=document_id,
            document_text=document_text,
        )
