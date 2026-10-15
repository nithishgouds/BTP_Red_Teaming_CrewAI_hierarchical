from __future__ import annotations

from ..core.config import get_settings
from ..crew.crew import build_crew
from ..services.document_service import DocumentService
from ..storage.local import LocalDocumentStorage


def get_settings_dependency():
    return get_settings()


def get_document_service() -> DocumentService:
    storage = LocalDocumentStorage(base_path=get_settings_dependency().storage_path)
    return DocumentService(storage=storage)


def get_agent_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
):
    """Dependency factory: build a fresh CrewAI Crew for each request."""
    return build_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )
