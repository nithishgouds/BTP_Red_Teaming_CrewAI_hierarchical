from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    content: str = Field(..., description="Extracted document text")
    metadata: dict = Field(..., description="Normalized metadata for the uploaded file")
    storage_reference: str = Field(..., description="Storage path or identifier")


class AgentRunRequest(BaseModel):
    user_query: str
    document_id: str | None = None
    document_text: str | None = None
