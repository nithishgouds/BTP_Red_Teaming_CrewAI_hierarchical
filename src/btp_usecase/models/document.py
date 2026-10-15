from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedDocument:
    text: str
    page_count: int = 0
    content_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    document_id: str
    original_filename: str
    file_type: str
    file_size: int
    created_at: str
    page_count: int
    content_length: int
    storage_reference: str


@dataclass
class DocumentUploadResult:
    document_id: str
    content: str
    metadata: DocumentMetadata
    storage_reference: str
