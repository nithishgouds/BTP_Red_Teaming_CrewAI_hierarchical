from __future__ import annotations

from datetime import datetime, timezone

from ..models.document import DocumentMetadata


class MetadataService:
    def build_metadata(
        self,
        *,
        document_id: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        page_count: int,
        content_length: int,
        storage_reference: str,
    ) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=document_id,
            original_filename=original_filename,
            file_type=file_type,
            file_size=file_size,
            created_at=datetime.now(timezone.utc).isoformat(),
            page_count=page_count,
            content_length=content_length,
            storage_reference=storage_reference,
        )
