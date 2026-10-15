from __future__ import annotations

import uuid
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile

from ..core.config import get_settings
from ..models.document import DocumentUploadResult
from ..services.extraction_service import ExtractionService
from ..services.metadata_service import MetadataService
from ..storage.base import DocumentStorage


class DocumentService:
    def __init__(self, storage: DocumentStorage, extraction_service: ExtractionService | None = None, metadata_service: MetadataService | None = None):
        self.storage = storage
        self.extraction_service = extraction_service or ExtractionService()
        self.metadata_service = metadata_service or MetadataService()
        self.settings = get_settings()
        self.max_file_size = self.settings.max_upload_size_mb * 1024 * 1024

    def upload_document(self, file: UploadFile) -> DocumentUploadResult:
        filename = file.filename or ""
        self._validate_filename(filename)
        file_type = self._detect_file_type(filename)
        content = self._read_file_bytes(file)
        self._validate_size(content)

        document_id = str(uuid.uuid4())
        storage_reference = self.storage.save(document_id, content, extension=file_type)

        extracted = self.extraction_service.extract(file_type, content)
        metadata = self.metadata_service.build_metadata(
            document_id=document_id,
            original_filename=filename,
            file_type=file_type,
            file_size=len(content),
            page_count=extracted.page_count,
            content_length=len(extracted.text or ""),
            storage_reference=storage_reference,
        )

        return DocumentUploadResult(
            document_id=document_id,
            content=extracted.text or "",
            metadata=metadata,
            storage_reference=storage_reference,
        )

    def _validate_filename(self, filename: str) -> None:
        if not filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename")

    def _detect_file_type(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in self.settings.allowed_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")
        return suffix

    def _read_file_bytes(self, file: UploadFile) -> bytes:
        file.file.seek(0)
        return file.file.read()

    def _validate_size(self, content: bytes) -> None:
        if len(content) > self.max_file_size:
            raise ValueError(f"File too large: {len(content)} bytes exceeds limit of {self.max_file_size} bytes")
