from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...models.api import DocumentUploadResponse
from ...services.document_service import DocumentService
from ...storage.local import LocalDocumentStorage

router = APIRouter()


def get_document_service() -> DocumentService:
    storage = LocalDocumentStorage(base_path="./data/documents")
    return DocumentService(storage=storage)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    service = get_document_service()
    try:
        result = service.upload_document(file)
        return DocumentUploadResponse(
            document_id=result.document_id,
            content=result.content,
            metadata=result.metadata.__dict__,
            storage_reference=result.storage_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents/{document_id}")
async def get_document(document_id: str) -> dict:
    storage = LocalDocumentStorage(base_path="./data/documents")
    blob = storage.get(document_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "storage_reference": str(storage.base_path / f"{document_id}.bin"), "size_bytes": len(blob)}
