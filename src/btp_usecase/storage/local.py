from __future__ import annotations

from pathlib import Path

from .base import DocumentStorage


class LocalDocumentStorage(DocumentStorage):
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, content: bytes, extension: str | None = None) -> str:
        suffix = (extension or "bin").strip().lower().lstrip(".")
        target_path = self.base_path / f"{document_id}.{suffix}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        return str(target_path)

    def get(self, document_id: str) -> bytes | None:
        candidates = [
            self.base_path / f"{document_id}.bin",
            *[self.base_path / f"{document_id}.{suffix}" for suffix in ["txt", "pdf", "docx"]],
        ]
        for target_path in candidates:
            if target_path.exists():
                return target_path.read_bytes()
        return None

    def delete(self, document_id: str) -> None:
        for suffix in ["bin", "txt", "pdf", "docx"]:
            target_path = self.base_path / f"{document_id}.{suffix}"
            if target_path.exists():
                target_path.unlink()
