from __future__ import annotations

from abc import ABC, abstractmethod


class DocumentStorage(ABC):
    @abstractmethod
    def save(self, document_id: str, content: bytes, extension: str | None = None) -> str:
        """Persist the document bytes and return the storage reference."""

    @abstractmethod
    def get(self, document_id: str) -> bytes | None:
        """Retrieve raw document bytes by document id."""

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Delete the associated document if it exists."""
