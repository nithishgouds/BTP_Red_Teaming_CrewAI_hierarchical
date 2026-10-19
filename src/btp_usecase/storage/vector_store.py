from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None


class TextChunker:
    """
    Splits document text into overlapping chunks with deterministic IDs.
    Preserves paragraph or sentence boundaries where possible.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, document_id: str, text: str, base_metadata: dict[str, Any] | None = None) -> list[ChunkRecord]:
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        step = self.chunk_size - self.chunk_overlap
        chunks: list[ChunkRecord] = []
        base_meta = base_metadata or {}

        # Split along whitespace/newlines to avoid severing words when possible
        start = 0
        text_len = len(clean_text)
        index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at the end of text, try to find a natural break near the end
            if end < text_len:
                break_point = -1
                for delim in ["\n\n", "\n", ". ", " "]:
                    pos = clean_text.rfind(delim, start + step // 2, end)
                    if pos != -1:
                        break_point = pos + len(delim)
                        break
                if break_point != -1 and break_point > start:
                    end = break_point

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunk_id = f"{document_id}_chunk_{index}"
                meta = {
                    **base_meta,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "start_offset": start,
                    "end_offset": end,
                    "char_count": len(chunk_content),
                }
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_content,
                        chunk_index=index,
                        metadata=meta,
                    )
                )
                index += 1

            start += step

        return chunks


class BaseVectorStore(ABC):
    """Abstract vector store contract for chunk-level semantic indexing and retrieval."""

    @abstractmethod
    def index_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """Chunk and index document text, returning the generated chunk IDs."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:
        """Perform semantic search against chunk embeddings."""

    @abstractmethod
    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        """Retrieve all indexed chunks belonging to a document."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks and embeddings for a document."""


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation providing persistent, chunk-level semantic search
    and metadata filtering.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = "./data/chroma",
        collection_name: str = "btp_document_chunks",
        in_memory: bool = False,
    ):
        self.collection_name = collection_name
        self.in_memory = in_memory

        if in_memory or persist_directory is None:
            self.client = chromadb.Client()
        else:
            persist_path = Path(persist_directory).resolve()
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_path))

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> list[str]:
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_text(document_id, text, metadata)
        if not chunks:
            return []

        # Remove existing chunks for this document first if re-indexing
        self.delete_document(document_id)

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
        )
        return ids

    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:
        if not query or not query.strip():
            return []

        where_filter: dict[str, Any] | None = None
        filters: list[dict[str, Any]] = []

        if document_id:
            filters.append({"document_id": document_id})
        if metadata_filter:
            for k, v in metadata_filter.items():
                filters.append({k: v})

        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []

        actual_k = min(top_k, total_chunks)

        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunk_records: list[ChunkRecord] = []
        if results and results.get("ids") and results["ids"][0]:
            matched_ids = results["ids"][0]
            matched_docs = results["documents"][0] if results.get("documents") else []
            matched_metas = results["metadatas"][0] if results.get("metadatas") else []
            matched_distances = results["distances"][0] if results.get("distances") else []

            for i, cid in enumerate(matched_ids):
                meta = matched_metas[i] if i < len(matched_metas) and matched_metas[i] else {}
                doc_text = matched_docs[i] if i < len(matched_docs) else ""
                dist = matched_distances[i] if i < len(matched_distances) else None
                doc_id = meta.get("document_id", document_id or "")
                c_idx = meta.get("chunk_index", 0)

                chunk_records.append(
                    ChunkRecord(
                        chunk_id=cid,
                        document_id=doc_id,
                        text=doc_text,
                        chunk_index=c_idx,
                        metadata=meta,
                        distance=round(dist, 4) if dist is not None else None,
                    )
                )

        return chunk_records

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        records: list[ChunkRecord] = []
        if results and results.get("ids"):
            for i, cid in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results.get("metadatas") and results["metadatas"][i] else {}
                doc_text = results["documents"][i] if results.get("documents") and results["documents"][i] else ""
                c_idx = meta.get("chunk_index", i)
                records.append(
                    ChunkRecord(
                        chunk_id=cid,
                        document_id=document_id,
                        text=doc_text,
                        chunk_index=c_idx,
                        metadata=meta,
                    )
                )
            records.sort(key=lambda x: x.chunk_index)
        return records

    def delete_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass
