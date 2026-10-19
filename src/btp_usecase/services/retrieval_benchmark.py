from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .privacy_guardrail import PrivacyGuardrail
from ..storage.vector_store import BaseVectorStore


@dataclass
class RetrievalBenchmarkResult:
    query: str
    document_id: str
    total_document_chars: int
    total_document_chunks: int
    retrieved_chunk_count: int
    retrieval_latency_ms: float
    full_scan_latency_ms: float
    recall_score: float
    baseline_pii_surface: int
    retrieved_pii_surface: int
    leakage_surface_reduction_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "document_id": self.document_id,
            "total_document_chars": self.total_document_chars,
            "total_document_chunks": self.total_document_chunks,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "full_scan_latency_ms": round(self.full_scan_latency_ms, 2),
            "recall_score": round(self.recall_score, 3),
            "baseline_pii_surface": self.baseline_pii_surface,
            "retrieved_pii_surface": self.retrieved_pii_surface,
            "leakage_surface_reduction_pct": round(self.leakage_surface_reduction_pct, 2),
        }


class RetrievalBenchmarkEngine:
    """
    Evaluates:
      1. Retrieval Latency (ChromaDB top-k query vs full-text scan).
      2. Retrieval Recall (coverage of query semantic terms).
      3. Leakage Surface Reduction (exposure of PII in top-k chunks vs entire document).
    """

    def __init__(self, guardrail: PrivacyGuardrail | None = None):
        self.guardrail = guardrail or PrivacyGuardrail()

    def benchmark(
        self,
        vector_store: BaseVectorStore,
        document_id: str,
        full_text: str,
        query: str,
        top_k: int = 3,
        relevant_keywords: list[str] | None = None,
    ) -> RetrievalBenchmarkResult:
        guardrail = self.guardrail

        # 1. Measure Baseline PII Surface on full document
        baseline_res = guardrail.redact(full_text)
        baseline_pii_count = len(baseline_res.raw_pii_values)

        # 2. Benchmark Full Text Scan Latency (baseline)
        t0 = time.perf_counter()
        _ = [w for w in query.lower().split() if w in full_text.lower()]
        full_scan_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Benchmark ChromaDB Semantic Retrieval Latency
        t1 = time.perf_counter()
        retrieved_chunks = vector_store.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000.0

        retrieved_text = "\n".join(c.text for c in retrieved_chunks)
        all_chunks = vector_store.get_document_chunks(document_id)

        # 4. Measure Retrieved PII Surface
        retrieved_res = guardrail.redact(retrieved_text)
        retrieved_pii_count = len(retrieved_res.raw_pii_values)

        # 5. Compute Leakage Surface Reduction %
        if baseline_pii_count == 0:
            reduction_pct = 0.0
        else:
            reduction = 1.0 - (retrieved_pii_count / baseline_pii_count)
            reduction_pct = max(0.0, reduction * 100.0)

        # 6. Compute Recall Score
        keywords = relevant_keywords or [w for w in query.lower().split() if len(w) > 3]
        if not keywords:
            recall = 1.0 if retrieved_chunks else 0.0
        else:
            hits = sum(1 for kw in keywords if kw in retrieved_text.lower())
            recall = hits / len(keywords)

        return RetrievalBenchmarkResult(
            query=query,
            document_id=document_id,
            total_document_chars=len(full_text),
            total_document_chunks=len(all_chunks),
            retrieved_chunk_count=len(retrieved_chunks),
            retrieval_latency_ms=retrieval_ms,
            full_scan_latency_ms=full_scan_ms,
            recall_score=recall,
            baseline_pii_surface=baseline_pii_count,
            retrieved_pii_surface=retrieved_pii_count,
            leakage_surface_reduction_pct=reduction_pct,
        )
