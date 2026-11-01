"""
Automated Adversarial Benchmarking package for Multi-Agent Systems.

Stage 3 implementation:
- fuzzer: Auto-generation of malicious PDFs/DOCX/TXT with indirect prompt injections.
- evaluator: Multi-model evaluation across llama3.2, mistral, phi3 logging 4 key telemetry metrics.
"""

from .fuzzer import (
    AdversarialDocumentFuzzer,
    FuzzedDocument,
    ThreatClass,
    DocumentDomain,
)
from .evaluator import (
    AdversarialBenchmarkEngine,
    BenchmarkRunResult,
    BenchmarkSummary,
)

__all__ = [
    "AdversarialDocumentFuzzer",
    "FuzzedDocument",
    "ThreatClass",
    "DocumentDomain",
    "AdversarialBenchmarkEngine",
    "BenchmarkRunResult",
    "BenchmarkSummary",
]

