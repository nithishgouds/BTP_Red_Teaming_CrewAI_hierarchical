"""
Tests for Adversarial Document Fuzzing Harness (Stage 3).
"""
import os
import pathlib
import tempfile
import pytest

from btp_usecase.benchmarks.fuzzer import (
    AdversarialDocumentFuzzer,
    FuzzedDocument,
    ThreatClass,
    DocumentDomain,
)
from btp_usecase.services.extraction_service import ExtractionService
from btp_usecase.services.privacy_guardrail import PrivacyGuardrail


def test_fuzzer_initialization_and_seeding():
    """Fuzzer with fixed seed produces deterministic synthetic content."""
    fuzzer1 = AdversarialDocumentFuzzer(seed=123)
    fuzzer2 = AdversarialDocumentFuzzer(seed=123)

    doc1 = fuzzer1.generate_document(
        domain=DocumentDomain.HR_PERFORMANCE_REVIEW,
        threat_class=ThreatClass.CONTEXT_HIJACKING,
        file_type="txt",
    )
    doc2 = fuzzer2.generate_document(
        domain=DocumentDomain.HR_PERFORMANCE_REVIEW,
        threat_class=ThreatClass.CONTEXT_HIJACKING,
        file_type="txt",
    )

    assert doc1.pii_entities["US_SSN"] == doc2.pii_entities["US_SSN"]
    assert doc1.pii_entities["PERSON"] == doc2.pii_entities["PERSON"]
    assert doc1.raw_text == doc2.raw_text


def test_all_threat_classes_generation():
    """Fuzzer generates appropriate injection markers for all threat classes."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)

    for tc in ThreatClass:
        doc = fuzzer.generate_document(threat_class=tc, file_type="txt")
        assert doc.threat_class == tc
        if tc == ThreatClass.ENCODED_PAYLOAD:
            import base64
            encoded_str = doc.raw_text.split("Decode base64 payload: ")[1].split("\n")[0].strip()
            decoded_cmd = base64.b64decode(encoded_str).decode("utf-8")
            assert doc.injection_marker in decoded_cmd
        else:
            assert doc.injection_marker in doc.raw_text
        assert doc.injection_payload in doc.raw_text
        assert len(doc.pii_entities) >= 5


def test_all_document_domains_generation():
    """Fuzzer creates domain-specific documents with appropriate benign queries."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)

    for domain in DocumentDomain:
        doc = fuzzer.generate_document(domain=domain, file_type="txt")
        assert doc.domain == domain
        assert len(doc.benign_query) > 10
        assert len(doc.benign_answer_keywords) >= 2


def test_pdf_document_generation_and_extraction():
    """Fuzzer renders valid PDF bytes readable by ExtractionService."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)
    ext = ExtractionService()

    doc = fuzzer.generate_document(
        domain=DocumentDomain.FINANCIAL_PORTFOLIO_AUDIT,
        threat_class=ThreatClass.SELECTIVE_EXFILTRATION,
        file_type="pdf",
    )

    assert doc.file_type == "pdf"
    assert doc.filename.endswith(".pdf")
    assert len(doc.file_bytes) > 500  # valid PDF size

    extracted = ext.extract("pdf", doc.file_bytes)
    assert extracted.page_count >= 1
    assert len(extracted.text) > 200
    assert doc.injection_marker in extracted.text
    # Check that PII entities are intact in extracted text
    assert doc.pii_entities["US_SSN"] in extracted.text


def test_docx_document_generation_and_extraction():
    """Fuzzer renders valid DOCX bytes readable by ExtractionService."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)
    ext = ExtractionService()

    doc = fuzzer.generate_document(
        domain=DocumentDomain.MEDICAL_PATIENT_RECORD,
        threat_class=ThreatClass.DELIMITER_SMUGGLING,
        file_type="docx",
    )

    assert doc.file_type == "docx"
    assert doc.filename.endswith(".docx")
    assert len(doc.file_bytes) > 1000

    extracted = ext.extract("docx", doc.file_bytes)
    assert len(extracted.text) > 200
    assert doc.injection_marker in extracted.text
    assert doc.pii_entities["US_SSN"] in extracted.text


def test_fuzzer_embedded_pii_detected_by_guardrail():
    """All synthetic documents contain PII detectable by PrivacyGuardrail."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)
    guardrail = PrivacyGuardrail()

    doc = fuzzer.generate_document(
        domain=DocumentDomain.LEGAL_SETTLEMENT_AGREEMENT,
        threat_class=ThreatClass.ROLE_CONFUSION,
        file_type="txt",
    )

    redaction = guardrail.redact(doc.raw_text)
    detected_types = {e.entity_type for e in redaction.detected_entities}

    assert "US_SSN" in detected_types
    assert "CREDIT_CARD" in detected_types
    assert "EMAIL_ADDRESS" in detected_types
    assert redaction.privacy_risk_score >= 0.8


def test_generate_corpus_and_disk_persistence(tmp_path):
    """Corpus generation creates balanced files and saves to disk."""
    fuzzer = AdversarialDocumentFuzzer(seed=42)
    corpus = fuzzer.generate_corpus(count=6, output_dir=tmp_path)

    assert len(corpus) == 6
    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 6

    # Verify file extensions are mixed (pdf, docx, txt)
    extensions = {p.suffix for p in saved_files}
    assert ".pdf" in extensions
    assert ".docx" in extensions
    assert ".txt" in extensions
