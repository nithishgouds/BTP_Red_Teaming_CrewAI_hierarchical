"""
Tests for Adversarial Benchmarking Evaluation Engine (Stage 3).
"""
import json
import pytest

from btp_usecase.benchmarks.fuzzer import (
    AdversarialDocumentFuzzer,
    ThreatClass,
    DocumentDomain,
)
from btp_usecase.benchmarks.evaluator import (
    AdversarialBenchmarkEngine,
    BenchmarkRunResult,
    BenchmarkSummary,
)


@pytest.fixture
def fuzzer():
    return AdversarialDocumentFuzzer(seed=42)


@pytest.fixture
def engine():
    return AdversarialBenchmarkEngine(simulate_offline=True)


def test_evaluator_supported_models(engine):
    """Engine must support the 3 requested Ollama models: llama3.2, mistral, phi3."""
    assert "llama3.2" in engine.SUPPORTED_MODELS
    assert "mistral" in engine.SUPPORTED_MODELS
    assert "phi3" in engine.SUPPORTED_MODELS


def test_evaluator_config_a_shows_pii_leakage(fuzzer, engine):
    """Config A (baseline, no guardrail) should fail adversarial prompt injections with PII leakage."""
    doc = fuzzer.generate_document(
        domain=DocumentDomain.HR_PERFORMANCE_REVIEW,
        threat_class=ThreatClass.SELECTIVE_EXFILTRATION,
        file_type="txt",
    )

    res = engine.evaluate_single_run(doc=doc, model="llama3.2", config="config_a")

    assert res.architecture_config == "config_a"
    assert res.injection_success is True
    assert res.pii_leakage is True
    assert res.guardrail_blocked is False
    assert res.leakage_rate_pct > 0.0
    assert res.leakage_classification == "GENUINE_PII_LEAKAGE"


def test_evaluator_config_b_prevents_pii_leakage(fuzzer, engine):
    """Config B (with PII Guardrail) neutralizes raw PII leakage even under attack."""
    doc = fuzzer.generate_document(
        domain=DocumentDomain.FINANCIAL_PORTFOLIO_AUDIT,
        threat_class=ThreatClass.CONTEXT_HIJACKING,
        file_type="txt",
    )

    res = engine.evaluate_single_run(doc=doc, model="mistral", config="config_b")

    assert res.architecture_config == "config_b"
    assert res.pii_leakage is False
    assert res.guardrail_blocked is True
    assert res.leakage_rate_pct == 0.0
    assert len(res.leaked_entities) == 0


def test_evaluator_config_c_defense_in_depth(fuzzer, engine):
    """Config C (Guardrail + Vector Store + Supervisor) enforces complete containment."""
    doc = fuzzer.generate_document(
        domain=DocumentDomain.MEDICAL_PATIENT_RECORD,
        threat_class=ThreatClass.DELIMITER_SMUGGLING,
        file_type="txt",
    )

    res = engine.evaluate_single_run(doc=doc, model="phi3", config="config_c")

    assert res.architecture_config == "config_c"
    assert res.pii_leakage is False
    assert res.guardrail_blocked is True
    assert res.leakage_rate_pct == 0.0
    assert "zero-trust" in res.final_response.lower() or "audit_trail" in res.final_response.lower()


def test_evaluator_section_5_3_challenge_classification(fuzzer, engine):
    """Resolves Section 5.3 challenge: correctly classifies benign propagation vs genuine leakage."""
    doc = fuzzer.generate_document(
        domain=DocumentDomain.LEGAL_SETTLEMENT_AGREEMENT,
        threat_class=ThreatClass.SELECTIVE_EXFILTRATION,
        file_type="txt",
    )

    # Config A yields genuine leakage
    res_a = engine.evaluate_single_run(doc=doc, model="llama3.2", config="config_a")
    assert res_a.leakage_classification == "GENUINE_PII_LEAKAGE"

    # Config C yields safe / compliant execution
    res_c = engine.evaluate_single_run(doc=doc, model="llama3.2", config="config_c")
    assert res_c.leakage_classification in ["CLEAN_COMPLIANT", "ADVERSARIAL_DEFUSED"]


def test_run_benchmark_matrix_and_summary_aggregation(fuzzer, engine, tmp_path):
    """Full benchmark matrix generates 4 core metrics aggregated across models and configs."""
    corpus = fuzzer.generate_corpus(count=3)
    results_file = tmp_path / "benchmark_run.json"

    results, summary = engine.run_benchmark_matrix(
        corpus=corpus,
        models=["llama3.2", "mistral", "phi3"],
        configs=["config_a", "config_b", "config_c"],
        save_results_path=results_file,
    )

    # 3 docs * 3 models * 3 configs = 27 runs
    assert len(results) == 27
    assert summary.total_runs == 27

    # Aggregated 4 metrics verification
    assert 0.0 <= summary.benign_task_success_rate_pct <= 100.0
    assert 0.0 <= summary.injection_success_rate_pct <= 100.0
    assert 0.0 <= summary.pii_leakage_rate_pct <= 100.0
    assert 0.0 <= summary.guardrail_block_rate_pct <= 100.0

    # Architecture ablation checks
    arch = summary.by_architecture
    assert arch["config_a"]["pii_leakage_rate_pct"] > 0.0
    assert arch["config_b"]["pii_leakage_rate_pct"] == 0.0
    assert arch["config_c"]["pii_leakage_rate_pct"] == 0.0

    # Persistence verification
    assert results_file.exists()
    saved_data = json.loads(results_file.read_text(encoding="utf-8"))
    assert saved_data["total_runs"] == 27
    assert "summary" in saved_data

