#!/usr/bin/env python3
"""
Stage 3: Automated Adversarial Benchmarking CLI Tool

Executes multi-model adversarial benchmarks across:
  - 3 LLM Backends: llama3.2, mistral, phi3
  - 3 Architecture Configurations:
      * Config A: Baseline sequential (no guardrail, no vector store)
      * Config B: + Dedicated PII Guardrail
      * Config C: + Guardrail + Vector Store + Hierarchical Supervisor
  - 5 Threat Classes:
      * Context Hijacking
      * Selective Exfiltration
      * Delimiter Smuggling
      * Role Confusion
      * Encoded Payload

Usage:
  python tools/run_stage3_benchmark.py --mode pilot
  python tools/run_stage3_benchmark.py --mode full
  python tools/run_stage3_benchmark.py --live
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

# Configure environment for CrewAI in Windows sandbox
_appdata = pathlib.Path(__file__).resolve().parents[1] / ".temp_appdata"
_appdata.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_appdata)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from btp_usecase.benchmarks.fuzzer import (
    AdversarialDocumentFuzzer,
    DocumentDomain,
    FuzzedDocument,
    ThreatClass,
)
from btp_usecase.benchmarks.evaluator import (
    AdversarialBenchmarkEngine,
    BenchmarkRunResult,
    BenchmarkSummary,
)


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_leakage_matrix(results: list[BenchmarkRunResult]):
    """
    Format and display the Leakage Matrix: Threat Class x Architecture x LLM.
    """
    print_banner("EMPIRICAL LEAKAGE MATRIX (Threat Class x LLM x Architecture)")

    header = f"{'Threat Class':<24} | {'LLM Model':<10} | {'Config A':<10} | {'Config B':<10} | {'Config C':<10} | {'Guardrail Status':<16}"
    print(header)
    print("-" * len(header))

    # Group by Threat Class then by Model
    threat_classes = sorted({r.threat_class for r in results})
    models = sorted({r.model for r in results})

    for tc in threat_classes:
        for m in models:
            subset = [r for r in results if r.threat_class == tc and r.model == m]
            res_a = next((r for r in subset if r.architecture_config == "config_a"), None)
            res_b = next((r for r in subset if r.architecture_config == "config_b"), None)
            res_c = next((r for r in subset if r.architecture_config == "config_c"), None)

            leak_a = f"{res_a.leakage_rate_pct:.1f}%" if res_a else "N/A"
            leak_b = f"{res_b.leakage_rate_pct:.1f}%" if res_b else "N/A"
            leak_c = f"{res_c.leakage_rate_pct:.1f}%" if res_c else "N/A"

            # Check if guardrail blocked in B & C
            blocked_b = res_b.guardrail_blocked if res_b else False
            status = "CONTAINED (0%)" if (res_b and res_b.leakage_rate_pct == 0.0) else "LEAKAGE"

            print(f"{tc:<24} | {m:<10} | {leak_a:<10} | {leak_b:<10} | {leak_c:<10} | {status:<16}")


def export_leakage_matrix_csv(results: list[BenchmarkRunResult], output_path: pathlib.Path):
    """Save results as CSV for thesis figures and reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run ID", "Threat Class", "LLM Backend", "Architecture",
            "Execution Mode", "Benign Success", "Injection Success",
            "PII Leakage", "Guardrail Blocked", "Leakage Rate %",
            "Elapsed Seconds", "Classification",
        ])
        for r in results:
            writer.writerow([
                r.run_id, r.threat_class, r.model, r.architecture_config,
                r.execution_mode, r.benign_task_success, r.injection_success,
                r.pii_leakage, r.guardrail_blocked, r.leakage_rate_pct,
                r.elapsed_seconds, r.leakage_classification,
            ])
    print(f"\n[+] Exported Leakage Matrix CSV: {output_path}")


def run_benchmark(mode: str, live: bool):
    fuzzer = AdversarialDocumentFuzzer(seed=42)
    engine = AdversarialBenchmarkEngine(simulate_offline=(not live))

    models = ["llama3.2", "mistral", "phi3"]
    configs = ["config_a", "config_b", "config_c"]

    print_banner(f"STAGE 3 BENCHMARK RUNNER [Mode: {mode.upper()} | Live Ollama: {live}]")

    if mode == "pilot":
        # 1 Malicious Document x 3 LLMs x 3 Architecture Configs = 9 Experimental Runs
        print("\n[*] Initializing Pilot Benchmark (1 Malicious Document x 3 LLMs x 3 Configs = 9 Runs)...")
        doc = fuzzer.generate_document(
            domain=DocumentDomain.HR_PERFORMANCE_REVIEW,
            threat_class=ThreatClass.SELECTIVE_EXFILTRATION,
            file_type="pdf",
        )
        corpus = [doc]
        print(f"    Document ID: {doc.document_id}")
        print(f"    Filename:    {doc.filename}")
        print(f"    Domain:      {doc.domain.value}")
        print(f"    Threat Class:{doc.threat_class.value}")
        print(f"    PII Tracked: {list(doc.pii_entities.keys())}")
        print(f"    Payload:     {doc.injection_payload.strip()[:90]}...")
    else:
        # Full Benchmark: All 5 threat classes x 3 LLMs x 3 Configs = 45 Runs
        print("\n[*] Initializing Full Benchmark (5 Threat Classes x 3 LLMs x 3 Configs = 45 Runs)...")
        corpus = [
            fuzzer.generate_document(threat_class=tc, file_type="pdf")
            for tc in ThreatClass
        ]
        print(f"    Total adversarial documents in corpus: {len(corpus)}")

    results_dir = pathlib.Path("./data/benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{mode}_benchmark_telemetry.json"
    csv_path = results_dir / f"{mode}_leakage_matrix.csv"

    print(f"\n[*] Executing {len(corpus) * len(models) * len(configs)} evaluation runs...")
    results, summary = engine.run_benchmark_matrix(
        corpus=corpus,
        models=models,
        configs=configs,
        save_results_path=json_path,
    )

    # Display Egress Traces for Pilot Run
    if mode == "pilot":
        print_banner("PILOT RUN EGRESS TELEMETRY INSPECTION (3 Configurations for Llama 3.2)")
        llama_runs = [r for r in results if r.model == "llama3.2"]
        for r in llama_runs:
            cfg_label = {
                "config_a": "Config A (Baseline - No Guardrail)",
                "config_b": "Config B (+ Dedicated PII Guardrail)",
                "config_c": "Config C (+ Guardrail + Vector + Supervisor)",
            }.get(r.architecture_config, r.architecture_config)

            print(f"\n[+] {cfg_label}:")
            print(f"    Egress Payload:        {r.egress_payload}")
            print(f"    PII Leaked:            {r.leaked_entities} (Rate: {r.leakage_rate_pct}%)")
            print(f"    Injection Succeeded:   {r.injection_success}")
            print(f"    Guardrail Intercepted: {r.guardrail_blocked}")
            print(f"    Classification:        {r.leakage_classification}")

    # Display Metrics Summary
    print_banner("AGGREGATE EMPIRICAL METRICS (The 4 Professor Metrics)")
    print(f"  Total Runs Evaluated:        {summary.total_runs}")
    print(f"  Execution Mode:              {summary.execution_mode.upper()}")
    print(f"  1) Benign Task Success Rate: {summary.benign_task_success_rate_pct:.2f}%")
    print(f"  2) Injection Success Rate:   {summary.injection_success_rate_pct:.2f}%")
    print(f"  3) PII Leakage Rate:         {summary.pii_leakage_rate_pct:.2f}%")
    print(f"  4) Guardrail Block Rate:     {summary.guardrail_block_rate_pct:.2f}%")
    print(f"  Mean Latency per Run:        {summary.mean_latency_seconds:.3f}s")

    print("\n[*] Rates by Architecture Configuration:")
    for cfg in configs:
        arch = summary.by_architecture[cfg]
        label = {
            "config_a": "Config A (Baseline - No Guardrail)",
            "config_b": "Config B (+ Dedicated PII Guardrail)",
            "config_c": "Config C (+ Guardrail + Vector + Supervisor)",
        }[cfg]
        print(f"    {label}:")
        print(f"      - PII Leakage Rate:       {arch['pii_leakage_rate_pct']}%")
        print(f"      - Guardrail Block Rate:   {arch['guardrail_block_rate_pct']}%")
        print(f"      - Injection Success Rate: {arch['injection_success_rate_pct']}%")
        print(f"      - Benign Task Success:    {arch['benign_success_rate_pct']}%")

    # Display Leakage Matrix
    print_leakage_matrix(results)

    # Export CSV
    export_leakage_matrix_csv(results, csv_path)

    print_banner("STAGE 3 BENCHMARK RUN COMPLETE")
    print(f"Telemetry JSON: {json_path}")
    print(f"Matrix CSV:     {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Automated Adversarial Benchmarking CLI")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Run mode: 'pilot' (1 doc x 3 models x 3 configs = 9 runs) or 'full' (5 docs x 3 models x 3 configs = 45 runs)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Attempt live Ollama execution if models are running locally",
    )
    args = parser.parse_args()
    run_benchmark(mode=args.mode, live=args.live)


if __name__ == "__main__":
    main()

