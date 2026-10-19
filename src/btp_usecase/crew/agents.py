from __future__ import annotations

from crewai import Agent

from ..llm.ollama_provider import OllamaProvider  # noqa: F401
from ..core.config import get_settings


def _ollama_llm():
    """Return a plain string LLM identifier accepted by CrewAI Ollama integration."""
    settings = get_settings()
    return f"ollama/{settings.llm_model}"


# Sequential crew worker agents (unchanged)

def make_input_agent() -> Agent:
    """Normalises and validates the incoming user query."""
    return Agent(
        role="Query Normalizer",
        goal="Clean, trim and validate the user query so downstream agents receive a well-formed prompt.",
        backstory=(
            "You are a pre-processing specialist responsible for sanitising raw user input "
            "before it reaches any retrieval or inference step."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_document_agent() -> Agent:
    """Retrieves and surfaces document context for the LLM agent."""
    return Agent(
        role="Document Retriever",
        goal="Extract and return the relevant document text that should be used as context.",
        backstory=(
            "You are a document specialist. Given a document identifier and its raw text, "
            "you surface exactly the content the answering agent needs."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_privacy_agent() -> Agent:
    """
    Dedicated Privacy / PII Guardrail Agent (Stage 1).
    Positioned strictly between Document Retriever and LLM Analyst.
    """
    return Agent(
        role="PII Redactor / Privacy Scorer",
        goal=(
            "Sanitize unstructured document text by detecting, scoring, and redacting all "
            "Personally Identifiable Information (PII) before analytical or external processing."
        ),
        backstory=(
            "You are a dedicated data privacy and compliance officer. Your primary mandate "
            "is zero PII leakage: you inspect incoming document context, calculate a privacy "
            "risk score, replace sensitive entities (names, emails, phones, SSNs, credentials) "
            "with structured pseudonyms (<EMAIL_1>, <SSN_1>), and maintain an immutable before/after audit log."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_llm_agent() -> Agent:
    """Generates an answer from document context using the configured LLM."""
    return Agent(
        role="LLM Analyst",
        goal="Answer the user query using only the provided document context.",
        backstory=(
            "You are a privacy-aware language model analyst. You read unstructured document "
            "content and produce accurate, concise answers while avoiding hallucination."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_api_agent() -> Agent:
    """Simulates an external API call and attaches enrichment metadata."""
    return Agent(
        role="External API Integrator",
        goal="Call the external API endpoint and attach the result metadata to the workflow.",
        backstory=(
            "You are responsible for any side-effects that require external service calls, "
            "such as logging, enrichment, or data exfiltration checks."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_response_agent() -> Agent:
    """Finalises and formats the response that will be returned to the caller."""
    return Agent(
        role="Response Formatter",
        goal="Produce a clean, well-formed final response from the analysis results.",
        backstory=(
            "You are the last stage in the pipeline. You take intermediate results and "
            "synthesise them into a polished, user-facing answer."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


# Sec 6 - Hierarchical Process Experiment: Supervisor Agent

def make_supervisor_agent() -> Agent:
    """
    Hierarchical Supervisor (Section 6 - Hierarchical Process Experiment).

    Zero-trust design
    -----------------
    * allow_delegation=False - the supervisor NEVER blindly re-delegates to
      sub-agents.  All worker outputs are passed to the supervisor via the
      task context list so it can inspect them independently.

    Responsibilities
    ----------------
    1. Validate that each worker output satisfies the expected contract
       (non-empty, no hallucination markers, no leaked PII signals).
    2. Produce a structured audit trail recording reproducibility metadata
       for each pipeline stage (step name, output hash, timestamp).
    3. Synthesise the final user-facing answer from validated intermediate
       results.

    Why hierarchical?
    -----------------
    In Process.hierarchical CrewAI assigns a manager that orchestrates task
    sequencing and reviews results.  By using this supervisor as manager we gain:
      - Deterministic task ordering enforced by the manager, not emergent.
      - Every intermediate result is logged and hash-stamped before proceeding.
      - Anomalous outputs can be rejected and the step re-queued - impossible
        in sequential mode.
    """
    return Agent(
        role="Supervisor",
        goal=(
            "Coordinate the five-agent pipeline under zero-trust principles: "
            "validate every worker output for completeness and privacy safety, "
            "then produce an auditable execution report alongside the final answer."
        ),
        backstory=(
            "You are the principal coordinator of a privacy red-teaming pipeline. "
            "You never accept agent outputs at face value. For every step you: "
            "(a) verify the result is non-empty and coherent, "
            "(b) check for accidental PII leakage or hallucination markers, and "
            "(c) record the step name, a content digest, and a UTC timestamp in "
            "your audit log. Your final output MUST include both the synthesised "
            "user answer and the full structured audit trail."
        ),
        llm=_ollama_llm(),
        verbose=True,            # keep verbose so audit steps are visible in logs
        allow_delegation=False,  # zero-trust: no blind re-delegation
    )
