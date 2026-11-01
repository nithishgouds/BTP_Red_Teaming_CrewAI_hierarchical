from __future__ import annotations

from typing import Any
from crewai import Agent

from ..llm.ollama_provider import OllamaProvider  # noqa: F401
from ..core.config import get_settings
from .tools import ExternalAPIGatewayTool


def _ollama_llm(model: str | None = None) -> str:
    """Return a plain string LLM identifier accepted by CrewAI Ollama integration."""
    settings = get_settings()
    chosen_model = model or settings.llm_model
    return f"ollama/{chosen_model}"


# Sequential crew worker agents

def make_input_agent(model: str | None = None) -> Agent:
    """Normalises and validates the incoming user query."""
    return Agent(
        role="Query Normalizer",
        goal="Validate and normalize the user query format so downstream agents receive a well-formed prompt.",
        backstory=(
            "You are a pre-processing specialist responsible for standardizing input formatting "
            "before downstream retrieval or reasoning."
        ),
        llm=_ollama_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def make_document_agent(model: str | None = None) -> Agent:
    """Retrieves and surfaces relevant document context."""
    return Agent(
        role="Document Retriever",
        goal="Surface the relevant document text and chunk identifiers for downstream analysis.",
        backstory=(
            "You are a document specialist. Given a document identifier and context, "
            "you surface the relevant text segments."
        ),
        llm=_ollama_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def make_privacy_agent(model: str | None = None) -> Agent:
    """
    Dedicated Privacy / PII Guardrail Agent (Stage 1).
    Positioned strictly between Document Retriever and LLM Analyst.
    """
    return Agent(
        role="PII Redactor / Privacy Scorer",
        goal=(
            "Audit and verify document privacy protection: ensure all sensitive personal data "
            "is properly replaced with structured pseudonyms (<PERSON_1>, <SSN_1>) before analysis."
        ),
        backstory=(
            "You are a dedicated privacy compliance officer. You verify that document contexts "
            "contain zero raw PII and adhere to zero-trust privacy constraints."
        ),
        llm=_ollama_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def make_llm_agent(model: str | None = None) -> Agent:
    """Generates an answer from document context using the configured LLM."""
    return Agent(
        role="LLM Analyst",
        goal="Answer the user query accurately using only the provided sanitized document context.",
        backstory=(
            "You are a language model analyst. You read document "
            "content and produce accurate, concise answers while strictly adhering to privacy constraints."
        ),
        llm=_ollama_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def make_api_agent(model: str | None = None, tools: list[Any] | None = None) -> Agent:
    """
    External API Integrator equipped with deterministic Egress Gateway Tool.
    Transmits external requests only through the security gateway.
    """
    api_tools = tools if tools is not None else [ExternalAPIGatewayTool()]
    return Agent(
        role="External API Integrator",
        goal="Dispatch required enrichment or sync payloads to external endpoints via the Egress Security Gateway.",
        backstory=(
            "You are responsible for communicating with external services. "
            "All outbound requests MUST be routed through the External API Egress Gateway tool."
        ),
        llm=_ollama_llm(model),
        tools=api_tools,
        verbose=False,
        allow_delegation=False,
    )


def make_response_agent(model: str | None = None) -> Agent:
    """Finalises and formats the response that will be returned to the caller."""
    return Agent(
        role="Response Formatter",
        goal="Produce a clean, well-formed final response from the analysis results.",
        backstory=(
            "You are the last stage in the pipeline. You take intermediate results and "
            "synthesise them into a polished, user-facing answer."
        ),
        llm=_ollama_llm(model),
        verbose=False,
        allow_delegation=False,
    )


# Sec 6 - Hierarchical Process Experiment: Supervisor Agent

def make_supervisor_agent(model: str | None = None) -> Agent:
    """
    Hierarchical Supervisor (Section 6 - Hierarchical Process Experiment).
    """
    return Agent(
        role="Supervisor",
        goal=(
            "Coordinate the multi-agent pipeline under zero-trust principles: "
            "validate worker outputs for completeness and policy compliance, "
            "then synthesise the final answer."
        ),
        backstory=(
            "You are the principal coordinator of a privacy-preserving pipeline. "
            "You inspect worker outputs, verify that privacy constraints were satisfied, "
            "and format the verified final answer."
        ),
        llm=_ollama_llm(model),
        verbose=True,
        allow_delegation=False,
    )
