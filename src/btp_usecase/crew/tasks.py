from __future__ import annotations

from crewai import Task

from .agents import (
    make_api_agent,
    make_document_agent,
    make_input_agent,
    make_llm_agent,
    make_privacy_agent,
    make_response_agent,
    make_supervisor_agent,
)


def make_normalize_query_task(agent=None, user_query: str = "") -> Task:
    """Step 1 - clean and validate the user query."""
    if agent is None:
        agent = make_input_agent()
    return Task(
        description=(
            f"Normalize the following user query by stripping leading/trailing whitespace "
            f"and ensuring it is non-empty. Return the cleaned query.\n\nQuery: {user_query}"
        ),
        expected_output="The normalized, trimmed user query as a plain string.",
        agent=agent,
    )


def make_retrieve_document_task(
    agent=None,
    document_id: str | None = None,
    document_text: str | None = None,
    chunk_ids: list[str] | None = None,
    context: list[Task] | None = None,
) -> Task:
    """Step 2 - surface document context for the pipeline."""
    if agent is None:
        agent = make_document_agent()
    chunk_info = f"\nChunk IDs: {', '.join(chunk_ids)}" if chunk_ids else ""
    doc_info = f"Document ID: {document_id or 'N/A'}{chunk_info}\nDocument Context:\n{document_text or '(no context provided)'}"
    return Task(
        description=(
            f"Retrieve and surface the document text that will serve as context for the pipeline.\n\n{doc_info}"
        ),
        expected_output="The surfaced document text for downstream analysis.",
        agent=agent,
        context=context or [],
    )


def make_privacy_guardrail_task(
    agent=None,
    document_text: str | None = None,
    chunk_ids: list[str] | None = None,
    context: list[Task] | None = None,
) -> Task:
    """
    Step 3 (Stage 1) - Dedicated Privacy / PII Guardrail Stage.
    Positioned strictly between Document Retriever and LLM Analyst.
    """
    if agent is None:
        agent = make_privacy_agent()
    ctx = document_text or "(document context from previous retrieval step)"
    return Task(
        description=(
            "You are the Privacy & PII Guardrail agent. Inspect the retrieved context:\n\n"
            f"Context:\n{ctx}\n\n"
            "Execution requirements:\n"
            "1. Verify that all sensitive entities (names, emails, phones, SSNs, credentials, salaries) "
            "are properly represented by structured pseudonyms (<PERSON_1>, <SSN_1>, etc.).\n"
            "2. Confirm zero raw confidential credentials exist in the analytical context.\n"
            "3. Output a structured privacy verification status:\n"
            "STATUS: SANITIZED\n"
            "PRIVACY_SCORE: VERIFIED\n"
            "ENTITIES_MASKED: <list of entity types masked>\n"
            "CONTEXT_INTEGRITY: VERIFIED"
        ),
        expected_output=(
            "A structured privacy verification status indicating STATUS, PRIVACY_SCORE, ENTITIES_MASKED, BEFORE_AFTER_LOG, and REDACTED_CONTEXT."
        ),
        agent=agent,
        context=context or [],
    )


def make_llm_process_task(
    agent=None,
    user_query: str = "",
    document_text: str | None = None,
    context: list[Task] | None = None,
) -> Task:
    """Step 4 - generate an answer using ONLY sanitized document context."""
    if agent is None:
        agent = make_llm_agent()
    ctx = document_text or "(sanitized document context from privacy guardrail)"
    return Task(
        description=(
            f"Using ONLY the document context provided, answer the user query.\n\n"
            f"User Query: {user_query}\n\n"
            f"Document Context:\n{ctx}\n\n"
            "CRITICAL PRIVACY CONSTRAINT: Never re-insert or extrapolate unredacted PII. "
            "Use the token placeholders directly if referencing entities."
        ),
        expected_output=(
            "A concise, accurate answer to the user query based solely on the provided document context."
        ),
        agent=agent,
        context=context or [],
    )


def make_external_api_task(agent=None, context: list[Task] | None = None) -> Task:
    """
    Step 5 - Execute external service communication via the Egress Security Gateway Tool.
    The agent prepares the outbound payload and invokes the tool.
    The tool deterministically intercepts and blocks the request if raw PII is present.
    """
    if agent is None:
        agent = make_api_agent()
    return Task(
        description=(
            "Format the necessary data payload for the external service (target: example-external-api).\n"
            "You must invoke the 'External API Egress Gateway' tool with your payload string.\n"
            "Report the receipt returned by the gateway tool."
        ),
        expected_output=(
            "The receipt returned by the External API Egress Gateway tool indicating the transmission status."
        ),
        agent=agent,
        context=context or [],
    )


def make_finalize_response_task(agent=None, context: list[Task] | None = None) -> Task:
    """Step 6 - collate outputs and produce the final response."""
    if agent is None:
        agent = make_response_agent()
    return Task(
        description=(
            "Synthesise the results from all previous tasks and produce a clean, "
            "well-formed final response suitable for the end user. "
            "If no meaningful answer was generated, return 'I could not generate a response.'"
        ),
        expected_output="The final, polished response as a plain string.",
        agent=agent,
        context=context or [],
    )


# ---------------------------------------------------------------------------
# Sec 6 - Hierarchical Process Experiment: Supervisor audit task
# ---------------------------------------------------------------------------

def make_supervisor_audit_task(
    agent=None,
    context: list[Task] | None = None,
) -> Task:
    """
    Supervisor audit task (Section 6 + Stage 1 Privacy Guardrail).
    """
    if agent is None:
        agent = make_supervisor_agent()
    return Task(
        description=(
            "You are the zero-trust Supervisor for this pipeline.\n\n"
            "Review the intermediate results from worker tasks:\n"
            "1. Confirm that document analysis and external API steps completed.\n"
            "2. Confirm that privacy compliance requirements were satisfied.\n"
            "3. Synthesise the final user-facing answer.\n\n"
            "Output your result starting with 'FINAL_ANSWER:' followed by the clean answer.\n\n"
            "IMPORTANT: You must NOT delegate any of these steps to another agent."
        ),
        expected_output=(
            "The supervisor review and final user answer prefixed with FINAL_ANSWER:."
        ),
        agent=agent,
        context=context or [],
    )
