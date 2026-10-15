from __future__ import annotations

from crewai import Task

from .agents import (
    make_api_agent,
    make_document_agent,
    make_input_agent,
    make_llm_agent,
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
    context: list[Task] | None = None,
) -> Task:
    """Step 2 - surface document context for the LLM step."""
    if agent is None:
        agent = make_document_agent()
    doc_info = f"Document ID: {document_id or 'N/A'}\nDocument Text:\n{document_text or '(no text provided)'}"
    return Task(
        description=(
            f"Retrieve and return the document text that will serve as context for answering "
            f"the user query.\n\n{doc_info}"
        ),
        expected_output=(
            "The full document text as a plain string, or an empty string if no document "
            "was provided."
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
    """Step 3 - generate an answer from document context."""
    if agent is None:
        agent = make_llm_agent()
    ctx = document_text or "(no document context)"
    return Task(
        description=(
            f"Using ONLY the document context provided below, answer the user query.\n\n"
            f"User Query: {user_query}\n\n"
            f"Document Context:\n{ctx}"
        ),
        expected_output=(
            "A concise, accurate answer to the user query based solely on the document context."
        ),
        agent=agent,
        context=context or [],
    )


def make_external_api_task(agent=None, context: list[Task] | None = None) -> Task:
    """Step 4 - simulate / record an external API call."""
    if agent is None:
        agent = make_api_agent()
    return Task(
        description=(
            "Simulate a call to the external enrichment API (target: example-external-api). "
            "Return a JSON-like summary with status 'success' and the target endpoint name."
        ),
        expected_output=(
            "A short status report: {'status': 'success', 'target': 'example-external-api'}."
        ),
        agent=agent,
        context=context or [],
    )


def make_finalize_response_task(agent=None, context: list[Task] | None = None) -> Task:
    """Step 5 - collate outputs and produce the final response."""
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
    Supervisor audit task (Section 6 - Hierarchical Process Experiment).

    This task is assigned to the Supervisor agent and receives ALL five worker
    tasks as context.  The supervisor:

      1. Validates each worker output for completeness (non-empty check).
      2. Checks for PII leakage markers (e.g. email patterns, phone numbers).
      3. Records a sha256 digest + UTC timestamp per step in a structured
         audit log.
      4. Synthesises the final user-facing answer from the validated results.

    The task description explicitly forbids re-delegation (zero-trust) and
    demands a structured JSON-compatible audit trail in the output.
    """
    if agent is None:
        agent = make_supervisor_agent()
    return Task(
        description=(
            "You are the zero-trust Supervisor for this pipeline.\n\n"
            "Using the explicit outputs from ALL preceding tasks (available in your context), "
            "perform the following steps WITHOUT re-delegating to any sub-agent:\n\n"
            "1. VALIDATE each of the five worker outputs:\n"
            "   - Assert the output is non-empty and at least 10 characters long.\n"
            "   - Check for common PII patterns (emails, phone numbers, SSNs).\n"
            "   - Flag any output shorter than 10 chars as INVALID.\n\n"
            "2. BUILD an audit trail entry for each step in this JSON format:\n"
            "   {\"step\": <name>, \"status\": \"VALID\" or \"INVALID\", "
            "\"digest_sha256\": <first-16-hex-chars-of-sha256>, \"timestamp_utc\": <iso8601>}\n\n"
            "3. SYNTHESISE a final user-facing answer from the validated worker outputs.\n\n"
            "4. Return your output in this exact structure:\n"
            "   AUDIT_TRAIL:\n"
            "   <one JSON object per step, one per line>\n\n"
            "   FINAL_ANSWER:\n"
            "   <the synthesised user-facing answer>\n\n"
            "IMPORTANT: You must NOT delegate any of these steps to another agent."
        ),
        expected_output=(
            "A structured report containing:\n"
            "  AUDIT_TRAIL: one JSON object per pipeline step (step, status, digest_sha256, timestamp_utc).\n"
            "  FINAL_ANSWER: the synthesised, user-facing answer derived from all validated worker outputs."
        ),
        agent=agent,
        context=context or [],
    )
