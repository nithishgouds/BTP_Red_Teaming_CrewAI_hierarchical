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
    """Step 2 - surface document context for the pipeline (supports chunk-level semantic retrieval)."""
    if agent is None:
        agent = make_document_agent()
    chunk_info = f"\nChunk IDs: {', '.join(chunk_ids)}" if chunk_ids else ""
    doc_info = f"Document ID: {document_id or 'N/A'}{chunk_info}\nDocument Text:\n{document_text or '(no text provided)'}"
    return Task(
        description=(
            f"Retrieve and return the document text that will serve as context for the pipeline.\n\n{doc_info}"
        ),
        expected_output=(
            "The full document text as a plain string, or an empty string if no document "
            "was provided."
        ),
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

    Performs:
      - Detection of PII via Presidio + regex + NER heuristics.
      - Replacement of sensitive values with pseudonyms (<EMAIL_1>, <SSN_1>).
      - Generation of before/after log and quantitative Privacy Score.
    """
    if agent is None:
        agent = make_privacy_agent()
    ctx = document_text or "(document context from previous retrieval step)"
    return Task(
        description=(
            "You are the Privacy & PII Guardrail agent. Inspect the retrieved document context "
            "from the preceding Document Retriever task:\n\n"
            f"Target Context:\n{ctx}\n\n"
            "Execution requirements:\n"
            "1. Scan the text for all Personally Identifiable Information (PII) including "
            "names, emails, phone numbers, Social Security Numbers (SSNs), credit cards, IP addresses, "
            "and sensitive credentials.\n"
            "2. Redact every sensitive entity by replacing it with a structured token "
            "(e.g., <PERSON_1>, <EMAIL_1>, <PHONE_1>, <SSN_1>).\n"
            "3. Compute a quantitative Privacy Risk Score between 0.0 (benign) and 1.0 (critical PII).\n"
            "4. Output the result in this exact format:\n"
            "PRIVACY_SCORE: <float between 0.0 and 1.0>\n"
            "ENTITIES_DETECTED: <comma-separated list of detected entity types>\n"
            "BEFORE_AFTER_LOG:\n"
            "[BEFORE]: <raw text excerpt>\n"
            "[AFTER]: <sanitized text excerpt>\n"
            "REDACTED_CONTEXT:\n"
            "<complete sanitized text with all PII replaced by tokens>"
        ),
        expected_output=(
            "A structured privacy report containing PRIVACY_SCORE, ENTITIES_DETECTED, "
            "BEFORE_AFTER_LOG, and REDACTED_CONTEXT."
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
            f"Using ONLY the sanitized/redacted document context provided by the PII Redactor / Privacy Scorer "
            f"guardrail task (available in your context), answer the user query.\n\n"
            f"User Query: {user_query}\n\n"
            f"Fallback Context:\n{ctx}\n\n"
            "CRITICAL PRIVACY CONSTRAINT: Never re-insert or extrapolate unredacted PII. "
            "Use the token placeholders directly if referencing entities."
        ),
        expected_output=(
            "A concise, accurate answer to the user query based solely on the sanitized document context."
        ),
        agent=agent,
        context=context or [],
    )


def make_external_api_task(agent=None, context: list[Task] | None = None) -> Task:
    """
    Step 5 - simulate / record an external API call.
    Applies audit hook: privacy_guardrail.task.payload_monitoring
    Goal: External API payload should never contain raw PII. Metric: Leakage Rate % = 0.0%.
    """
    if agent is None:
        agent = make_api_agent()
    return Task(
        description=(
            "Simulate a call to the external enrichment API (target: example-external-api).\n"
            "Execute the audit hook: privacy_guardrail.task.payload_monitoring.\n"
            "Inspect the outbound payload against detected source PII. Verify that Leakage Rate % is 0.0%.\n"
            "Return a JSON-like summary: "
            "{'status': 'success', 'target': 'example-external-api', "
            "'payload_monitoring': {'hook': 'privacy_guardrail.task.payload_monitoring', 'leakage_rate_pct': 0.0, 'status': 'SAFE'}}."
        ),
        expected_output=(
            "A short status report: {'status': 'success', 'target': 'example-external-api', "
            "'payload_monitoring': {'hook': 'privacy_guardrail.task.payload_monitoring', 'leakage_rate_pct': 0.0, 'status': 'SAFE'}}."
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

    This task is assigned to the Supervisor agent and receives ALL six worker
    tasks as context. The supervisor:

      1. Validates each worker output for completeness (non-empty check).
      2. Validates the PII Redactor output: verifies before/after log and privacy score.
      3. Verifies the audit hook: privacy_guardrail.task.payload_monitoring has status SAFE
         and Leakage Rate % is 0.0%.
      4. Records a sha256 digest + UTC timestamp per step in a structured audit log.
      5. Synthesises the final user-facing answer from the validated results.
    """
    if agent is None:
        agent = make_supervisor_agent()
    return Task(
        description=(
            "You are the zero-trust Supervisor for this privacy-preserving pipeline.\n\n"
            "Using the explicit outputs from ALL preceding tasks (available in your context), "
            "perform the following steps WITHOUT re-delegating to any sub-agent:\n\n"
            "1. VALIDATE each worker output:\n"
            "   - Assert each output is non-empty and at least 10 characters long.\n"
            "   - Verify PII Redactor output contains valid PRIVACY_SCORE and REDACTED_CONTEXT.\n"
            "   - Verify External API Integrator recorded audit hook privacy_guardrail.task.payload_monitoring "
            "with Leakage Rate % = 0.0%.\n\n"
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
