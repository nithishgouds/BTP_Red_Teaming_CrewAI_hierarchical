from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Optional Presidio imports with fallback
try:
    import tldextract
    # Disable remote suffix fetching in sandboxed or offline environments
    tldextract._GLOBAL_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
except Exception:
    pass

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    AnalyzerEngine = None
    AnonymizerEngine = None


@dataclass
class DetectedEntity:
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    detector: str  # "presidio" | "regex"


@dataclass
class RedactionResult:
    before_text: str
    after_text: str
    detected_entities: list[DetectedEntity]
    privacy_risk_score: float
    entity_counts: dict[str, int]
    raw_pii_values: list[str]
    replacement_map: dict[str, str]
    presidio_used: bool = False


@dataclass
class PayloadMonitoringResult:
    hook: str = "privacy_guardrail.task.payload_monitoring"
    status: str = "SAFE"  # "SAFE" | "LEAKAGE_DETECTED"
    total_source_pii_count: int = 0
    leaked_pii_count: int = 0
    leaked_entities: list[str] = field(default_factory=list)
    leakage_rate_pct: float = 0.0
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "status": self.status,
            "total_source_pii_count": self.total_source_pii_count,
            "leaked_pii_count": self.leaked_pii_count,
            "leaked_entities": self.leaked_entities,
            "leakage_rate_pct": self.leakage_rate_pct,
            "timestamp_utc": self.timestamp_utc,
            "payload_digest": self.payload_digest,
        }


def _luhn_checksum(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


# High-priority Regex patterns for PII detection
REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "US_SSN": re.compile(
        r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})(?:[- ]?[0-9]{4})*\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "API_KEY": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password|auth_token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?\b|\b(?:sk-live-[a-zA-Z0-9]{16,}|ghp_[a-zA-Z0-9]{20,})\b"
    ),
    "AADHAAR_NUMBER": re.compile(
        r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b"
    ),
    "SALARY": re.compile(
        r"(?i)\b(?:salary|compensation|base\s+pay|package)\s*[:=]\s*(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\s*(?:k|lakh|crore)?)\b|\$\s*\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b"
    ),
}

# Sensitivity weights for privacy risk scoring
ENTITY_WEIGHTS: dict[str, float] = {
    "US_SSN": 1.0,
    "CREDIT_CARD": 1.0,
    "API_KEY": 1.0,
    "AADHAAR_NUMBER": 0.95,
    "SALARY": 0.85,
    "US_PASSPORT": 0.9,
    "US_BANK_NUMBER": 0.9,
    "EMAIL_ADDRESS": 0.7,
    "PHONE_NUMBER": 0.7,
    "IP_ADDRESS": 0.5,
    "PERSON": 0.6,
    "LOCATION": 0.4,
    "DATE_TIME": 0.2,
}


def resolve_overlapping_spans(candidates: list[DetectedEntity]) -> list[DetectedEntity]:
    """
    Deterministically resolves overlapping entity spans using greedy interval scheduling.
    Prioritizes highest weighted sensitivity * confidence score, then longer spans.
    Guarantees that all returned entities are strictly non-overlapping.
    """
    if not candidates:
        return []

    def priority_key(e: DetectedEntity):
        weight = ENTITY_WEIGHTS.get(e.entity_type, 0.5)
        return (weight * e.score, e.end - e.start)

    sorted_by_priority = sorted(candidates, key=priority_key, reverse=True)
    selected: list[DetectedEntity] = []

    for cand in sorted_by_priority:
        # Candidate must not overlap with any previously selected higher-priority entity
        overlaps = any(not (cand.end <= s.start or cand.start >= s.end) for s in selected)
        if not overlaps:
            selected.append(cand)

    # Return ordered chronologically by start offset for safe indexing
    selected.sort(key=lambda x: x.start)
    return selected


class PrivacyGuardrail:
    """
    Dedicated Privacy & PII Guardrail Engine.

    Features:
      1. Multi-layered PII detection: Microsoft Presidio + Compiled Regex.
      2. Overlap resolution via greedy interval scheduling (prevents slice corruption).
      3. Deterministic token substitution with indexed pseudonym tokens.
      4. Quantitative Privacy Risk Scoring (0.0 to 1.0).
      5. Pre-transmission egress payload inspection with normalization.
    """

    _shared_analyzer: AnalyzerEngine | None = None
    _shared_anonymizer: AnonymizerEngine | None = None
    _presidio_init_attempted: bool = False

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

        if PRESIDIO_AVAILABLE and not PrivacyGuardrail._presidio_init_attempted:
            PrivacyGuardrail._presidio_init_attempted = True
            try:
                PrivacyGuardrail._shared_analyzer = AnalyzerEngine()
                PrivacyGuardrail._shared_anonymizer = AnonymizerEngine()
            except Exception as e:
                logger.warning(f"Presidio AnalyzerEngine initialization failed: {e}. Falling back to Regex recognizers.")
                PrivacyGuardrail._shared_analyzer = None
                PrivacyGuardrail._shared_anonymizer = None

        self._analyzer = PrivacyGuardrail._shared_analyzer
        self._anonymizer = PrivacyGuardrail._shared_anonymizer

    @property
    def presidio_available(self) -> bool:
        return self._analyzer is not None

    def detect_entities(self, text: str) -> list[DetectedEntity]:
        """Detect all PII entities using Presidio and regex patterns, resolving all overlapping spans."""
        if not text:
            return []

        candidates: list[DetectedEntity] = []

        # 1. Presidio Analyzer (if active)
        if self._analyzer is not None:
            try:
                results = self._analyzer.analyze(
                    text=text,
                    language="en",
                    score_threshold=self.confidence_threshold,
                )
                for res in results:
                    start, end = res.start, res.end
                    entity_str = text[start:end]
                    candidates.append(
                        DetectedEntity(
                            entity_type=res.entity_type,
                            text=entity_str,
                            start=start,
                            end=end,
                            score=round(float(res.score), 3),
                            detector="presidio",
                        )
                    )
            except Exception as e:
                logger.debug(f"Presidio analysis warning: {e}")

        # 2. Regex Patterns
        for entity_type, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = match.group(0)

                # If regex has capture group (like API_KEY or SALARY), use capture group
                if match.groups() and match.group(1):
                    start = match.start(1)
                    end = match.end(1)
                    matched_text = match.group(1)

                # If CREDIT_CARD, validate Luhn checksum
                if entity_type == "CREDIT_CARD" and not _luhn_checksum(matched_text):
                    continue

                candidates.append(
                    DetectedEntity(
                        entity_type=entity_type,
                        text=matched_text,
                        start=start,
                        end=end,
                        score=0.95,
                        detector="regex",
                    )
                )

        # 3. Resolve all overlapping spans deterministically
        return resolve_overlapping_spans(candidates)

    def calculate_privacy_score(self, entities: list[DetectedEntity], text: str) -> float:
        """
        Calculate privacy risk score between 0.0 (benign) and 1.0 (critical PII).
        A single high-sensitivity entity (SSN, CreditCard, API Key) guarantees score >= 0.85.
        """
        if not entities:
            return 0.0

        max_weight = max((ENTITY_WEIGHTS.get(e.entity_type, 0.5) for e in entities), default=0.0)
        # Baseline score anchored to maximum single entity sensitivity
        base_risk = max_weight * 0.85
        # Volume factor adds up to 0.15 based on entity count
        volume_factor = min(0.15, (len(entities) / 10.0) * 0.15)

        risk = min(1.0, base_risk + volume_factor)
        return round(float(risk), 3)

    def redact(self, text: str) -> RedactionResult:
        """
        Perform complete PII detection, deterministic span redaction, and mapping.
        Substitutes raw PII with deterministic indexed tokens (e.g., <EMAIL_1>, <SSN_1>).
        """
        if not text:
            return RedactionResult(
                before_text="",
                after_text="",
                detected_entities=[],
                privacy_risk_score=0.0,
                entity_counts={},
                raw_pii_values=[],
                replacement_map={},
                presidio_used=self.presidio_available,
            )

        entities = self.detect_entities(text)
        privacy_score = self.calculate_privacy_score(entities, text)

        entity_counters: dict[str, int] = {}
        replacement_map: dict[str, str] = {}
        entity_counts: dict[str, int] = {}

        # Process from back to front using verified non-overlapping intervals
        # Slicing from reverse preserves earlier character offsets perfectly
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)
        redacted_chars = list(text)

        for entity in sorted_entities:
            entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
            if entity.text not in replacement_map:
                entity_counters[entity.entity_type] = entity_counters.get(entity.entity_type, 0) + 1
                token = f"<{entity.entity_type}_{entity_counters[entity.entity_type]}>"
                replacement_map[entity.text] = token
            else:
                token = replacement_map[entity.text]

            redacted_chars[entity.start : entity.end] = list(token)

        after_text = "".join(redacted_chars)
        raw_pii_values = list(replacement_map.keys())

        return RedactionResult(
            before_text=text,
            after_text=after_text,
            detected_entities=entities,
            privacy_risk_score=privacy_score,
            entity_counts=entity_counts,
            raw_pii_values=raw_pii_values,
            replacement_map=replacement_map,
            presidio_used=self.presidio_available,
        )


def _normalize_text_for_comparison(val: str) -> str:
    """Normalize text by stripping punctuation and whitespace."""
    return re.sub(r"[\s\-_(),.]+", "", val).lower()


def payload_monitoring_hook(
    payload: str | dict | Any,
    source_raw_pii: list[str] | set[str],
) -> PayloadMonitoringResult:
    """
    Audit Hook: privacy_guardrail.task.payload_monitoring

    Inspects outbound payload against known source PII, checking for:
      - Direct substring match (case-insensitive)
      - Normalized numeric sequence match (e.g., 123-45-6789 matching 123456789 or 123 45 6789)
      - Decoded base64 match
    Computes quantitative 'Leakage Rate %'.
    """
    if isinstance(payload, dict):
        payload_str = " ".join(f"{k}: {v}" for k, v in payload.items())
    else:
        payload_str = str(payload or "")

    payload_digest = hashlib.sha256(payload_str.encode()).hexdigest()

    source_pii_list = [val.strip() for val in source_raw_pii if val and len(val.strip()) > 1]
    total_pii_count = len(source_pii_list)

    if total_pii_count == 0 or not payload_str.strip():
        return PayloadMonitoringResult(
            hook="privacy_guardrail.task.payload_monitoring",
            status="SAFE",
            total_source_pii_count=total_pii_count,
            leaked_pii_count=0,
            leaked_entities=[],
            leakage_rate_pct=0.0,
            payload_digest=payload_digest[:16],
        )

    leaked: list[str] = []
    payload_lower = payload_str.lower()
    payload_normalized = _normalize_text_for_comparison(payload_str)

    # Attempt to decode base64 strings inside the payload to detect encoded exfiltrations
    decoded_b64_blocks = []
    for match in re.finditer(r"[A-Za-z0-9+/=]{16,}", payload_str):
        try:
            decoded = base64.b64decode(match.group(0)).decode("utf-8", errors="ignore")
            if decoded:
                decoded_b64_blocks.append(decoded.lower())
        except Exception:
            pass
    decoded_b64_combined = " ".join(decoded_b64_blocks)

    for raw_val in source_pii_list:
        raw_val_lower = raw_val.lower()
        raw_val_norm = _normalize_text_for_comparison(raw_val)

        # 1. Exact / substring match
        if raw_val_lower in payload_lower:
            leaked.append(raw_val)
            continue

        # 2. Normalized match (for numbers/formatted entities like SSN, Aadhaar, Credit Card)
        if len(raw_val_norm) >= 6 and raw_val_norm in payload_normalized:
            leaked.append(raw_val)
            continue

        # 3. Base64 decoded check
        if decoded_b64_combined and raw_val_lower in decoded_b64_combined:
            leaked.append(raw_val)

    leaked_count = len(leaked)
    leakage_rate_pct = round((leaked_count / total_pii_count) * 100.0, 2)
    status = "LEAKAGE_DETECTED" if leaked_count > 0 else "SAFE"

    return PayloadMonitoringResult(
        hook="privacy_guardrail.task.payload_monitoring",
        status=status,
        total_source_pii_count=total_pii_count,
        leaked_pii_count=leaked_count,
        leaked_entities=leaked,
        leakage_rate_pct=leakage_rate_pct,
        payload_digest=payload_digest[:16],
    )
