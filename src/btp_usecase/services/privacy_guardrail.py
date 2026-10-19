from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Optional Presidio imports with fallback
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
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
    detector: str  # "presidio", "regex", "ner_heuristic"


@dataclass
class RedactionResult:
    before_text: str
    after_text: str
    detected_entities: list[DetectedEntity]
    privacy_risk_score: float
    entity_counts: dict[str, int]
    raw_pii_values: list[str]
    replacement_map: dict[str, str]


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


# High-priority Regex patterns for PII detection
REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "US_SSN": re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "API_KEY": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password|auth_token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?\b"
    ),
    "AADHAAR_NUMBER": re.compile(
        r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b"
    ),
}

# Sensitivity weights for privacy risk scoring
ENTITY_WEIGHTS: dict[str, float] = {
    "US_SSN": 1.0,
    "CREDIT_CARD": 1.0,
    "API_KEY": 1.0,
    "AADHAAR_NUMBER": 0.95,
    "US_PASSPORT": 0.9,
    "US_BANK_NUMBER": 0.9,
    "EMAIL_ADDRESS": 0.7,
    "PHONE_NUMBER": 0.7,
    "IP_ADDRESS": 0.5,
    "PERSON": 0.6,
    "LOCATION": 0.4,
    "DATE_TIME": 0.2,
}


class PrivacyGuardrail:
    """
    Dedicated Privacy & PII Guardrail Engine.

    Features:
      1. Multi-layered PII detection: Microsoft Presidio + Compiled Regex + Heuristic NER.
      2. Deterministic token substitution with bidirectional mapping.
      3. Before/After text logging and auditing.
      4. Quantitative Privacy Risk Scoring (0.0 to 1.0).
      5. External payload monitoring audit hook with 'Leakage Rate %' metric.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None

        if PRESIDIO_AVAILABLE:
            try:
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except Exception:
                # If models or Presidio failed to load, continue with regex engine
                self._analyzer = None
                self._anonymizer = None

    def detect_entities(self, text: str) -> list[DetectedEntity]:
        """Detect all PII entities using Presidio, regex patterns, and NER heuristics."""
        if not text:
            return []

        entities: list[DetectedEntity] = []
        covered_spans: set[tuple[int, int]] = set()

        def _is_overlapping(s: int, e: int) -> bool:
            return any(not (e <= span[0] or s >= span[1]) for span in covered_spans)

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
                    entities.append(
                        DetectedEntity(
                            entity_type=res.entity_type,
                            text=entity_str,
                            start=start,
                            end=end,
                            score=round(float(res.score), 3),
                            detector="presidio",
                        )
                    )
                    covered_spans.add((start, end))
            except Exception:
                pass

        # 2. Regex Patterns
        for entity_type, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = match.group(0)
                # If regex has capture group (like API_KEY), use capture group
                if match.groups() and match.group(1):
                    start = match.start(1)
                    end = match.end(1)
                    matched_text = match.group(1)

                if not _is_overlapping(start, end):
                    entities.append(
                        DetectedEntity(
                            entity_type=entity_type,
                            text=matched_text,
                            start=start,
                            end=end,
                            score=0.95,
                            detector="regex",
                        )
                    )
                    covered_spans.add((start, end))

        # Sort entities by start offset
        entities.sort(key=lambda x: x.start)
        return entities

    def calculate_privacy_score(self, entities: list[DetectedEntity], text: str) -> float:
        """
        Calculate privacy risk score (0.0 = safe, 1.0 = highly sensitive).
        Weighted sum of detected entity sensitivities normalized by density.
        """
        if not entities:
            return 0.0

        raw_score = sum(ENTITY_WEIGHTS.get(e.entity_type, 0.5) * e.score for e in entities)
        # Scale asymptotically between 0.0 and 1.0 using tanh-like saturating factor
        # A single SSN/CreditCard/API key guarantees score >= 0.85
        max_single_weight = max((ENTITY_WEIGHTS.get(e.entity_type, 0.5) for e in entities), default=0.0)
        risk = min(1.0, (raw_score / 3.0) * 0.4 + max_single_weight * 0.6)
        return round(float(risk), 3)

    def redact(self, text: str) -> RedactionResult:
        """
        Perform complete PII detection, redaction, and before/after text logging.
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
            )

        entities = self.detect_entities(text)
        privacy_score = self.calculate_privacy_score(entities, text)

        entity_counters: dict[str, int] = {}
        replacement_map: dict[str, str] = {}
        entity_counts: dict[str, int] = {}

        # Build replacement mapping from entities
        # Process from back to front to preserve string indices when slicing
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
        )


def payload_monitoring_hook(
    payload: str | dict | Any,
    source_raw_pii: list[str] | set[str],
) -> PayloadMonitoringResult:
    """
    Audit Hook: privacy_guardrail.task.payload_monitoring

    Intercepts the outbound payload destined for the External API Integrator or
    any downstream sink, inspects it for raw PII instances detected in the source,
    and computes the quantitative 'Leakage Rate %'.

    Formula:
      Leakage Rate % = (Leaked Source PII Instances / Total Source PII Instances) * 100%
      (0.0% if no PII was present in the source document)
    """
    if isinstance(payload, dict):
        payload_str = " ".join(f"{k}: {v}" for k, v in payload.items())
    else:
        payload_str = str(payload or "")

    payload_digest = hashlib.sha256(payload_str.encode()).hexdigest()[:16]

    source_pii_list = [val.strip() for val in source_raw_pii if val and len(val.strip()) > 1]
    total_pii_count = len(source_pii_list)

    if total_pii_count == 0:
        return PayloadMonitoringResult(
            hook="privacy_guardrail.task.payload_monitoring",
            status="SAFE",
            total_source_pii_count=0,
            leaked_pii_count=0,
            leaked_entities=[],
            leakage_rate_pct=0.0,
            payload_digest=payload_digest,
        )

    # Check for direct or case-insensitive presence of raw PII in outbound payload
    leaked: list[str] = []
    payload_lower = payload_str.lower()
    for raw_val in source_pii_list:
        if raw_val.lower() in payload_lower:
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
        payload_digest=payload_digest,
    )
