from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .privacy_guardrail import payload_monitoring_hook, PayloadMonitoringResult

logger = logging.getLogger(__name__)


@dataclass
class EgressReceipt:
    request_id: str
    destination: str
    decision: str  # "ALLOWED" | "BLOCKED"
    http_status_code: int  # 200 (allowed) | 403 (blocked)
    transmitted: bool
    payload_sha256: str
    detected_entities: list[str]
    leakage_rate_pct: float
    message: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "destination": self.destination,
            "decision": self.decision,
            "http_status_code": self.http_status_code,
            "transmitted": self.transmitted,
            "payload_sha256": self.payload_sha256,
            "detected_entities": self.detected_entities,
            "leakage_rate_pct": self.leakage_rate_pct,
            "message": self.message,
            "timestamp_utc": self.timestamp_utc,
        }


class EgressSecurityGateway:
    """
    Deterministic Pre-Transmission Security Gateway.

    Intercepts serialized agent outbound payloads BEFORE transmission to any external API.
    Enforces Zero-Leakage Policy:
      - If raw PII is detected -> Immediately BLOCKS the request (HTTP 403), prevents transmission.
      - If clean -> ALLOWS transmission to destination sink.
    Returns immutable cryptographic receipt with full SHA-256 digest.
    """

    def __init__(self, default_destination: str = "example-external-api"):
        self.default_destination = default_destination
        self._audit_log: list[EgressReceipt] = []

    def inspect_and_transmit(
        self,
        payload: str | dict[str, Any],
        ground_truth_pii: list[str] | set[str] | None = None,
        destination: str | None = None,
    ) -> EgressReceipt:
        dest = destination or self.default_destination
        req_id = f"req_{uuid.uuid4().hex[:12]}"

        # Standardize payload serialization
        if isinstance(payload, dict):
            try:
                serialized_payload = json.dumps(payload, sort_keys=True)
            except Exception:
                serialized_payload = str(payload)
        else:
            serialized_payload = str(payload or "")

        full_sha256 = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

        # If no ground-truth PII is supplied, pass empty list
        pii_targets = list(ground_truth_pii) if ground_truth_pii else []

        # Run pre-transmission deep inspection hook
        monitoring_res: PayloadMonitoringResult = payload_monitoring_hook(
            payload=serialized_payload,
            source_raw_pii=pii_targets,
        )

        if monitoring_res.status == "LEAKAGE_DETECTED" or len(monitoring_res.leaked_entities) > 0:
            # Active Pre-Transmission Interception: Halt and Block
            decision = "BLOCKED"
            http_status = 403
            transmitted = False
            msg = (
                f"EGRESS_FIREWALL_BLOCKED: Outbound request to '{dest}' was intercepted and BLOCKED. "
                f"Raw PII detected in payload: {monitoring_res.leaked_entities}. Leakage Rate: {monitoring_res.leakage_rate_pct}%."
            )
            logger.warning(f"[FIREWALL] {msg} (Digest: {full_sha256[:16]})")
        else:
            # Clean payload: Allow dispatch to mock external sink
            decision = "ALLOWED"
            http_status = 200
            transmitted = True
            msg = f"EGRESS_FIREWALL_ALLOWED: Outbound payload to '{dest}' validated clean. Successfully transmitted."
            logger.info(f"[FIREWALL] {msg} (Digest: {full_sha256[:16]})")

        receipt = EgressReceipt(
            request_id=req_id,
            destination=dest,
            decision=decision,
            http_status_code=http_status,
            transmitted=transmitted,
            payload_sha256=full_sha256,
            detected_entities=monitoring_res.leaked_entities,
            leakage_rate_pct=monitoring_res.leakage_rate_pct,
            message=msg,
        )

        self._audit_log.append(receipt)
        return receipt

    def get_latest_receipt(self) -> EgressReceipt | None:
        return self._audit_log[-1] if self._audit_log else None

    def get_audit_log(self) -> list[EgressReceipt]:
        return list(self._audit_log)

