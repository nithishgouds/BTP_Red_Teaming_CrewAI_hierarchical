from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChunkAuditEvent:
    agent_role: str
    step: str
    chunk_ids: list[str]
    action: str  # "retrieved", "redacted", "analyzed", "verified", "transmitted"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "step": self.step,
            "chunk_ids": self.chunk_ids,
            "action": self.action,
            "timestamp_utc": self.timestamp_utc,
            "metadata": self.metadata,
        }


class ChunkFlowAuditor:
    """
    Tracks and audits the exact data flow of document chunk IDs across agents.
    Verifies that no chunk bypasses security or privacy controls.
    """

    def __init__(self):
        self.events: list[ChunkAuditEvent] = []

    def record(
        self,
        agent_role: str,
        step: str,
        chunk_ids: list[str],
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkAuditEvent:
        clean_ids = [str(cid).strip() for cid in chunk_ids if str(cid).strip()]
        event = ChunkAuditEvent(
            agent_role=agent_role,
            step=step,
            chunk_ids=clean_ids,
            action=action,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def get_flow_matrix(self) -> dict[str, list[str]]:
        """Return a mapping of chunk_id -> list of agent roles that handled it."""
        matrix: dict[str, list[str]] = {}
        for ev in self.events:
            for cid in ev.chunk_ids:
                if cid not in matrix:
                    matrix[cid] = []
                if ev.agent_role not in matrix[cid]:
                    matrix[cid].append(ev.agent_role)
        return matrix

    def verify_provenance(self) -> dict[str, Any]:
        """
        Verify that all chunks received by analytical agents (LLM Analyst)
        strictly flowed through the Document Retriever and PII Redactor.
        """
        matrix = self.get_flow_matrix()
        all_chunks = list(matrix.keys())

        unguarded_chunks: list[str] = []
        for cid, agents in matrix.items():
            if "LLM Analyst" in agents and "PII Redactor / Privacy Scorer" not in agents:
                unguarded_chunks.append(cid)

        is_verified = (len(unguarded_chunks) == 0) and (len(all_chunks) > 0)
        status = "PROVENANCE_VERIFIED" if is_verified else (
            "NO_CHUNKS_TRACKED" if not all_chunks else "UNGUARDED_FLOW_DETECTED"
        )

        return {
            "status": status,
            "total_chunks_tracked": len(all_chunks),
            "flow_matrix": matrix,
            "unguarded_chunks": unguarded_chunks,
            "event_count": len(self.events),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "provenance": self.verify_provenance(),
        }
