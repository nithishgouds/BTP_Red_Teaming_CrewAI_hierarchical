from __future__ import annotations

from typing import Type, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..services.egress_gateway import EgressSecurityGateway, EgressReceipt


class ExternalAPIPayloadSchema(BaseModel):
    payload: str = Field(..., description="The outbound payload or data dictionary to be sent to the external API.")
    destination: str = Field(default="example-external-api", description="The target external API endpoint name.")


class ExternalAPIGatewayTool(BaseTool):
    """
    Deterministic Egress Gateway Tool.

    Enforces runtime pre-transmission privacy policy:
      - Intercepts and parses the agent's serialized payload.
      - Inspects against detected/ground-truth PII.
      - BLOCKS transmission (HTTP 403) if raw PII is present.
      - ALLOWS transmission (HTTP 200) to external sink if clean.
      - Produces an immutable cryptographic receipt.
    """
    name: str = "External API Egress Gateway"
    description: str = (
        "Call this tool to securely transmit data to the external API service. "
        "The gateway performs deterministic pre-transmission inspection and blocks unauthorized PII."
    )
    args_schema: Type[BaseModel] = ExternalAPIPayloadSchema
    gateway: EgressSecurityGateway = Field(default_factory=EgressSecurityGateway)
    ground_truth_pii: list[str] = Field(default_factory=list)

    def _run(self, payload: str, destination: str = "example-external-api") -> str:
        receipt: EgressReceipt = self.gateway.inspect_and_transmit(
            payload=payload,
            ground_truth_pii=self.ground_truth_pii,
            destination=destination,
        )
        return (
            f"RECEIPT_ID: {receipt.request_id} | STATUS: {receipt.decision} ({receipt.http_status_code}) | "
            f"DIGEST: {receipt.payload_sha256[:16]} | {receipt.message}"
        )

