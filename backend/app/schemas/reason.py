"""Reason schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ReasonResponse(BaseModel):
    """KYC reason code response."""

    id: UUID
    session_id: UUID
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]

    model_config = {"from_attributes": True}






