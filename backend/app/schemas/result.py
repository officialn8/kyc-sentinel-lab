"""Result schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ResultResponse(BaseModel):
    """KYC result response."""

    id: UUID
    session_id: UUID
    created_at: datetime

    risk_score: int
    decision: str

    face_similarity: Optional[float] = None
    pad_score: Optional[float] = None
    doc_score: Optional[float] = None

    model_version: str
    rules_version: str

    model_config = {"from_attributes": True}




