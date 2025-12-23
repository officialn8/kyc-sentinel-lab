"""Session schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.result import ResultResponse
from app.schemas.reason import ReasonResponse


class SessionCreate(BaseModel):
    """Schema for creating a new session."""

    source: str = Field(
        default="upload", pattern="^(upload|synthetic)$"
    )
    attack_family: Optional[str] = Field(
        default=None,
        pattern="^(replay|injection|face_swap|doc_tamper|benign)$",
    )
    attack_severity: Optional[str] = Field(
        default=None, pattern="^(low|medium|high)$"
    )
    device_os: Optional[str] = None
    device_model: Optional[str] = None
    ip_country: Optional[str] = None


class PresignedUrlResponse(BaseModel):
    """Presigned URL response for uploading assets."""

    selfie_upload_url: str
    selfie_asset_key: str
    id_upload_url: str
    id_asset_key: str
    expires_in: int


class SessionResponse(BaseModel):
    """Basic session response."""

    id: UUID
    created_at: datetime
    updated_at: datetime
    status: str
    source: str
    attack_family: Optional[str] = None
    attack_severity: Optional[str] = None
    selfie_asset_key: Optional[str] = None
    id_asset_key: Optional[str] = None

    model_config = {"from_attributes": True}


class SessionCreateResponse(BaseModel):
    """Response when creating a new session."""

    session: SessionResponse
    upload_urls: PresignedUrlResponse


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    items: list[SessionResponse]
    total: int
    page: int
    page_size: int
    pages: int


class SessionDetail(BaseModel):
    """Detailed session response with result and reasons."""

    id: UUID
    created_at: datetime
    updated_at: datetime
    status: str
    source: str
    attack_family: Optional[str] = None
    attack_severity: Optional[str] = None

    # Metadata
    device_os: Optional[str] = None
    device_model: Optional[str] = None
    ip_country: Optional[str] = None
    capture_fps: Optional[float] = None
    resolution: Optional[str] = None

    # Assets
    selfie_asset_key: Optional[str] = None
    id_asset_key: Optional[str] = None
    selfie_url: Optional[str] = None
    id_url: Optional[str] = None

    # Results
    result: Optional[ResultResponse] = None
    reasons: list[ReasonResponse] = []

    model_config = {"from_attributes": True}


class SessionFilter(BaseModel):
    """Filters for listing sessions."""

    status: Optional[str] = None
    source: Optional[str] = None
    attack_family: Optional[str] = None
    decision: Optional[str] = None
    min_risk_score: Optional[int] = None
    max_risk_score: Optional[int] = None



