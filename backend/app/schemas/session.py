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
    
    # Phase 2: Fraud detection fields
    device_fingerprint: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Unique device identifier for velocity tracking",
    )
    device_timezone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Client timezone (e.g., 'America/New_York')",
    )

    # Optional upload metadata (recommended for production)
    # Used to generate keys with file extensions and to sign presigned URLs
    # with Content-Type constraints.
    selfie_filename: Optional[str] = None
    selfie_content_type: Optional[str] = None
    id_filename: Optional[str] = None
    id_content_type: Optional[str] = None


class PresignedUpload(BaseModel):
    """Presigned upload configuration (POST or PUT)."""

    method: str = "POST"
    url: str
    fields: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    asset_key: str
    expires_in: int


class PresignedUrlResponse(BaseModel):
    """Presigned URL response for uploading assets (deprecated)."""

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
    selfie_upload: PresignedUpload
    id_upload: PresignedUpload


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
    
    # Face crops (generated during processing)
    selfie_crop_url: Optional[str] = None
    id_crop_url: Optional[str] = None

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


class SimilarFaceResponse(BaseModel):
    """Response for similar face search (fraud ring detection)."""

    session_id: UUID
    created_at: datetime
    status: str
    source: str
    attack_family: Optional[str] = None
    similarity_score: float = Field(
        description="Cosine similarity score (0-1, higher = more similar)"
    )
    distance: float = Field(
        description="Cosine distance (0-2, lower = more similar)"
    )
    
    # Optional result summary
    decision: Optional[str] = None
    risk_score: Optional[int] = None

    model_config = {"from_attributes": True}


class SimilarFacesListResponse(BaseModel):
    """Response for similar faces endpoint."""

    source_session_id: UUID
    threshold: float
    matches: list[SimilarFaceResponse]
    total_matches: int


