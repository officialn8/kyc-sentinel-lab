"""KYC Session model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Float, DateTime, BigInteger, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import SessionStatus, AttackFamily, AttackSeverity

if TYPE_CHECKING:
    from app.models.result import KYCResult
    from app.models.reason import KYCReason
    from app.models.frame_metric import KYCFrameMetric


class KYCSession(Base):
    """KYC verification session model."""

    __tablename__ = "kyc_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    
    # Status uses enum for type safety, stored as string in DB
    # Valid values: pending, processing, completed, failed
    status: Mapped[str] = mapped_column(
        String(20), default=SessionStatus.PENDING.value
    )

    # Source
    source: Mapped[str] = mapped_column(String(20))  # "upload" | "synthetic"
    
    # Attack family - validated via AttackFamily enum
    # Valid values: replay, injection, face_swap, doc_tamper, benign
    attack_family: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    
    # Attack severity - validated via AttackSeverity enum
    # Valid values: low, medium, high
    attack_severity: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )

    # Device/capture metadata
    device_os: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    device_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_country: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    capture_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Phase 2: Fraud detection fields
    device_fingerprint: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True, index=True
    )  # Unique device identifier for velocity tracking
    client_ip: Mapped[Optional[str]] = mapped_column(
        String(45), nullable=True, index=True
    )  # Client IP address (IPv6 max length = 45)
    device_timezone: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Client timezone (e.g., "America/New_York")

    # Asset references (R2 keys)
    selfie_asset_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    id_asset_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    selfie_expected_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    id_expected_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    selfie_expected_content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    id_expected_content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    selfie_upload_ticket_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    id_upload_ticket_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    upload_ticket_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Soft delete / retention fields
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    media_purged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Face embedding for similarity search (pgvector)
    face_embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(512), nullable=True
    )

    # Relationships
    result: Mapped[Optional["KYCResult"]] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )
    reasons: Mapped[list["KYCReason"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    frame_metrics: Mapped[list["KYCFrameMetric"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<KYCSession(id={self.id}, status={self.status}, source={self.source})>"










