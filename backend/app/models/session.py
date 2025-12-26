"""KYC Session model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

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
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, processing, completed, failed

    # Source
    source: Mapped[str] = mapped_column(String(20))  # "upload" | "synthetic"
    attack_family: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # replay | injection | face_swap | doc_tamper | benign
    attack_severity: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # low | medium | high

    # Device/capture metadata
    device_os: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    device_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_country: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    capture_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Asset references (R2 keys)
    selfie_asset_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    id_asset_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

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






