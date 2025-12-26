"""KYC Result model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Integer, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.session import KYCSession


class KYCResult(Base):
    """KYC verification result model."""

    __tablename__ = "kyc_results"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kyc_sessions.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Scores
    risk_score: Mapped[int] = mapped_column(Integer)  # 0-100
    decision: Mapped[str] = mapped_column(String(20))  # pass | review | fail

    # Component scores
    face_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-1
    pad_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # 0-1 (higher = more suspicious)
    doc_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # 0-1 (higher = more suspicious)

    # Versioning
    model_version: Mapped[str] = mapped_column(String(20), default="v1")
    rules_version: Mapped[str] = mapped_column(String(20), default="v1")

    # Relationship
    session: Mapped["KYCSession"] = relationship(back_populates="result")

    def __repr__(self) -> str:
        return f"<KYCResult(id={self.id}, risk_score={self.risk_score}, decision={self.decision})>"






