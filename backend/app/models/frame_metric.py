"""KYC Frame Metric model for timeline visualization."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.session import KYCSession


class KYCFrameMetric(Base):
    """Per-frame metrics for timeline visualization."""

    __tablename__ = "kyc_frame_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kyc_sessions.id", ondelete="CASCADE")
    )
    frame_idx: Mapped[int] = mapped_column(Integer)

    motion_entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpness: Mapped[float | None] = mapped_column(Float, nullable=True)
    noise_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    color_shift: Mapped[float | None] = mapped_column(Float, nullable=True)
    pad_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationship
    session: Mapped["KYCSession"] = relationship(back_populates="frame_metrics")

    def __repr__(self) -> str:
        return f"<KYCFrameMetric(session_id={self.session_id}, frame={self.frame_idx})>"











