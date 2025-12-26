"""KYC processing job model (durable queue)."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class KYCJob(Base):
    """Durable job queue for session processing."""

    __tablename__ = "kyc_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kyc_sessions.id", ondelete="CASCADE"), nullable=False
    )

    job_type: Mapped[str] = mapped_column(String(50))  # process_session | generate_synthetic_session
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | processing | completed | failed

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Optional fields for synthetic job
    attack_family: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    attack_severity: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    def __repr__(self) -> str:
        return f"<KYCJob(id={self.id}, type={self.job_type}, status={self.status}, session_id={self.session_id})>"


