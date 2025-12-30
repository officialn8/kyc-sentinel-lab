"""KYC Reason model."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.session import KYCSession


class KYCReason(Base):
    """KYC reason code with evidence."""

    __tablename__ = "kyc_reasons"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kyc_sessions.id", ondelete="CASCADE")
    )

    code: Mapped[str] = mapped_column(
        String(50)
    )  # FACE_MISMATCH, PAD_SUSPECT_REPLAY, etc.
    severity: Mapped[str] = mapped_column(String(20))  # info | warn | high
    message: Mapped[str] = mapped_column(Text)  # Human-readable explanation

    # Evidence (JSONB)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    # e.g., {"frame_indices": [12, 13, 14], "crop_key": "...", "similarity": 0.34}

    # Relationship
    session: Mapped["KYCSession"] = relationship(back_populates="reasons")

    def __repr__(self) -> str:
        return f"<KYCReason(code={self.code}, severity={self.severity})>"











