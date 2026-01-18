"""Webhook model for async notifications."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Webhook(Base):
    """Webhook configuration for processing completion notifications."""

    __tablename__ = "webhooks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Target URL for webhook delivery
    url: Mapped[str] = mapped_column(String(2048))
    
    # Event types to subscribe to (stored as JSON array for portability)
    # Valid values: session.completed, session.failed
    events: Mapped[list[str]] = mapped_column(
        JSON, default=list
    )
    
    # Optional description/name for the webhook
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Whether the webhook is active
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Failure tracking for auto-disable after repeated failures
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_failure: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_success: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Webhook(id={self.id}, url={self.url[:50]}..., enabled={self.enabled})>"
