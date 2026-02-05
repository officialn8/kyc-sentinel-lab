"""Audit logging models."""

import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import Date, DateTime, String, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AuditChain(Base):
    """Daily hash chain cursor for audit events."""

    __tablename__ = "audit_chains"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    chain_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    last_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuditEvent(Base):
    """Immutable audit event record (append-only)."""

    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    actor_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    actor_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(nullable=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    request_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # "metadata" is reserved by SQLAlchemy Declarative; use a different attribute
    event_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    archive_error: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
