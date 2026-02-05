"""Data retention and GDPR erasure utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.session import KYCSession
from app.services.audit import log_audit_event, AuditActor
from app.services.storage import StorageService


async def purge_expired_media(
    db: AsyncSession,
    storage: StorageService,
) -> dict:
    """Purge media assets older than retention policy."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.retention_media_days)
    stmt = (
        select(KYCSession)
        .where(KYCSession.created_at < cutoff)
        .where(KYCSession.media_purged_at.is_(None))
    )
    result = await db.execute(stmt)
    sessions = result.scalars().all()

    deleted_objects = 0
    for session in sessions:
        deleted_objects += await storage.delete_prefix(f"sessions/{session.id}/")
        deleted_objects += await storage.delete_prefix(f"crops/{session.id}/")
        session.media_purged_at = datetime.now(timezone.utc)
        session.selfie_asset_key = None
        session.id_asset_key = None

    if sessions:
        await db.commit()

    await log_audit_event(
        db,
        event_type="retention.media_purged",
        status="success",
        actor=AuditActor(actor_type="system"),
        metadata={
            "session_count": len(sessions),
            "deleted_objects": deleted_objects,
            "retention_days": settings.retention_media_days,
            "sample_session_ids": [str(s.id) for s in sessions[:5]],
        },
    )

    return {
        "session_count": len(sessions),
        "deleted_objects": deleted_objects,
    }


async def purge_expired_metadata(
    db: AsyncSession,
    storage: StorageService,
) -> dict:
    """Purge session metadata older than retention policy."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.retention_metadata_days)
    stmt = select(KYCSession).where(KYCSession.created_at < cutoff)
    result = await db.execute(stmt)
    sessions = result.scalars().all()

    deleted_objects = 0
    for session in sessions:
        deleted_objects += await storage.delete_prefix(f"sessions/{session.id}/")
        deleted_objects += await storage.delete_prefix(f"crops/{session.id}/")
        await db.delete(session)

    if sessions:
        await db.commit()

    await log_audit_event(
        db,
        event_type="retention.metadata_purged",
        status="success",
        actor=AuditActor(actor_type="system"),
        metadata={
            "session_count": len(sessions),
            "deleted_objects": deleted_objects,
            "retention_days": settings.retention_metadata_days,
            "sample_session_ids": [str(s.id) for s in sessions[:5]],
        },
    )

    return {
        "session_count": len(sessions),
        "deleted_objects": deleted_objects,
    }

