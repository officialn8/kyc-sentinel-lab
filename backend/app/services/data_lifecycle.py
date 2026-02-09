"""Data retention and GDPR erasure utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError
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


def _normalize_content_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower().strip()
    if ";" in normalized:
        normalized = normalized.split(";", 1)[0].strip()
    return normalized or None


async def cleanup_stale_pending_uploads(
    db: AsyncSession,
    storage: StorageService,
) -> dict:
    """Delete stale/invalid pending uploads to limit storage abuse."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.presigned_url_expiration)
    stmt = (
        select(KYCSession)
        .where(KYCSession.status == "pending")
        .where(KYCSession.created_at < cutoff)
        .where(KYCSession.deleted_at.is_(None))
        .where(
            (KYCSession.selfie_upload_ticket_hash.is_not(None))
            | (KYCSession.id_upload_ticket_hash.is_not(None))
        )
    )
    sessions = (await db.execute(stmt)).scalars().all()

    cleaned_sessions = 0
    deleted_objects = 0
    for session in sessions:
        invalid = False
        for key, expected_type, expected_size in (
            (session.selfie_asset_key, session.selfie_expected_content_type, session.selfie_expected_size_bytes),
            (session.id_asset_key, session.id_expected_content_type, session.id_expected_size_bytes),
        ):
            if not key:
                invalid = True
                continue
            try:
                metadata = await storage.get_object_metadata(key)
            except ClientError:
                invalid = True
                continue

            content_length = int(metadata.get("content_length") or 0)
            content_type = _normalize_content_type(metadata.get("content_type"))
            expected_type_normalized = _normalize_content_type(expected_type)

            if content_length > settings.max_upload_size_bytes:
                invalid = True
            if expected_size is not None and content_length != expected_size:
                invalid = True
            if expected_type_normalized and content_type != expected_type_normalized:
                invalid = True

        if not invalid:
            continue

        cleaned_sessions += 1
        deleted_objects += await storage.delete_prefix(f"sessions/{session.id}/")
        deleted_objects += await storage.delete_prefix(f"crops/{session.id}/")
        session.status = "failed"
        session.deleted_reason = "stale_invalid_upload_cleanup"

    if cleaned_sessions:
        await db.commit()

    await log_audit_event(
        db,
        event_type="retention.pending_upload_cleanup",
        status="success",
        actor=AuditActor(actor_type="system"),
        metadata={
            "session_count": cleaned_sessions,
            "deleted_objects": deleted_objects,
            "cutoff": cutoff.isoformat(),
        },
    )

    return {
        "session_count": cleaned_sessions,
        "deleted_objects": deleted_objects,
    }
