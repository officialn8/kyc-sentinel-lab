"""Database-backed job queue utilities."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.job import KYCJob


def _is_job_stale(job: KYCJob) -> bool:
    """Detect jobs stuck in processing beyond the configured timeout."""
    if job.status != "processing" or job.updated_at is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.processing_stuck_timeout_seconds
    )
    return job.updated_at < cutoff


def _normalize_session_id(session_id: uuid.UUID | str) -> uuid.UUID:
    """Coerce session identifiers to UUID for DB driver compatibility."""
    if isinstance(session_id, uuid.UUID):
        return session_id
    return uuid.UUID(str(session_id))


async def enqueue_process_session(
    db: AsyncSession,
    session_id: uuid.UUID | str,
) -> KYCJob:
    normalized_session_id = _normalize_session_id(session_id)
    existing = await db.scalar(
        select(KYCJob)
        .where(
            KYCJob.session_id == normalized_session_id,
            KYCJob.job_type == "process_session",
            KYCJob.status.in_(("pending", "processing")),
        )
        .limit(1)
    )
    if existing:
        if _is_job_stale(existing):
            existing.status = "failed"
            existing.last_error = "Stuck processing timeout"
        else:
            return existing

    job = KYCJob(
        session_id=normalized_session_id,
        job_type="process_session",
        status="pending",
    )
    db.add(job)
    return job


async def enqueue_generate_synthetic_session(
    db: AsyncSession,
    *,
    session_id: uuid.UUID | str,
    attack_family: str,
    attack_severity: str,
) -> KYCJob:
    normalized_session_id = _normalize_session_id(session_id)
    existing = await db.scalar(
        select(KYCJob)
        .where(
            KYCJob.session_id == normalized_session_id,
            KYCJob.job_type == "generate_synthetic_session",
            KYCJob.status.in_(("pending", "processing")),
        )
        .limit(1)
    )
    if existing:
        if _is_job_stale(existing):
            existing.status = "failed"
            existing.last_error = "Stuck processing timeout"
        else:
            return existing

    job = KYCJob(
        session_id=normalized_session_id,
        job_type="generate_synthetic_session",
        status="pending",
        attack_family=attack_family,
        attack_severity=attack_severity,
    )
    db.add(job)
    return job
