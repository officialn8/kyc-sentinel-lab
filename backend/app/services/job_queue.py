"""Database-backed job queue utilities."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import KYCJob


async def enqueue_process_session(db: AsyncSession, session_id: str) -> KYCJob:
    job = KYCJob(session_id=session_id, job_type="process_session", status="pending")
    db.add(job)
    return job


async def enqueue_generate_synthetic_session(
    db: AsyncSession,
    *,
    session_id: str,
    attack_family: str,
    attack_severity: str,
) -> KYCJob:
    job = KYCJob(
        session_id=session_id,
        job_type="generate_synthetic_session",
        status="pending",
        attack_family=attack_family,
        attack_severity=attack_severity,
    )
    db.add(job)
    return job


