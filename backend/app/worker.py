"""Database-backed worker for durable processing jobs.

Run this as a separate Railway service:
  python -m app.worker

It polls `kyc_jobs` for pending work, claims a job with SKIP LOCKED, then calls
the configured processing backend (Modal in prod).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from sqlalchemy import select

from app.database import async_session_maker
from app.models.job import KYCJob
from app.models.session import KYCSession
from app.services.processing import get_processing_backend

logger = logging.getLogger("kyc.worker")


POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL", "1.0"))
MAX_ATTEMPTS = int(os.environ.get("WORKER_MAX_ATTEMPTS", "3"))


async def _claim_next_job() -> Optional[KYCJob]:
    async with async_session_maker() as db:
        async with db.begin():
            stmt = (
                select(KYCJob)
                .where(KYCJob.status == "pending")
                .order_by(KYCJob.created_at.asc())
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            res = await db.execute(stmt)
            job = res.scalar_one_or_none()
            if job is None:
                return None

            if (job.attempts or 0) >= MAX_ATTEMPTS:
                job.status = "failed"
                job.last_error = f"Max attempts exceeded ({MAX_ATTEMPTS})"
                return None

            job.status = "processing"
            job.attempts = (job.attempts or 0) + 1
            return job


async def _mark_job_completed(job_id) -> None:
    async with async_session_maker() as db:
        async with db.begin():
            job = await db.get(KYCJob, job_id)
            if job:
                job.status = "completed"


async def _mark_job_failed(job_id, error: str) -> None:
    async with async_session_maker() as db:
        async with db.begin():
            job = await db.get(KYCJob, job_id)
            if job:
                job.status = "failed"
                job.last_error = error[:4000]


async def _ensure_session_failed(session_id) -> None:
    async with async_session_maker() as db:
        async with db.begin():
            sess = await db.get(KYCSession, session_id)
            if sess and sess.status not in ("completed", "failed"):
                sess.status = "failed"


async def _run_job(job: KYCJob) -> None:
    backend = get_processing_backend()

    if job.job_type == "process_session":
        await backend.process_session(str(job.session_id))
        return

    if job.job_type == "generate_synthetic_session":
        await backend.generate_synthetic_session(
            str(job.session_id),
            job.attack_family or "benign",
            job.attack_severity or "medium",
        )
        return

    raise RuntimeError(f"Unknown job_type: {job.job_type}")


async def run_forever() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger.info(
        "worker starting (poll=%.2fs, max_attempts=%d)", POLL_INTERVAL_SECONDS, MAX_ATTEMPTS
    )

    while True:
        job = await _claim_next_job()
        if job is None:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        logger.info(
            "claimed job id=%s type=%s session_id=%s attempt=%s",
            str(job.id),
            job.job_type,
            str(job.session_id),
            job.attempts,
        )

        try:
            await _run_job(job)
            await _mark_job_completed(job.id)
            logger.info("completed job id=%s", str(job.id))
        except Exception as e:
            logger.exception("job failed id=%s", str(job.id))
            await _mark_job_failed(job.id, str(e))
            await _ensure_session_failed(job.session_id)


def main() -> None:
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()


