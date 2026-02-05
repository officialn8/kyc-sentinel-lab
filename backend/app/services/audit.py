"""Audit logging service with hash chaining and immutable archive."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.models.audit import AuditEvent, AuditChain
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)


GENESIS_HASH = "0" * 64


@dataclass
class AuditActor:
    """Actor context for audit logs."""

    actor_type: str
    actor_id: Optional[str] = None


def _canonical_json(payload: dict) -> str:
    """Serialize payload with stable ordering for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: dict) -> str:
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _archive_key(event: AuditEvent, created_at: datetime) -> str:
    prefix = (settings.audit_log_prefix or "audit").strip().strip("/")
    date_path = created_at.strftime("%Y/%m/%d")
    return f"{prefix}/{date_path}/{event.id}.json"


def _extract_actor_from_request(request) -> AuditActor:
    """Best-effort actor extraction without logging secrets."""
    if request is None:
        return AuditActor(actor_type="system")

    api_key = (request.headers.get("X-API-Key") or "").strip()
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        return AuditActor(actor_type="api_key", actor_id=digest)

    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("basic "):
        return AuditActor(actor_type="basic_auth")

    return AuditActor(actor_type="unknown")


async def _get_or_create_chain(
    db: AsyncSession,
    chain_date: date,
) -> AuditChain:
    stmt = select(AuditChain).where(AuditChain.chain_date == chain_date).with_for_update()
    chain = await db.scalar(stmt)
    if chain:
        return chain

    chain = AuditChain(chain_date=chain_date, last_hash=GENESIS_HASH)
    db.add(chain)
    try:
        await db.flush()
        return chain
    except IntegrityError:
        await db.rollback()
        chain = await db.scalar(stmt)
        if chain:
            return chain
        raise


async def log_audit_event(
    db: AsyncSession,
    *,
    event_type: str,
    status: Optional[str] = None,
    actor: Optional[AuditActor] = None,
    session_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[dict] = None,
    request=None,
) -> AuditEvent:
    """Create a hash-chained audit event and archive to R2."""
    created_at = datetime.now(timezone.utc)
    actor = actor or _extract_actor_from_request(request)
    metadata = metadata or {}
    session_uuid = None
    if session_id:
        try:
            session_uuid = uuid.UUID(str(session_id))
        except ValueError:
            session_uuid = None

    chain = await _get_or_create_chain(db, created_at.date())
    prev_hash = chain.last_hash or GENESIS_HASH

    payload_for_hash = {
        "event_type": event_type,
        "created_at": created_at.isoformat(),
        "status": status,
        "actor_type": actor.actor_type,
        "actor_id": actor.actor_id,
        "session_id": str(session_uuid) if session_uuid else None,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "request_id": request_id,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "metadata": metadata,
        "prev_hash": prev_hash,
    }
    event_hash = _hash_payload(payload_for_hash)

    event = AuditEvent(
        event_type=event_type,
        status=status,
        actor_type=actor.actor_type,
        actor_id=actor.actor_id,
        session_id=session_uuid,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=request_id,
        ip_address=ip_address,
        user_agent=user_agent,
        prev_hash=prev_hash,
        event_hash=event_hash,
        event_metadata=metadata,
    )
    db.add(event)
    chain.last_hash = event_hash

    await db.commit()
    await db.refresh(event)

    try:
        storage = get_storage_service()
        archive_payload = {
            **payload_for_hash,
            "event_id": str(event.id),
            "event_hash": event_hash,
        }
        archive_key = _archive_key(event, created_at)
        await storage.upload_file(
            archive_key,
            json.dumps(archive_payload, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        event.archived_at = datetime.now(timezone.utc)
        event.archive_error = None
        await db.commit()
    except Exception as e:
        event.archive_error = str(e)[:500]
        await db.commit()
        logger.error("Audit archive failed for %s: %s", event.id, e)

    return event
