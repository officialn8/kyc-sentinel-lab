"""Session CRUD endpoints."""

import os
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import select, func, or_, cast, String
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Storage
from app.api.security import rate_limiter, extract_client_ip
from app.config import settings
from app.models.session import KYCSession
from app.services.geolocation import lookup_ip_country
from app.models.result import KYCResult
from app.services.job_queue import enqueue_process_session
from app.services.processing import get_processing_backend
from app.services.audit import log_audit_event
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionCreateResponse,
    SessionListResponse,
    SessionDetail,
    PresignedUpload,
    SimilarFaceResponse,
    SimilarFacesListResponse,
)

router = APIRouter()


_SELFIE_EXT_ALLOWLIST = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm"}
_ID_EXT_ALLOWLIST = {".jpg", ".jpeg", ".png", ".webp"}
_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    # iOS often uses QuickTime for .mov uploads
    "video/quicktime": ".mov",
}
_EXT_TO_CONTENT_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


def _ext_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    _, ext = os.path.splitext(filename)
    ext = (ext or "").lower().strip()
    return ext or None


def _pick_extension(
    *,
    filename: str | None,
    content_type: str | None,
    allowlist: set[str],
) -> str:
    ext = _ext_from_filename(filename)
    if ext and ext in allowlist:
        return ext
    guessed = _CONTENT_TYPE_TO_EXT.get((content_type or "").lower().strip())
    if guessed and guessed in allowlist:
        return guessed
    return ""


def _normalize_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.lower().strip() or None


def _canonical_content_type_for_ext(ext: str | None) -> str | None:
    if not ext:
        return None
    return _EXT_TO_CONTENT_TYPE.get(ext)


def _normalize_ext(ext: str | None) -> str | None:
    if not ext:
        return None
    # Treat .jpeg and .jpg as equivalent
    if ext == ".jpeg":
        return ".jpg"
    return ext


def _get_request_id(request: Request | None) -> str | None:
    if request is None:
        return None
    return (
        request.headers.get("X-Request-ID")
        or request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Trace-ID")
    )


@router.post(
    "",
    response_model=SessionCreateResponse,
    dependencies=[Depends(rate_limiter(limit=20, window_seconds=60))],
)
async def create_session(
    data: SessionCreate,
    request: Request,
    db: DbSession,
    storage: Storage,
) -> SessionCreateResponse:
    """Create a new KYC session and get presigned URLs for upload.

    SECURITY NOTE:
    - client_ip is captured server-side from the request, NOT from client input
    - ip_country is enriched server-side via GeoIP lookup (trusted)
    - device_fingerprint and device_timezone are client-provided (untrusted)
    """
    # SECURITY: Capture client IP server-side using trusted proxy config
    client_ip = extract_client_ip(request)

    # SECURITY: Server-side IP geolocation (trusted, unlike client-provided)
    # Falls back to client-provided ip_country if GeoIP unavailable
    server_ip_country = lookup_ip_country(client_ip)
    ip_country = server_ip_country or data.ip_country  # Prefer server-side

    # Create session with server-side enrichment
    session = KYCSession(
        source=data.source,
        attack_family=data.attack_family,
        attack_severity=data.attack_severity,
        device_os=data.device_os,
        device_model=data.device_model,
        ip_country=ip_country,  # Server-enriched (trusted) or client fallback
        device_fingerprint=data.device_fingerprint,  # Client-provided (untrusted)
        device_timezone=data.device_timezone,  # Client-provided (untrusted)
        client_ip=client_ip,  # SERVER-SIDE capture (trusted)
        status="pending",
    )
    db.add(session)
    await db.flush()

    # Generate presigned URLs for uploads
    selfie_ext = _pick_extension(
        filename=data.selfie_filename,
        content_type=data.selfie_content_type,
        allowlist=_SELFIE_EXT_ALLOWLIST,
    )
    id_ext = _pick_extension(
        filename=data.id_filename,
        content_type=data.id_content_type,
        allowlist=_ID_EXT_ALLOWLIST,
    )

    # If client provided metadata but we couldn't validate it, reject.
    if (data.selfie_filename or data.selfie_content_type) and not selfie_ext:
        raise HTTPException(status_code=400, detail="Unsupported selfie file type")
    if (data.id_filename or data.id_content_type) and not id_ext:
        raise HTTPException(status_code=400, detail="Unsupported ID document file type")

    # Default to JPEG if no metadata provided (ensures safe Content-Type + extension)
    if not selfie_ext:
        selfie_ext = ".jpg"
    if not id_ext:
        id_ext = ".jpg"

    # Canonical content types derived from allowlisted extensions
    selfie_content_type = _canonical_content_type_for_ext(selfie_ext)
    id_content_type = _canonical_content_type_for_ext(id_ext)
    if not selfie_content_type:
        raise HTTPException(status_code=400, detail="Unsupported selfie file type")
    if not id_content_type:
        raise HTTPException(status_code=400, detail="Unsupported ID document file type")

    # Validate client-provided Content-Type matches the allowlisted extension
    selfie_ct_client = _normalize_content_type(data.selfie_content_type)
    if selfie_ct_client:
        if selfie_ct_client not in _CONTENT_TYPE_TO_EXT:
            raise HTTPException(status_code=400, detail="Unsupported selfie content type")
        if _normalize_ext(_CONTENT_TYPE_TO_EXT[selfie_ct_client]) != _normalize_ext(selfie_ext):
            raise HTTPException(status_code=400, detail="Selfie content type does not match file extension")

    id_ct_client = _normalize_content_type(data.id_content_type)
    if id_ct_client:
        if id_ct_client not in _CONTENT_TYPE_TO_EXT:
            raise HTTPException(status_code=400, detail="Unsupported ID document content type")
        if _normalize_ext(_CONTENT_TYPE_TO_EXT[id_ct_client]) != _normalize_ext(id_ext):
            raise HTTPException(status_code=400, detail="ID document content type does not match file extension")

    selfie_key = f"sessions/{session.id}/selfie{selfie_ext}"
    id_key = f"sessions/{session.id}/id{id_ext}"

    # Generate presigned upload config (POST for S3/MinIO, PUT for R2)
    selfie_upload = await storage.generate_presigned_upload(
        selfie_key,
        content_type=selfie_content_type,
        max_size_bytes=settings.max_upload_size_bytes,
    )
    id_upload = await storage.generate_presigned_upload(
        id_key,
        content_type=id_content_type,
        max_size_bytes=settings.max_upload_size_bytes,
    )

    # Store asset keys
    session.selfie_asset_key = selfie_key
    session.id_asset_key = id_key

    await db.commit()
    await db.refresh(session)

    await log_audit_event(
        db,
        event_type="session.created",
        status="success",
        session_id=str(session.id),
        resource_type="session",
        resource_id=str(session.id),
        request_id=_get_request_id(request),
        ip_address=client_ip,
        user_agent=request.headers.get("User-Agent"),
        metadata={
            "source": session.source,
            "attack_family": session.attack_family,
            "attack_severity": session.attack_severity,
        },
        request=request,
    )

    return SessionCreateResponse(
        session=SessionResponse.model_validate(session),
        selfie_upload=PresignedUpload(
            method=selfie_upload.get("method", "POST"),
            url=selfie_upload["url"],
            fields=selfie_upload["fields"],
            headers=selfie_upload.get("headers", {}),
            asset_key=selfie_key,
            expires_in=settings.presigned_url_expiration,
        ),
        id_upload=PresignedUpload(
            method=id_upload.get("method", "POST"),
            url=id_upload["url"],
            fields=id_upload["fields"],
            headers=id_upload.get("headers", {}),
            asset_key=id_key,
            expires_in=settings.presigned_url_expiration,
        ),
    )


@router.post(
    "/{session_id}/finalize",
    dependencies=[Depends(rate_limiter(limit=30, window_seconds=60))],
)
async def finalize_session(
    session_id: UUID,
    db: DbSession,
    background_tasks: BackgroundTasks,
    force: bool = False,
    request: Request = None,
) -> dict:
    """Mark uploads complete and start processing.

    SECURITY: Uses row-level locking (SELECT FOR UPDATE) to prevent race conditions.
    Multiple concurrent finalize calls will serialize, ensuring only one wins.

    Args:
        force: If True, allows retrying failed sessions.
               Processing sessions are only retried if they exceed the stuck timeout.
    """
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select

    # IDEMPOTENCY: Use SELECT FOR UPDATE to prevent race conditions
    # This locks the row until the transaction commits, serializing concurrent calls
    query = (
        select(KYCSession)
        .where(KYCSession.id == session_id)
        .with_for_update(nowait=False)  # Block until lock is available
    )
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Detect potentially stuck processing using updated_at as a lightweight lease.
    # This avoids infinite "processing" when a worker crashes or a task never starts.
    now = datetime.now(timezone.utc)
    stuck_cutoff = now - timedelta(seconds=settings.processing_stuck_timeout_seconds)
    is_stuck_processing = (
        session.status == "processing"
        and session.updated_at is not None
        and session.updated_at < stuck_cutoff
    )

    # Check status AFTER acquiring lock to prevent TOCTOU race
    if session.status == "processing":
        if is_stuck_processing:
            # Session appears stuck, allow retry
            pass
        else:
            raise HTTPException(
                status_code=409,  # Conflict
                detail="Session is already being processed. Retry after completion.",
            )
    elif session.status == "completed":
        # Already done - idempotent success
        return {"status": "completed", "message": "Session already processed"}
    elif session.status == "failed" and not force:
        raise HTTPException(
            status_code=400,
            detail="Session processing failed. Use ?force=true to retry.",
        )
    elif session.status != "pending" and not force:
        raise HTTPException(
            status_code=400,
            detail=f"Session is in unexpected state: {session.status}",
        )

    # Atomically update status within the same transaction (lock still held)
    session.status = "processing"
    await db.flush()  # Ensure status is written before committing

    if settings.use_worker:
        # Production: use durable job queue (requires separate worker service)
        await enqueue_process_session(db, str(session_id))
        await db.commit()  # Releases the row lock
        await log_audit_event(
            db,
            event_type="session.processing_started",
            status="queued",
            session_id=str(session.id),
            resource_type="session",
            resource_id=str(session.id),
            request_id=_get_request_id(request),
            ip_address=extract_client_ip(request) if request else None,
            user_agent=request.headers.get("User-Agent") if request else None,
            metadata={"mode": "worker", "force": force, "stuck_retry": is_stuck_processing},
            request=request,
        )
        return {"status": "processing", "message": "Session queued for worker processing"}
    else:
        # Development: process directly via background task (no worker needed)
        await db.commit()  # Releases the row lock
        backend = get_processing_backend()
        background_tasks.add_task(backend.process_session, str(session_id))
        await log_audit_event(
            db,
            event_type="session.processing_started",
            status="processing",
            session_id=str(session.id),
            resource_type="session",
            resource_id=str(session.id),
            request_id=_get_request_id(request),
            ip_address=extract_client_ip(request) if request else None,
            user_agent=request.headers.get("User-Agent") if request else None,
            metadata={"mode": "background_task", "force": force, "stuck_retry": is_stuck_processing},
            request=request,
        )
        return {"status": "processing", "message": "Session processing started"}


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    db: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    attack_family: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None, description="Search by session ID, attack family, or status"),
    include_deleted: bool = Query(default=False),
) -> SessionListResponse:
    """List sessions with filters, search, and pagination."""
    # Build query
    query = select(KYCSession)

    # Exclude soft-deleted sessions by default
    if not include_deleted:
        query = query.where(KYCSession.deleted_at.is_(None))

    # Text search across multiple fields
    # PERFORMANCE: Uses ILIKE which triggers GIN trigram indexes (ix_kyc_sessions_*_gin)
    # See migration 005_search_performance.py for index definitions.
    if search:
        search_term = f"%{search}%"
        
        # Build OR conditions for text columns that have GIN indexes
        search_conditions = [
            # Search by attack family (GIN indexed)
            KYCSession.attack_family.ilike(search_term),
            # Search by status (GIN indexed)
            KYCSession.status.ilike(search_term),
            # Search by source (GIN indexed)
            KYCSession.source.ilike(search_term),
        ]
        
        # UUID search: Only do prefix matching if input looks like a UUID prefix
        # This avoids expensive full-text UUID search while still allowing partial ID lookup
        if len(search) >= 8 and search.replace("-", "").isalnum():
            # Prefix search on ID (uses btree index on primary key for exact match,
            # or falls back to GIN text index for substring)
            search_conditions.append(
                cast(KYCSession.id, String).ilike(search_term)
            )
        
        query = query.where(or_(*search_conditions))

    if status:
        query = query.where(KYCSession.status == status)
    if source:
        query = query.where(KYCSession.source == source)
    if attack_family:
        query = query.where(KYCSession.attack_family == attack_family)
    if decision:
        query = query.join(KYCResult).where(KYCResult.decision == decision)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Apply pagination
    query = query.order_by(KYCSession.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    sessions = result.scalars().all()

    return SessionListResponse(
        items=[SessionResponse.model_validate(s) for s in sessions],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: UUID,
    db: DbSession,
    storage: Storage,
    include_deleted: bool = Query(default=False),
) -> SessionDetail:
    """Get full session details with results and reasons."""
    query = (
        select(KYCSession)
        .options(
            selectinload(KYCSession.result),
            selectinload(KYCSession.reasons),
        )
        .where(KYCSession.id == session_id)
    )

    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.deleted_at and not include_deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    # Generate presigned download URLs for media
    selfie_url = None
    id_url = None
    if session.selfie_asset_key:
        selfie_url = await storage.generate_presigned_download_url(
            session.selfie_asset_key
        )
    if session.id_asset_key:
        id_url = await storage.generate_presigned_download_url(session.id_asset_key)

    # Check for face crops and generate presigned URLs
    selfie_crop_url = None
    id_crop_url = None
    selfie_crop_key = f"crops/{session_id}/selfie_face.jpg"
    id_crop_key = f"crops/{session_id}/id_face.jpg"
    
    if await storage.object_exists(selfie_crop_key):
        selfie_crop_url = await storage.generate_presigned_download_url(selfie_crop_key)
    if await storage.object_exists(id_crop_key):
        id_crop_url = await storage.generate_presigned_download_url(id_crop_key)

    return SessionDetail(
        id=session.id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        status=session.status,
        source=session.source,
        attack_family=session.attack_family,
        attack_severity=session.attack_severity,
        device_os=session.device_os,
        device_model=session.device_model,
        ip_country=session.ip_country,
        capture_fps=session.capture_fps,
        resolution=session.resolution,
        selfie_asset_key=session.selfie_asset_key,
        id_asset_key=session.id_asset_key,
        selfie_url=selfie_url,
        id_url=id_url,
        selfie_crop_url=selfie_crop_url,
        id_crop_url=id_crop_url,
        result=session.result,
        reasons=session.reasons,
    )


@router.delete(
    "/{session_id}",
    dependencies=[Depends(rate_limiter(limit=30, window_seconds=60))],
)
async def delete_session(
    session_id: UUID,
    db: DbSession,
    storage: Storage,
    request: Request,
    reason: Optional[str] = Query(default=None, max_length=200),
) -> dict:
    """Soft-delete a session for operational removal (data retained)."""
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.deleted_at:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.deleted_at:
        return {"status": "deleted", "id": str(session_id)}

    from datetime import datetime, timezone

    session.deleted_at = datetime.now(timezone.utc)
    session.deleted_reason = reason or "operational_delete"
    await db.commit()

    await log_audit_event(
        db,
        event_type="session.deleted",
        status="success",
        session_id=str(session.id),
        resource_type="session",
        resource_id=str(session.id),
        request_id=_get_request_id(request),
        ip_address=extract_client_ip(request),
        user_agent=request.headers.get("User-Agent"),
        metadata={"reason": session.deleted_reason},
        request=request,
    )

    return {"status": "deleted", "id": str(session_id)}


@router.post(
    "/{session_id}/erase",
    dependencies=[Depends(rate_limiter(limit=10, window_seconds=60))],
)
async def erase_session(
    session_id: UUID,
    db: DbSession,
    storage: Storage,
    request: Request,
    reason: str = Query(..., min_length=3, max_length=200),
) -> dict:
    """GDPR erasure: hard delete all data and assets with audit logging."""
    from sqlalchemy import select

    query = (
        select(KYCSession)
        .where(KYCSession.id == session_id)
        .with_for_update(nowait=False)
    )
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await log_audit_event(
        db,
        event_type="session.erasure_requested",
        status="requested",
        session_id=str(session.id),
        resource_type="session",
        resource_id=str(session.id),
        request_id=_get_request_id(request),
        ip_address=extract_client_ip(request),
        user_agent=request.headers.get("User-Agent"),
        metadata={"reason": reason},
        request=request,
    )

    # Delete all stored assets for this session
    deleted_objects = 0
    deleted_objects += await storage.delete_prefix(f"sessions/{session_id}/")
    deleted_objects += await storage.delete_prefix(f"crops/{session_id}/")

    # Delete session and related records (cascade)
    await db.delete(session)
    await db.commit()

    await log_audit_event(
        db,
        event_type="session.erasure_completed",
        status="success",
        session_id=str(session_id),
        resource_type="session",
        resource_id=str(session_id),
        request_id=_get_request_id(request),
        ip_address=extract_client_ip(request),
        user_agent=request.headers.get("User-Agent"),
        metadata={
            "reason": reason,
            "deleted_prefixes": [f"sessions/{session_id}/", f"crops/{session_id}/"],
            "deleted_objects": deleted_objects,
        },
        request=request,
    )

    return {"status": "erased", "id": str(session_id), "deleted_objects": deleted_objects}

@router.get("/{session_id}/similar", response_model=SimilarFacesListResponse)
async def find_similar_sessions(
    session_id: UUID,
    db: DbSession,
    threshold: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0-1, default 0.7 = 70% similar)",
    ),
    limit: int = Query(default=10, ge=1, le=50),
) -> SimilarFacesListResponse:
    """Find sessions with similar face embeddings using pgvector.
    
    This endpoint is useful for fraud ring detection - finding sessions
    that may belong to the same person or use similar faces.
    
    Args:
        session_id: Source session to find similar faces for
        threshold: Minimum similarity score (0-1). Default 0.7 means
            only return sessions with >= 70% face similarity.
        limit: Maximum number of results to return
    
    Returns:
        List of similar sessions with similarity scores
    """
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.face_embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Session does not have a face embedding",
        )

    # Cosine distance to similarity: similarity = 1 - distance
    # So for threshold=0.7, we want distance < 0.3
    max_distance = 1.0 - threshold

    # Use pgvector cosine distance for similarity search with threshold filtering
    # Select both the session and the computed distance for sorting/filtering
    distance_expr = KYCSession.face_embedding.cosine_distance(session.face_embedding)

    # PERFORMANCE: Use selectinload to prevent N+1 query when accessing session.result
    query = (
        select(
            KYCSession,
            distance_expr.label("distance"),
        )
        .options(selectinload(KYCSession.result))  # Eager load results
        .where(KYCSession.id != session_id)
        .where(KYCSession.face_embedding.isnot(None))
        .where(KYCSession.deleted_at.is_(None))
        .where(distance_expr < max_distance)  # Filter by threshold
        .order_by(distance_expr)
        .limit(limit)
    )

    result = await db.execute(query)
    rows = result.all()

    # Build response with similarity scores
    matches: list[SimilarFaceResponse] = []
    for row in rows:
        similar_session = row[0]  # KYCSession object
        distance = float(row[1])  # Cosine distance
        similarity = 1.0 - distance
        
        # Get decision and risk score from result if exists
        decision = None
        risk_score = None
        if similar_session.result:
            decision = similar_session.result.decision
            risk_score = similar_session.result.risk_score
        
        matches.append(
            SimilarFaceResponse(
                session_id=similar_session.id,
                created_at=similar_session.created_at,
                status=similar_session.status,
                source=similar_session.source,
                attack_family=similar_session.attack_family,
                similarity_score=round(similarity, 4),
                distance=round(distance, 4),
                decision=decision,
                risk_score=risk_score,
            )
        )

    return SimilarFacesListResponse(
        source_session_id=session_id,
        threshold=threshold,
        matches=matches,
        total_matches=len(matches),
    )
