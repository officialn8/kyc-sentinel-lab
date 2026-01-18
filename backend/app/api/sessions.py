"""Session CRUD endpoints."""

import os
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import select, func, or_, cast, String
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Storage
from app.api.security import rate_limiter, extract_client_ip
from app.models.session import KYCSession
from app.services.geolocation import lookup_ip_country
from app.models.result import KYCResult
from app.services.job_queue import enqueue_process_session
from app.services.processing import get_processing_backend
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionCreateResponse,
    SessionListResponse,
    SessionDetail,
    PresignedUrlResponse,
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

    selfie_key = f"sessions/{session.id}/selfie{selfie_ext}"
    id_key = f"sessions/{session.id}/id{id_ext}"

    selfie_url, selfie_expiry = await storage.generate_presigned_upload_url(
        selfie_key, content_type=data.selfie_content_type
    )
    id_url, id_expiry = await storage.generate_presigned_upload_url(
        id_key, content_type=data.id_content_type
    )

    # Store asset keys
    session.selfie_asset_key = selfie_key
    session.id_asset_key = id_key

    await db.commit()
    await db.refresh(session)

    return SessionCreateResponse(
        session=SessionResponse.model_validate(session),
        upload_urls=PresignedUrlResponse(
            selfie_upload_url=selfie_url,
            selfie_asset_key=selfie_key,
            id_upload_url=id_url,
            id_asset_key=id_key,
            expires_in=selfie_expiry,
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
) -> dict:
    """Mark uploads complete and start processing.

    SECURITY: Uses row-level locking (SELECT FOR UPDATE) to prevent race conditions.
    Multiple concurrent finalize calls will serialize, ensuring only one wins.

    Args:
        force: If True, allows restarting processing for stuck sessions.
    """
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

    # Check status AFTER acquiring lock to prevent TOCTOU race
    if session.status == "processing":
        if force:
            # Session is stuck, allow retry
            pass
        else:
            raise HTTPException(
                status_code=409,  # Conflict
                detail="Session is already being processed. Use ?force=true to retry stuck sessions.",
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

    # Import settings to determine processing mode
    from app.config import settings

    if settings.use_worker:
        # Production: use durable job queue (requires separate worker service)
        await enqueue_process_session(db, str(session_id))
        await db.commit()  # Releases the row lock
        return {"status": "processing", "message": "Session queued for worker processing"}
    else:
        # Development: process directly via background task (no worker needed)
        await db.commit()  # Releases the row lock
        backend = get_processing_backend()
        background_tasks.add_task(backend.process_session, str(session_id))
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
) -> SessionListResponse:
    """List sessions with filters, search, and pagination."""
    # Build query
    query = select(KYCSession)

    # Text search across multiple fields
    if search:
        search_term = f"%{search.lower()}%"
        query = query.where(
            or_(
                # Search by session ID (UUID cast to string for LIKE matching)
                cast(KYCSession.id, String).ilike(search_term),
                # Search by attack family
                func.lower(KYCSession.attack_family).like(search_term),
                # Search by status
                func.lower(KYCSession.status).like(search_term),
                # Search by source
                func.lower(KYCSession.source).like(search_term),
            )
        )

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
) -> dict:
    """Delete a session and its associated data."""
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete assets from storage
    if session.selfie_asset_key:
        await storage.delete_object(session.selfie_asset_key)
    if session.id_asset_key:
        await storage.delete_object(session.id_asset_key)

    # Delete session (cascades to result, reasons, frame_metrics)
    await db.delete(session)
    await db.commit()

    return {"status": "deleted", "id": str(session_id)}


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



