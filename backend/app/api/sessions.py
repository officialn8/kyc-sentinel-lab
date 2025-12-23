"""Session CRUD endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Storage, Processing
from app.models.session import KYCSession
from app.models.result import KYCResult
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionCreateResponse,
    SessionListResponse,
    SessionDetail,
    PresignedUrlResponse,
)

router = APIRouter()


@router.post("", response_model=SessionCreateResponse)
async def create_session(
    data: SessionCreate,
    db: DbSession,
    storage: Storage,
) -> SessionCreateResponse:
    """Create a new KYC session and get presigned URLs for upload."""
    # Create session
    session = KYCSession(
        source=data.source,
        attack_family=data.attack_family,
        attack_severity=data.attack_severity,
        device_os=data.device_os,
        device_model=data.device_model,
        ip_country=data.ip_country,
        status="pending",
    )
    db.add(session)
    await db.flush()

    # Generate presigned URLs for uploads
    selfie_key = f"sessions/{session.id}/selfie"
    id_key = f"sessions/{session.id}/id"

    selfie_url, selfie_expiry = await storage.generate_presigned_upload_url(selfie_key)
    id_url, id_expiry = await storage.generate_presigned_upload_url(id_key)

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


@router.post("/{session_id}/finalize")
async def finalize_session(
    session_id: UUID,
    db: DbSession,
    processing: Processing,
    background_tasks: BackgroundTasks,
) -> dict:
    """Mark uploads complete and start processing."""
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Session is already {session.status}",
        )

    # Update status
    session.status = "processing"
    await db.commit()

    # Start background processing
    background_tasks.add_task(processing.process_session, str(session_id))

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
) -> SessionListResponse:
    """List sessions with filters and pagination."""
    # Build query
    query = select(KYCSession)

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
        result=session.result,
        reasons=session.reasons,
    )


@router.delete("/{session_id}")
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


@router.get("/{session_id}/similar", response_model=list[SessionResponse])
async def find_similar_sessions(
    session_id: UUID,
    db: DbSession,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[SessionResponse]:
    """Find sessions with similar face embeddings using pgvector."""
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.face_embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Session does not have a face embedding",
        )

    # Use pgvector cosine distance for similarity search
    query = (
        select(KYCSession)
        .where(KYCSession.id != session_id)
        .where(KYCSession.face_embedding.isnot(None))
        .order_by(KYCSession.face_embedding.cosine_distance(session.face_embedding))
        .limit(limit)
    )

    result = await db.execute(query)
    similar = result.scalars().all()

    return [SessionResponse.model_validate(s) for s in similar]



