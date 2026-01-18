"""Upload-related endpoints."""

import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import Storage, DbSession
from app.api.security import rate_limiter
from app.models.session import KYCSession

router = APIRouter()

# SECURITY: Pattern for allowed storage keys
# Only allow keys within session directories with controlled filenames
_ALLOWED_KEY_PATTERN = re.compile(
    r"^sessions/(?P<session_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/"
    r"(selfie|id|frames/\d{4})\.(jpg|jpeg|png|webp|mp4|mov|webm)$",
    re.IGNORECASE,
)


def _validate_storage_key(key: str) -> UUID:
    """
    Validate storage key to prevent path injection attacks.

    SECURITY: Only allow keys matching the session asset pattern.
    Rejects path traversal (..), absolute paths, and keys outside session scope.

    Returns:
        The session UUID extracted from the key for ownership validation.
    """
    if not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Storage key is required",
        )

    # Block path traversal attempts
    if ".." in key or key.startswith("/") or "\\" in key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key: path traversal not allowed",
        )

    # Validate against allowed pattern and extract session ID
    match = _ALLOWED_KEY_PATTERN.match(key)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid storage key: must be sessions/{uuid}/[selfie|id].[ext] format",
        )

    try:
        return UUID(match.group("session_id"))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID in storage key",
        )


@router.post(
    "/presigned",
    dependencies=[Depends(rate_limiter(limit=30, window_seconds=60))],
)
async def get_presigned_url(
    key: str,
    db: DbSession,
    storage: Storage,
    content_type: str | None = None,
) -> dict:
    """Get a presigned URL for uploading a file.

    SECURITY:
    - Keys are validated to prevent storage vandalism
    - Session must exist and be in pending state (prevents writes to processed sessions)
    - Only session-scoped keys are allowed

    NOTE: In production, consider removing this endpoint entirely and only
    issuing presigned URLs during session creation. This provides better
    ownership enforcement since the URL is tied to the session transaction.
    """
    session_id = _validate_storage_key(key)

    # SECURITY: Verify session exists and is still pending
    # This prevents writing to arbitrary session UUIDs or already-processed sessions
    session = await db.get(KYCSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upload to session in '{session.status}' state",
        )

    url, expires_in = await storage.generate_presigned_upload_url(
        key, content_type=content_type
    )
    return {
        "upload_url": url,
        "key": key,
        "expires_in": expires_in,
    }






