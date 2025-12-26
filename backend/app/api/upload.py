"""Upload-related endpoints."""

from fastapi import APIRouter, Depends

from app.api.deps import Storage
from app.api.security import rate_limiter

router = APIRouter()


@router.post(
    "/presigned",
    dependencies=[Depends(rate_limiter(limit=30, window_seconds=60))],
)
async def get_presigned_url(
    key: str,
    content_type: str | None = None,
    storage: Storage,
) -> dict:
    """Get a presigned URL for uploading a file."""
    url, expires_in = await storage.generate_presigned_upload_url(
        key, content_type=content_type
    )
    return {
        "upload_url": url,
        "key": key,
        "expires_in": expires_in,
    }






