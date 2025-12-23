"""Upload-related endpoints."""

from fastapi import APIRouter

from app.api.deps import Storage

router = APIRouter()


@router.post("/presigned")
async def get_presigned_url(
    key: str,
    storage: Storage,
) -> dict:
    """Get a presigned URL for uploading a file."""
    url, expires_in = await storage.generate_presigned_upload_url(key)
    return {
        "upload_url": url,
        "key": key,
        "expires_in": expires_in,
    }




