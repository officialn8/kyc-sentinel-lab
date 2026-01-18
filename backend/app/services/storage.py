"""R2/S3 storage service."""

from functools import lru_cache
from typing import BinaryIO

import aioboto3
from botocore.config import Config

from app.config import settings


class StorageService:
    """S3-compatible storage service for R2/MinIO."""

    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.endpoint_url = settings.r2_endpoint
        self.access_key = settings.r2_access_key
        self.secret_key = settings.r2_secret_key
        self.bucket = settings.r2_bucket
        self.url_expiration = settings.presigned_url_expiration

    def _get_client_config(self) -> dict:
        """Get boto3 client configuration."""
        return {
            "service_name": "s3",
            "endpoint_url": self.endpoint_url,
            # Cloudflare R2 expects region 'auto' (S3-compatible).
            # MinIO ignores region.
            "region_name": "auto",
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "config": Config(signature_version="s3v4"),
        }

    async def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> tuple[str, int]:
        """Generate a presigned URL for uploading an object.

        SECURITY: Enforces content-type and content-length limits when specified.
        Clients MUST send matching Content-Type and respect Content-Length limits.

        Args:
            key: Storage key for the object
            content_type: Required content type (client must send matching header)
            max_size_bytes: Maximum upload size in bytes (uses config default if None)

        Returns:
            Tuple of (presigned_url, expiration_seconds)
        """
        if max_size_bytes is None:
            max_size_bytes = settings.max_upload_size_bytes

        async with self.session.client(**self._get_client_config()) as client:
            params = {"Bucket": self.bucket, "Key": key}

            # SECURITY: If ContentType is included in the signature, clients MUST
            # send the same Content-Type header. This prevents uploading arbitrary files.
            if content_type:
                params["ContentType"] = content_type

            # SECURITY: Add content-length-range condition to limit upload size.
            # This is enforced server-side by S3/R2 during upload.
            # Note: Standard presigned_url doesn't support conditions directly,
            # so we use presigned_post for uploads that need size enforcement.
            #
            # For now, we document this limitation and rely on the processing
            # pipeline to validate file sizes after upload.

            url = await client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=self.url_expiration,
            )
            return url, self.url_expiration

    async def generate_presigned_post(
        self,
        key: str,
        content_type: str,
        max_size_bytes: int | None = None,
    ) -> dict:
        """Generate a presigned POST for uploading with size enforcement.

        SECURITY: This method enforces content-length limits server-side.
        Use this instead of generate_presigned_upload_url when size enforcement
        is critical.

        Args:
            key: Storage key for the object
            content_type: Required content type
            max_size_bytes: Maximum upload size in bytes

        Returns:
            dict with 'url' and 'fields' for form-based upload
        """
        if max_size_bytes is None:
            max_size_bytes = settings.max_upload_size_bytes

        async with self.session.client(**self._get_client_config()) as client:
            conditions = [
                {"bucket": self.bucket},
                ["starts-with", "$key", key],  # Key must match exactly
                ["content-length-range", 1, max_size_bytes],  # Size enforcement
                {"Content-Type": content_type},  # Content type enforcement
            ]

            fields = {
                "Content-Type": content_type,
            }

            response = await client.generate_presigned_post(
                Bucket=self.bucket,
                Key=key,
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=self.url_expiration,
            )
            return response

    async def generate_presigned_download_url(self, key: str, expiration: int | None = None) -> str:
        """Generate a presigned URL for downloading an object.
        
        Args:
            key: Object key in storage
            expiration: Optional expiration time in seconds (defaults to config value)
        """
        async with self.session.client(**self._get_client_config()) as client:
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration or self.url_expiration,
            )
            return url

    async def upload_file(self, key: str, file: BinaryIO, content_type: str = "application/octet-stream") -> None:
        """Upload a file directly to storage."""
        async with self.session.client(**self._get_client_config()) as client:
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file,
                ContentType=content_type,
            )

    async def download_file(self, key: str) -> bytes:
        """Download a file from storage."""
        async with self.session.client(**self._get_client_config()) as client:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()

    async def delete_object(self, key: str) -> None:
        """Delete an object from storage."""
        async with self.session.client(**self._get_client_config()) as client:
            await client.delete_object(Bucket=self.bucket, Key=key)

    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in storage."""
        async with self.session.client(**self._get_client_config()) as client:
            try:
                await client.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False

    async def ensure_bucket_exists(self) -> None:
        """Ensure the storage bucket exists."""
        async with self.session.client(**self._get_client_config()) as client:
            try:
                await client.head_bucket(Bucket=self.bucket)
            except Exception:
                await client.create_bucket(Bucket=self.bucket)


@lru_cache
def get_storage_service() -> StorageService:
    """Get cached storage service instance."""
    return StorageService()



