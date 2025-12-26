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
        self, key: str, content_type: str | None = None
    ) -> tuple[str, int]:
        """Generate a presigned URL for uploading an object.
        
        Returns:
            Tuple of (presigned_url, expiration_seconds)
        """
        async with self.session.client(**self._get_client_config()) as client:
            params = {"Bucket": self.bucket, "Key": key}
            # If ContentType is included in the signature, clients must send the
            # same Content-Type header (helps prevent obvious misuse).
            if content_type:
                params["ContentType"] = content_type
            url = await client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=self.url_expiration,
            )
            return url, self.url_expiration

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



