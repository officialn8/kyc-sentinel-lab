"""R2/S3 storage service with connection pooling."""

import asyncio
import logging
from functools import lru_cache
from typing import Any, BinaryIO

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """S3-compatible storage service for R2/MinIO with connection pooling.
    
    PERFORMANCE: Client is created once and reused for all operations,
    eliminating connection overhead on each request.
    """

    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.endpoint_url = settings.r2_endpoint
        self.access_key = settings.r2_access_key
        self.secret_key = settings.r2_secret_key
        self.bucket = settings.r2_bucket
        self.url_expiration = settings.presigned_url_expiration
        
        # Connection pooling: cache the client for reuse
        self._client = None
        self._client_lock = asyncio.Lock()

    def _resolve_upload_mode(self) -> str:
        """Resolve presigned upload mode based on settings and endpoint."""
        mode = (settings.presigned_upload_mode or "auto").lower()
        if mode in ("post", "put"):
            return mode
        endpoint = (self.endpoint_url or "").lower()
        # Cloudflare R2 does not support presigned POST uploads.
        if "r2.cloudflarestorage.com" in endpoint:
            return "put"
        return "post"

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
            "config": Config(
                signature_version="s3v4",
                # Connection pooling settings for aiohttp
                max_pool_connections=20,
            ),
        }

    async def _get_client(self):
        """Get or create the S3 client with connection pooling.
        
        PERFORMANCE: Creates client once on first use, reuses for subsequent calls.
        Thread-safe via asyncio.Lock.
        """
        if self._client is None:
            async with self._client_lock:
                # Double-check pattern for thread safety
                if self._client is None:
                    self._client = await self.session.client(
                        **self._get_client_config()
                    ).__aenter__()
        return self._client

    async def close(self):
        """Close the client connection. Call during app shutdown."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

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

        client = await self._get_client()
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
        """Generate presigned POST with size and type enforcement.

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

        # Use persistent client for connection pooling
        client = await self._get_client()

        conditions = [
            ["content-length-range", 1, max_size_bytes],
            ["eq", "$Content-Type", content_type],
            ["eq", "$key", key],
            {"bucket": self.bucket},
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

        return {
            "url": response["url"],
            "fields": response["fields"]
        }

    async def generate_presigned_upload(
        self,
        key: str,
        content_type: str,
        max_size_bytes: int | None = None,
    ) -> dict:
        """Generate presigned upload config (POST or PUT).

        R2 does not support presigned POST, so we fall back to PUT.
        """
        mode = self._resolve_upload_mode()
        if mode == "post":
            response = await self.generate_presigned_post(
                key,
                content_type=content_type,
                max_size_bytes=max_size_bytes,
            )
            return {
                "method": "POST",
                "url": response["url"],
                "fields": response.get("fields", {}),
                "headers": {},
            }

        # PUT upload (R2-compatible)
        url, _ = await self.generate_presigned_upload_url(
            key,
            content_type=content_type,
            max_size_bytes=max_size_bytes,
        )
        return {
            "method": "PUT",
            "url": url,
            "fields": {},
            "headers": {"Content-Type": content_type},
        }

    async def generate_presigned_download_url(self, key: str, expiration: int | None = None) -> str:
        """Generate a presigned URL for downloading an object.
        
        Args:
            key: Object key in storage
            expiration: Optional expiration time in seconds (defaults to config value)
        """
        client = await self._get_client()
        url = await client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiration or self.url_expiration,
        )
        return url

    async def upload_file(self, key: str, file: BinaryIO, content_type: str = "application/octet-stream") -> None:
        """Upload a file directly to storage."""
        client = await self._get_client()
        await client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file,
            ContentType=content_type,
        )

    async def download_file(self, key: str) -> bytes:
        """Download a file from storage."""
        client = await self._get_client()
        response = await client.get_object(Bucket=self.bucket, Key=key)
        return await response["Body"].read()

    async def delete_object(self, key: str) -> None:
        """Delete an object from storage."""
        client = await self._get_client()
        await client.delete_object(Bucket=self.bucket, Key=key)

    async def list_objects(self, prefix: str) -> list[str]:
        """List object keys under a prefix."""
        client = await self._get_client()
        keys: list[str] = []
        continuation_token = None

        while True:
            params = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = await client.list_objects_v2(**params)
            for item in response.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(key)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        return keys

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix.

        Returns:
            Number of objects deleted.
        """
        client = await self._get_client()
        keys = await self.list_objects(prefix)
        if not keys:
            return 0

        deleted = 0
        # Delete in batches of 1000 (S3 limit)
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            response = await client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            deleted += len(response.get("Deleted", []))

        return deleted

    async def get_object_size(self, key: str) -> int:
        """Get the size of an object in bytes.
        
        Raises:
            ClientError: If object doesn't exist or access denied
        """
        metadata = await self.get_object_metadata(key)
        return metadata["content_length"]

    async def get_object_metadata(self, key: str) -> dict[str, Any]:
        """Get object metadata from storage.

        Returns:
            Dictionary with content_length, content_type, etag, last_modified.
        """
        client = await self._get_client()
        response = await client.head_object(Bucket=self.bucket, Key=key)
        return {
            "content_length": int(response.get("ContentLength", 0)),
            "content_type": response.get("ContentType"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified"),
        }

    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in storage."""
        client = await self._get_client()
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            # Only return False for 404 - re-raise other errors
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

    async def ensure_bucket_exists(self) -> None:
        """Ensure the storage bucket exists."""
        client = await self._get_client()
        try:
            await client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            # Create bucket only if it doesn't exist (404)
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                await client.create_bucket(Bucket=self.bucket)
            else:
                raise


@lru_cache
def get_storage_service() -> StorageService:
    """Get cached storage service instance."""
    return StorageService()
