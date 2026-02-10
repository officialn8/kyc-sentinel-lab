"""Application configuration via pydantic-settings."""

from pathlib import Path
from functools import lru_cache
import json
from typing import Literal
from enum import Enum

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    _repo_root = Path(__file__).resolve().parents[2]

    model_config = SettingsConfigDict(
        # Support running from repo root or from within `backend/`
        env_file=(
            str(_repo_root / ".env"),
            str(_repo_root / "backend" / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow shared/root env files to contain frontend-only vars (e.g., Clerk).
        extra="ignore",
    )

    # Application
    app_name: str = "KYC Sentinel Lab"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/kyc_sentinel"

    # R2/S3 Storage
    r2_endpoint: str = "http://127.0.0.1:9000"  # MinIO for local dev
    r2_access_key: str = "minioadmin"
    r2_secret_key: str = "minioadmin"
    r2_bucket: str = "kyc-sentinel-media"

    # Processing backend
    use_modal: bool = False
    modal_environment: str = "dev"  # dev | prod

    # Worker configuration
    # Set to true in production to use durable job queue (requires separate worker service)
    # Set to false in development for direct background processing
    use_worker: bool = False
    # Processing reliability: consider a job/session "stuck" after this many seconds
    # Used by worker reaper and finalize retry logic
    processing_stuck_timeout_seconds: int = 15 * 60

    # Scoring configuration
    # Available profiles: default, fintech_high_risk, crypto_exchange, social_verification
    scoring_profile: str = "default"

    # PAD (Presentation Attack Detection) configuration
    # Available profiles: default, strict, lenient
    pad_profile: str = "default"
    
    # Velocity rule thresholds for fraud detection
    velocity_device_1h_limit: int = 3  # Max submissions per device per hour
    velocity_device_24h_limit: int = 10  # Max submissions per device per 24 hours
    velocity_ip_1h_limit: int = 5  # Max submissions per IP per hour
    velocity_ip_24h_limit: int = 20  # Max submissions per IP per 24 hours

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Presigned URL expiration (seconds)
    presigned_url_expiration: int = 3600

    # Maximum upload file size (bytes) - enforced in presigned URL conditions
    # Default: 50MB for images/videos
    max_upload_size_bytes: int = 50 * 1024 * 1024

    # Upload signing mode for presigned uploads:
    # - "auto": use POST for MinIO/local, PUT for Cloudflare R2
    # - "post": always use presigned POST (S3/MinIO)
    # - "put": always use presigned PUT (R2-compatible)
    presigned_upload_mode: Literal["auto", "post", "put"] = "auto"

    # Audit logging
    audit_log_prefix: str = "audit"

    # Data retention defaults
    retention_media_days: int = 30
    retention_metadata_days: int = 90

    # Fraud ring detection (advisory)
    fraud_ring_similarity_threshold: float = 0.80
    fraud_ring_min_matches: int = 2

    # Local ML hardware toggles (safe defaults for laptops)
    # Set to True only if you have GPU-enabled builds installed.
    face_use_gpu: bool = False
    ocr_use_gpu: bool = False

    # Optional shared secret to protect backend endpoints when running publicly.
    backend_api_key: str = ""

    # Optional Basic Auth credentials (recommended for public demo deployments).
    basic_auth_username: str = ""
    basic_auth_password: str = ""

    # SECURITY: Auth is required by default in production.
    # Set to True explicitly to disable auth (development/testing only).
    # This prevents accidental exposure of unauthenticated APIs.
    auth_disabled: bool = False

    # Trusted proxy CIDRs for X-Forwarded-For parsing.
    # Only trust X-Forwarded-For from these networks.
    # Empty list = trust direct connection only (safest default).
    # Example: "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16" for RFC1918
    # Railway/Render/Vercel typically use private ranges.
    trusted_proxy_cidrs: str = ""

    # GeoIP database path for IP geolocation (MaxMind GeoLite2 or GeoIP2)
    # Download free GeoLite2-Country.mmdb from:
    # https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
    # If not set, tries common system locations.
    geoip_database_path: str = ""

    # Redis URL for distributed rate limiting (optional)
    # If not set, falls back to in-memory rate limiting (single-instance only)
    # Format: redis://[[username]:[password]@]host[:port][/database]
    redis_url: str = ""

    # Webhook signing secret for HMAC signatures
    # Required in production for secure webhook delivery
    webhook_secret: str = ""

    # Secret used to sign upload validation tickets.
    # Required in production to prevent client-side tampering of upload metadata.
    upload_ticket_secret: str = ""

    @model_validator(mode="after")
    def _validate_production_config(self) -> "Settings":
        """
        SECURITY: Fail fast at startup if production config is invalid.

        Production and staging environments require:
        - Authentication configured
        - Webhook secret configured
        - Debug mode disabled
        - Auth cannot be disabled
        """
        if self.environment in (Environment.STAGING, Environment.PRODUCTION):
            errors = []

            # Auth must not be disabled
            if self.auth_disabled:
                errors.append("AUTH_DISABLED=true forbidden in production")

            # Check for valid authentication
            if not self._has_valid_auth():
                errors.append("No authentication configured")

            # Webhook secret required
            if not self.webhook_secret:
                errors.append("WEBHOOK_SECRET required for production")

            # Upload ticket secret required
            if not self.upload_ticket_secret:
                errors.append("UPLOAD_TICKET_SECRET required for production")

            # Debug must be off
            if self.debug:
                errors.append("DEBUG=true forbidden in production")

            if errors:
                raise ValueError(f"Production config errors: {'; '.join(errors)}")

        # For non-production environments, still validate auth if not explicitly disabled
        elif not self.auth_disabled:
            if not self._has_valid_auth():
                raise ValueError(
                    "SECURITY: No authentication configured. "
                    "Set BACKEND_API_KEY or BASIC_AUTH_USERNAME+BASIC_AUTH_PASSWORD, "
                    "or explicitly set AUTH_DISABLED=true for development/testing."
                )

        return self

    def _has_valid_auth(self) -> bool:
        """Check if at least one auth method is configured."""
        has_api_key = bool((self.backend_api_key or "").strip())
        has_basic = bool(
            (self.basic_auth_username or "").strip()
            and (self.basic_auth_password or "").strip()
        )
        return has_api_key or has_basic

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """
        Allow CORS_ORIGINS to be set as:
        - JSON array string: ["https://a.com","https://b.com"]
        - Comma-separated: https://a.com,https://b.com
        - Single string: https://a.com
        """
        if v is None:
            return ["http://localhost:3000"]
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    # Fall back to comma-splitting below
                    pass
            return [p.strip() for p in s.split(",") if p.strip()]
        return v

    @field_validator("fraud_ring_similarity_threshold")
    @classmethod
    def _validate_fraud_ring_similarity_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("FRAUD_RING_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        return v

    @field_validator("fraud_ring_min_matches")
    @classmethod
    def _validate_fraud_ring_min_matches(cls, v: int) -> int:
        if v < 1:
            raise ValueError("FRAUD_RING_MIN_MATCHES must be >= 1")
        return v

    @property
    def async_database_url(self) -> str:
        """Ensure database URL uses asyncpg driver."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
