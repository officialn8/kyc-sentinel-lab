"""Application configuration via pydantic-settings."""

from pathlib import Path
from functools import lru_cache
import json
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    )

    # Application
    app_name: str = "KYC Sentinel Lab"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/kyc_sentinel"

    # R2/S3 Storage
    r2_endpoint: str = "http://localhost:9000"  # MinIO for local dev
    r2_access_key: str = "minioadmin"
    r2_secret_key: str = "minioadmin"
    r2_bucket: str = "kyc-sentinel-media"

    # Processing backend
    use_modal: bool = False
    modal_environment: str = "dev"  # dev | prod

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Presigned URL expiration (seconds)
    presigned_url_expiration: int = 3600

    # Optional shared secret to protect backend endpoints when running publicly.
    # If unset/empty, auth is disabled.
    backend_api_key: str = ""

    # Optional Basic Auth credentials (recommended for public demo deployments).
    # If unset/empty, Basic Auth is disabled.
    basic_auth_username: str = ""
    basic_auth_password: str = ""

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



