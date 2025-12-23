"""Application configuration via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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



