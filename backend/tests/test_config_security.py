"""Test environment-aware configuration security."""

import pytest
import os
from app.config import Settings, Environment


class TestEnvironmentConfig:
    """Test production configuration validation."""

    def test_local_env_allows_auth_disabled(self):
        """Local environment should allow AUTH_DISABLED=true."""
        settings = Settings(
            environment=Environment.LOCAL,
            auth_disabled=True,
            cors_origins=["http://localhost:3000"],
        )
        assert settings.environment == Environment.LOCAL
        assert settings.auth_disabled is True

    def test_production_env_requires_auth(self):
        """Production environment must have authentication configured."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                auth_disabled=False,
                # No auth configured
                backend_api_key="",
                basic_auth_username="",
                basic_auth_password="",
                cors_origins=["https://app.example.com"],
            )
        assert "No authentication configured" in str(exc_info.value)

    def test_production_env_forbids_auth_disabled(self):
        """Production environment must not allow AUTH_DISABLED=true."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                auth_disabled=True,
                backend_api_key="test-key",
                webhook_secret="test-secret",
                cors_origins=["https://app.example.com"],
            )
        assert "AUTH_DISABLED=true forbidden in production" in str(exc_info.value)

    def test_production_env_requires_webhook_secret(self):
        """Production environment must have webhook secret configured."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                auth_disabled=False,
                backend_api_key="test-key",
                webhook_secret="",  # Empty webhook secret
                cors_origins=["https://app.example.com"],
            )
        assert "WEBHOOK_SECRET required for production" in str(exc_info.value)

    def test_production_env_forbids_debug(self):
        """Production environment must have debug disabled."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                auth_disabled=False,
                backend_api_key="test-key",
                webhook_secret="test-secret",
                debug=True,  # Debug enabled
                cors_origins=["https://app.example.com"],
            )
        assert "DEBUG=true forbidden in production" in str(exc_info.value)

    def test_staging_env_same_requirements_as_production(self):
        """Staging environment should have same requirements as production."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.STAGING,
                auth_disabled=True,  # Should fail
                backend_api_key="test-key",
                webhook_secret="test-secret",
                cors_origins=["https://staging.example.com"],
            )
        assert "AUTH_DISABLED=true forbidden in production" in str(exc_info.value)

    def test_valid_production_config(self):
        """Valid production configuration should pass validation."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            auth_disabled=False,
            backend_api_key="secure-api-key-here",
            webhook_secret="secure-webhook-secret",
            upload_ticket_secret="secure-upload-ticket-secret",
            debug=False,
            cors_origins=["https://app.example.com"],
        )
        assert settings.environment == Environment.PRODUCTION
        assert settings.auth_disabled is False
        assert settings.backend_api_key == "secure-api-key-here"
        assert settings.webhook_secret == "secure-webhook-secret"
        assert settings.upload_ticket_secret == "secure-upload-ticket-secret"
        assert settings.debug is False

    def test_ci_env_allows_relaxed_config(self):
        """CI environment should allow relaxed configuration for testing."""
        settings = Settings(
            environment=Environment.CI,
            auth_disabled=True,
            webhook_secret="",  # Not required in CI
            debug=True,  # Allowed in CI
            cors_origins=["http://localhost:3000"],
        )
        assert settings.environment == Environment.CI
        assert settings.auth_disabled is True
        assert settings.debug is True

    def test_multiple_production_errors_reported(self):
        """All production config errors should be reported together."""
        with pytest.raises(ValueError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                auth_disabled=True,
                backend_api_key="",
                webhook_secret="",
                debug=True,
                cors_origins=["https://app.example.com"],
            )
        error_msg = str(exc_info.value)
        assert "AUTH_DISABLED=true forbidden" in error_msg
        assert "No authentication configured" in error_msg
        assert "WEBHOOK_SECRET required" in error_msg
        assert "DEBUG=true forbidden" in error_msg
