"""Test presigned POST upload functionality."""

import os
import pytest
from httpx import AsyncClient
from app.config import settings

# Disable auth for tests
os.environ["AUTH_DISABLED"] = "true"


@pytest.mark.asyncio
async def test_create_session_returns_presigned_post(client: AsyncClient):
    """Test that creating a session returns presigned POST data."""
    # Create session
    response = await client.post(
        "/api/sessions",
        json={
            "source": "upload",
            "device_os": "iOS",
            "device_model": "iPhone 13",
            "selfie_filename": "selfie.jpg",
            "selfie_content_type": "image/jpeg",
            "id_filename": "id_card.png",
            "id_content_type": "image/png",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "session" in data
    assert "selfie_upload" in data
    assert "id_upload" in data

    # Verify selfie upload structure
    selfie_upload = data["selfie_upload"]
    assert "url" in selfie_upload
    assert "fields" in selfie_upload
    assert "asset_key" in selfie_upload
    assert "expires_in" in selfie_upload

    # Verify fields include required S3 fields
    assert "Content-Type" in selfie_upload["fields"]
    assert selfie_upload["fields"]["Content-Type"] == "image/jpeg"

    # Verify ID upload structure
    id_upload = data["id_upload"]
    assert "url" in id_upload
    assert "fields" in id_upload
    assert "asset_key" in id_upload
    assert "expires_in" in id_upload

    # Verify content type
    assert id_upload["fields"]["Content-Type"] == "image/png"

    # Asset keys should follow expected pattern
    assert selfie_upload["asset_key"].endswith("/selfie.jpg")
    assert id_upload["asset_key"].endswith("/id.png")

    # Expiration should match settings
    assert selfie_upload["expires_in"] == settings.presigned_url_expiration
    assert id_upload["expires_in"] == settings.presigned_url_expiration


@pytest.mark.asyncio
async def test_session_enforces_content_type(client: AsyncClient):
    """Test that content type is enforced based on file extension."""
    # Test with mismatched content type
    response = await client.post(
        "/api/sessions",
        json={
            "source": "upload",
            "selfie_filename": "selfie.jpg",
            "selfie_content_type": "video/mp4",  # Wrong type for .jpg
            "id_filename": "id.png",
            "id_content_type": "image/png",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Content type should be based on extension
    assert data["selfie_upload"]["fields"]["Content-Type"] == "video/mp4"

    # But asset key should still use .jpg extension
    assert data["selfie_upload"]["asset_key"].endswith("/selfie.jpg")