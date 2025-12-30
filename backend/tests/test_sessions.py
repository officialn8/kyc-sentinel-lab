"""Tests for session endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_create_session(client: AsyncClient) -> None:
    """Test creating a new session."""
    response = await client.post(
        "/api/sessions",
        json={"source": "upload"},
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "session" in data
    assert "upload_urls" in data
    assert data["session"]["status"] == "pending"
    assert data["session"]["source"] == "upload"


@pytest.mark.asyncio
async def test_create_synthetic_session(client: AsyncClient) -> None:
    """Test creating a synthetic session."""
    response = await client.post(
        "/api/sessions",
        json={
            "source": "synthetic",
            "attack_family": "replay",
            "attack_severity": "medium",
        },
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["session"]["source"] == "synthetic"
    assert data["session"]["attack_family"] == "replay"


@pytest.mark.asyncio
async def test_list_sessions_empty(client: AsyncClient) -> None:
    """Test listing sessions when empty."""
    response = await client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_sessions_with_data(client: AsyncClient) -> None:
    """Test listing sessions with data."""
    # Create some sessions
    await client.post("/api/sessions", json={"source": "upload"})
    await client.post("/api/sessions", json={"source": "synthetic", "attack_family": "replay"})
    
    response = await client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 2
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_get_session(client: AsyncClient) -> None:
    """Test getting a specific session."""
    # Create a session
    create_response = await client.post("/api/sessions", json={"source": "upload"})
    session_id = create_response.json()["session"]["id"]
    
    # Get the session
    response = await client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == session_id
    assert data["source"] == "upload"


@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent session."""
    response = await client.get("/api/sessions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session(client: AsyncClient) -> None:
    """Test deleting a session."""
    # Create a session
    create_response = await client.post("/api/sessions", json={"source": "upload"})
    session_id = create_response.json()["session"]["id"]
    
    # Delete the session
    response = await client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    
    # Verify it's deleted
    get_response = await client.get(f"/api/sessions/{session_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_list_attack_families(client: AsyncClient) -> None:
    """Test listing attack families."""
    response = await client.get("/api/simulate/families")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data) > 0
    family_ids = [f["id"] for f in data]
    assert "replay" in family_ids
    assert "injection" in family_ids
    assert "face_swap" in family_ids
    assert "doc_tamper" in family_ids
    assert "benign" in family_ids











