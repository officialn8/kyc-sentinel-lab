"""Tests for trusted GeoIP behavior."""

import pytest
from httpx import AsyncClient

from app.api import sessions as sessions_api


@pytest.mark.asyncio
async def test_create_session_does_not_trust_client_country_when_geoip_unavailable(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions_api, "lookup_ip_country", lambda _ip: None)

    create = await client.post(
        "/api/sessions",
        json={"source": "upload", "ip_country": "US"},
    )
    assert create.status_code == 200
    session_id = create.json()["session"]["id"]

    detail = await client.get(f"/api/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["ip_country"] is None
