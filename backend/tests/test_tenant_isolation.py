"""Tests for tenant-scoped API behavior."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_session_without_tenant_header_rejected(
    no_tenant_client: AsyncClient,
) -> None:
    response = await no_tenant_client.post("/api/sessions", json={"source": "upload"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Tenant context required"


@pytest.mark.asyncio
async def test_cross_tenant_get_session_returns_404(client: AsyncClient) -> None:
    create = await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"source": "upload"},
    )
    assert create.status_code == 200
    session_id = create.json()["session"]["id"]

    get_other = await client.get(
        f"/api/sessions/{session_id}",
        headers={"X-Authenticated-Org-Id": "org_beta"},
    )
    assert get_other.status_code == 404


@pytest.mark.asyncio
async def test_cross_tenant_similar_source_hidden(client: AsyncClient) -> None:
    create = await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"source": "upload"},
    )
    assert create.status_code == 200
    session_id = create.json()["session"]["id"]

    response = await client.get(
        f"/api/sessions/{session_id}/similar",
        headers={"X-Authenticated-Org-Id": "org_beta"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions_tenant_scoped(client: AsyncClient) -> None:
    await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"source": "upload"},
    )
    await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_beta"},
        json={"source": "upload"},
    )

    alpha = await client.get(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
    )
    beta = await client.get(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_beta"},
    )

    assert alpha.status_code == 200
    assert beta.status_code == 200
    assert alpha.json()["total"] == 1
    assert beta.json()["total"] == 1


@pytest.mark.asyncio
async def test_metrics_summary_tenant_scoped(client: AsyncClient) -> None:
    await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"source": "upload"},
    )
    await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_beta"},
        json={"source": "upload"},
    )

    alpha_summary = await client.get(
        "/api/metrics/summary",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
    )
    beta_summary = await client.get(
        "/api/metrics/summary",
        headers={"X-Authenticated-Org-Id": "org_beta"},
    )

    assert alpha_summary.status_code == 200
    assert beta_summary.status_code == 200
    assert alpha_summary.json()["total_sessions"] == 1
    assert beta_summary.json()["total_sessions"] == 1


@pytest.mark.asyncio
async def test_simulate_endpoint_tenant_scoped(client: AsyncClient) -> None:
    simulate = await client.post(
        "/api/simulate",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"attack_family": "benign", "attack_severity": "low", "count": 1},
    )
    assert simulate.status_code == 200

    alpha_list = await client.get(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
    )
    beta_list = await client.get(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_beta"},
    )

    assert alpha_list.status_code == 200
    assert beta_list.status_code == 200
    assert alpha_list.json()["total"] == 1
    assert beta_list.json()["total"] == 0


@pytest.mark.asyncio
async def test_upload_presigned_without_tenant_header_rejected(
    no_tenant_client: AsyncClient,
) -> None:
    response = await no_tenant_client.post(
        "/api/upload/presigned",
        params={"key": "sessions/00000000-0000-0000-0000-000000000001/selfie.jpg"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Tenant context required"


@pytest.mark.asyncio
async def test_upload_presigned_cross_tenant_returns_404(client: AsyncClient) -> None:
    create = await client.post(
        "/api/sessions",
        headers={"X-Authenticated-Org-Id": "org_alpha"},
        json={"source": "upload"},
    )
    assert create.status_code == 200
    session_id = create.json()["session"]["id"]

    response = await client.post(
        "/api/upload/presigned",
        headers={"X-Authenticated-Org-Id": "org_beta"},
        params={"key": f"sessions/{session_id}/selfie.jpg"},
    )
    assert response.status_code == 404
