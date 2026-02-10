"""Tests for fraud ring evidence redaction."""

import json

import pytest

from app.services.fraud_detection import check_fraud_ring


class _FakeResult:
    def __init__(self, row: tuple[int, float]):
        self._row = row

    def one(self) -> tuple[int, float]:
        return self._row


class _FakeDB:
    async def execute(self, _stmt):
        return _FakeResult((3, 0.02))


@pytest.mark.asyncio
async def test_fraud_ring_payload_contains_only_aggregate_fields() -> None:
    result = await check_fraud_ring(
        db=_FakeDB(),
        tenant_id="org_alpha",
        session_id="00000000-0000-0000-0000-000000000001",
        face_embedding=[0.1] * 512,
        similarity_threshold=0.8,
        min_matches=2,
    )

    assert result is not None
    assert set(result.keys()) == {"match_count", "max_similarity", "threshold"}
    assert "samples" not in result
    assert "session_id" not in json.dumps(result)
