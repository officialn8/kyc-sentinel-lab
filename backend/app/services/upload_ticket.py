"""Signed upload ticket utilities for finalize-time validation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from typing import Any

from app.config import settings


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _ticket_secret() -> str:
    secret = (
        (settings.upload_ticket_secret or "").strip()
        or (settings.backend_api_key or "").strip()
        or (settings.webhook_secret or "").strip()
        or (settings.r2_secret_key or "").strip()
    )
    if not secret:
        raise ValueError("No upload ticket secret configured")
    return secret


def _sign(payload_json: str) -> str:
    secret = _ticket_secret().encode("utf-8")
    return hmac.new(secret, payload_json.encode("utf-8"), hashlib.sha256).hexdigest()


def ticket_hash(ticket: str) -> str:
    return hashlib.sha256(ticket.encode("utf-8")).hexdigest()


@dataclass
class UploadTicketPayload:
    version: int
    session_id: str
    asset_key: str
    content_type: str
    max_size_bytes: int
    claimed_size_bytes: int | None
    expires_at: int
    nonce: str

    @property
    def expires_at_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.expires_at, tz=timezone.utc)


class UploadTicketError(ValueError):
    """Ticket validation failure with stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def generate_upload_ticket(
    *,
    session_id: str,
    asset_key: str,
    content_type: str,
    max_size_bytes: int,
    claimed_size_bytes: int | None,
    expires_in_seconds: int,
) -> tuple[str, UploadTicketPayload]:
    """Generate an opaque signed ticket for upload validation."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=expires_in_seconds)

    payload = {
        "v": 1,
        "session_id": str(session_id),
        "asset_key": str(asset_key),
        "content_type": (content_type or "").strip().lower(),
        "max_size_bytes": int(max_size_bytes),
        "claimed_size_bytes": int(claimed_size_bytes) if claimed_size_bytes else None,
        "expires_at": int(expires_at.timestamp()),
        "nonce": secrets.token_urlsafe(12),
    }
    payload_json = _json_dumps(payload)
    signature = _sign(payload_json)
    token = f"{_b64url_encode(payload_json.encode('utf-8'))}.{_b64url_encode(signature.encode('utf-8'))}"

    parsed = UploadTicketPayload(
        version=payload["v"],
        session_id=payload["session_id"],
        asset_key=payload["asset_key"],
        content_type=payload["content_type"],
        max_size_bytes=payload["max_size_bytes"],
        claimed_size_bytes=payload["claimed_size_bytes"],
        expires_at=payload["expires_at"],
        nonce=payload["nonce"],
    )
    return token, parsed


def validate_upload_ticket(
    ticket: str,
    *,
    expected_session_id: str,
    expected_asset_key: str,
    expected_content_type: str,
    expected_hash: str | None = None,
    now: datetime | None = None,
) -> UploadTicketPayload:
    """Validate ticket signature and claims."""
    if not ticket:
        raise UploadTicketError("ticket_missing", "Upload ticket is required")

    if expected_hash and not hmac.compare_digest(ticket_hash(ticket), expected_hash):
        raise UploadTicketError("ticket_hash_mismatch", "Upload ticket does not match the expected token")

    parts = ticket.split(".", 1)
    if len(parts) != 2:
        raise UploadTicketError("ticket_malformed", "Upload ticket is malformed")

    try:
        payload_json = _b64url_decode(parts[0]).decode("utf-8")
        provided_sig = _b64url_decode(parts[1]).decode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise UploadTicketError("ticket_decode_failed", "Upload ticket decoding failed") from exc

    expected_sig = _sign(payload_json)
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise UploadTicketError("ticket_signature_invalid", "Upload ticket signature is invalid")

    try:
        payload = json.loads(payload_json)
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise UploadTicketError("ticket_payload_invalid", "Upload ticket payload is invalid") from exc

    required_fields = {
        "v",
        "session_id",
        "asset_key",
        "content_type",
        "max_size_bytes",
        "expires_at",
        "nonce",
    }
    if not required_fields.issubset(set(payload)):
        raise UploadTicketError("ticket_payload_missing_fields", "Upload ticket payload is missing required fields")

    current = now or datetime.now(timezone.utc)
    expires_at = datetime.fromtimestamp(int(payload["expires_at"]), tz=timezone.utc)
    if expires_at <= current:
        raise UploadTicketError("ticket_expired", "Upload ticket has expired")

    if str(payload["session_id"]) != str(expected_session_id):
        raise UploadTicketError("ticket_session_mismatch", "Upload ticket session does not match request session")

    if str(payload["asset_key"]) != str(expected_asset_key):
        raise UploadTicketError("ticket_asset_mismatch", "Upload ticket asset key does not match expected asset")

    normalized_expected_type = (expected_content_type or "").strip().lower()
    normalized_ticket_type = (str(payload["content_type"]) or "").strip().lower()
    if normalized_ticket_type != normalized_expected_type:
        raise UploadTicketError("ticket_content_type_mismatch", "Upload ticket content type does not match expected type")

    max_size = int(payload["max_size_bytes"])
    if max_size <= 0:
        raise UploadTicketError("ticket_max_size_invalid", "Upload ticket max size is invalid")

    claimed = payload.get("claimed_size_bytes")
    if claimed is not None:
        claimed = int(claimed)
        if claimed <= 0:
            raise UploadTicketError("ticket_claimed_size_invalid", "Upload ticket claimed size is invalid")
        if claimed > max_size:
            raise UploadTicketError("ticket_claimed_size_exceeds_max", "Upload ticket claimed size exceeds max size")

    return UploadTicketPayload(
        version=int(payload["v"]),
        session_id=str(payload["session_id"]),
        asset_key=str(payload["asset_key"]),
        content_type=normalized_ticket_type,
        max_size_bytes=max_size,
        claimed_size_bytes=claimed,
        expires_at=int(payload["expires_at"]),
        nonce=str(payload["nonce"]),
    )
