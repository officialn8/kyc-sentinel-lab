"""Webhook service for dispatching HTTP callbacks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import time
from typing import Optional
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select

from app.config import settings
from app.database import async_session_maker
from app.models.webhook import Webhook
from app.services.webhook_security import WebhookUrlValidationError, validate_webhook_url

logger = logging.getLogger(__name__)

# Configuration
WEBHOOK_TIMEOUT_SECONDS = 10
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds
WEBHOOK_MAX_FAILURES = 10  # Disable webhook after this many consecutive failures
WEBHOOK_MAX_CONCURRENCY = 8


@dataclass
class WebhookDispatchResult:
    webhook_id: str
    success: bool
    error: str | None = None


def _build_signed_headers(
    *,
    event: str,
    body_bytes: bytes,
    delivery_id: str,
    timestamp: str,
) -> dict[str, str]:
    secret = (settings.webhook_secret or "").strip()
    if not secret:
        raise RuntimeError("WEBHOOK_SECRET is required for signed webhook delivery")

    body_text = body_bytes.decode("utf-8")
    signing_input = f"{timestamp}.{delivery_id}.{body_text}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).hexdigest()

    return {
        "Content-Type": "application/json",
        "User-Agent": "KYC-Sentinel-Webhook/1.0",
        # Backward-compatible event header
        "X-Webhook-Event": event,
        # Signed delivery headers
        "X-KYC-Event": event,
        "X-KYC-Timestamp": timestamp,
        "X-KYC-Delivery-Id": delivery_id,
        "X-KYC-Signature": f"v1={signature}",
    }


async def _record_webhook_success(webhook_id: UUID) -> None:
    async with async_session_maker() as db:
        async with db.begin():
            stmt = select(Webhook).where(Webhook.id == webhook_id).with_for_update()
            webhook = (await db.execute(stmt)).scalar_one_or_none()
            if not webhook:
                return
            webhook.failure_count = 0
            webhook.last_failure = None
            webhook.last_success = datetime.now(timezone.utc)


async def _record_webhook_failure(webhook_id: UUID, error: str) -> None:
    async with async_session_maker() as db:
        async with db.begin():
            stmt = select(Webhook).where(Webhook.id == webhook_id).with_for_update()
            webhook = (await db.execute(stmt)).scalar_one_or_none()
            if not webhook:
                return
            webhook.failure_count = int(webhook.failure_count or 0) + 1
            webhook.last_failure = datetime.now(timezone.utc)
            if webhook.failure_count >= WEBHOOK_MAX_FAILURES:
                webhook.enabled = False
                logger.error(
                    "Webhook %s disabled after %d consecutive failures (last_error=%s)",
                    str(webhook.id),
                    webhook.failure_count,
                    error,
                )


async def _send_webhook_with_retry(
    *,
    webhook_id: UUID,
    event: str,
    payload: dict,
) -> WebhookDispatchResult:
    async with async_session_maker() as db:
        webhook = await db.get(Webhook, webhook_id)
        if not webhook or not webhook.enabled:
            return WebhookDispatchResult(webhook_id=str(webhook_id), success=False, error="webhook_not_enabled")
        if event not in (webhook.events or []):
            return WebhookDispatchResult(webhook_id=str(webhook_id), success=False, error="event_not_subscribed")

    try:
        await validate_webhook_url(webhook.url)
    except WebhookUrlValidationError as exc:
        await _record_webhook_failure(webhook_id, f"url_validation:{exc.code}")
        return WebhookDispatchResult(
            webhook_id=str(webhook_id),
            success=False,
            error=f"url_validation:{exc.code}",
        )

    full_payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }
    body_bytes = json.dumps(full_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    delivery_id = str(uuid4())
    delivery_timestamp = str(int(time.time()))

    try:
        headers = _build_signed_headers(
            event=event,
            body_bytes=body_bytes,
            delivery_id=delivery_id,
            timestamp=delivery_timestamp,
        )
    except Exception as exc:
        await _record_webhook_failure(webhook_id, "signature_generation_failed")
        return WebhookDispatchResult(
            webhook_id=str(webhook_id),
            success=False,
            error=f"signature_generation_failed:{exc}",
        )

    last_error = "unknown"
    async with httpx.AsyncClient(
        timeout=WEBHOOK_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:
        for attempt in range(WEBHOOK_MAX_RETRIES):
            try:
                response = await client.post(
                    webhook.url,
                    content=body_bytes,
                    headers=headers,
                )
                if response.is_success:
                    await _record_webhook_success(webhook_id)
                    logger.info(
                        "Webhook delivered id=%s url=%s status=%d delivery_id=%s",
                        str(webhook_id),
                        webhook.url,
                        response.status_code,
                        delivery_id,
                    )
                    return WebhookDispatchResult(webhook_id=str(webhook_id), success=True)

                # 3xx are treated as failure to prevent redirect-based SSRF pivots.
                last_error = f"http_{response.status_code}"
                logger.warning(
                    "Webhook failed id=%s attempt=%d status=%d",
                    str(webhook_id),
                    attempt + 1,
                    response.status_code,
                )
            except Exception as exc:
                last_error = f"request_error:{exc}"
                logger.warning(
                    "Webhook error id=%s attempt=%d error=%s",
                    str(webhook_id),
                    attempt + 1,
                    exc,
                )

            if attempt < WEBHOOK_MAX_RETRIES - 1:
                await asyncio.sleep(WEBHOOK_RETRY_DELAYS[attempt])

    await _record_webhook_failure(webhook_id, last_error)
    return WebhookDispatchResult(webhook_id=str(webhook_id), success=False, error=last_error)


async def dispatch_webhooks(event: str, payload: dict) -> None:
    """Dispatch webhooks for an event to all subscribed, enabled handlers."""
    async with async_session_maker() as db:
        rows = (
            await db.execute(select(Webhook.id, Webhook.events).where(Webhook.enabled.is_(True)))
        ).all()
        webhook_ids = [row[0] for row in rows if event in (row[1] or [])]

    if not webhook_ids:
        return

    logger.info("Dispatching event '%s' to %d webhook(s)", event, len(webhook_ids))

    semaphore = asyncio.Semaphore(WEBHOOK_MAX_CONCURRENCY)

    async def _deliver(webhook_id: UUID) -> WebhookDispatchResult:
        async with semaphore:
            return await _send_webhook_with_retry(
                webhook_id=webhook_id,
                event=event,
                payload=payload,
            )

    results = await asyncio.gather(*[_deliver(webhook_id) for webhook_id in webhook_ids])
    success_count = sum(1 for result in results if result.success)
    failed = [result for result in results if not result.success]

    if failed:
        logger.warning(
            "Webhook dispatch completed with failures event=%s success=%d failed=%d failed_ids=%s",
            event,
            success_count,
            len(failed),
            [result.webhook_id for result in failed],
        )


async def create_webhook(
    url: str,
    events: list[str],
    name: Optional[str] = None,
) -> Webhook:
    """Create a new webhook configuration."""
    await validate_webhook_url(url)
    async with async_session_maker() as db:
        webhook = Webhook(
            url=url,
            events=events,
            name=name,
            enabled=True,
        )
        db.add(webhook)
        await db.commit()
        await db.refresh(webhook)
        return webhook


async def list_webhooks() -> list[Webhook]:
    """List all webhook configurations."""
    async with async_session_maker() as db:
        result = await db.execute(select(Webhook).order_by(Webhook.created_at.desc()))
        return list(result.scalars().all())


async def get_webhook(webhook_id: UUID) -> Optional[Webhook]:
    """Get a webhook by ID."""
    async with async_session_maker() as db:
        return await db.get(Webhook, webhook_id)


async def delete_webhook(webhook_id: UUID) -> bool:
    """Delete a webhook configuration."""
    async with async_session_maker() as db:
        webhook = await db.get(Webhook, webhook_id)
        if webhook:
            await db.delete(webhook)
            await db.commit()
            return True
        return False


async def update_webhook(
    webhook_id: UUID,
    url: Optional[str] = None,
    events: Optional[list[str]] = None,
    name: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Optional[Webhook]:
    """Update a webhook configuration."""
    if url is not None:
        await validate_webhook_url(url)

    async with async_session_maker() as db:
        webhook = await db.get(Webhook, webhook_id)
        if not webhook:
            return None

        if url is not None:
            webhook.url = url
        if events is not None:
            webhook.events = events
        if name is not None:
            webhook.name = name
        if enabled is not None:
            webhook.enabled = enabled
            if enabled:
                webhook.failure_count = 0

        await db.commit()
        await db.refresh(webhook)
        return webhook
