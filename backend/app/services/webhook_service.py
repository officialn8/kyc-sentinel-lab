"""Webhook service for dispatching HTTP callbacks.

Sends HTTPS POST requests to registered webhook URLs when
session processing events occur.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models.webhook import Webhook

logger = logging.getLogger(__name__)

# Configuration
WEBHOOK_TIMEOUT_SECONDS = 10
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds
WEBHOOK_MAX_FAILURES = 10  # Disable webhook after this many consecutive failures


async def dispatch_webhooks(event: str, payload: dict) -> None:
    """Dispatch webhooks for an event to all registered handlers.
    
    Args:
        event: Event type (e.g., "session.completed", "session.failed")
        payload: Event payload to send in POST body
    """
    async with async_session_maker() as db:
        # Find all enabled webhooks
        stmt = select(Webhook).where(Webhook.enabled == True)
        result = await db.execute(stmt)
        all_webhooks = result.scalars().all()
        
        # Filter webhooks that subscribe to this event (in-memory for JSON column portability)
        webhooks = [w for w in all_webhooks if event in (w.events or [])]
        
        if not webhooks:
            return
        
        logger.info(f"Dispatching event '{event}' to {len(webhooks)} webhook(s)")
        
        # Send webhooks concurrently
        tasks = [
            _send_webhook_with_retry(db, webhook, event, payload)
            for webhook in webhooks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_webhook_with_retry(
    db: AsyncSession,
    webhook: Webhook,
    event: str,
    payload: dict,
) -> bool:
    """Send webhook with retry logic and failure tracking.
    
    Returns:
        True if delivery succeeded, False otherwise
    """
    full_payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }
    
    for attempt in range(WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    webhook.url,
                    json=full_payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "KYC-Sentinel-Webhook/1.0",
                        "X-Webhook-Event": event,
                    },
                )
                
                if response.is_success:
                    # Reset failure count on success
                    await db.execute(
                        update(Webhook)
                        .where(Webhook.id == webhook.id)
                        .values(
                            failure_count=0,
                            last_success=datetime.now(timezone.utc),
                        )
                    )
                    await db.commit()
                    
                    logger.info(f"Webhook delivered to {webhook.url} (status={response.status_code})")
                    return True
                else:
                    logger.warning(
                        f"Webhook failed: {webhook.url} returned {response.status_code}"
                    )
                    
        except Exception as e:
            logger.warning(f"Webhook error (attempt {attempt + 1}): {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(WEBHOOK_RETRY_DELAYS[attempt])
    
    # All retries failed - update failure tracking
    new_failure_count = webhook.failure_count + 1
    updates = {
        "failure_count": new_failure_count,
        "last_failure": datetime.now(timezone.utc),
    }
    
    # Disable webhook after too many failures
    if new_failure_count >= WEBHOOK_MAX_FAILURES:
        updates["enabled"] = False
        logger.error(
            f"Webhook {webhook.id} disabled after {new_failure_count} consecutive failures"
        )
    
    await db.execute(
        update(Webhook)
        .where(Webhook.id == webhook.id)
        .values(**updates)
    )
    await db.commit()
    
    return False


async def create_webhook(
    url: str,
    events: list[str],
    name: Optional[str] = None,
) -> Webhook:
    """Create a new webhook configuration.
    
    Args:
        url: Target URL for webhook delivery (must be HTTPS in production)
        events: List of event types to subscribe to
        name: Optional friendly name for the webhook
        
    Returns:
        Created Webhook instance
    """
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
    """Delete a webhook configuration.
    
    Returns:
        True if deleted, False if not found
    """
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
    """Update a webhook configuration.
    
    Returns:
        Updated Webhook or None if not found
    """
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
            if enabled:  # Reset failure count when re-enabling
                webhook.failure_count = 0
        
        await db.commit()
        await db.refresh(webhook)
        return webhook
