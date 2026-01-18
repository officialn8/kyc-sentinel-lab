"""Webhook CRUD API endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.services import webhook_service

router = APIRouter()


# --- Schemas ---

class WebhookCreate(BaseModel):
    """Schema for creating a webhook."""
    
    url: str = Field(..., description="Target URL for webhook delivery (HTTPS recommended)")
    events: list[str] = Field(
        ...,
        description="Event types to subscribe to: session.completed, session.failed"
    )
    name: Optional[str] = Field(None, max_length=255, description="Friendly name")


class WebhookUpdate(BaseModel):
    """Schema for updating a webhook."""
    
    url: Optional[str] = None
    events: Optional[list[str]] = None
    name: Optional[str] = None
    enabled: Optional[bool] = None


class WebhookResponse(BaseModel):
    """Webhook response schema."""
    
    id: UUID
    url: str
    events: list[str]
    name: Optional[str]
    enabled: bool
    failure_count: int
    
    model_config = {"from_attributes": True}


class WebhookListResponse(BaseModel):
    """List of webhooks response."""
    
    items: list[WebhookResponse]
    total: int


# --- Endpoints ---

@router.post("/", response_model=WebhookResponse)
async def create_webhook(data: WebhookCreate) -> WebhookResponse:
    """Create a new webhook configuration.
    
    Webhooks receive POST requests when subscribed events occur.
    
    Available events:
    - `session.completed`: Session processing finished successfully
    - `session.failed`: Session processing encountered an error
    
    Webhooks are automatically disabled after 10 consecutive delivery failures.
    """
    # Validate events
    valid_events = {"session.completed", "session.failed"}
    invalid_events = set(data.events) - valid_events
    if invalid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid events: {invalid_events}. Valid events: {valid_events}"
        )
    
    webhook = await webhook_service.create_webhook(
        url=data.url,
        events=data.events,
        name=data.name,
    )
    return WebhookResponse.model_validate(webhook)


@router.get("/", response_model=WebhookListResponse)
async def list_webhooks() -> WebhookListResponse:
    """List all webhook configurations."""
    webhooks = await webhook_service.list_webhooks()
    return WebhookListResponse(
        items=[WebhookResponse.model_validate(w) for w in webhooks],
        total=len(webhooks),
    )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(webhook_id: UUID) -> WebhookResponse:
    """Get a webhook by ID."""
    webhook = await webhook_service.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return WebhookResponse.model_validate(webhook)


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(webhook_id: UUID, data: WebhookUpdate) -> WebhookResponse:
    """Update a webhook configuration.
    
    Re-enabling a disabled webhook resets its failure count.
    """
    if data.events is not None:
        valid_events = {"session.completed", "session.failed"}
        invalid_events = set(data.events) - valid_events
        if invalid_events:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid events: {invalid_events}. Valid events: {valid_events}"
            )
    
    webhook = await webhook_service.update_webhook(
        webhook_id=webhook_id,
        url=data.url,
        events=data.events,
        name=data.name,
        enabled=data.enabled,
    )
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return WebhookResponse.model_validate(webhook)


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: UUID) -> dict:
    """Delete a webhook configuration."""
    deleted = await webhook_service.delete_webhook(webhook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"status": "deleted", "id": str(webhook_id)}
