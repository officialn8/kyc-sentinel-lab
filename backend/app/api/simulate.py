"""Synthetic session generation endpoints."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.deps import DbSession, Processing
from app.models.session import KYCSession
from app.schemas.session import SessionResponse

router = APIRouter()


class SimulateRequest(BaseModel):
    """Request to generate a synthetic session."""

    attack_family: str = Field(
        pattern="^(replay|injection|face_swap|doc_tamper|benign)$"
    )
    attack_severity: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
    )
    count: int = Field(default=1, ge=1, le=10)


class AttackFamily(BaseModel):
    """Attack family information."""

    id: str
    name: str
    description: str
    severities: list[str]


ATTACK_FAMILIES = [
    AttackFamily(
        id="replay",
        name="Replay Attack",
        description="Screen capture of a genuine user's session played back to the camera",
        severities=["low", "medium", "high"],
    ),
    AttackFamily(
        id="injection",
        name="Injection Attack",
        description="Artificial video stream injected at the device or API level",
        severities=["low", "medium", "high"],
    ),
    AttackFamily(
        id="face_swap",
        name="Face Swap",
        description="Deepfake or face-swap overlays blended onto a genuine face",
        severities=["low", "medium", "high"],
    ),
    AttackFamily(
        id="doc_tamper",
        name="Document Tampering",
        description="Modified or fabricated identity documents",
        severities=["low", "medium", "high"],
    ),
    AttackFamily(
        id="benign",
        name="Benign",
        description="Legitimate verification session with no attack artifacts",
        severities=["low", "medium", "high"],
    ),
]


@router.get("/families", response_model=list[AttackFamily])
async def list_attack_families() -> list[AttackFamily]:
    """List available attack families for simulation."""
    return ATTACK_FAMILIES


@router.post("", response_model=list[SessionResponse])
async def generate_synthetic_sessions(
    request: SimulateRequest,
    db: DbSession,
    processing: Processing,
    background_tasks: BackgroundTasks,
) -> list[SessionResponse]:
    """Generate synthetic KYC sessions with specified attack patterns."""
    sessions = []

    for _ in range(request.count):
        session = KYCSession(
            source="synthetic",
            attack_family=request.attack_family,
            attack_severity=request.attack_severity,
            status="processing",
        )
        db.add(session)
        sessions.append(session)

    await db.flush()

    # Start processing for each session
    for session in sessions:
        background_tasks.add_task(
            processing.generate_synthetic_session,
            str(session.id),
            request.attack_family,
            request.attack_severity,
        )

    await db.commit()

    return [SessionResponse.model_validate(s) for s in sessions]




