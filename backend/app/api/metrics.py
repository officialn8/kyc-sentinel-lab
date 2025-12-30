"""Metrics and analytics endpoints."""

from fastapi import APIRouter
from sqlalchemy import select, func, case

from app.api.deps import DbSession
from app.models.session import KYCSession
from app.models.result import KYCResult
from app.schemas.metrics import (
    MetricsSummary,
    AttackFamilyBreakdown,
    AttackFamilyMetrics,
    ConfusionMatrixData,
    ConfusionCell,
)

router = APIRouter()


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary(db: DbSession) -> MetricsSummary:
    """Get aggregate metrics summary."""
    # Total sessions
    total = await db.scalar(select(func.count()).select_from(KYCSession)) or 0

    # Completed sessions
    completed = (
        await db.scalar(
            select(func.count())
            .select_from(KYCSession)
            .where(KYCSession.status == "completed")
        )
        or 0
    )

    # Decision counts
    pass_count = (
        await db.scalar(
            select(func.count()).select_from(KYCResult).where(KYCResult.decision == "pass")
        )
        or 0
    )
    review_count = (
        await db.scalar(
            select(func.count()).select_from(KYCResult).where(KYCResult.decision == "review")
        )
        or 0
    )
    fail_count = (
        await db.scalar(
            select(func.count()).select_from(KYCResult).where(KYCResult.decision == "fail")
        )
        or 0
    )

    # Average risk score
    avg_score = await db.scalar(select(func.avg(KYCResult.risk_score))) or 0.0

    # Detection rate (synthetic attacks that were flagged as review or fail)
    synthetic_attacks = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .where(KYCSession.source == "synthetic")
        .where(KYCSession.attack_family != "benign")
    ) or 0

    detected_attacks = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .join(KYCResult)
        .where(KYCSession.source == "synthetic")
        .where(KYCSession.attack_family != "benign")
        .where(KYCResult.decision.in_(["review", "fail"]))
    ) or 0

    detection_rate = (
        (detected_attacks / synthetic_attacks * 100) if synthetic_attacks > 0 else 0.0
    )

    return MetricsSummary(
        total_sessions=total,
        completed_sessions=completed,
        pass_count=pass_count,
        review_count=review_count,
        fail_count=fail_count,
        avg_risk_score=round(float(avg_score), 2),
        detection_rate=round(detection_rate, 2),
    )


@router.get("/by-attack-family", response_model=AttackFamilyBreakdown)
async def get_attack_family_breakdown(db: DbSession) -> AttackFamilyBreakdown:
    """Get metrics breakdown by attack family."""
    families = ["replay", "injection", "face_swap", "doc_tamper", "benign"]
    results = []

    for family in families:
        # Total for this family
        total = (
            await db.scalar(
                select(func.count())
                .select_from(KYCSession)
                .where(KYCSession.attack_family == family)
                .where(KYCSession.status == "completed")
            )
            or 0
        )

        if total == 0:
            results.append(
                AttackFamilyMetrics(
                    family=family,
                    total=0,
                    detected=0,
                    missed=0,
                    detection_rate=0.0,
                    avg_risk_score=0.0,
                )
            )
            continue

        # For attacks: detected = review or fail
        # For benign: detected = pass (correctly identified as safe)
        if family == "benign":
            detected = (
                await db.scalar(
                    select(func.count())
                    .select_from(KYCSession)
                    .join(KYCResult)
                    .where(KYCSession.attack_family == family)
                    .where(KYCResult.decision == "pass")
                )
                or 0
            )
        else:
            detected = (
                await db.scalar(
                    select(func.count())
                    .select_from(KYCSession)
                    .join(KYCResult)
                    .where(KYCSession.attack_family == family)
                    .where(KYCResult.decision.in_(["review", "fail"]))
                )
                or 0
            )

        missed = total - detected

        avg_score = (
            await db.scalar(
                select(func.avg(KYCResult.risk_score))
                .select_from(KYCSession)
                .join(KYCResult)
                .where(KYCSession.attack_family == family)
            )
            or 0.0
        )

        results.append(
            AttackFamilyMetrics(
                family=family,
                total=total,
                detected=detected,
                missed=missed,
                detection_rate=round((detected / total) * 100, 2) if total > 0 else 0.0,
                avg_risk_score=round(float(avg_score), 2),
            )
        )

    return AttackFamilyBreakdown(families=results)


@router.get("/confusion-matrix", response_model=ConfusionMatrixData)
async def get_confusion_matrix(db: DbSession) -> ConfusionMatrixData:
    """Get confusion matrix data for visualization."""
    attack_families = ["benign", "replay", "injection", "face_swap", "doc_tamper"]
    decisions = ["pass", "review", "fail"]

    cells = []
    total = 0

    for family in attack_families:
        for decision in decisions:
            count = (
                await db.scalar(
                    select(func.count())
                    .select_from(KYCSession)
                    .join(KYCResult)
                    .where(KYCSession.attack_family == family)
                    .where(KYCResult.decision == decision)
                )
                or 0
            )
            cells.append(
                ConfusionCell(
                    actual=family,
                    predicted=decision,
                    count=count,
                )
            )
            total += count

    return ConfusionMatrixData(cells=cells, total=total)











