"""Metrics and analytics endpoints."""

from fastapi import APIRouter
from sqlalchemy import select, func, case, and_

from app.api.deps import DbSession
from app.models.session import KYCSession
from app.models.result import KYCResult
from app.schemas.metrics import (
    MetricsSummary,
    AttackFamilyBreakdown,
    AttackFamilyMetrics,
    ConfusionMatrixData,
    ConfusionCell,
    ClassificationMetrics,
    ScoreDistribution,
    ScoreDistributionBucket,
    ActiveProfiles,
    ScoringProfileInfo,
    PADProfileInfo,
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


@router.get("/classification", response_model=ClassificationMetrics)
async def get_classification_metrics(db: DbSession) -> ClassificationMetrics:
    """Get classification performance metrics (precision, recall, F1, FPR).

    Binary classification where:
    - Positive = Attack (attack_family != "benign")
    - Negative = Benign (attack_family == "benign")
    - Predicted Positive = decision in ["review", "fail"]
    - Predicted Negative = decision == "pass"
    """
    # True Positives: Attack correctly flagged as review/fail
    tp = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .join(KYCResult)
        .where(KYCSession.attack_family != "benign")
        .where(KYCResult.decision.in_(["review", "fail"]))
    ) or 0

    # True Negatives: Benign correctly passed
    tn = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .join(KYCResult)
        .where(KYCSession.attack_family == "benign")
        .where(KYCResult.decision == "pass")
    ) or 0

    # False Positives: Benign incorrectly flagged as review/fail
    fp = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .join(KYCResult)
        .where(KYCSession.attack_family == "benign")
        .where(KYCResult.decision.in_(["review", "fail"]))
    ) or 0

    # False Negatives: Attack incorrectly passed
    fn = await db.scalar(
        select(func.count())
        .select_from(KYCSession)
        .join(KYCResult)
        .where(KYCSession.attack_family != "benign")
        .where(KYCResult.decision == "pass")
    ) or 0

    # Calculate metrics (handle division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ClassificationMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        specificity=round(specificity, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        total_attacks=tp + fn,
        total_benign=tn + fp,
    )


@router.get("/score-distribution", response_model=ScoreDistribution)
async def get_score_distribution(db: DbSession) -> ScoreDistribution:
    """Get risk score distribution bucketed by score range and attack family.

    Returns histogram data showing how risk scores are distributed,
    broken down by attack family. Useful for threshold calibration.
    """
    attack_families = ["benign", "replay", "injection", "face_swap", "doc_tamper"]
    bucket_size = 10
    buckets = []
    total = 0

    for bucket_min in range(0, 100, bucket_size):
        bucket_max = bucket_min + bucket_size
        family_counts: dict[str, int] = {}

        for family in attack_families:
            count = await db.scalar(
                select(func.count())
                .select_from(KYCSession)
                .join(KYCResult)
                .where(KYCSession.attack_family == family)
                .where(
                    and_(
                        KYCResult.risk_score >= bucket_min,
                        KYCResult.risk_score < bucket_max,
                    )
                )
            ) or 0
            family_counts[family] = count

        bucket_total = sum(family_counts.values())
        total += bucket_total

        buckets.append(
            ScoreDistributionBucket(
                min_score=bucket_min,
                max_score=bucket_max,
                count=bucket_total,
                attack_families=family_counts,
            )
        )

    return ScoreDistribution(buckets=buckets, total=total)


@router.get("/profiles", response_model=ActiveProfiles)
async def get_active_profiles() -> ActiveProfiles:
    """Get currently active scoring and PAD configuration profiles.

    Returns the active profile configuration for transparency
    and debugging. Useful for understanding which thresholds
    are being used in production.
    """
    from app.config import settings
    from app.services.scoring import get_scoring_config, SCORING_PROFILES, ScoringProfile
    from app.detection.pad_heuristics import get_pad_config, PAD_PROFILES, PADProfile

    # Get scoring profile info
    scoring_config = get_scoring_config(settings.scoring_profile)
    scoring_profiles_info = []
    for profile_enum, config in SCORING_PROFILES.items():
        is_active = profile_enum.value == settings.scoring_profile
        if is_active:
            active_scoring = ScoringProfileInfo(
                name=config.name,
                description=config.description,
                is_active=True,
                face_weight=config.face_weight,
                pad_weight=config.pad_weight,
                doc_weight=config.doc_weight,
                fail_threshold=config.fail_threshold,
                review_threshold=config.review_threshold,
            )

    # Get PAD profile info
    pad_config = get_pad_config(settings.pad_profile)
    for profile_enum, config in PAD_PROFILES.items():
        is_active = profile_enum.value == settings.pad_profile
        if is_active:
            active_pad = PADProfileInfo(
                name=config.name,
                description=config.description,
                is_active=True,
                moire_threshold=config.moire_threshold,
                motion_entropy_min=config.motion_entropy_min,
                ear_blink_threshold=config.ear_blink_threshold,
                min_blinks_expected=config.min_blinks_expected,
                lbp_texture_threshold=config.lbp_texture_threshold,
            )

    return ActiveProfiles(
        scoring_profile=active_scoring,
        pad_profile=active_pad,
    )












