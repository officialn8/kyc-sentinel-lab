"""Metrics schemas."""

from typing import Optional

from pydantic import BaseModel


class MetricsSummary(BaseModel):
    """Aggregate metrics summary."""

    total_sessions: int
    completed_sessions: int
    pass_count: int
    review_count: int
    fail_count: int
    avg_risk_score: float
    detection_rate: float  # % of synthetic attacks correctly flagged


class AttackFamilyMetrics(BaseModel):
    """Metrics for a specific attack family."""

    family: str
    total: int
    detected: int
    missed: int
    detection_rate: float
    avg_risk_score: float


class AttackFamilyBreakdown(BaseModel):
    """Breakdown of metrics by attack family."""

    families: list[AttackFamilyMetrics]


class ConfusionCell(BaseModel):
    """Single cell in confusion matrix."""

    actual: str  # benign, replay, injection, face_swap, doc_tamper
    predicted: str  # pass, review, fail
    count: int


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data for visualization."""

    cells: list[ConfusionCell]
    total: int


class ScoreDistributionBucket(BaseModel):
    """Score distribution bucket."""

    min_score: int
    max_score: int
    count: int
    attack_families: dict[str, int]


class ScoreDistribution(BaseModel):
    """Risk score distribution data."""

    buckets: list[ScoreDistributionBucket]
    total: int






