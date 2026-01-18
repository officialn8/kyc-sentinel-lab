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


class ClassificationMetrics(BaseModel):
    """Classification performance metrics.

    Binary classification where:
    - Positive = Attack (attack_family != "benign")
    - Negative = Benign (attack_family == "benign")
    - Predicted Positive = decision in ["review", "fail"]
    - Predicted Negative = decision == "pass"
    """

    true_positives: int  # Attack correctly flagged as review/fail
    true_negatives: int  # Benign correctly passed
    false_positives: int  # Benign incorrectly flagged as review/fail
    false_negatives: int  # Attack incorrectly passed

    precision: float  # TP / (TP + FP) - of flagged sessions, % actual attacks
    recall: float  # TP / (TP + FN) - of attacks, % correctly flagged (sensitivity)
    specificity: float  # TN / (TN + FP) - of benign, % correctly passed
    f1_score: float  # 2 * (precision * recall) / (precision + recall)
    false_positive_rate: float  # FP / (FP + TN) - of benign, % incorrectly flagged

    total_attacks: int
    total_benign: int


class ProfileInfo(BaseModel):
    """Information about a configuration profile."""

    name: str
    description: str
    is_active: bool


class ScoringProfileInfo(ProfileInfo):
    """Scoring profile configuration details."""

    face_weight: float
    pad_weight: float
    doc_weight: float
    fail_threshold: int
    review_threshold: int


class PADProfileInfo(ProfileInfo):
    """PAD profile configuration details."""

    moire_threshold: float
    motion_entropy_min: float
    ear_blink_threshold: float
    min_blinks_expected: int
    lbp_texture_threshold: float


class ActiveProfiles(BaseModel):
    """Currently active configuration profiles."""

    scoring_profile: ScoringProfileInfo
    pad_profile: PADProfileInfo












