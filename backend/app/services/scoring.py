"""Risk scoring and decision logic."""

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.models.reason import KYCReason


# =============================================================================
# Phase 2: Configurable Scoring Profiles
# =============================================================================

class ScoringProfile(str, Enum):
    """Available scoring profiles for different use cases."""
    
    DEFAULT = "default"
    FINTECH_HIGH_RISK = "fintech_high_risk"
    CRYPTO_EXCHANGE = "crypto_exchange"
    SOCIAL_VERIFICATION = "social_verification"


@dataclass
class ScoringConfig:
    """Configurable scoring weights and thresholds.
    
    Attributes:
        face_weight: Weight for face similarity score (0-1)
        pad_weight: Weight for PAD score (0-1)
        doc_weight: Weight for document analysis score (0-1)
        fail_threshold: Risk score threshold for automatic fail (0-100)
        review_threshold: Risk score threshold for manual review (0-100)
        name: Human-readable profile name
        description: Profile description
    """
    
    face_weight: float
    pad_weight: float
    doc_weight: float
    fail_threshold: int
    review_threshold: int
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        # Validate weights sum to ~1.0 (allow small tolerance for floating point)
        total = self.face_weight + self.pad_weight + self.doc_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        # Validate thresholds
        if not (0 <= self.review_threshold < self.fail_threshold <= 100):
            raise ValueError(
                f"Invalid thresholds: review={self.review_threshold}, fail={self.fail_threshold}. "
                "Must satisfy: 0 <= review < fail <= 100"
            )


# Predefined scoring profiles for common use cases
SCORING_PROFILES: dict[ScoringProfile, ScoringConfig] = {
    ScoringProfile.DEFAULT: ScoringConfig(
        face_weight=0.45,
        pad_weight=0.35,
        doc_weight=0.20,
        fail_threshold=70,
        review_threshold=40,
        name="Default",
        description="Balanced scoring for general KYC verification",
    ),
    ScoringProfile.FINTECH_HIGH_RISK: ScoringConfig(
        face_weight=0.50,
        pad_weight=0.35,
        doc_weight=0.15,
        fail_threshold=60,
        review_threshold=35,
        name="Fintech High Risk",
        description="Stricter scoring for high-value financial services (lower fail threshold)",
    ),
    ScoringProfile.CRYPTO_EXCHANGE: ScoringConfig(
        face_weight=0.40,
        pad_weight=0.40,
        doc_weight=0.20,
        fail_threshold=55,
        review_threshold=30,
        name="Crypto Exchange",
        description="Enhanced PAD weight for injection attack detection common in crypto",
    ),
    ScoringProfile.SOCIAL_VERIFICATION: ScoringConfig(
        face_weight=0.60,
        pad_weight=0.25,
        doc_weight=0.15,
        fail_threshold=70,
        review_threshold=45,
        name="Social Verification",
        description="Face-focused scoring for social platform identity verification",
    ),
}


def get_scoring_config(profile: ScoringProfile | str | None = None) -> ScoringConfig:
    """Get scoring configuration for a profile.
    
    Args:
        profile: Profile enum, string name, or None for default
        
    Returns:
        ScoringConfig for the requested profile
    """
    if profile is None:
        return SCORING_PROFILES[ScoringProfile.DEFAULT]
    
    if isinstance(profile, str):
        try:
            profile = ScoringProfile(profile)
        except ValueError:
            # Unknown profile, use default
            return SCORING_PROFILES[ScoringProfile.DEFAULT]
    
    return SCORING_PROFILES.get(profile, SCORING_PROFILES[ScoringProfile.DEFAULT])


# =============================================================================
# Phase 1: Reason Code Categories
# =============================================================================

# Phase 1: Critical liveness indicators that increase suspicion
CRITICAL_LIVENESS_CODES = {
    "PAD_PRINTED_TEXTURE",
    "PAD_FLAT_DEPTH",
    "PAD_NO_BLINK_DETECTED",
}

# Phase 1: Quality-related codes that may reduce confidence
QUALITY_CONCERN_CODES = {
    "FACE_LOW_QUALITY_SELFIE",
    "FACE_LOW_QUALITY_ID",
    "FACE_EXTREME_POSE",
}

# Phase 1: Document integrity codes
DOCUMENT_INTEGRITY_CODES = {
    "DOC_MRZ_INVALID",
    "DOC_MRZ_MISMATCH",
    "DOC_METADATA_SUSPICIOUS",
}

# Phase 2: Fraud pattern codes
FRAUD_PATTERN_CODES = {
    "FRAUD_HIGH_VELOCITY",
    "FRAUD_GEO_MISMATCH",
}


def compute_risk_score(
    face_similarity: float,
    pad_score: float,
    doc_score: float,
    reasons: list["KYCReason"],
    liveness_score: Optional[float] = None,
    face_quality_score: Optional[float] = None,
    config: Optional[ScoringConfig] = None,
) -> tuple[int, str]:
    """
    Compute overall risk score and decision.
    
    Args:
        face_similarity: Face match score (0-1, higher = more similar)
        pad_score: Presentation attack detection score (0-1, higher = more suspicious)
        doc_score: Document analysis score (0-1, higher = more suspicious)
        reasons: List of reason codes generated during analysis
        liveness_score: Optional explicit liveness score (0-1, higher = more confident live)
        face_quality_score: Optional face quality score (0-1, higher = better quality)
        config: Optional scoring configuration. Uses DEFAULT profile if not provided.
    
    Returns:
        (risk_score, decision) where decision is "pass" | "review" | "fail"
    """
    # Use default config if not provided
    if config is None:
        config = SCORING_PROFILES[ScoringProfile.DEFAULT]
    
    # PRECISION: Use Decimal arithmetic for financial-grade accuracy.
    # Decimal ensures deterministic, reproducible scoring.
    
    # Convert inputs to Decimal for precise arithmetic
    d_face = Decimal(str(face_similarity))
    d_pad = Decimal(str(pad_score))
    d_doc = Decimal(str(doc_score))
    
    # Convert weights to Decimal
    w_face = Decimal(str(config.face_weight))
    w_pad = Decimal(str(config.pad_weight))
    w_doc = Decimal(str(config.doc_weight))
    
    # Convert face similarity to risk (invert: low similarity = high risk)
    face_risk = Decimal("1.0") - d_face

    # Weighted combination using Decimal for precision
    raw_score = (
        w_face * face_risk +
        w_pad * d_pad +
        w_doc * d_doc
    )

    # Phase 1: Incorporate explicit liveness score if provided
    if liveness_score is not None:
        d_liveness = Decimal(str(liveness_score))
        liveness_risk = Decimal("1.0") - d_liveness
        liveness_weight = w_pad * Decimal("0.20")
        raw_score += liveness_weight * liveness_risk

    # Phase 1: Quality penalty - low quality reduces confidence in match
    if face_quality_score is not None and face_quality_score < 0.5:
        d_quality = Decimal(str(face_quality_score))
        quality_penalty = (Decimal("0.5") - d_quality) * Decimal("0.1")
        raw_score += quality_penalty

    # Scale to 0-100 with ceiling (risk-increasing bias)
    # SECURITY: Ceiling ensures we NEVER under-report risk.
    # - Decimal("39.01") * 100 → ceiling → 40 (not 39)
    # - Decimal("69.999") * 100 → ceiling → 70
    scaled = raw_score * Decimal("100")
    risk_score = int(scaled.to_integral_value(rounding=ROUND_CEILING))

    # Clamp to valid range
    risk_score = max(0, min(100, risk_score))

    # Extract reason codes as strings for checking
    reason_codes = {r.code for r in reasons}

    # Phase 1: Check for critical liveness failures
    liveness_failures = reason_codes & CRITICAL_LIVENESS_CODES
    if liveness_failures:
        # Boost risk score for liveness failures (relative to review threshold)
        liveness_boost = config.review_threshold + 25
        risk_score = max(risk_score, liveness_boost)

    # Phase 1: Check for document integrity issues
    doc_integrity_failures = reason_codes & DOCUMENT_INTEGRITY_CODES
    if doc_integrity_failures:
        # Boost risk score for document integrity failures
        doc_boost = config.review_threshold + 15
        risk_score = max(risk_score, doc_boost)
    
    # Phase 2: Check for fraud pattern signals
    fraud_signals = reason_codes & FRAUD_PATTERN_CODES
    if fraud_signals:
        # Boost risk score for fraud patterns
        fraud_boost = config.review_threshold + 10
        risk_score = max(risk_score, fraud_boost)

    # Hard fail rules: Any high-severity reason forces fail
    high_severity_reasons = [r for r in reasons if r.severity == "high"]
    if high_severity_reasons:
        return max(risk_score, config.fail_threshold + 5), "fail"

    # Decision thresholds using configurable values
    if risk_score >= config.fail_threshold:
        return risk_score, "fail"
    elif risk_score >= config.review_threshold:
        return risk_score, "review"
    else:
        return risk_score, "pass"


def compute_component_scores(
    face_similarity: float,
    pad_score: float,
    doc_score: float,
    liveness_score: Optional[float] = None,
    face_quality_score: Optional[float] = None,
    config: Optional[ScoringConfig] = None,
) -> dict[str, float]:
    """
    Get individual component contributions to the risk score.
    
    Useful for explainability and debugging.
    
    Args:
        face_similarity: Face match score (0-1)
        pad_score: PAD suspicion score (0-1)
        doc_score: Document suspicion score (0-1)
        liveness_score: Optional liveness confidence (0-1)
        face_quality_score: Optional face quality score (0-1)
        config: Optional scoring configuration. Uses DEFAULT profile if not provided.
    """
    # Use default config if not provided
    if config is None:
        config = SCORING_PROFILES[ScoringProfile.DEFAULT]
    
    face_risk = 1.0 - face_similarity

    result = {
        "face_contribution": round(config.face_weight * face_risk * 100, 2),
        "pad_contribution": round(config.pad_weight * pad_score * 100, 2),
        "doc_contribution": round(config.doc_weight * doc_score * 100, 2),
        "face_weight": config.face_weight,
        "pad_weight": config.pad_weight,
        "doc_weight": config.doc_weight,
        "fail_threshold": config.fail_threshold,
        "review_threshold": config.review_threshold,
    }
    
    total = (
        config.face_weight * face_risk +
        config.pad_weight * pad_score +
        config.doc_weight * doc_score
    )
    
    # Phase 1: Include liveness contribution if provided
    if liveness_score is not None:
        liveness_risk = 1.0 - liveness_score
        liveness_weight = config.pad_weight * 0.20
        liveness_contribution = liveness_weight * liveness_risk * 100
        result["liveness_contribution"] = round(liveness_contribution, 2)
        total += liveness_weight * liveness_risk
    
    # Phase 1: Include quality penalty if applicable
    if face_quality_score is not None and face_quality_score < 0.5:
        quality_penalty = (0.5 - face_quality_score) * 0.1 * 100
        result["quality_penalty"] = round(quality_penalty, 2)
        total += (0.5 - face_quality_score) * 0.1
    
    result["total"] = round(total * 100, 2)
    
    return result


def compute_liveness_score(
    has_blinks: bool,
    blink_count: int,
    texture_score: float,
    motion_entropy: float,
) -> float:
    """
    Compute overall liveness confidence score.
    
    Combines multiple liveness signals into a single score.
    
    Args:
        has_blinks: Whether natural blinks were detected
        blink_count: Number of blinks detected
        texture_score: LBP texture score (0-1, higher = more like real skin)
        motion_entropy: Motion entropy score
    
    Returns:
        Liveness score (0-1, higher = more confident subject is live)
    """
    score = 0.0
    
    # Blink detection is a strong liveness signal
    if has_blinks:
        score += 0.4
        # Bonus for multiple blinks
        if blink_count >= 2:
            score += 0.1
    
    # Texture analysis
    if texture_score >= 0.6:
        score += 0.3
    elif texture_score >= 0.4:
        score += 0.15
    
    # Motion entropy
    if motion_entropy >= 0.1:
        score += 0.2
    elif motion_entropy >= 0.05:
        score += 0.1
    
    return min(score, 1.0)












