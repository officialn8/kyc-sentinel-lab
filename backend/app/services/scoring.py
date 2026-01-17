"""Risk scoring and decision logic."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.models.reason import KYCReason


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


def compute_risk_score(
    face_similarity: float,
    pad_score: float,
    doc_score: float,
    reasons: list["KYCReason"],
    liveness_score: Optional[float] = None,
    face_quality_score: Optional[float] = None,
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
    
    Returns:
        (risk_score, decision) where decision is "pass" | "review" | "fail"
    """
    # Convert face similarity to risk (invert: low similarity = high risk)
    face_risk = 1.0 - face_similarity

    # Weighted combination
    # Face matching is most critical (45%)
    # PAD is important for liveness (35%)
    # Document analysis is supporting (20%)
    raw_score = (
        0.45 * face_risk +
        0.35 * pad_score +
        0.20 * doc_score
    )

    # Phase 1: Incorporate explicit liveness score if provided
    if liveness_score is not None:
        # Liveness score adjustment: lack of liveness increases risk
        liveness_risk = 1.0 - liveness_score
        # Blend into PAD portion (20% of PAD weight)
        raw_score += 0.07 * liveness_risk  # 0.35 * 0.20 = 0.07

    # Phase 1: Quality penalty - low quality reduces confidence in match
    if face_quality_score is not None and face_quality_score < 0.5:
        # Poor quality means face match is less reliable
        # Add small penalty for low quality
        quality_penalty = (0.5 - face_quality_score) * 0.1
        raw_score += quality_penalty

    # Scale to 0-100
    risk_score = int(raw_score * 100)

    # Clamp to valid range
    risk_score = max(0, min(100, risk_score))

    # Extract reason codes as strings for checking
    reason_codes = {r.code for r in reasons}

    # Phase 1: Check for critical liveness failures
    liveness_failures = reason_codes & CRITICAL_LIVENESS_CODES
    if liveness_failures:
        # Boost risk score for liveness failures
        risk_score = max(risk_score, 65)

    # Phase 1: Check for document integrity issues
    doc_integrity_failures = reason_codes & DOCUMENT_INTEGRITY_CODES
    if doc_integrity_failures:
        # Boost risk score for document integrity failures
        risk_score = max(risk_score, 55)

    # Hard fail rules: Any high-severity reason forces fail
    high_severity_reasons = [r for r in reasons if r.severity == "high"]
    if high_severity_reasons:
        return max(risk_score, 75), "fail"

    # Decision thresholds
    if risk_score >= 70:
        return risk_score, "fail"
    elif risk_score >= 40:
        return risk_score, "review"
    else:
        return risk_score, "pass"


def compute_component_scores(
    face_similarity: float,
    pad_score: float,
    doc_score: float,
    liveness_score: Optional[float] = None,
    face_quality_score: Optional[float] = None,
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
    """
    face_risk = 1.0 - face_similarity

    result = {
        "face_contribution": round(0.45 * face_risk * 100, 2),
        "pad_contribution": round(0.35 * pad_score * 100, 2),
        "doc_contribution": round(0.20 * doc_score * 100, 2),
    }
    
    total = 0.45 * face_risk + 0.35 * pad_score + 0.20 * doc_score
    
    # Phase 1: Include liveness contribution if provided
    if liveness_score is not None:
        liveness_risk = 1.0 - liveness_score
        liveness_contribution = 0.07 * liveness_risk * 100
        result["liveness_contribution"] = round(liveness_contribution, 2)
        total += 0.07 * liveness_risk
    
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












