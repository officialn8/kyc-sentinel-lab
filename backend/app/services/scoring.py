"""Risk scoring and decision logic."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.reason import KYCReason


def compute_risk_score(
    face_similarity: float,
    pad_score: float,
    doc_score: float,
    reasons: list["KYCReason"],
) -> tuple[int, str]:
    """
    Compute overall risk score and decision.
    
    Args:
        face_similarity: Face match score (0-1, higher = more similar)
        pad_score: Presentation attack detection score (0-1, higher = more suspicious)
        doc_score: Document analysis score (0-1, higher = more suspicious)
        reasons: List of reason codes generated during analysis
    
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

    # Scale to 0-100
    risk_score = int(raw_score * 100)

    # Clamp to valid range
    risk_score = max(0, min(100, risk_score))

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
) -> dict[str, float]:
    """
    Get individual component contributions to the risk score.
    
    Useful for explainability and debugging.
    """
    face_risk = 1.0 - face_similarity

    return {
        "face_contribution": round(0.45 * face_risk * 100, 2),
        "pad_contribution": round(0.35 * pad_score * 100, 2),
        "doc_contribution": round(0.20 * doc_score * 100, 2),
        "total": round(
            (0.45 * face_risk + 0.35 * pad_score + 0.20 * doc_score) * 100, 2
        ),
    }




