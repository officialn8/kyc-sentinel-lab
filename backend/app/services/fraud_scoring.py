"""Fraud scoring service for graduated fraud-pattern risk contributions."""

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.services.fraud_detection import FraudSignals


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class RiskFactor:
    """Single fraud-risk factor contribution."""

    name: str
    raw_value: float
    normalized_value: float
    weight: float
    points: int
    evidence: dict = field(default_factory=dict)


@dataclass
class FraudScoreResult:
    """Result from graduated fraud-pattern scoring."""

    risk_fraction: float
    boost_points: int
    force_fail: bool
    factors: list[RiskFactor]
    reason_codes: list[str] = field(default_factory=list)


class FraudScoringService:
    """Compute graduated risk boosts from fraud pattern signals."""

    def __init__(self, min_matches: int | None = None) -> None:
        self.min_matches = max(1, min_matches or settings.fraud_ring_min_matches)

    def score(self, signals: "FraudSignals") -> FraudScoreResult:
        factors: list[RiskFactor] = []

        breached_limit_count = 0
        if signals.velocity_check and signals.velocity_check.flagged:
            breached_limit_count = len(signals.velocity_check.flags)
        velocity_points = min(10, 3 * breached_limit_count)
        velocity_norm = _clamp(velocity_points / 10, 0.0, 1.0)
        factors.append(
            RiskFactor(
                name="velocity",
                raw_value=float(breached_limit_count),
                normalized_value=velocity_norm,
                weight=10 / 35,
                points=velocity_points,
                evidence={
                    "breached_limit_count": breached_limit_count,
                    "flags": signals.velocity_check.flags if signals.velocity_check else [],
                },
            )
        )

        geo_confidence = 0.0
        if signals.geo_anomaly and signals.geo_anomaly.is_anomalous:
            geo_confidence = float(signals.geo_anomaly.confidence)
        geo_points = int(math.ceil(8 * geo_confidence)) if geo_confidence > 0 else 0
        geo_norm = _clamp(geo_confidence, 0.0, 1.0)
        factors.append(
            RiskFactor(
                name="geo",
                raw_value=geo_confidence,
                normalized_value=geo_norm,
                weight=8 / 35,
                points=geo_points,
                evidence={
                    "is_anomalous": bool(signals.geo_anomaly and signals.geo_anomaly.is_anomalous),
                    "confidence": geo_confidence,
                    "anomaly_type": signals.geo_anomaly.anomaly_type if signals.geo_anomaly else None,
                },
            )
        )

        ring_points = 0
        ring_severity = 0.0
        ring_evidence: dict = {}
        max_similarity = 0.0
        match_count = 0
        threshold = float(settings.fraud_ring_similarity_threshold)
        if signals.fraud_ring:
            max_similarity = float(signals.fraud_ring.get("max_similarity", 0.0))
            match_count = int(signals.fraud_ring.get("match_count", 0))
            threshold = float(signals.fraud_ring.get("threshold", threshold))

            denom = 1.0 - threshold
            if denom <= 0:
                ring_similarity_norm = 1.0 if max_similarity >= threshold else 0.0
            else:
                ring_similarity_norm = _clamp((max_similarity - threshold) / denom, 0.0, 1.0)

            ring_match_norm = _clamp(
                (match_count - self.min_matches + 1) / 5,
                0.0,
                1.0,
            )

            ring_severity = 0.65 * ring_similarity_norm + 0.35 * ring_match_norm
            ring_points = int(math.ceil(17 * ring_severity))
            ring_evidence = {
                "max_similarity": max_similarity,
                "match_count": match_count,
                "threshold": threshold,
                "similarity_norm": round(ring_similarity_norm, 4),
                "match_norm": round(ring_match_norm, 4),
            }

        factors.append(
            RiskFactor(
                name="fraud_ring",
                raw_value=ring_severity,
                normalized_value=_clamp(ring_severity, 0.0, 1.0),
                weight=17 / 35,
                points=ring_points,
                evidence=ring_evidence,
            )
        )

        boost_points = min(35, velocity_points + geo_points + ring_points)
        force_fail = bool(max_similarity >= 0.97 and match_count >= 5)

        return FraudScoreResult(
            risk_fraction=round(boost_points / 35, 4),
            boost_points=boost_points,
            force_fail=force_fail,
            factors=factors,
            reason_codes=list(signals.reason_codes),
        )
