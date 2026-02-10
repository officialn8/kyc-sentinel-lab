"""Unit tests for graduated fraud scoring service."""

from app.services.fraud_detection import FraudSignals
from app.services.fraud_scoring import FraudScoringService


def test_ring_signal_monotonicity() -> None:
    """Higher ring similarity/match count should increase boost points."""
    service = FraudScoringService(min_matches=2)

    low = FraudSignals(
        fraud_ring={"match_count": 2, "max_similarity": 0.81, "threshold": 0.80}
    )
    high = FraudSignals(
        fraud_ring={"match_count": 6, "max_similarity": 0.95, "threshold": 0.80}
    )

    low_result = service.score(low)
    high_result = service.score(high)

    assert high_result.boost_points > low_result.boost_points


def test_force_fail_threshold() -> None:
    """Very strong fraud ring linkage should trigger force-fail."""
    service = FraudScoringService(min_matches=2)

    signals = FraudSignals(
        fraud_ring={"match_count": 5, "max_similarity": 0.98, "threshold": 0.80}
    )

    result = service.score(signals)
    assert result.force_fail is True


def test_geo_signal_unavailable_has_no_geo_points() -> None:
    """Geo-unavailable informational signals should not add scoring points."""
    service = FraudScoringService(min_matches=2)

    signals = FraudSignals(reason_codes=["FRAUD_GEO_SIGNAL_UNAVAILABLE"])
    result = service.score(signals)

    geo_factor = next(f for f in result.factors if f.name == "geo")
    assert geo_factor.points == 0
    assert result.boost_points == 0
