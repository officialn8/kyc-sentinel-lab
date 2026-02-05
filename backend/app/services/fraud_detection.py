"""
Fraud pattern detection service.

Implements:
- Velocity rules (submission rate limiting per device/IP)
- Geographic anomaly detection
- Device fingerprinting analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import KYCSession
from app.config import settings


@dataclass
class VelocityThresholds:
    """Configurable thresholds for velocity rules."""
    
    device_1h_limit: int = 3  # Max submissions per device per hour
    device_24h_limit: int = 10  # Max submissions per device per 24 hours
    ip_1h_limit: int = 5  # Max submissions per IP per hour
    ip_24h_limit: int = 20  # Max submissions per IP per 24 hours
    
    @classmethod
    def from_settings(cls) -> "VelocityThresholds":
        """Create thresholds from application settings."""
        return cls(
            device_1h_limit=settings.velocity_device_1h_limit,
            device_24h_limit=settings.velocity_device_24h_limit,
            ip_1h_limit=settings.velocity_ip_1h_limit,
            ip_24h_limit=settings.velocity_ip_24h_limit,
        )


@dataclass
class VelocityCheck:
    """Result of velocity rule evaluation."""
    
    device_submissions_1h: int = 0
    device_submissions_24h: int = 0
    ip_submissions_1h: int = 0
    ip_submissions_24h: int = 0
    flagged: bool = False
    flags: list[str] = field(default_factory=list)
    thresholds_used: Optional[VelocityThresholds] = None
    
    @property
    def is_high_velocity(self) -> bool:
        """Check if any velocity threshold is exceeded."""
        return self.flagged


@dataclass
class GeoAnomalyResult:
    """Result of geographic anomaly detection."""
    
    document_country: Optional[str] = None
    device_country: Optional[str] = None
    device_timezone: Optional[str] = None
    is_anomalous: bool = False
    anomaly_type: Optional[str] = None
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class FraudSignals:
    """Aggregated fraud signals for a session."""
    
    velocity_check: Optional[VelocityCheck] = None
    geo_anomaly: Optional[GeoAnomalyResult] = None
    fraud_ring: Optional[dict] = None
    reason_codes: list[str] = field(default_factory=list)
    overall_fraud_score: float = 0.0  # 0-1, higher = more suspicious


async def check_velocity(
    db: AsyncSession,
    device_fingerprint: Optional[str],
    client_ip: Optional[str],
    thresholds: Optional[VelocityThresholds] = None,
    exclude_session_id: Optional[str] = None,
) -> VelocityCheck:
    """
    Check submission velocity from device and IP.
    
    Queries recent session counts to detect potential fraud rings
    or automated submission attacks.
    
    Args:
        db: Database session
        device_fingerprint: Unique device identifier (can be None)
        client_ip: Client IP address (can be None)
        thresholds: Velocity thresholds to use (defaults to settings)
        exclude_session_id: Session ID to exclude from count (the current session)
    
    Returns:
        VelocityCheck with submission counts and flag status
    """
    if thresholds is None:
        thresholds = VelocityThresholds.from_settings()
    
    from datetime import timezone
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    twenty_four_hours_ago = now - timedelta(hours=24)
    
    result = VelocityCheck(thresholds_used=thresholds)
    
    # Check device velocity
    if device_fingerprint:
        # Count submissions in last hour
        query_1h = select(func.count()).select_from(KYCSession).where(
            and_(
                KYCSession.device_fingerprint == device_fingerprint,
                KYCSession.created_at >= one_hour_ago,
            )
        )
        if exclude_session_id:
            query_1h = query_1h.where(KYCSession.id != exclude_session_id)
        
        result.device_submissions_1h = await db.scalar(query_1h) or 0
        
        # Count submissions in last 24 hours
        query_24h = select(func.count()).select_from(KYCSession).where(
            and_(
                KYCSession.device_fingerprint == device_fingerprint,
                KYCSession.created_at >= twenty_four_hours_ago,
            )
        )
        if exclude_session_id:
            query_24h = query_24h.where(KYCSession.id != exclude_session_id)
        
        result.device_submissions_24h = await db.scalar(query_24h) or 0
        
        # Check thresholds
        if result.device_submissions_1h >= thresholds.device_1h_limit:
            result.flagged = True
            result.flags.append(
                f"device_1h:{result.device_submissions_1h}>={thresholds.device_1h_limit}"
            )
        if result.device_submissions_24h >= thresholds.device_24h_limit:
            result.flagged = True
            result.flags.append(
                f"device_24h:{result.device_submissions_24h}>={thresholds.device_24h_limit}"
            )
    
    # Check IP velocity
    if client_ip:
        # Count submissions in last hour
        query_1h = select(func.count()).select_from(KYCSession).where(
            and_(
                KYCSession.client_ip == client_ip,
                KYCSession.created_at >= one_hour_ago,
            )
        )
        if exclude_session_id:
            query_1h = query_1h.where(KYCSession.id != exclude_session_id)
        
        result.ip_submissions_1h = await db.scalar(query_1h) or 0
        
        # Count submissions in last 24 hours
        query_24h = select(func.count()).select_from(KYCSession).where(
            and_(
                KYCSession.client_ip == client_ip,
                KYCSession.created_at >= twenty_four_hours_ago,
            )
        )
        if exclude_session_id:
            query_24h = query_24h.where(KYCSession.id != exclude_session_id)
        
        result.ip_submissions_24h = await db.scalar(query_24h) or 0
        
        # Check thresholds
        if result.ip_submissions_1h >= thresholds.ip_1h_limit:
            result.flagged = True
            result.flags.append(
                f"ip_1h:{result.ip_submissions_1h}>={thresholds.ip_1h_limit}"
            )
        if result.ip_submissions_24h >= thresholds.ip_24h_limit:
            result.flagged = True
            result.flags.append(
                f"ip_24h:{result.ip_submissions_24h}>={thresholds.ip_24h_limit}"
            )
    
    return result


def check_geo_anomaly(
    document_country: Optional[str],
    device_country: Optional[str],
    device_timezone: Optional[str] = None,
) -> GeoAnomalyResult:
    """
    Detect geographic inconsistencies between document and device.
    
    Flags cases where the document's country doesn't match the
    device's geolocation, which could indicate fraud.
    
    Args:
        document_country: Country code from document MRZ or OCR (ISO 3166-1 alpha-2/3)
        device_country: Country code from IP geolocation
        device_timezone: Timezone string from device (e.g., "America/New_York")
    
    Returns:
        GeoAnomalyResult with anomaly assessment
    """
    result = GeoAnomalyResult(
        document_country=document_country,
        device_country=device_country,
        device_timezone=device_timezone,
    )
    
    # Need both countries to compare
    if not document_country or not device_country:
        result.details["reason"] = "insufficient_data"
        return result
    
    # Normalize country codes (handle 3-letter to 2-letter conversion)
    doc_country_norm = _normalize_country_code(document_country)
    dev_country_norm = _normalize_country_code(device_country)
    
    result.details["doc_country_normalized"] = doc_country_norm
    result.details["dev_country_normalized"] = dev_country_norm
    
    # Direct mismatch
    if doc_country_norm and dev_country_norm and doc_country_norm != dev_country_norm:
        # Check if countries are in the same region (less suspicious)
        same_region = _countries_in_same_region(doc_country_norm, dev_country_norm)
        
        if same_region:
            result.is_anomalous = False
            result.anomaly_type = "cross_border_same_region"
            result.confidence = 0.3
            result.details["region"] = same_region
        else:
            result.is_anomalous = True
            result.anomaly_type = "country_mismatch"
            result.confidence = 0.8
    
    # Check timezone consistency if available
    if device_timezone and doc_country_norm:
        tz_country = _get_country_from_timezone(device_timezone)
        if tz_country and tz_country != doc_country_norm:
            result.details["timezone_country"] = tz_country
            if not result.is_anomalous:
                result.is_anomalous = True
                result.anomaly_type = "timezone_mismatch"
                result.confidence = 0.6
            else:
                # Both IP and timezone mismatch - higher confidence
                result.confidence = min(result.confidence + 0.15, 1.0)
    
    return result


def _normalize_country_code(code: str) -> Optional[str]:
    """Normalize country code to 2-letter ISO format."""
    code = code.upper().strip()
    
    # Common 3-letter to 2-letter mappings
    alpha3_to_alpha2 = {
        "USA": "US", "GBR": "GB", "DEU": "DE", "FRA": "FR",
        "CAN": "CA", "AUS": "AU", "JPN": "JP", "CHN": "CN",
        "IND": "IN", "BRA": "BR", "MEX": "MX", "RUS": "RU",
        "KOR": "KR", "ITA": "IT", "ESP": "ES", "NLD": "NL",
        "CHE": "CH", "SWE": "SE", "NOR": "NO", "DNK": "DK",
        "FIN": "FI", "POL": "PL", "AUT": "AT", "BEL": "BE",
        "PRT": "PT", "GRC": "GR", "TUR": "TR", "ZAF": "ZA",
        "NGA": "NG", "EGY": "EG", "KEN": "KE", "ARG": "AR",
        "CHL": "CL", "COL": "CO", "PER": "PE", "VEN": "VE",
        "PHL": "PH", "IDN": "ID", "THA": "TH", "VNM": "VN",
        "MYS": "MY", "SGP": "SG", "PAK": "PK", "BGD": "BD",
        "ARE": "AE", "SAU": "SA", "ISR": "IL", "NZL": "NZ",
        "IRL": "IE", "CZE": "CZ", "HUN": "HU", "ROU": "RO",
        "UKR": "UA", "TWN": "TW", "HKG": "HK",
    }
    
    if len(code) == 3:
        return alpha3_to_alpha2.get(code, code[:2])
    elif len(code) == 2:
        return code
    
    return None


def _countries_in_same_region(country1: str, country2: str) -> Optional[str]:
    """Check if two countries are in the same geographic region."""
    regions = {
        "EU": {"DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "GR", "PL",
               "CZ", "HU", "RO", "SE", "DK", "FI", "IE", "SK", "BG", "HR",
               "LT", "LV", "EE", "SI", "LU", "MT", "CY"},
        "NORTH_AMERICA": {"US", "CA", "MX"},
        "SOUTH_AMERICA": {"BR", "AR", "CL", "CO", "PE", "VE", "EC", "UY", "PY", "BO"},
        "EAST_ASIA": {"JP", "KR", "CN", "TW", "HK", "MO"},
        "SOUTHEAST_ASIA": {"TH", "VN", "PH", "ID", "MY", "SG", "MM", "KH", "LA"},
        "SOUTH_ASIA": {"IN", "PK", "BD", "LK", "NP"},
        "MIDDLE_EAST": {"AE", "SA", "IL", "TR", "EG", "JO", "LB", "KW", "QA", "BH", "OM"},
        "OCEANIA": {"AU", "NZ", "FJ", "PG"},
        "UK_IRELAND": {"GB", "IE"},
        "SCANDINAVIA": {"SE", "NO", "DK", "FI", "IS"},
    }
    
    for region, countries in regions.items():
        if country1 in countries and country2 in countries:
            return region
    
    return None


def _get_country_from_timezone(timezone: str) -> Optional[str]:
    """Get country code from timezone string."""
    # Common timezone to country mappings
    tz_to_country = {
        "America/New_York": "US", "America/Los_Angeles": "US",
        "America/Chicago": "US", "America/Denver": "US",
        "America/Toronto": "CA", "America/Vancouver": "CA",
        "America/Mexico_City": "MX",
        "Europe/London": "GB", "Europe/Paris": "FR",
        "Europe/Berlin": "DE", "Europe/Rome": "IT",
        "Europe/Madrid": "ES", "Europe/Amsterdam": "NL",
        "Europe/Brussels": "BE", "Europe/Vienna": "AT",
        "Europe/Zurich": "CH", "Europe/Stockholm": "SE",
        "Europe/Oslo": "NO", "Europe/Copenhagen": "DK",
        "Europe/Helsinki": "FI", "Europe/Warsaw": "PL",
        "Europe/Prague": "CZ", "Europe/Budapest": "HU",
        "Europe/Athens": "GR", "Europe/Istanbul": "TR",
        "Europe/Moscow": "RU", "Europe/Kiev": "UA",
        "Asia/Tokyo": "JP", "Asia/Seoul": "KR",
        "Asia/Shanghai": "CN", "Asia/Hong_Kong": "HK",
        "Asia/Taipei": "TW", "Asia/Singapore": "SG",
        "Asia/Bangkok": "TH", "Asia/Jakarta": "ID",
        "Asia/Manila": "PH", "Asia/Kuala_Lumpur": "MY",
        "Asia/Ho_Chi_Minh": "VN", "Asia/Kolkata": "IN",
        "Asia/Dubai": "AE", "Asia/Riyadh": "SA",
        "Asia/Jerusalem": "IL",
        "Australia/Sydney": "AU", "Australia/Melbourne": "AU",
        "Pacific/Auckland": "NZ",
        "America/Sao_Paulo": "BR", "America/Buenos_Aires": "AR",
        "America/Santiago": "CL", "America/Bogota": "CO",
        "America/Lima": "PE",
        "Africa/Johannesburg": "ZA", "Africa/Lagos": "NG",
        "Africa/Cairo": "EG", "Africa/Nairobi": "KE",
    }
    
    return tz_to_country.get(timezone)


async def check_fraud_ring(
    db: AsyncSession,
    *,
    session_id: Optional[str],
    face_embedding: Optional[list[float]],
    similarity_threshold: Optional[float] = None,
    min_matches: Optional[int] = None,
    sample_limit: int = 3,
) -> Optional[dict]:
    """Detect fraud ring linkage via similar face embeddings.

    Returns a dict with match_count and max_similarity if threshold met.
    """
    if not session_id or not face_embedding:
        return None

    threshold = similarity_threshold or settings.fraud_ring_similarity_threshold
    required_matches = min_matches or settings.fraud_ring_min_matches
    if required_matches <= 0:
        return None

    max_distance = 1.0 - threshold
    distance_expr = KYCSession.face_embedding.cosine_distance(face_embedding)

    count_stmt = (
        select(
            func.count().label("match_count"),
            func.min(distance_expr).label("min_distance"),
        )
        .where(KYCSession.id != session_id)
        .where(KYCSession.face_embedding.isnot(None))
        .where(KYCSession.deleted_at.is_(None))
        .where(distance_expr < max_distance)
    )
    count_row = await db.execute(count_stmt)
    match_count, min_distance = count_row.one()

    if not match_count or match_count < required_matches:
        return None

    # Sample a few closest matches for evidence
    sample_stmt = (
        select(KYCSession.id, distance_expr.label("distance"))
        .where(KYCSession.id != session_id)
        .where(KYCSession.face_embedding.isnot(None))
        .where(KYCSession.deleted_at.is_(None))
        .where(distance_expr < max_distance)
        .order_by(distance_expr)
        .limit(sample_limit)
    )
    sample_rows = (await db.execute(sample_stmt)).all()
    samples = [
        {
            "session_id": str(row[0]),
            "similarity": round(1.0 - float(row[1]), 4),
        }
        for row in sample_rows
    ]

    max_similarity = 1.0 - float(min_distance) if min_distance is not None else threshold

    return {
        "match_count": int(match_count),
        "max_similarity": round(max_similarity, 4),
        "threshold": threshold,
        "samples": samples,
    }


async def compute_fraud_signals(
    db: AsyncSession,
    device_fingerprint: Optional[str] = None,
    client_ip: Optional[str] = None,
    session_id: Optional[str] = None,
    face_embedding: Optional[list[float]] = None,
    document_country: Optional[str] = None,
    device_country: Optional[str] = None,
    device_timezone: Optional[str] = None,
    exclude_session_id: Optional[str] = None,
) -> FraudSignals:
    """
    Compute all fraud signals for a session.

    Aggregates velocity checks and geographic anomaly detection
    into a single fraud assessment.

    Args:
        db: Database session
        device_fingerprint: Unique device identifier (client-provided, untrusted)
        client_ip: Client IP address (server-captured, trusted)
        document_country: Country from document MRZ/OCR
        device_country: Country from IP geolocation (trusted if server-enriched)
        device_timezone: Timezone from device (client-provided, for cross-validation)
        session_id: Session ID for fraud ring detection
        face_embedding: Face embedding for similarity search
        exclude_session_id: Session ID to exclude from velocity counts

    SECURITY NOTE:
        - client_ip should be captured server-side, not from client input
        - device_country should ideally be enriched server-side via IP geolocation
        - device_fingerprint and device_timezone are client-controlled (untrusted)

    Returns:
        FraudSignals with all fraud indicators
    """
    result = FraudSignals()
    fraud_score = 0.0

    # Check velocity using server-captured IP
    velocity = await check_velocity(
        db=db,
        device_fingerprint=device_fingerprint,
        client_ip=client_ip,
        exclude_session_id=exclude_session_id,
    )
    result.velocity_check = velocity

    if velocity.flagged:
        result.reason_codes.append("FRAUD_HIGH_VELOCITY")
        # Score based on how many limits exceeded
        fraud_score += len(velocity.flags) * 0.2

    # Check geographic anomaly
    # NOTE: For production, device_country should be enriched via IP geolocation service
    # (e.g., MaxMind GeoIP2, IP2Location) using the server-captured client_ip.
    # The caller should pass the server-enriched device_country, not client input.

    geo = check_geo_anomaly(
        document_country=document_country,
        device_country=device_country,
        device_timezone=device_timezone,
    )
    result.geo_anomaly = geo

    if geo.is_anomalous:
        result.reason_codes.append("FRAUD_GEO_MISMATCH")
        fraud_score += geo.confidence * 0.3

    # Fraud ring detection (advisory only)
    fraud_ring = await check_fraud_ring(
        db=db,
        session_id=session_id,
        face_embedding=face_embedding,
    )
    if fraud_ring:
        result.fraud_ring = fraud_ring
        result.reason_codes.append("FRAUD_RING_LINKED")

    result.overall_fraud_score = min(fraud_score, 1.0)

    return result
