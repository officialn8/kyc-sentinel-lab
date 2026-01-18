"""
IP Geolocation service using MaxMind GeoIP2.

Provides server-side IP-to-country lookup for fraud detection,
replacing unreliable client-provided country information.

IMPORTANT: Requires a MaxMind GeoIP2 database file.
- Free GeoLite2-Country database: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
- Commercial GeoIP2-Country for better accuracy in production

Database should be updated regularly (MaxMind updates weekly).
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Optional import - geoip2 may not be installed
try:
    import geoip2.database
    import geoip2.errors

    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False
    logger.warning(
        "geoip2 package not installed. IP geolocation will be disabled. "
        "Install with: pip install geoip2"
    )


@dataclass
class GeoIPResult:
    """Result of IP geolocation lookup."""

    ip_address: str
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2 (e.g., "US")
    country_name: Optional[str] = None  # Full country name (e.g., "United States")
    continent_code: Optional[str] = None  # Continent code (e.g., "NA")
    is_anonymous_proxy: bool = False
    is_satellite_provider: bool = False
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if lookup returned a valid country."""
        return self.country_code is not None and self.error is None


class GeoIPService:
    """
    Service for IP geolocation using MaxMind GeoIP2.

    SECURITY: This service provides SERVER-SIDE IP geolocation,
    which is trusted (unlike client-provided country information).

    Usage:
        service = GeoIPService("/path/to/GeoLite2-Country.mmdb")
        result = service.lookup("8.8.8.8")
        if result.is_valid:
            print(f"Country: {result.country_code}")  # "US"
    """

    _instance: Optional["GeoIPService"] = None
    _reader: Optional["geoip2.database.Reader"] = None

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the GeoIP service.

        Args:
            database_path: Path to MaxMind database file (.mmdb)
                          If None, tries common locations or disables geolocation.
        """
        self._database_path = database_path
        self._reader = None
        self._init_error: Optional[str] = None

        if not GEOIP2_AVAILABLE:
            self._init_error = "geoip2 package not installed"
            return

        # Try to find database
        db_path = self._find_database(database_path)
        if db_path:
            try:
                self._reader = geoip2.database.Reader(str(db_path))
                logger.info(f"GeoIP2 database loaded: {db_path}")
            except Exception as e:
                self._init_error = f"Failed to load GeoIP2 database: {e}"
                logger.error(self._init_error)
        else:
            self._init_error = "GeoIP2 database not found"
            logger.warning(
                f"{self._init_error}. Set GEOIP_DATABASE_PATH or download from "
                "https://dev.maxmind.com/geoip/geolite2-free-geolocation-data"
            )

    def _find_database(self, explicit_path: Optional[str]) -> Optional[Path]:
        """Find the GeoIP database file."""
        # 1. Explicit path from config
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            logger.warning(f"Explicit GeoIP path does not exist: {explicit_path}")

        # 2. Common locations
        common_paths = [
            # Project-local
            Path("data/GeoLite2-Country.mmdb"),
            Path("GeoLite2-Country.mmdb"),
            # System-wide (Linux)
            Path("/usr/share/GeoIP/GeoLite2-Country.mmdb"),
            Path("/var/lib/GeoIP/GeoLite2-Country.mmdb"),
            # macOS Homebrew
            Path("/opt/homebrew/share/GeoIP/GeoLite2-Country.mmdb"),
            Path("/usr/local/share/GeoIP/GeoLite2-Country.mmdb"),
        ]

        for path in common_paths:
            if path.exists():
                return path

        return None

    @property
    def is_available(self) -> bool:
        """Check if geolocation service is available."""
        return self._reader is not None

    def lookup(self, ip_address: str) -> GeoIPResult:
        """
        Look up country information for an IP address.

        Args:
            ip_address: IPv4 or IPv6 address string

        Returns:
            GeoIPResult with country information or error details
        """
        result = GeoIPResult(ip_address=ip_address)

        # Handle unavailable service
        if not self._reader:
            result.error = self._init_error or "GeoIP service not initialized"
            return result

        # Validate input
        if not ip_address or ip_address == "unknown":
            result.error = "Invalid IP address"
            return result

        # Skip private/local IPs
        if self._is_private_ip(ip_address):
            result.error = "Private IP address"
            return result

        try:
            response = self._reader.country(ip_address)

            result.country_code = response.country.iso_code
            result.country_name = response.country.name
            result.continent_code = response.continent.code

            # Check for anonymous proxies (if using full GeoIP2 database)
            if hasattr(response, "traits"):
                result.is_anonymous_proxy = getattr(
                    response.traits, "is_anonymous_proxy", False
                )
                result.is_satellite_provider = getattr(
                    response.traits, "is_satellite_provider", False
                )

        except geoip2.errors.AddressNotFoundError:
            result.error = "IP address not found in database"
        except Exception as e:
            result.error = f"Lookup failed: {str(e)}"
            logger.error(f"GeoIP lookup failed for {ip_address}: {e}")

        return result

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is a private/local address."""
        import ipaddress

        try:
            addr = ipaddress.ip_address(ip)
            return addr.is_private or addr.is_loopback or addr.is_link_local
        except ValueError:
            return False

    def close(self):
        """Close the database reader."""
        if self._reader:
            self._reader.close()
            self._reader = None


# Singleton accessor
@lru_cache(maxsize=1)
def get_geoip_service() -> GeoIPService:
    """
    Get or create the singleton GeoIP service instance.

    Uses GEOIP_DATABASE_PATH from settings if configured.
    """
    from app.config import settings

    return GeoIPService(
        database_path=getattr(settings, "geoip_database_path", None)
    )


def lookup_ip_country(ip_address: str) -> Optional[str]:
    """
    Convenience function to look up country code for an IP.

    Args:
        ip_address: IP address to look up

    Returns:
        ISO 3166-1 alpha-2 country code (e.g., "US") or None if unavailable
    """
    service = get_geoip_service()
    result = service.lookup(ip_address)
    return result.country_code if result.is_valid else None


def enrich_session_with_geoip(client_ip: str) -> dict:
    """
    Enrich session data with server-side geolocation.

    Args:
        client_ip: Server-captured client IP address

    Returns:
        Dict with geolocation fields to merge into session data:
        {
            "ip_country": "US",  # Server-enriched (trusted)
            "ip_continent": "NA",
            "ip_anonymous_proxy": False,
        }
    """
    service = get_geoip_service()
    result = service.lookup(client_ip)

    enrichment = {
        "ip_country_server": result.country_code,  # Trusted server-side value
        "ip_continent": result.continent_code,
        "ip_anonymous_proxy": result.is_anonymous_proxy,
        "ip_satellite_provider": result.is_satellite_provider,
    }

    if result.error:
        enrichment["ip_geo_error"] = result.error

    return enrichment
