"""Security dependencies (auth, rate limiting)."""

from __future__ import annotations

import ipaddress
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Deque, DefaultDict, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import settings

logger = logging.getLogger(__name__)
_CLERK_ORG_ID_RE = re.compile(r"^org_[A-Za-z0-9]+$")

# Optional Redis import
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("redis package not installed. Using in-memory rate limiting only.")


@lru_cache(maxsize=1)
def _get_trusted_proxies() -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse trusted proxy CIDRs from settings (cached)."""
    if not settings.trusted_proxy_cidrs:
        return []

    networks = []
    for cidr in settings.trusted_proxy_cidrs.split(","):
        cidr = cidr.strip()
        if cidr:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                # Invalid CIDR, skip it (could log warning)
                pass
    return networks


def _is_trusted_proxy(ip: str) -> bool:
    """Check if an IP is from a trusted proxy network."""
    trusted = _get_trusted_proxies()
    if not trusted:
        return False

    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in network for network in trusted)
    except ValueError:
        return False


_basic = HTTPBasic(auto_error=False)


def _basic_enabled() -> bool:
    return bool((settings.basic_auth_username or "").strip() and (settings.basic_auth_password or "").strip())


def _api_key_enabled() -> bool:
    return bool((settings.backend_api_key or "").strip())


async def require_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Auth dependency that protects API endpoints.

    SECURITY: Auth configuration is validated at startup (see config.py).
    If we reach this code, either auth is disabled or credentials are configured.

    Accepts either:
    - Basic Auth (BASIC_AUTH_USERNAME / BASIC_AUTH_PASSWORD)
    - API key header (BACKEND_API_KEY via X-API-Key)
    """
    # Explicit opt-out (validated at startup to require explicit setting)
    if settings.auth_disabled:
        return

    # Note: If we get here and no auth is configured, startup validation failed.
    # This is a defensive check that should never trigger.

    # Option A: API key
    if _api_key_enabled():
        expected = (settings.backend_api_key or "").strip()
        provided = (x_api_key or "").strip()
        if provided and secrets.compare_digest(provided, expected):
            return

    # Option B: Basic Auth
    if _basic_enabled() and credentials is not None:
        user_ok = secrets.compare_digest(
            (credentials.username or "").strip(),
            (settings.basic_auth_username or "").strip(),
        )
        pass_ok = secrets.compare_digest(
            (credentials.password or "").strip(),
            (settings.basic_auth_password or "").strip(),
        )
        if user_ok and pass_ok:
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


# Backwards-compatible alias used by older code paths
async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    await require_auth(credentials=None, x_api_key=x_api_key)


async def require_tenant_context(
    x_authenticated_org_id: Optional[str] = Header(
        default=None, alias="X-Authenticated-Org-Id"
    ),
) -> str:
    """Require and validate tenant context forwarded by the trusted proxy."""
    tenant_id = (x_authenticated_org_id or "").strip()
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context required",
        )
    if len(tenant_id) > 128 or not _CLERK_ORG_ID_RE.fullmatch(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid tenant context",
        )
    return tenant_id


# =============================================================================
# Rate Limiting
# =============================================================================
#
# LIMITATIONS (documented for operators):
# - Per-process: Does NOT work across multiple workers/instances (use Redis for distributed)
# - Memory-bounded: Evicts stale keys periodically to prevent unbounded growth
# - Best-effort: Not a hard security boundary, but prevents obvious abuse
#
# For production at scale, replace with Redis-based rate limiting (e.g., redis-rate-limit).

_RATE_BUCKETS: DefaultDict[str, Deque[float]] = defaultdict(deque)
_RATE_BUCKET_LAST_CLEANUP = 0.0
_RATE_BUCKET_CLEANUP_INTERVAL = 60.0  # Cleanup every 60 seconds
_RATE_BUCKET_MAX_KEYS = 10000  # Maximum unique keys to track


def extract_client_ip(request: Request) -> str:
    """
    Extract client IP from request, only trusting proxy headers from trusted networks.

    SECURITY: X-Forwarded-For is only trusted if the direct connection is from
    a configured trusted proxy CIDR. This prevents header spoofing attacks.

    Configure TRUSTED_PROXY_CIDRS with your proxy network ranges, e.g.:
    - Railway/Render: Use their documented internal ranges
    - Cloudflare: Use their published IP ranges
    - Local dev: "127.0.0.0/8,::1/128"

    If TRUSTED_PROXY_CIDRS is empty, always uses direct connection IP (safest).
    """
    direct_ip = request.client.host if request.client else None

    # Only trust proxy headers if the direct connection is from a trusted proxy
    if direct_ip and _is_trusted_proxy(direct_ip):
        # Check X-Forwarded-For (set by reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For: client, proxy1, proxy2, ...
            # SECURITY: Walk from RIGHT to LEFT to find the first untrusted IP.
            # This is the actual client IP. Attackers can prepend fake IPs to XFF,
            # but they can't insert IPs after the trusted proxy chain.
            #
            # Example attack: Attacker sends "X-Forwarded-For: spoofed.ip"
            # After proxy chain: "spoofed.ip, real.client.ip, proxy1, proxy2"
            # Rightmost untrusted = real.client.ip (correct)
            # Leftmost = spoofed.ip (WRONG
            ips = [ip.strip() for ip in forwarded_for.split(",")]
            for ip in reversed(ips):
                if ip and not _is_trusted_proxy(ip):
                    return ip
            # All IPs in chain are trusted proxies - use leftmost as last resort
            if ips and ips[0]:
                return ips[0]

        # Check X-Real-IP (alternative header)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

    # Fall back to direct connection (or if no trusted proxy configured)
    return direct_ip or "unknown"


# Alias for backwards compatibility
_extract_client_ip_for_rate_limit = extract_client_ip


def _cleanup_stale_buckets(now: float, window_seconds: int) -> None:
    """
    Remove stale rate limit buckets to prevent memory growth.

    Called periodically during rate limit checks.
    """
    global _RATE_BUCKET_LAST_CLEANUP

    # Only cleanup periodically
    if now - _RATE_BUCKET_LAST_CLEANUP < _RATE_BUCKET_CLEANUP_INTERVAL:
        return

    _RATE_BUCKET_LAST_CLEANUP = now
    cutoff = now - window_seconds

    # Find and remove empty/stale buckets
    stale_keys = []
    for key, q in _RATE_BUCKETS.items():
        # Remove old entries
        while q and q[0] < cutoff:
            q.popleft()
        # Mark empty buckets for removal
        if not q:
            stale_keys.append(key)

    for key in stale_keys:
        del _RATE_BUCKETS[key]

    # Emergency eviction if too many keys (prevents memory exhaustion under attack)
    if len(_RATE_BUCKETS) > _RATE_BUCKET_MAX_KEYS:
        # Remove oldest half of buckets
        keys_by_oldest = sorted(
            _RATE_BUCKETS.keys(),
            key=lambda k: _RATE_BUCKETS[k][0] if _RATE_BUCKETS[k] else 0
        )
        for key in keys_by_oldest[:len(keys_by_oldest) // 2]:
            del _RATE_BUCKETS[key]


def _rate_limiter_memory(*, limit: int, window_seconds: int):
    """
    In-memory rate limiter (single-instance only).

    Uses sliding window algorithm with per-request cleanup.
    """

    async def _dep(request: Request) -> None:
        ip = _extract_client_ip_for_rate_limit(request)
        key = f"{ip}:{request.url.path}"
        now = time.time()

        # Periodic cleanup to bound memory
        _cleanup_stale_buckets(now, window_seconds)

        q = _RATE_BUCKETS[key]
        # Drop old entries for this key
        cutoff = now - window_seconds
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(window_seconds)},
            )

        q.append(now)

    return _dep


# =============================================================================
# Redis Rate Limiting (for multi-instance deployments)
# =============================================================================
#
# Uses Redis sorted sets with sliding window algorithm.
# Automatically falls back to in-memory if Redis is unavailable.
#
# Configuration:
#   REDIS_URL=redis://localhost:6379/0
#
# Redis keys:
#   rate_limit:{ip}:{path} -> sorted set of timestamps
#   TTL: window_seconds + buffer
#

_REDIS_CLIENT: Optional["redis.Redis"] = None
_REDIS_INIT_ATTEMPTED = False


@lru_cache(maxsize=1)
def _get_redis_url() -> Optional[str]:
    """Get Redis URL from settings (cached)."""
    url = (settings.redis_url or "").strip()
    return url if url else None


async def _get_redis_client() -> Optional["redis.Redis"]:
    """
    Get or create Redis client connection.

    Returns None if Redis is unavailable or not configured.
    Connection errors are logged but don't raise exceptions.
    """
    global _REDIS_CLIENT, _REDIS_INIT_ATTEMPTED

    if not REDIS_AVAILABLE:
        return None

    redis_url = _get_redis_url()
    if not redis_url:
        return None

    if _REDIS_CLIENT is not None:
        try:
            # Quick health check
            await _REDIS_CLIENT.ping()
            return _REDIS_CLIENT
        except Exception:
            # Connection lost, will reconnect below
            _REDIS_CLIENT = None

    # Only log once about connection attempts
    if not _REDIS_INIT_ATTEMPTED:
        _REDIS_INIT_ATTEMPTED = True
        logger.info(f"Connecting to Redis for rate limiting: {redis_url[:50]}...")

    try:
        _REDIS_CLIENT = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=1.0,
        )
        await _REDIS_CLIENT.ping()
        logger.info("Redis rate limiting connected successfully")
        return _REDIS_CLIENT
    except Exception as e:
        logger.warning(f"Redis connection failed, falling back to in-memory: {e}")
        _REDIS_CLIENT = None
        return None


async def _rate_limit_redis(
    client: "redis.Redis",
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Check rate limit using Redis sorted set sliding window.

    Args:
        client: Redis client
        key: Rate limit key (e.g., "rate_limit:1.2.3.4:/api/sessions")
        limit: Maximum requests allowed in window
        window_seconds: Time window in seconds

    Returns:
        Tuple of (is_allowed, current_count)
    """
    now = time.time()
    window_start = now - window_seconds
    redis_key = f"rate_limit:{key}"

    # Use pipeline for atomic operations
    pipe = client.pipeline(transaction=True)

    # Remove entries outside the window
    pipe.zremrangebyscore(redis_key, "-inf", window_start)

    # Count current entries in window
    pipe.zcard(redis_key)

    # Execute the first two commands
    results = await pipe.execute()
    current_count = results[1]

    if current_count >= limit:
        return False, current_count

    # Add current timestamp and set TTL
    pipe = client.pipeline(transaction=True)
    pipe.zadd(redis_key, {str(now): now})
    pipe.expire(redis_key, window_seconds + 10)  # Extra buffer
    await pipe.execute()

    return True, current_count + 1


def _rate_limiter_redis(*, limit: int, window_seconds: int):
    """
    Redis-based rate limiter for multi-instance deployments.

    Falls back to in-memory if Redis is unavailable.
    Uses sliding window algorithm with Redis sorted sets.
    """

    async def _dep(request: Request) -> None:
        ip = _extract_client_ip_for_rate_limit(request)
        key = f"{ip}:{request.url.path}"

        # Try Redis first
        redis_client = await _get_redis_client()
        if redis_client:
            try:
                is_allowed, count = await _rate_limit_redis(
                    redis_client, key, limit, window_seconds
                )
                if not is_allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": str(window_seconds)},
                    )
                return
            except HTTPException:
                raise
            except Exception as e:
                # Redis error, fall back to memory
                logger.warning(f"Redis rate limit error, falling back to memory: {e}")

        # Fall back to in-memory rate limiting
        now = time.time()
        _cleanup_stale_buckets(now, window_seconds)

        q = _RATE_BUCKETS[key]
        cutoff = now - window_seconds
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(window_seconds)},
            )

        q.append(now)

    return _dep


def rate_limiter(*, limit: int, window_seconds: int):
    """
    Create a rate limiter dependency.

    Automatically selects backend:
    - Redis (if REDIS_URL configured) - works across multiple instances
    - In-memory (fallback) - single-instance only

    Args:
        limit: Maximum requests allowed in window
        window_seconds: Time window in seconds

    Usage:
        @router.post("/endpoint", dependencies=[Depends(rate_limiter(limit=10, window_seconds=60))])
        async def endpoint():
            ...
    """
    # If Redis is configured, use Redis-based limiter
    if _get_redis_url():
        return _rate_limiter_redis(limit=limit, window_seconds=window_seconds)

    # Otherwise use in-memory (single-instance)
    return _rate_limiter_memory(limit=limit, window_seconds=window_seconds)

