"""Security dependencies (auth, rate limiting)."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from typing import Deque, DefaultDict, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import settings


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
    Optional auth for public/demo deployments.

    Enabled if either is configured:
    - Basic Auth (BASIC_AUTH_USERNAME / BASIC_AUTH_PASSWORD)
    - API key header (BACKEND_API_KEY via X-API-Key)
    """

    if not _basic_enabled() and not _api_key_enabled():
        return

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


# Simple in-memory rate limiter (best-effort).
# NOTE: This is per-process; with multiple Railway instances/workers it is not global.
_RATE_BUCKETS: DefaultDict[str, Deque[float]] = defaultdict(deque)


def rate_limiter(*, limit: int, window_seconds: int):
    """
    Create a dependency that rate-limits requests per client IP and path.
    """

    async def _dep(request: Request) -> None:
        ip = getattr(getattr(request, "client", None), "host", None) or "unknown"
        key = f"{ip}:{request.url.path}"
        now = time.time()

        q = _RATE_BUCKETS[key]
        # Drop old entries
        cutoff = now - window_seconds
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        q.append(now)

    return _dep


