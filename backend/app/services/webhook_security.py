"""Webhook URL security validation utilities."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


class WebhookUrlValidationError(ValueError):
    """Webhook destination URL validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("198.18.0.0/15"),  # Benchmarking
)


def _ip_is_blocked(ip_text: str) -> bool:
    addr = ipaddress.ip_address(ip_text)
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return True
    return any(addr in network for network in _EXTRA_BLOCKED_NETWORKS)


def _normalize_hostname(hostname: str | None) -> str:
    if not hostname:
        raise WebhookUrlValidationError("host_missing", "Webhook URL must include a hostname")
    return hostname.strip().rstrip(".").lower()


async def validate_webhook_url(url: str) -> None:
    """Validate that a webhook URL targets a public HTTPS destination."""
    parsed = urlsplit((url or "").strip())

    if parsed.scheme.lower() != "https":
        raise WebhookUrlValidationError("scheme_not_https", "Webhook URL must use https")

    if parsed.username or parsed.password:
        raise WebhookUrlValidationError("userinfo_forbidden", "Webhook URL cannot include username/password")

    hostname = _normalize_hostname(parsed.hostname)
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise WebhookUrlValidationError("localhost_forbidden", "Webhook URL cannot target localhost")

    # Reject IP literal hosts directly.
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise WebhookUrlValidationError("ip_literal_forbidden", "Webhook URL cannot use IP literal hostnames")

    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(
            hostname,
            parsed.port or 443,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise WebhookUrlValidationError("dns_resolution_failed", "Webhook hostname could not be resolved") from exc

    addresses = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_text = str(sockaddr[0])
        if ip_text not in addresses:
            addresses.append(ip_text)

    if not addresses:
        raise WebhookUrlValidationError("dns_no_addresses", "Webhook hostname resolved to no addresses")

    blocked = [addr for addr in addresses if _ip_is_blocked(addr)]
    if blocked:
        raise WebhookUrlValidationError(
            "private_destination_forbidden",
            "Webhook destination resolves to a non-public network address",
        )
