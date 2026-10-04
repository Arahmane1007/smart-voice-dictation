"""API key verification and client IP resolution."""

import hashlib
import hmac
from collections.abc import Iterable, Sequence
from ipaddress import IPv4Network, IPv6Network, ip_address


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header."""
    if not header:
        return None
    scheme, _, token = header.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def fingerprint(key: str) -> str:
    """Short, non-reversible identifier of a key, safe to write in logs."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class KeyVerifier:
    def __init__(self, keys: Iterable[str]) -> None:
        self._keys = [key.encode() for key in keys]

    def verify(self, token: str | None) -> str | None:
        """Return the fingerprint of the matching key, or None.

        Every configured key is compared, each with a comparison that is
        constant-time for tokens of the same length as the key, so the response
        time does not reveal which key (or how much of it) matched. The length
        itself may be observable; generated keys all have the same fixed length.
        """
        if token is None:
            return None
        candidate = token.encode()
        matched: bytes | None = None
        for key in self._keys:
            if hmac.compare_digest(candidate, key):
                matched = key
        return fingerprint(matched.decode()) if matched is not None else None


def client_ip(
    peer: str | None,
    forwarded_for: str | None,
    trusted_proxies: Sequence[IPv4Network | IPv6Network],
) -> str:
    """Return the real client IP.

    `X-Forwarded-For` is only honoured when the direct peer is a trusted proxy
    (e.g. Traefik); otherwise any client could forge it. The last hop is the one
    appended by our own proxy.
    """
    if peer is None:
        return "unknown"
    if forwarded_for and _is_trusted(peer, trusted_proxies):
        hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
        if hops:
            return hops[-1]
    return peer


def _is_trusted(peer: str, trusted_proxies: Sequence[IPv4Network | IPv6Network]) -> bool:
    try:
        address = ip_address(peer)
    except ValueError:
        return False
    return any(address in network for network in trusted_proxies)
