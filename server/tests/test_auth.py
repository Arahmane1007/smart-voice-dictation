from ipaddress import ip_network

import pytest
from svd_server.auth import KeyVerifier, client_ip, fingerprint, parse_bearer

KEY_A = "a" * 40
KEY_B = "b" * 40
TRUSTED = (ip_network("10.0.1.0/24"),)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (f"Bearer {KEY_A}", KEY_A),
        (f"bearer {KEY_A}", KEY_A),
        (f"Bearer   {KEY_A}  ", KEY_A),
        (None, None),
        ("", None),
        ("Bearer", None),
        ("Bearer    ", None),
        (f"Basic {KEY_A}", None),
        (KEY_A, None),
    ],
)
def test_parse_bearer(header: str | None, expected: str | None) -> None:
    assert parse_bearer(header) == expected


def test_fingerprint_is_short_stable_and_not_the_key() -> None:
    assert fingerprint(KEY_A) == fingerprint(KEY_A)
    assert fingerprint(KEY_A) != fingerprint(KEY_B)
    assert len(fingerprint(KEY_A)) == 8
    assert KEY_A[:8] not in fingerprint(KEY_A)


def test_verifier_accepts_any_configured_key() -> None:
    verifier = KeyVerifier([KEY_A, KEY_B])
    assert verifier.verify(KEY_A) == fingerprint(KEY_A)
    assert verifier.verify(KEY_B) == fingerprint(KEY_B)


@pytest.mark.parametrize("token", [None, "", "c" * 40, KEY_A[:-1], KEY_A + "x"])
def test_verifier_rejects_unknown_tokens(token: str | None) -> None:
    assert KeyVerifier([KEY_A, KEY_B]).verify(token) is None


def test_client_ip_without_proxy_uses_peer() -> None:
    assert client_ip("203.0.113.5", None, TRUSTED) == "203.0.113.5"


def test_client_ip_ignores_forwarded_for_from_untrusted_peer() -> None:
    assert client_ip("203.0.113.5", "1.2.3.4", TRUSTED) == "203.0.113.5"


def test_client_ip_uses_last_hop_from_trusted_proxy() -> None:
    assert client_ip("10.0.1.7", "6.6.6.6, 198.51.100.9", TRUSTED) == "198.51.100.9"


def test_client_ip_trusted_proxy_without_header_uses_peer() -> None:
    assert client_ip("10.0.1.7", None, TRUSTED) == "10.0.1.7"
    assert client_ip("10.0.1.7", " , ", TRUSTED) == "10.0.1.7"


def test_client_ip_handles_non_ip_peer_and_missing_peer() -> None:
    assert client_ip("testclient", "1.2.3.4", TRUSTED) == "testclient"
    assert client_ip(None, "1.2.3.4", TRUSTED) == "unknown"
