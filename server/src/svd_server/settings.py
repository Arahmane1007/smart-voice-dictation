"""Server configuration, read from environment variables."""

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_network

MIN_KEY_LENGTH = 32


class SettingsError(ValueError):
    """Raised when the environment does not describe a valid configuration."""


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    cpu_threads: int = 0
    max_upload_mb: int = 10
    max_audio_seconds: int = 125
    rate_limit_per_minute: int = 30
    auth_failures_per_minute: int = 10
    queue_size: int = 2
    trusted_proxies: tuple[IPv4Network | IPv6Network, ...] = ()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        keys = _split(env.get("API_KEYS", ""))
        if not keys:
            raise SettingsError("API_KEYS must contain at least one key")
        if any(len(key) < MIN_KEY_LENGTH for key in keys):
            raise SettingsError(
                f"every key in API_KEYS must be at least {MIN_KEY_LENGTH} characters long "
                "(generate one with `svd-server generate-key`)"
            )
        return cls(
            api_keys=keys,
            whisper_model=env.get("WHISPER_MODEL", "").strip() or "small",
            whisper_compute_type=env.get("WHISPER_COMPUTE_TYPE", "").strip() or "int8",
            whisper_beam_size=_int(env, "WHISPER_BEAM_SIZE", 5, minimum=1),
            cpu_threads=_int(env, "CPU_THREADS", 0, minimum=0),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 10, minimum=1),
            max_audio_seconds=_int(env, "MAX_AUDIO_SECONDS", 125, minimum=1),
            rate_limit_per_minute=_int(env, "RATE_LIMIT_PER_MINUTE", 30, minimum=1),
            auth_failures_per_minute=_int(env, "AUTH_FAILURES_PER_MINUTE", 10, minimum=1),
            queue_size=_int(env, "QUEUE_SIZE", 2, minimum=0),
            trusted_proxies=_networks(env.get("TRUSTED_PROXIES", "")),
        )


def _split(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _int(env: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SettingsError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {value}")
    return value


def _networks(raw: str) -> tuple[IPv4Network | IPv6Network, ...]:
    networks = []
    for entry in _split(raw):
        try:
            networks.append(ip_network(entry, strict=False))
        except ValueError:
            raise SettingsError(f"TRUSTED_PROXIES contains an invalid address: {entry!r}") from None
    return tuple(networks)
