from ipaddress import ip_network

import pytest
from svd_server.settings import MIN_KEY_LENGTH, Settings, SettingsError

KEY_A = "a" * MIN_KEY_LENGTH
KEY_B = "b" * 40


def test_defaults_with_single_key() -> None:
    settings = Settings.from_env({"API_KEYS": KEY_A})
    assert settings.api_keys == (KEY_A,)
    assert settings.whisper_model == "small"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_beam_size == 5
    assert settings.cpu_threads == 0
    assert settings.max_upload_mb == 10
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_audio_seconds == 125
    assert settings.rate_limit_per_minute == 30
    assert settings.auth_failures_per_minute == 10
    assert settings.queue_size == 2
    assert settings.trusted_proxies == ()


def test_several_keys_are_split_and_stripped() -> None:
    settings = Settings.from_env({"API_KEYS": f" {KEY_A} , {KEY_B} ,"})
    assert settings.api_keys == (KEY_A, KEY_B)


@pytest.mark.parametrize("value", [None, "", "  ", " , ,"])
def test_missing_or_blank_keys_are_rejected(value: str | None) -> None:
    env = {} if value is None else {"API_KEYS": value}
    with pytest.raises(SettingsError, match="API_KEYS"):
        Settings.from_env(env)


def test_short_key_is_rejected_without_leaking_it() -> None:
    with pytest.raises(SettingsError) as info:
        Settings.from_env({"API_KEYS": f"{KEY_A},tooshort"})
    assert "tooshort" not in str(info.value)
    assert str(MIN_KEY_LENGTH) in str(info.value)


def test_integer_overrides() -> None:
    settings = Settings.from_env(
        {
            "API_KEYS": KEY_A,
            "WHISPER_MODEL": "medium",
            "WHISPER_BEAM_SIZE": "1",
            "CPU_THREADS": "4",
            "MAX_UPLOAD_MB": "5",
            "MAX_AUDIO_SECONDS": "60",
            "RATE_LIMIT_PER_MINUTE": "10",
            "AUTH_FAILURES_PER_MINUTE": "3",
            "QUEUE_SIZE": "0",
        }
    )
    assert settings.whisper_model == "medium"
    assert settings.whisper_beam_size == 1
    assert settings.cpu_threads == 4
    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert settings.max_audio_seconds == 60
    assert settings.rate_limit_per_minute == 10
    assert settings.auth_failures_per_minute == 3
    assert settings.queue_size == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [("QUEUE_SIZE", "two"), ("MAX_UPLOAD_MB", "0"), ("WHISPER_BEAM_SIZE", "-1")],
)
def test_invalid_integers_are_rejected(name: str, value: str) -> None:
    with pytest.raises(SettingsError, match=name):
        Settings.from_env({"API_KEYS": KEY_A, name: value})


def test_trusted_proxies_accept_ips_and_networks() -> None:
    settings = Settings.from_env({"API_KEYS": KEY_A, "TRUSTED_PROXIES": "10.0.1.0/24, 127.0.0.1"})
    assert settings.trusted_proxies == (ip_network("10.0.1.0/24"), ip_network("127.0.0.1/32"))


def test_invalid_trusted_proxy_is_rejected() -> None:
    with pytest.raises(SettingsError, match="TRUSTED_PROXIES"):
        Settings.from_env({"API_KEYS": KEY_A, "TRUSTED_PROXIES": "not-an-ip"})
