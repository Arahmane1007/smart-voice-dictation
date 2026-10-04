import pytest
from svd_server import cli
from svd_server.settings import MIN_KEY_LENGTH


def test_generate_key_is_long_random_and_url_safe() -> None:
    first, second = cli.generate_key(), cli.generate_key()
    assert first != second
    assert len(first) >= MIN_KEY_LENGTH
    assert all(c.isalnum() or c in "-_" for c in first)


def test_generate_key_command_prints_a_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-key"]) == 0
    printed = capsys.readouterr().out.strip()
    assert len(printed) >= MIN_KEY_LENGTH


def test_serve_without_keys_exits_with_error_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(cli, "FasterWhisperEngine", fail)
    assert cli.main(["serve"]) == 2
    assert "API_KEYS" in capsys.readouterr().err


def test_serve_starts_uvicorn_with_our_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "k" * 40)
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli, "FasterWhisperEngine", lambda settings: object())
    monkeypatch.setattr(cli, "create_app", lambda settings, engine: "the-app")

    def fake_run(app: object, **kwargs: object) -> None:
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    assert cli.main(["serve", "--port", "9000"]) == 0
    assert calls == {
        "app": "the-app",
        "host": "0.0.0.0",
        "port": 9000,
        "proxy_headers": False,
        "access_log": False,
    }
