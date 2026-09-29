import asyncio
import threading
from ipaddress import ip_network
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser
from svd_server.app import create_app
from svd_server.auth import fingerprint
from svd_server.engine import SAMPLE_RATE, AudioDecodeError, AudioTooLongError, Transcript
from svd_server.settings import Settings

KEY = "k" * 40
OTHER_KEY = "o" * 40
URL = "/v1/audio/transcriptions"
AUTH = {"Authorization": f"Bearer {KEY}"}
SECRET_TEXT = "mon texte ultra confidentiel"


def audio_file() -> dict[str, Any]:
    return {"file": ("clip.flac", b"fake-audio-bytes", "audio/flac")}


class FakeEngine:
    def __init__(
        self,
        text: str = SECRET_TEXT,
        seconds: float = 2.0,
        decode_error: bool = False,
        too_long: bool = False,
        crash: bool = False,
        gate: threading.Event | None = None,
        started: threading.Event | None = None,
    ) -> None:
        self.text = text
        self.seconds = seconds
        self.decode_error = decode_error
        self.too_long = too_long
        self.crash = crash
        self.gate = gate
        self.started = started
        self.warmed = False
        self.calls: list[tuple[str | None, str | None]] = []

    def warmup(self) -> None:
        self.warmed = True

    def decode(self, data: bytes) -> np.ndarray:
        if self.too_long:
            raise AudioTooLongError("long")
        if self.decode_error:
            raise AudioDecodeError("bad")
        return np.zeros(int(self.seconds * SAMPLE_RATE), dtype=np.float32)

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None) -> Transcript:
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.crash:
            raise RuntimeError("model exploded")
        self.calls.append((language, prompt))
        return Transcript(self.text, language or "fr", len(audio) / SAMPLE_RATE)


def settings(**overrides: Any) -> Settings:
    return Settings(api_keys=(KEY, OTHER_KEY), **overrides)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> Any:
    with TestClient(create_app(settings(), engine)) as test_client:
        yield test_client


def test_health_is_ok_after_warmup(client: TestClient, engine: FakeEngine) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.text == "ok"
    assert engine.warmed


def test_health_is_503_before_startup() -> None:
    app = create_app(settings(), FakeEngine())
    response = TestClient(app).get("/health")  # no `with`: lifespan not run
    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_are_disabled(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_missing_key_is_401(client: TestClient) -> None:
    response = client.post(URL, files=audio_file())
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Invalid or missing API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }


def test_wrong_key_is_401(client: TestClient) -> None:
    response = client.post(URL, headers={"Authorization": "Bearer " + "x" * 40}, files=audio_file())
    assert response.status_code == 401


def test_json_response(client: TestClient) -> None:
    response = client.post(URL, headers=AUTH, files=audio_file())
    assert response.status_code == 200
    assert response.json() == {"text": SECRET_TEXT}


def test_verbose_json_response_and_options(client: TestClient, engine: FakeEngine) -> None:
    response = client.post(
        URL,
        headers=AUTH,
        files=audio_file(),
        data={
            "response_format": "verbose_json",
            "language": "fr",
            "prompt": "Polyspace",
            "model": "whisper-1",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"text": SECRET_TEXT, "language": "fr", "duration": 2.0}
    assert engine.calls == [("fr", "Polyspace")]


def test_blank_language_and_prompt_mean_none(client: TestClient, engine: FakeEngine) -> None:
    client.post(URL, headers=AUTH, files=audio_file(), data={"language": " ", "prompt": ""})
    assert engine.calls == [(None, None)]


def test_second_key_also_works(client: TestClient) -> None:
    response = client.post(
        URL, headers={"Authorization": f"Bearer {OTHER_KEY}"}, files=audio_file()
    )
    assert response.status_code == 200


def test_unsupported_response_format_is_422(client: TestClient) -> None:
    response = client.post(URL, headers=AUTH, files=audio_file(), data={"response_format": "srt"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_response_format"


def test_missing_file_is_422(client: TestClient) -> None:
    response = client.post(URL, headers=AUTH, data={"language": "fr"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_file"


def test_undecodable_audio_is_422() -> None:
    with TestClient(create_app(settings(), FakeEngine(decode_error=True))) as client:
        response = client.post(URL, headers=AUTH, files=audio_file())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_audio"


def test_audio_too_long_is_413() -> None:
    engine = FakeEngine(seconds=126)
    with TestClient(create_app(settings(max_audio_seconds=125), engine)) as client:
        response = client.post(URL, headers=AUTH, files=audio_file())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_long"
    assert engine.calls == []  # never transcribed


def test_upload_too_large_is_413() -> None:
    with TestClient(create_app(settings(max_upload_mb=1), FakeEngine())) as client:
        big = {"file": ("clip.flac", b"x" * (2 * 1024 * 1024), "audio/flac")}
        response = client.post(URL, headers=AUTH, files=big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_uploads_stay_in_memory() -> None:
    create_app(settings(max_upload_mb=10), FakeEngine())
    assert MultiPartParser.spool_max_size >= 10 * 1024 * 1024


def test_engine_crash_is_500_without_details() -> None:
    with TestClient(create_app(settings(), FakeEngine(crash=True))) as client:
        response = client.post(URL, headers=AUTH, files=audio_file())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "exploded" not in response.text


def test_rate_limit_per_key() -> None:
    with TestClient(create_app(settings(rate_limit_per_minute=2), FakeEngine())) as client:
        codes = [client.post(URL, headers=AUTH, files=audio_file()).status_code for _ in range(3)]
        other = client.post(
            URL, headers={"Authorization": f"Bearer {OTHER_KEY}"}, files=audio_file()
        )
    assert codes == [200, 200, 429]
    assert other.status_code == 200


def test_auth_failures_block_bad_keys_but_never_a_valid_key() -> None:
    bad = {"Authorization": "Bearer " + "x" * 40}
    with TestClient(create_app(settings(auth_failures_per_minute=3), FakeEngine())) as client:
        failures = [client.post(URL, headers=bad, files=audio_file()).status_code for _ in range(3)]
        blocked = client.post(URL, headers=bad, files=audio_file())
        missing = client.post(URL, files=audio_file())
        valid = client.post(URL, headers=AUTH, files=audio_file())
    assert failures == [401, 401, 401]
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "auth_rate_limited"
    assert missing.status_code == 429
    assert valid.status_code == 200  # a stranger sharing the IP cannot lock the owner out


async def post_from(app: Any, peer: str, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=(peer, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(URL, headers=headers, files=audio_file())


async def test_forwarded_for_is_honoured_only_from_trusted_proxy() -> None:
    app = create_app(
        settings(auth_failures_per_minute=1, trusted_proxies=(ip_network("10.0.1.0/24"),)),
        FakeEngine(),
    )
    bad = {"Authorization": "Bearer " + "x" * 40}

    # Through the trusted proxy, the client 198.51.100.9 fails once and is blocked...
    await post_from(app, "10.0.1.7", {**bad, "X-Forwarded-For": "198.51.100.9"})
    blocked = await post_from(app, "10.0.1.7", {**bad, "X-Forwarded-For": "198.51.100.9"})
    assert blocked.status_code == 429
    # ...while another client behind the same proxy is not.
    other = await post_from(app, "10.0.1.7", {**bad, "X-Forwarded-For": "198.51.100.10"})
    assert other.status_code == 401

    # A direct, untrusted peer cannot escape its block by forging the header.
    await post_from(app, "203.0.113.5", {**bad, "X-Forwarded-For": "1.1.1.1"})
    forged = await post_from(app, "203.0.113.5", {**bad, "X-Forwarded-For": "2.2.2.2"})
    assert forged.status_code == 429
    # A valid key is still served from a blocked IP.
    owner = await post_from(app, "203.0.113.5", {**AUTH, "X-Forwarded-For": "2.2.2.2"})
    assert owner.status_code == 200


async def test_busy_server_returns_503_with_retry_after() -> None:
    gate, started = threading.Event(), threading.Event()
    app = create_app(settings(queue_size=0), FakeEngine(gate=gate, started=started))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post(URL, headers=AUTH, files=audio_file()))
        assert await asyncio.to_thread(started.wait, 5)
        second = await client.post(URL, headers=AUTH, files=audio_file())
        gate.set()
        assert (await first).status_code == 200
    assert second.status_code == 503
    assert second.headers["retry-after"] == "5"
    assert second.json()["error"]["code"] == "server_busy"


def test_logs_never_contain_text_or_keys(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="svd_server")
    client.post(URL, headers=AUTH, files=audio_file())
    client.post(URL, headers={"Authorization": "Bearer " + "x" * 40}, files=audio_file())
    records = [r for r in caplog.records if r.name == "svd_server.access"]
    assert [r.svd["status"] for r in records] == [200, 401]  # type: ignore[attr-defined]
    assert records[0].svd["key"] == fingerprint(KEY)  # type: ignore[attr-defined]
    assert records[0].svd["audio_seconds"] == 2.0  # type: ignore[attr-defined]
    full_log = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    assert SECRET_TEXT not in full_log
    assert KEY not in full_log


def test_engine_declared_too_long_is_413() -> None:
    with TestClient(create_app(settings(), FakeEngine(too_long=True))) as client:
        response = client.post(URL, headers=AUTH, files=audio_file())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_long"


def test_two_file_parts_is_400_invalid_multipart(client: TestClient) -> None:
    files = [
        ("file", ("a.flac", b"a", "audio/flac")),
        ("file", ("b.flac", b"b", "audio/flac")),
    ]
    response = client.post(URL, headers=AUTH, files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"


def test_multipart_without_boundary_is_400(client: TestClient) -> None:
    response = client.post(
        URL, headers={**AUTH, "Content-Type": "multipart/form-data"}, content=b"junk"
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"


def test_unknown_route_is_404_in_openai_format(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chunked_oversized_upload_is_413() -> None:
    def body() -> Any:
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a"\r\n\r\n'
        for _ in range(4):
            yield b"x" * (512 * 1024)
        yield b"\r\n--b--\r\n"

    with TestClient(create_app(settings(max_upload_mb=1), FakeEngine())) as client:
        response = client.post(
            URL,
            headers={**AUTH, "Content-Type": "multipart/form-data; boundary=b"},
            content=body(),
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_every_request_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="svd_server")
    with TestClient(create_app(settings(max_upload_mb=1), FakeEngine())) as client:
        client.post(URL, headers={**AUTH, "Content-Type": "multipart/form-data"}, content=b"j")
        big = {"file": ("clip.flac", b"x" * (2 * 1024 * 1024), "audio/flac")}
        client.post(URL, headers=AUTH, files=big)
    statuses = [r.svd["status"] for r in caplog.records if r.name == "svd_server.access"]  # type: ignore[attr-defined]
    assert statuses == [400, 413]


def test_trusted_proxies_are_logged_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="svd_server")
    create_app(settings(trusted_proxies=(ip_network("10.0.1.0/24"),)), FakeEngine())
    records = [r for r in caplog.records if r.getMessage() == "trusted proxies"]
    assert len(records) == 1
    assert records[0].name == "svd_server"
    assert records[0].svd == {"trusted_proxies": ["10.0.1.0/24"]}  # type: ignore[attr-defined]
    assert KEY not in caplog.text + str(records[0].__dict__)
