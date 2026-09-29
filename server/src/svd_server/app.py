"""FastAPI application: the transcription route and the health check."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartParser
from starlette.responses import JSONResponse, PlainTextResponse, Response

from svd_server.auth import KeyVerifier, client_ip, parse_bearer
from svd_server.body_limit import BodySizeLimitMiddleware, BodyTooLarge, too_large_response
from svd_server.engine import (
    SAMPLE_RATE,
    AudioDecodeError,
    AudioTooLongError,
    TranscriptionEngine,
)
from svd_server.errors import openai_error
from svd_server.limits import QueueFull, SlidingWindowLimiter, TranscriptionQueue
from svd_server.settings import Settings

logger = logging.getLogger("svd_server.access")

ROUTE = "/v1/audio/transcriptions"
MULTIPART_OVERHEAD = 64 * 1024
RESPONSE_FORMATS = {"json", "verbose_json"}
RETRY_AFTER_SECONDS = "5"
MAX_FIELD_BYTES = 16 * 1024
HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def create_app(settings: Settings, engine: TranscriptionEngine) -> FastAPI:
    # Uploaded files are spooled in memory up to this size: audio never touches the disk.
    MultiPartParser.spool_max_size = settings.max_upload_bytes + MULTIPART_OVERHEAD

    # Behind Traefik/Dokploy a wrong value makes every client share the proxy's IP.
    logging.getLogger("svd_server").info(
        "trusted proxies",
        extra={"svd": {"trusted_proxies": [str(net) for net in settings.trusted_proxies]}},
    )

    verifier = KeyVerifier(settings.api_keys)
    key_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    auth_failures = SlidingWindowLimiter(settings.auth_failures_per_minute)
    queue = TranscriptionQueue(settings.queue_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(engine.warmup)
        app.state.ready = True
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ready = False
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD
    )

    async def on_body_too_large(request: Request, exc: Exception) -> Response:
        return too_large_response()

    app.add_exception_handler(BodyTooLarge, on_body_too_large)

    async def on_http_exception(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return openai_error(
            exc.status_code,
            str(exc.detail),
            "invalid_request_error",
            code,
            headers=dict(exc.headers) if exc.headers else None,
        )

    app.add_exception_handler(HTTPException, on_http_exception)

    @app.get("/health")
    async def health() -> Response:
        if app.state.ready:
            return PlainTextResponse("ok")
        return PlainTextResponse("starting", status_code=503)

    @app.post(ROUTE)
    async def transcriptions(request: Request) -> Response:
        started = time.perf_counter()
        key: str | None = None
        audio_seconds: float | None = None

        def done(response: Response) -> Response:
            logger.info(
                "request",
                extra={
                    "svd": {
                        "route": ROUTE,
                        "status": response.status_code,
                        "key": key,
                        "audio_seconds": audio_seconds,
                        "processing_ms": round((time.perf_counter() - started) * 1000),
                    }
                },
            )
            return response

        def too_long() -> Response:
            return openai_error(
                413,
                f"Audio longer than {settings.max_audio_seconds} seconds",
                "invalid_request_error",
                "audio_too_long",
            )

        ip = client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            settings.trusted_proxies,
        )
        # The key is checked first: a valid key is never blocked by its IP, so a
        # stranger behind the same NAT (or proxy) cannot lock the owner out.
        key = verifier.verify(parse_bearer(request.headers.get("authorization")))
        if key is None:
            if auth_failures.blocked(ip):
                return done(
                    openai_error(
                        429,
                        "Too many failed authentication attempts",
                        "rate_limit_error",
                        "auth_rate_limited",
                    )
                )
            auth_failures.record(ip)
            return done(
                openai_error(
                    401, "Invalid or missing API key", "invalid_request_error", "invalid_api_key"
                )
            )

        if not key_limiter.allow(key):
            return done(
                openai_error(429, "Rate limit exceeded", "rate_limit_error", "rate_limit_exceeded")
            )

        try:
            form = await request.form(max_files=1, max_fields=10, max_part_size=MAX_FIELD_BYTES)
        except BodyTooLarge:
            return done(too_large_response())
        except HTTPException:  # Starlette's MultiPartException, e.g. no boundary, two files
            return done(
                openai_error(
                    400, "Invalid multipart body", "invalid_request_error", "invalid_multipart"
                )
            )
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                return done(
                    openai_error(422, "Missing audio file", "invalid_request_error", "missing_file")
                )
            response_format = _text_field(form.get("response_format")) or "json"
            if response_format not in RESPONSE_FORMATS:
                return done(
                    openai_error(
                        422,
                        "response_format must be 'json' or 'verbose_json'",
                        "invalid_request_error",
                        "unsupported_response_format",
                    )
                )
            language = _text_field(form.get("language"))
            prompt = _text_field(form.get("prompt"))
            data = await upload.read()
            if len(data) > settings.max_upload_bytes:
                return done(too_large_response())

            try:
                async with queue.slot():
                    audio = await asyncio.to_thread(engine.decode, data)
                    audio_seconds = round(len(audio) / SAMPLE_RATE, 2)
                    if audio_seconds > settings.max_audio_seconds:
                        return done(too_long())
                    transcript = await asyncio.to_thread(engine.transcribe, audio, language, prompt)
            except QueueFull:
                return done(
                    openai_error(
                        503,
                        "Server busy, retry shortly",
                        "server_error",
                        "server_busy",
                        headers={"Retry-After": RETRY_AFTER_SECONDS},
                    )
                )
            except AudioTooLongError:
                return done(too_long())
            except AudioDecodeError:
                return done(
                    openai_error(
                        422, "Could not decode audio", "invalid_request_error", "invalid_audio"
                    )
                )
            except Exception as exc:  # never leak internals to the client or the logs
                logger.error("transcription failed", extra={"svd": {"error": type(exc).__name__}})
                return done(openai_error(500, "Internal error", "server_error", "internal_error"))
        finally:
            await form.close()

        if response_format == "json":
            return done(JSONResponse({"text": transcript.text}))
        return done(
            JSONResponse(
                {
                    "text": transcript.text,
                    "language": transcript.language,
                    "duration": round(transcript.duration, 2),
                }
            )
        )

    return app


def _text_field(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
