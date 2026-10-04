"""ASGI middleware that caps the size of request bodies."""

import logging
import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from svd_server.errors import openai_error

logger = logging.getLogger("svd_server.access")


class BodyTooLarge(Exception):
    """Raised while streaming a request body that exceeds the limit."""


def too_large_response() -> JSONResponse:
    return openai_error(413, "Request body too large", "invalid_request_error", "file_too_large")


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            started = time.perf_counter()
            await too_large_response()(scope, receive, send)
            logger.info(
                "request",
                extra={
                    "svd": {
                        "route": scope["path"],
                        "status": 413,
                        "key": None,
                        "audio_seconds": None,
                        "processing_ms": round((time.perf_counter() - started) * 1000),
                    }
                },
            )
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
