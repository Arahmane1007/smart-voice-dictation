from collections.abc import Iterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from svd_server.body_limit import BodySizeLimitMiddleware, BodyTooLarge, too_large_response
from svd_server.errors import openai_error

LIMIT = 1000


async def echo_length(request: Request) -> Response:
    body = await request.body()
    return JSONResponse({"received": len(body)})


async def on_too_large(request: Request, exc: Exception) -> Response:
    return too_large_response()


def make_client() -> TestClient:
    app = Starlette(
        routes=[Route("/", echo_length, methods=["POST"])],
        exception_handlers={BodyTooLarge: on_too_large},
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=LIMIT)
    return TestClient(app)


def chunks(total: int, size: int = 100) -> Iterator[bytes]:
    for _ in range(total // size):
        yield b"x" * size


def test_openai_error_shape() -> None:
    response = openai_error(401, "Invalid key", "invalid_request_error", "invalid_api_key")
    assert response.status_code == 401
    assert response.body == (
        b'{"error":{"message":"Invalid key","type":"invalid_request_error",'
        b'"code":"invalid_api_key"}}'
    )


def test_small_body_passes() -> None:
    response = make_client().post("/", content=b"x" * LIMIT)
    assert response.status_code == 200
    assert response.json() == {"received": LIMIT}


def test_declared_length_over_limit_is_rejected() -> None:
    response = make_client().post("/", content=b"x" * (LIMIT + 1))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_chunked_body_over_limit_is_rejected() -> None:
    # A generator body is sent without Content-Length (chunked transfer).
    response = make_client().post("/", content=chunks(LIMIT * 5))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_chunked_body_under_limit_passes() -> None:
    response = make_client().post("/", content=chunks(LIMIT))
    assert response.status_code == 200
    assert response.json() == {"received": LIMIT}
