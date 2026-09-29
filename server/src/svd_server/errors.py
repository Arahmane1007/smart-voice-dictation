"""Error responses in the OpenAI API format."""

from starlette.responses import JSONResponse


def openai_error(
    status: int,
    message: str,
    type_: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
        headers=headers,
    )
