"""RFC 7807-style error handling for the API service.

All failures are rendered as the error envelope defined in ``docs/API.md``
section 2.4 (``{"error": {"code", "type", "message", "param", "request_id"}}``).
The ``request_id`` is read from ``request.state`` where it was placed by
:class:`api.middleware.RequestIDMiddleware`, so it always matches the
``X-Request-Id`` response header.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.middleware import REQUEST_ID_HEADER
from api.schemas import ErrorDetail, ErrorEnvelope

logger = logging.getLogger(__name__)

#: Error code for client-side request failures (API.md example value).
CODE_INVALID_REQUEST = "invalid_request_error"
#: Error type for client-side request failures (API.md example value).
TYPE_INVALID_REQUEST = "invalid_request_error"
#: Error type for request validation failures (API.md example value).
TYPE_VALIDATION_ERROR = "validation_error"
#: Error type for resources that do not exist.
TYPE_NOT_FOUND_ERROR = "not_found_error"
#: Error code for unexpected server-side failures.
CODE_API_ERROR = "api_error"
#: Error type for unexpected server-side failures.
TYPE_SERVER_ERROR = "server_error"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    status_code: int,
    code: str,
    error_type: str,
    message: str,
    param: str | None,
    request_id: str | None,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            type=error_type,
            message=message,
            param=param,
            request_id=request_id,
        )
    ).model_dump(mode="json")
    response = JSONResponse(status_code=status_code, content=payload)
    # API.md section 2.7 requires ``X-Request-Id`` on every response. For
    # unhandled 500s the generic ``Exception`` handler runs on Starlette's
    # outermost ServerErrorMiddleware, outside ``RequestIDMiddleware``, so the
    # middleware's post-``call_next`` header assignment never executes. Set the
    # header here so every RFC 7807 error response carries it consistently.
    if request_id is not None:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = first.get("loc", ())
    param = str(loc[-1]) if loc else None
    message = str(first.get("msg", "Request validation failed"))
    return _error_response(
        422,
        CODE_INVALID_REQUEST,
        TYPE_VALIDATION_ERROR,
        message,
        param,
        _request_id(request),
    )


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    error_type = TYPE_NOT_FOUND_ERROR if exc.status_code == 404 else TYPE_INVALID_REQUEST
    return _error_response(
        exc.status_code,
        CODE_INVALID_REQUEST,
        error_type,
        str(exc.detail),
        None,
        _request_id(request),
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception while processing %s", request.url)
    return _error_response(
        500,
        CODE_API_ERROR,
        TYPE_SERVER_ERROR,
        "An unexpected error occurred",
        None,
        _request_id(request),
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register the RFC 7807 exception handlers on the application."""
    app.exception_handler(RequestValidationError)(_validation_error_handler)
    app.exception_handler(StarletteHTTPException)(_http_exception_handler)
    app.exception_handler(Exception)(_unhandled_exception_handler)
