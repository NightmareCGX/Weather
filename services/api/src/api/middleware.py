"""Request ID middleware for request tracing.

Every response carries an ``X-Request-Id`` header (API.md section 2.7). The
generated or client-supplied ID is also stored on ``request.state`` so error
handlers can include it in the RFC 7807 error envelope, and bound to a
context variable so structured JSON logs (MONITORING.md section 8) carry the
same correlation ID.
"""

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.monitoring.logging import set_request_id

#: The tracing header name used on every response.
REQUEST_ID_HEADER = "X-Request-Id"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a request ID on every request/response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex}"
        request.state.request_id = request_id
        # Bound before ``call_next`` so both the downstream application task
        # (spawned by ``call_next``) and Uvicorn's access log (emitted in this
        # task after the response) observe the value. Deliberately not reset:
        # each request runs in its own asyncio task with a fresh context copy
        # (see api.monitoring.logging).
        set_request_id(request_id)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
