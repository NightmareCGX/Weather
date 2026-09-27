"""Structured JSON logging with request-ID correlation for the API service.

Every log line is emitted as a single JSON object (one line per record) so a
log shipper can index fields without regex parsing (MONITORING.md section 8).
The ``request_id`` field is populated from a context variable that
:class:`api.middleware.RequestIDMiddleware` sets per request, which correlates
application logs, access logs, and the ``X-Request-Id`` response header.

The context variable is deliberately never reset inside the middleware:
Uvicorn runs each HTTP request cycle in its own asyncio task with a fresh
context copy, so the value cannot leak into a later request, while it stays
visible to Uvicorn's access logger (which runs after the application returns,
in the same task).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

#: Context variable carrying the current request ID (empty outside requests).
_request_id_contextvar: ContextVar[str] = ContextVar("api_request_id", default="")

#: Extra ``LogRecord`` fields copied into the JSON payload. Only fields with
#: these prefixes/names are forwarded, so ``logger.info(..., extra={...})``
#: cannot inject arbitrary or conflicting keys.
_EXTRA_FIELD_PREFIXES: tuple[str, ...] = ("telemetry_",)
_EXTRA_FIELD_ALLOWLIST: frozenset[str] = frozenset({"http_method", "http_path", "http_status"})


def set_request_id(request_id: str) -> Token[str]:
    """Bind ``request_id`` to the current execution context."""
    return _request_id_contextvar.set(request_id)


def get_request_id() -> str:
    """Return the request ID bound to the current context (``""`` if none)."""
    return _request_id_contextvar.get("")


class JsonLogFormatter(logging.Formatter):
    """Render ``LogRecord`` instances as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if value is None:
                continue
            if key.startswith(_EXTRA_FIELD_PREFIXES) or key in _EXTRA_FIELD_ALLOWLIST:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(log_format: str) -> None:
    """Install process-wide handlers for the API and Uvicorn loggers.

    Called once from :func:`api.main.create_app`. Replaces the handlers of the
    root logger and the three Uvicorn loggers so access logs, startup
    messages, and application logs share one format. Safe to call repeatedly
    (idempotent handler replacement, not accumulation).

    Args:
        log_format: ``"json"`` (default, production) or ``"text"`` (local
            escape hatch, configured via ``API_LOG_FORMAT``).
    """
    handler = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        handler.setFormatter(JsonLogFormatter())
    elif log_format == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    else:
        raise ValueError(
            f"Unsupported API_LOG_FORMAT {log_format!r} (expected 'json' or 'text')"
        )

    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers = [handler]
        # Uvicorn loggers default to propagating to root; with the root
        # handler replaced here that would double-emit every record.
        target.propagate = name == ""
