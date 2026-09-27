"""Process-local HTTP request metrics for the API serving layer (RED method).

Implements rate / errors / duration telemetry for every served request with
strictly bounded label cardinality (MONITORING.md section 1, cardinality
safety guarantee):

- ``route``: the FastAPI route template (e.g. ``/v1/points``), never the raw
  request path. Requests that match no route (404 scans, probes) collapse to
  the single ``unmatched`` label, so the label space cannot grow.
- ``method``: normalized to the common HTTP verb set; anything outside it
  collapses to ``OTHER`` (the method header is client-controlled).
- ``status_class``: the response status class (``2xx`` .. ``5xx``). Unhandled
  exceptions that never produce a response are recorded as ``5xx``.

The ``/v1/metrics`` endpoint itself is excluded from collection: every
Prometheus scrape would otherwise add a self-referential datapoint, and the
scrape-time dependency probes would distort the latency distribution.

The primitives are deliberately hand-rolled (thread-safe dicts, Prometheus
text exposition 0.0.4) instead of importing the ingestion metrics registry:
the API and ingestion packages are independently installable and the API does
not depend on the ingestion package (ENGINEERING_CONTRACT section 6).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

#: Metric names exposed on ``/v1/metrics`` (MONITORING.md section 2.7).
REQUESTS_TOTAL_METRIC = "weather_api_http_requests_total"
REQUEST_DURATION_METRIC = "weather_api_http_request_duration_seconds"
REQUESTS_IN_FLIGHT_METRIC = "weather_api_http_requests_in_flight"

#: Route templates whose traffic is not counted: scraping the metrics
#: endpoint must not generate self-referential request datapoints.
EXCLUDED_ROUTE_TEMPLATES: frozenset[str] = frozenset({"/v1/metrics"})

#: Label reported when no FastAPI route matched (404 scans, health bots
#: hitting unknown paths). A single constant label keeps cardinality bounded.
UNMATCHED_ROUTE_LABEL = "unmatched"

#: Known HTTP verbs; anything outside this set collapses to ``OTHER``.
_KNOWN_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
)
_OTHER_METHOD_LABEL = "OTHER"

#: Latency histogram buckets (seconds). Chosen to separate tile-fast
#: (~millisecond) requests from ensemble-slow (~tens of seconds) fan-outs;
#: documented in MONITORING.md section 2.7.
DURATION_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


def _status_class(status_code: int | None) -> str:
    """Map a response status code (or absence of one) to its status class."""
    if status_code is None or not 100 <= status_code < 600:
        # No response was produced (unhandled exception) -> server error.
        return "5xx"
    return f"{status_code // 100}xx"


def _method_label(method: str) -> str:
    """Normalize the client-controlled HTTP method to a bounded label."""
    normalized = method.upper()
    return normalized if normalized in _KNOWN_METHODS else _OTHER_METHOD_LABEL


def _route_template(scope: MutableMapping[str, Any]) -> str:
    """Return the matched FastAPI route template, or ``unmatched``.

    FastAPI populates ``scope["route"]`` during routing (``fastapi/routing.py``
    sets ``child_scope["route"]``), which mutates the same scope dict this
    middleware holds, so the template is read *after* ``call_next`` returns.
    """
    route: Any = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return UNMATCHED_ROUTE_LABEL


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_labels(labels: dict[str, str]) -> str:
    """Render sorted ``k="v"`` pairs in Prometheus exposition format."""
    items = [
        f'{key}="{_escape_label_value(value)}"' for key, value in sorted(labels.items())
    ]
    return "{" + ",".join(items) + "}"


class HttpMetrics:
    """Thread-safe, bounded-cardinality HTTP request metrics registry.

    Storage is process-local: with multiple Uvicorn workers each worker
    reports its own values from ``/v1/metrics`` (the endpoint is served by a
    single worker per scrape), mirroring the existing process-local resource
    gauges. Prometheus aggregates per-worker series via the ``instance`` label.
    """

    def __init__(self, buckets: tuple[float, ...] = DURATION_BUCKETS) -> None:
        self._lock = threading.Lock()
        resolved = list(buckets)
        # Prometheus semantics: the +Inf bucket must always exist and equal
        # the observed count, so durations above the highest bound are counted.
        if float("inf") not in resolved:
            resolved.append(float("inf"))
        self._buckets: tuple[float, ...] = tuple(sorted(resolved))
        # (route, method, status_class) -> cumulative request count.
        self._counts: dict[tuple[str, str, str], int] = {}
        # (route, method) -> per-bucket cumulative counts, index-aligned with
        # ``self._buckets``.
        self._bucket_counts: dict[tuple[str, str], list[int]] = {}
        self._sums: dict[tuple[str, str], float] = {}
        self._totals: dict[tuple[str, str], int] = {}
        self._in_flight = 0

    def enter(self) -> None:
        """Mark one request as in flight."""
        with self._lock:
            self._in_flight += 1

    def exit(self) -> None:
        """Mark one in-flight request as finished."""
        with self._lock:
            self._in_flight -= 1

    def record(
        self, route: str, method: str, status_class: str, duration_seconds: float
    ) -> None:
        """Record one completed (or failed) request observation."""
        with self._lock:
            self._counts[(route, method, status_class)] = (
                self._counts.get((route, method, status_class), 0) + 1
            )
            series = (route, method)
            if series not in self._bucket_counts:
                self._bucket_counts[series] = [0] * len(self._buckets)
            bucket_counts = self._bucket_counts[series]
            for index, bound in enumerate(self._buckets):
                if duration_seconds <= bound:
                    bucket_counts[index] += 1
            self._sums[series] = self._sums.get(series, 0.0) + duration_seconds
            self._totals[series] = self._totals.get(series, 0) + 1

    def in_flight(self) -> int:
        """Return the number of requests currently being served."""
        with self._lock:
            return self._in_flight

    def render_lines(self) -> list[str]:
        """Render the Prometheus 0.0.4 text exposition lines for this registry."""
        with self._lock:
            counts = dict(self._counts)
            bucket_counts = {
                series: list(values) for series, values in self._bucket_counts.items()
            }
            sums = dict(self._sums)
            totals = dict(self._totals)
            in_flight = self._in_flight

        lines: list[str] = []
        lines.append(
            f"# HELP {REQUESTS_TOTAL_METRIC} HTTP requests processed by the API serving layer"
        )
        lines.append(f"# TYPE {REQUESTS_TOTAL_METRIC} counter")
        for (route, method, status_class), count in sorted(counts.items()):
            labels = _format_labels(
                {"method": method, "route": route, "status_class": status_class}
            )
            lines.append(f"{REQUESTS_TOTAL_METRIC}{labels} {count}")

        lines.append(
            f"# HELP {REQUEST_DURATION_METRIC} HTTP request duration in seconds"
        )
        lines.append(f"# TYPE {REQUEST_DURATION_METRIC} histogram")
        for (route, method), values in sorted(bucket_counts.items()):
            base_labels = _format_labels({"method": method, "route": route})
            for bound, count in zip(self._buckets, values, strict=True):
                le = "+Inf" if bound == float("inf") else str(bound)
                le_labels = _format_labels(
                    {"le": le, "method": method, "route": route}
                )
                lines.append(f"{REQUEST_DURATION_METRIC}_bucket{le_labels} {count}")
            lines.append(f"{REQUEST_DURATION_METRIC}_sum{base_labels} {sums[(route, method)]}")
            lines.append(
                f"{REQUEST_DURATION_METRIC}_count{base_labels} {totals[(route, method)]}"
            )

        lines.append(
            f"# HELP {REQUESTS_IN_FLIGHT_METRIC} HTTP requests currently in flight"
        )
        lines.append(f"# TYPE {REQUESTS_IN_FLIGHT_METRIC} gauge")
        lines.append(f"{REQUESTS_IN_FLIGHT_METRIC} {in_flight}")
        return lines


class HTTPMetricsMiddleware(BaseHTTPMiddleware):
    """Outermost observability middleware: counts every served request.

    Added after :class:`api.middleware.RequestIDMiddleware` so it sits
    outermost among the user middlewares: request durations include all
    middleware overhead, and requests that fail inside inner middlewares are
    still counted (as ``5xx``).
    """

    def __init__(
        self,
        app: Any,
        metrics: HttpMetrics | None = None,
    ) -> None:
        super().__init__(app)
        self._metrics = metrics if metrics is not None else HTTP_METRICS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Excluded endpoints (the metrics exposition itself) bypass collection
        # entirely — including the in-flight gauge, otherwise every scrape
        # would observe a constant floor of 1. The raw path is matched exactly
        # against the constant exclusion set before routing; the bounded
        # route template is still used for recording below.
        if request.url.path in EXCLUDED_ROUTE_TEMPLATES:
            return await call_next(request)

        started = time.perf_counter()
        self._metrics.enter()
        try:
            response = await call_next(request)
        except Exception:
            self._metrics.record(
                _route_template(request.scope),
                _method_label(request.method),
                _status_class(None),
                time.perf_counter() - started,
            )
            self._metrics.exit()
            raise
        self._metrics.exit()
        # ``scope["route"]`` is populated by the router inside ``call_next``,
        # so the route template is only readable after the call completes.
        self._metrics.record(
            _route_template(request.scope),
            _method_label(request.method),
            _status_class(response.status_code),
            time.perf_counter() - started,
        )
        return response


#: Process-wide HTTP metrics singleton rendered by ``/v1/metrics``.
HTTP_METRICS = HttpMetrics()
