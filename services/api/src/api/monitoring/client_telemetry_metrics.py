"""Bounded-cardinality counter for client telemetry events.

Rendered by ``/v1/metrics`` alongside the HTTP request metrics. Only the
``event_type`` label (``error`` | ``web_vital``) is bounded and allowed;
free-form event names are routed exclusively to structured logs
(MONITORING.md section 1, cardinality safety guarantee).
"""

from __future__ import annotations

import threading

#: Metric name exposed on ``/v1/metrics`` (MONITORING.md section 2.8).
CLIENT_TELEMETRY_METRIC_NAME = "weather_client_telemetry_events_total"

#: The bounded label values this counter accepts.
ALLOWED_EVENT_TYPES: tuple[str, ...] = ("error", "web_vital")


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class ClientTelemetryCounter:
    """Thread-safe counter with a single bounded ``event_type`` label."""

    def __init__(
        self,
        name: str,
        documentation: str,
        allowed_values: tuple[str, ...] = ALLOWED_EVENT_TYPES,
    ) -> None:
        self._name = name
        self._documentation = documentation
        self._allowed_values = allowed_values
        self._lock = threading.Lock()
        self._counts: dict[str, int] = dict.fromkeys(allowed_values, 0)

    def labels(self, event_type: str) -> _CounterChild:
        """Return a handle for one ``event_type`` label value."""
        if event_type not in self._allowed_values:
            raise ValueError(
                f"Unbounded telemetry label value {event_type!r}; "
                f"allowed values: {self._allowed_values}"
            )
        return _CounterChild(self, event_type)

    def render_lines(self) -> list[str]:
        """Render the Prometheus 0.0.4 text exposition lines for this counter."""
        with self._lock:
            counts = dict(self._counts)
        lines = [
            f"# HELP {self._name} {self._documentation}",
            f"# TYPE {self._name} counter",
        ]
        for event_type, count in sorted(counts.items()):
            lines.append(f'{self._name}{{event_type="{_escape_label_value(event_type)}"}} {count}')
        return lines


class _CounterChild:
    """Increment handle for one label value of :class:`ClientTelemetryCounter`."""

    def __init__(self, counter: ClientTelemetryCounter, event_type: str) -> None:
        self._counter = counter
        self._event_type = event_type

    def inc(self, amount: int = 1) -> None:
        with self._counter._lock:
            self._counter._counts[self._event_type] += amount


#: Process-wide client telemetry counter rendered by ``/v1/metrics``.
CLIENT_TELEMETRY_EVENTS_TOTAL = ClientTelemetryCounter(
    CLIENT_TELEMETRY_METRIC_NAME,
    "Client telemetry events accepted by POST /v1/telemetry/client",
)
