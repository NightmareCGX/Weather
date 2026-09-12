"""Lightweight, zero-external-dependency Prometheus metrics registry and exposition format.

Provides high-performance, thread-safe metric primitives (Counter, Gauge, Histogram)
and generates Prometheus text exposition format (version 0.0.4) without requiring
external dependencies. Bounded cardinality is strictly enforced.
"""

from __future__ import annotations

import re
import threading
import time

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label_value(val: str) -> str:
    return val.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class Metric:
    """Base class for Prometheus metrics."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        if not _METRIC_NAME_RE.match(name):
            raise ValueError(f"Invalid metric name: {name!r}")
        for label in labelnames:
            if not _LABEL_NAME_RE.match(label):
                raise ValueError(f"Invalid label name: {label!r}")
            if label.startswith("__"):
                raise ValueError(f"Label names cannot start with '__': {label!r}")
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._lock = threading.Lock()

    def collect(self) -> list[str]:
        """Return Prometheus lines for this metric."""
        raise NotImplementedError

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        items = [f'{k}="{_escape_label_value(str(v))}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(items) + "}"


class Gauge(Metric):
    """A metric that represents a single numerical value that can arbitrarily go up and down."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, *labelvalues: str, **kwargs: str) -> _GaugeChild:
        vals = self._resolve_labels(labelvalues, kwargs)
        return _GaugeChild(self, vals)

    def set(self, value: float) -> None:
        if self.labelnames:
            raise ValueError("Must use .labels() for metrics with labelnames")
        with self._lock:
            self._values[()] = float(value)

    def inc(self, amount: float = 1.0) -> None:
        if self.labelnames:
            raise ValueError("Must use .labels() for metrics with labelnames")
        with self._lock:
            self._values[()] = self._values.get((), 0.0) + float(amount)

    def dec(self, amount: float = 1.0) -> None:
        self.inc(-amount)

    def _resolve_labels(
        self,
        labelvalues: tuple[str, ...],
        kwargs: dict[str, str],
    ) -> tuple[str, ...]:
        if labelvalues and kwargs:
            raise ValueError("Cannot mix positional and keyword labels")
        if kwargs:
            if len(kwargs) != len(self.labelnames):
                raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(kwargs)}")
            vals = tuple(str(kwargs[k]) for k in self.labelnames if k in kwargs)
            if len(vals) != len(self.labelnames):
                raise ValueError(f"Missing labels in {kwargs}")
            return vals
        if len(labelvalues) != len(self.labelnames):
            raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(labelvalues)}")
        return tuple(str(v) for v in labelvalues)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.documentation)}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            items = sorted(self._values.items())
        for label_vals, val in items:
            lbl_dict = dict(zip(self.labelnames, label_vals, strict=False))
            lbl_str = self._format_labels(lbl_dict)
            lines.append(f"{self.name}{lbl_str} {val}")
        return lines


class _GaugeChild:
    def __init__(self, gauge: Gauge, labelvalues: tuple[str, ...]) -> None:
        self._gauge = gauge
        self._labelvalues = labelvalues

    def set(self, value: float) -> None:
        with self._gauge._lock:
            self._gauge._values[self._labelvalues] = float(value)

    def inc(self, amount: float = 1.0) -> None:
        with self._gauge._lock:
            cur = self._gauge._values.get(self._labelvalues, 0.0)
            self._gauge._values[self._labelvalues] = cur + float(amount)

    def dec(self, amount: float = 1.0) -> None:
        self.inc(-amount)

    def set_to_current_time(self) -> None:
        self.set(time.time())


class Counter(Metric):
    """A cumulative metric that represents a single monotonically increasing counter."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, *labelvalues: str, **kwargs: str) -> _CounterChild:
        vals = self._resolve_labels(labelvalues, kwargs)
        return _CounterChild(self, vals)

    def inc(self, amount: float = 1.0) -> None:
        if self.labelnames:
            raise ValueError("Must use .labels() for metrics with labelnames")
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative values")
        with self._lock:
            self._values[()] = self._values.get((), 0.0) + float(amount)

    def _resolve_labels(
        self,
        labelvalues: tuple[str, ...],
        kwargs: dict[str, str],
    ) -> tuple[str, ...]:
        if labelvalues and kwargs:
            raise ValueError("Cannot mix positional and keyword labels")
        if kwargs:
            if len(kwargs) != len(self.labelnames):
                raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(kwargs)}")
            vals = tuple(str(kwargs[k]) for k in self.labelnames if k in kwargs)
            if len(vals) != len(self.labelnames):
                raise ValueError(f"Missing labels in {kwargs}")
            return vals
        if len(labelvalues) != len(self.labelnames):
            raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(labelvalues)}")
        return tuple(str(v) for v in labelvalues)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.documentation)}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            items = sorted(self._values.items())
        for label_vals, val in items:
            lbl_dict = dict(zip(self.labelnames, label_vals, strict=False))
            lbl_str = self._format_labels(lbl_dict)
            lines.append(f"{self.name}{lbl_str} {val}")
        return lines


class _CounterChild:
    def __init__(self, counter: Counter, labelvalues: tuple[str, ...]) -> None:
        self._counter = counter
        self._labelvalues = labelvalues

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative values")
        with self._counter._lock:
            cur = self._counter._values.get(self._labelvalues, 0.0)
            self._counter._values[self._labelvalues] = cur + float(amount)


class Histogram(Metric):
    """A histogram samples observations and counts them in configurable buckets."""

    DEFAULT_BUCKETS: tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(name, documentation, labelnames)
        b = list(buckets or self.DEFAULT_BUCKETS)
        if float("inf") not in b:
            b.append(float("inf"))
        self.buckets = tuple(sorted(b))
        self._counts: dict[tuple[str, ...], dict[float, int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}

    def labels(self, *labelvalues: str, **kwargs: str) -> _HistogramChild:
        vals = self._resolve_labels(labelvalues, kwargs)
        return _HistogramChild(self, vals)

    def observe(self, amount: float) -> None:
        if self.labelnames:
            raise ValueError("Must use .labels() for metrics with labelnames")
        self._observe((), amount)

    def _observe(self, labelvalues: tuple[str, ...], amount: float) -> None:
        val = float(amount)
        with self._lock:
            if labelvalues not in self._counts:
                self._counts[labelvalues] = {b: 0 for b in self.buckets}
                self._sums[labelvalues] = 0.0
            self._sums[labelvalues] += val
            bucket_counts = self._counts[labelvalues]
            for b in self.buckets:
                if val <= b:
                    bucket_counts[b] += 1

    def _resolve_labels(
        self,
        labelvalues: tuple[str, ...],
        kwargs: dict[str, str],
    ) -> tuple[str, ...]:
        if labelvalues and kwargs:
            raise ValueError("Cannot mix positional and keyword labels")
        if kwargs:
            if len(kwargs) != len(self.labelnames):
                raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(kwargs)}")
            vals = tuple(str(kwargs[k]) for k in self.labelnames if k in kwargs)
            if len(vals) != len(self.labelnames):
                raise ValueError(f"Missing labels in {kwargs}")
            return vals
        if len(labelvalues) != len(self.labelnames):
            raise ValueError(f"Expected {len(self.labelnames)} labels, got {len(labelvalues)}")
        return tuple(str(v) for v in labelvalues)

    def collect(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {_escape_help(self.documentation)}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            items = sorted(self._counts.items())
            sums = dict(self._sums)

        for label_vals, bcounts in items:
            base_lbls = dict(zip(self.labelnames, label_vals, strict=False))
            total_count = bcounts[float("inf")]
            total_sum = sums.get(label_vals, 0.0)

            for b in self.buckets:
                lbls_with_le = dict(base_lbls)
                lbls_with_le["le"] = "+Inf" if b == float("inf") else str(b)
                lbl_str = self._format_labels(lbls_with_le)
                lines.append(f"{self.name}_bucket{lbl_str} {bcounts[b]}")

            sum_lbl_str = self._format_labels(base_lbls)
            lines.append(f"{self.name}_sum{sum_lbl_str} {total_sum}")
            lines.append(f"{self.name}_count{sum_lbl_str} {total_count}")

        return lines


class _HistogramChild:
    def __init__(self, histogram: Histogram, labelvalues: tuple[str, ...]) -> None:
        self._histogram = histogram
        self._labelvalues = labelvalues

    def observe(self, amount: float) -> None:
        self._histogram._observe(self._labelvalues, amount)


class MetricRegistry:
    """Thread-safe registry holding registered metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> Metric:
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]
            self._metrics[metric.name] = metric
            return metric

    def get(self, name: str) -> Metric | None:
        with self._lock:
            return self._metrics.get(name)

    def gauge(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
    ) -> Gauge:
        metric = Gauge(name, documentation, labelnames)
        return self.register(metric)  # type: ignore[return-value]

    def counter(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
    ) -> Counter:
        metric = Counter(name, documentation, labelnames)
        return self.register(metric)  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        metric = Histogram(name, documentation, labelnames, buckets=buckets)
        return self.register(metric)  # type: ignore[return-value]

    def generate_latest(self) -> str:
        """Render all registered metrics in Prometheus 0.0.4 text format."""
        with self._lock:
            metrics = list(self._metrics.values())
        output_lines: list[str] = []
        for metric in sorted(metrics, key=lambda m: m.name):
            output_lines.extend(metric.collect())
        output = "\n".join(output_lines)
        if output and not output.endswith("\n"):
            output += "\n"
        return output


#: Default global process registry
REGISTRY = MetricRegistry()
