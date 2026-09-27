"""Runtime observability primitives for the API serving layer.

Currently provides:

- :mod:`api.monitoring.http_metrics` — process-local HTTP request metrics
  (RED method) with bounded label cardinality, exposed on ``/v1/metrics``.
- :mod:`api.monitoring.logging` — structured JSON logging with request-ID
  correlation (MONITORING.md section 8).
"""
