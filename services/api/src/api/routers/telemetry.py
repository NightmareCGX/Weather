"""Client telemetry ingestion endpoint (API.md section 8.2).

Receives batched browser telemetry (JavaScript errors, unhandled rejections,
Web Vitals) from the frontend beacon and turns it into two observability
signals:

- a bounded-cardinality Prometheus counter (``weather_client_telemetry_events_total``,
  labeled only by event type) exposed on ``/v1/metrics``;
- structured JSON log records carrying the free-form event details (name,
  message, stack, page URL, session ID).

The endpoint is deliberately fail-open and side-effect-free: it never touches
databases or caches, accepts a small bounded payload (capped field lengths,
at most 20 events per batch), and responds ``202`` immediately. Invalid
payloads are rejected by request validation as RFC 7807 ``422`` responses
(API.md section 2.4). This is an open endpoint — in production the edge
gateway is the place to layer rate limiting on top.
"""

import logging

from fastapi import APIRouter, Response

from api.monitoring.client_telemetry_metrics import CLIENT_TELEMETRY_EVENTS_TOTAL
from api.schemas import (
    ClientTelemetryBatch,
    ClientTelemetryEnvelope,
    ClientTelemetryReceipt,
)

router = APIRouter()

logger = logging.getLogger(__name__)

#: Cache policy for telemetry receipts (never cached, mirrors API.md 8.1).
CACHE_CONTROL_TELEMETRY = "no-store"


@router.post(
    "/telemetry/client",
    status_code=202,
    response_model=ClientTelemetryEnvelope,
    summary="Record client telemetry events (JS errors, Web Vitals)",
)
def record_client_telemetry(
    batch: ClientTelemetryBatch, response: Response
) -> ClientTelemetryEnvelope:
    """Accept a batch of client telemetry events.

    Each event increments the bounded ``type`` metric and is logged with its
    free-form details. The response reports the number of accepted events.
    """
    for event in batch.events:
        CLIENT_TELEMETRY_EVENTS_TOTAL.labels(event_type=event.type).inc()
        # Free-form fields go to structured logs only (never to metric
        # labels) so hostile payloads cannot grow the exposition cardinality.
        logger.info(
            "client telemetry event: %s",
            event.name,
            extra={
                "telemetry_type": event.type,
                "telemetry_name": event.name,
                "telemetry_message": event.message or "",
                "telemetry_stack": event.stack or "",
                "telemetry_page_url": event.page_url or "",
                "telemetry_session_id": event.session_id or "",
                "telemetry_value": event.value,
                "telemetry_rating": event.rating or "",
            },
        )
    response.headers["Cache-Control"] = CACHE_CONTROL_TELEMETRY
    return ClientTelemetryEnvelope(data=ClientTelemetryReceipt(accepted=len(batch.events)))
