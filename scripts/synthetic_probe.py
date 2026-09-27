#!/usr/bin/env python3
"""Synthetic end-to-end probe for the weather platform serving tier.

Drives real HTTP requests against the deployed serving surface on a schedule
(cron, systemd timer, or ``--interval`` loop) so the "every dependency is
healthy but the product is broken" failure class — bad gateway config, stale
TLS certificates, misrouted upstreams — is caught from the outside in,
independently of the internal metrics pipeline (MONITORING.md section 9).

Per URL the probe reports:
- HTTP reachability (2xx/3xx = pass, 4xx = warning, 5xx/network = critical);
- response latency (warning above ``--max-latency-seconds``);
- optional JSON body sanity (``--expect-json``);
- TLS certificate expiry for https targets (warning inside
  ``--tls-expiry-warn-days``, critical when already expired).

Failures are optionally dispatched to the platform alert webhook in the same
JSON payload shape the ingestion AlertEngine uses, so operators get one
uniform alert stream (MONITORING.md section 6).

Usage (see ``--help`` for every switch)::

    # Cron-style single pass against local services (exit 0/1/2):
    uv run --no-sync python scripts/synthetic_probe.py \
        --url http://127.0.0.1:8000/v1/health \
        --url http://127.0.0.1:8000/v1/forecast/availability --expect-json

    # Against the TLS gateway every 60s, alerting a webhook:
    uv run --no-sync python scripts/synthetic_probe.py \
        --url https://weather.example.com/v1/health --insecure \
        --interval 60 --webhook https://alerts.example.com/webhook

Exit codes mirror ``weather-ingest alert-check``: ``0`` all probes passed,
``1`` at least one warning (slow response, 4xx, certificate expiring),
``2`` at least one critical failure (probe failed, certificate expired).
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

#: Default latency budget per probe (seconds) before a WARNING is raised.
DEFAULT_MAX_LATENCY_SECONDS = 5.0

#: Default number of days before TLS certificate expiry to raise a WARNING.
DEFAULT_TLS_EXPIRY_WARN_DAYS = 14

#: Per-request socket timeout (seconds).
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one synthetic probe against one URL."""

    url: str
    severity: str  # "ok", "warning", "critical"
    latency_seconds: float
    status_code: int | None
    detail: str

    @property
    def exit_rank(self) -> int:
        """Sort rank matching the exit-code contract: ok=0, warning=1, critical=2."""
        return {"ok": 0, "warning": 1, "critical": 2}[self.severity]


def _http_probe(url: str, timeout: float, insecure: bool) -> ProbeResult:
    """Perform one GET probe and classify the outcome."""
    started = time.monotonic()
    status_code: int | None = None
    try:
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": "weather-synthetic-probe/1.0"})
        context = None
        if url.startswith("https") and insecure:
            context = ssl._create_unverified_context()  # noqa: SLF001 - opt-in for self-signed gateways
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            status_code = response.status
            response.read()
        latency = time.monotonic() - started
        if 200 <= status_code < 400:
            return ProbeResult(url, "ok", latency, status_code, "reachable")
        return ProbeResult(url, "warning", latency, status_code, f"unexpected HTTP {status_code}")
    except urllib.error.HTTPError as exc:
        latency = time.monotonic() - started
        status_code = exc.code
        if 400 <= status_code < 500:
            return ProbeResult(url, "warning", latency, status_code, f"client error HTTP {status_code}")
        return ProbeResult(url, "critical", latency, status_code, f"server error HTTP {status_code}")
    except Exception as exc:  # noqa: BLE001 - every probe failure is a result, never a crash
        latency = time.monotonic() - started
        return ProbeResult(url, "critical", latency, None, f"request failed: {exc}")


def _expect_json_check(result: ProbeResult, timeout: float, insecure: bool) -> ProbeResult:
    """Downgrade a passing probe whose body is not JSON (``--expect-json``)."""
    if result.severity != "ok":
        return result
    try:
        request = urllib.request.Request(result.url, headers={"User-Agent": "weather-synthetic-probe/1.0"})
        context = None
        if result.url.startswith("https") and insecure:
            context = ssl._create_unverified_context()  # noqa: SLF001
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - classification, never a crash
        return ProbeResult(
            result.url, "warning", result.latency_seconds, result.status_code, f"body is not valid JSON: {exc}"
        )
    return result


def _tls_expiry_days(url: str) -> tuple[int | None, str]:
    """Return (days_until_expiry, detail) for the https target's certificate.

    Uses the stdlib ``ssl`` stack only: fetches the peer certificate chain and
    decodes ``notAfter`` via ``ssl.cert_time_to_seconds``. Returns
    ``(None, detail)`` when the certificate cannot be inspected.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return None, "not an https target"
    host = parsed.hostname
    if not host:
        return None, "URL has no hostname"
    port = parsed.port or 443
    try:
        pem = ssl.get_server_certificate((host, port))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as handle:
            handle.write(pem)
            path = handle.name
        # ``_test_decode_cert`` is the stdlib's PEM-to-dict decoder (a private
        # helper, but the only dependency-free option on the standard ssl stack).
        cert = ssl._ssl._test_decode_cert(path)  # type: ignore[attr-defined]  # noqa: SLF001
        not_after = parsedate_to_datetime(cert["notAfter"])
        remaining_days = int((not_after.timestamp() - time.time()) / 86400)
        return remaining_days, f"certificate expires in {remaining_days} days"
    except Exception as exc:  # noqa: BLE001 - classification, never a crash
        return None, f"certificate inspection failed: {exc}"


def _classify_tls(result: ProbeResult, warn_days: int) -> ProbeResult:
    """Attach TLS expiry severity to a probe result for an https target."""
    remaining, detail = _tls_expiry_days(result.url)
    if remaining is None:
        if result.severity == "ok":
            return ProbeResult(result.url, "warning", result.latency_seconds, result.status_code, detail)
        return result
    if remaining < 0:
        return ProbeResult(result.url, "critical", result.latency_seconds, result.status_code, f"{detail} (EXPIRED)")
    if remaining <= warn_days:
        return ProbeResult(
            result.url,
            "warning" if result.severity == "ok" else result.severity,
            result.latency_seconds,
            result.status_code,
            f"{detail} (warn threshold {warn_days} days)",
        )
    return result


def probe_url(url: str, *, timeout: float, insecure: bool, expect_json: bool, max_latency: float, tls_warn_days: int) -> ProbeResult:
    """Run the full probe chain for one URL and return the worst outcome."""
    result = _http_probe(url, timeout, insecure)
    if result.severity == "ok" and result.latency_seconds > max_latency:
        result = ProbeResult(
            url, "warning", result.latency_seconds, result.status_code,
            f"latency {result.latency_seconds:.2f}s exceeds budget {max_latency:.2f}s",
        )
    if expect_json:
        result = _expect_json_check(result, timeout, insecure)
    if url.startswith("https"):
        result = _classify_tls(result, tls_warn_days)
    return result


def dispatch_webhook(webhook_url: str, results: list[ProbeResult]) -> None:
    """POST critical/warning probe outcomes to the platform alert webhook.

    Payload shape matches the ingestion AlertEngine webhook contract
    (MONITORING.md section 6). Fail-open: dispatch errors are logged, never
    raised — the probe's exit code must stay deterministic.
    """
    if not webhook_url:
        return
    import urllib.request as request_module

    for result in results:
        if result.severity == "ok":
            continue
        payload = {
            "event_type": "triggered",
            "name": "synthetic_probe_failed" if result.severity == "critical" else "synthetic_probe_degraded",
            "severity": "CRITICAL" if result.severity == "critical" else "WARNING",
            "summary": f"Synthetic probe {result.severity}: {result.url}",
            "description": f"{result.detail} (latency {result.latency_seconds:.2f}s)",
            "scope": "synthetic-probe",
            "value": result.latency_seconds,
            "threshold": None,
            "timestamp": time.time(),
            "runbook": "#synthetic-probe-failure",
        }
        try:
            request = request_module.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request_module.urlopen(request, timeout=5.0) as response:
                response.read()
        except Exception as exc:  # noqa: BLE001 - fail-open delivery
            print(json.dumps({"level": "warning", "message": f"webhook dispatch failed: {exc}"}), file=sys.stderr)


def run_pass(urls: list[str], args: argparse.Namespace) -> int:
    """Run one probe pass and return the process exit code (0/1/2)."""
    results = [
        probe_url(
            url,
            timeout=args.timeout,
            insecure=args.insecure,
            expect_json=args.expect_json,
            max_latency=args.max_latency_seconds,
            tls_warn_days=args.tls_expiry_warn_days,
        )
        for url in urls
    ]
    for result in results:
        print(
            json.dumps(
                {
                    "level": result.severity,
                    "url": result.url,
                    "status_code": result.status_code,
                    "latency_seconds": round(result.latency_seconds, 4),
                    "detail": result.detail,
                }
            )
        )
    dispatch_webhook(args.webhook, results)
    worst = max((result.exit_rank for result in results), default=0)
    return worst


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Target URL to probe; repeatable (e.g. the /v1/health and /v1/forecast/availability endpoints).",
    )
    parser.add_argument("--expect-json", action="store_true", help="Require a parseable JSON body from every target.")
    parser.add_argument("--max-latency-seconds", type=float, default=DEFAULT_MAX_LATENCY_SECONDS, help="Latency budget before a WARNING (default: %(default)s).")
    parser.add_argument("--tls-expiry-warn-days", type=int, default=DEFAULT_TLS_EXPIRY_WARN_DAYS, help="Days before TLS expiry to WARN (default: %(default)s).")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification (self-signed gateway certificates).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds (default: %(default)s).")
    parser.add_argument("--webhook", default="", help="Optional alert webhook URL; failures are POSTed in the platform alert payload shape.")
    parser.add_argument("--interval", type=float, default=0.0, help="Loop cadence in seconds; 0 runs a single pass (cron mode).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if not args.url:
        print("error: at least one --url is required", file=sys.stderr)
        return 2
    if args.interval > 0:
        while True:
            run_pass(args.url, args)
            time.sleep(args.interval)
    return run_pass(args.url, args)


if __name__ == "__main__":
    sys.exit(main())
