/**
 * Client telemetry beacon: JavaScript errors and Web Vitals delivered to
 * POST /v1/telemetry/client (API.md section 8.2).
 *
 * The beacon is fail-open by design: every send is wrapped so a failing or
 * missing backend can never break the application, batches are capped to the
 * server-side limits (20 events, bounded field lengths), and the whole
 * feature is disabled with NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED=0.
 */

import { API_PREFIX } from "@/lib/api/client";
import { onCLS, onFCP, onINP, onLCP, onTTFB, type Metric } from "web-vitals";

/** Mirrors the API schema field caps (api.schemas.ClientTelemetryEvent). */
const MAX_BATCH_SIZE = 20;
const MAX_NAME_LENGTH = 64;
const MAX_MESSAGE_LENGTH = 512;
const MAX_STACK_LENGTH = 2048;
const MAX_PAGE_URL_LENGTH = 512;
const MAX_SESSION_ID_LENGTH = 64;

/** Queue drain cadence: fast enough to survive a crash, rare enough to batch. */
const FLUSH_INTERVAL_MS = 5000;

type TelemetryEventType = "error" | "web_vital";

/** Wire shape of one event; matches api.schemas.ClientTelemetryEvent. */
interface TelemetryEvent {
  type: TelemetryEventType;
  name: string;
  timestamp: number;
  message?: string;
  stack?: string;
  page_url?: string;
  session_id?: string;
  value?: number;
  rating?: "good" | "needs-improvement" | "poor";
}

let initialized = false;
let queue: TelemetryEvent[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
/** Registered global listeners, kept for removal in the test reset hook. */
let activeTeardown: Array<() => void> = [];

/** The beacon endpoint (same origin: the gateway/proxy routes /v1/* to the API). */
export const TELEMETRY_ENDPOINT = `${API_PREFIX}/telemetry/client`;

/** Build (once per tab) a pseudo-anonymous session id for event grouping. */
function sessionId(): string | undefined {
  try {
    const existing = window.sessionStorage.getItem("telemetry_session_id");
    if (existing) {
      return existing;
    }
    const generated = `sess_${Math.random().toString(36).slice(2, 12)}`;
    window.sessionStorage.setItem("telemetry_session_id", generated);
    return generated;
  } catch {
    // Storage can be unavailable (private mode, permissions) — grouping is optional.
    return undefined;
  }
}

function truncate(value: string, max: number): string {
  return value.length <= max ? value : value.slice(0, max);
}

/** Whether the beacon is enabled (runtime check so tests can toggle it). */
export function isClientTelemetryEnabled(): boolean {
  return process.env.NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED !== "0";
}

function enqueue(event: TelemetryEvent): void {
  queue.push(event);
  if (queue.length >= MAX_BATCH_SIZE) {
    flush();
    return;
  }
  if (flushTimer === null) {
    flushTimer = setTimeout(() => {
      flushTimer = null;
      flush();
    }, FLUSH_INTERVAL_MS);
  }
}

/**
 * Drain the queue in one POST batch. `useSendBeacon` switches to
 * navigator.sendBeacon for page-exit flushes (the request must survive the
 * document unloading; fetch keepalive already covers ordinary flushes).
 */
export function flush(useSendBeacon = false): void {
  if (queue.length === 0) {
    return;
  }
  const batch = queue.splice(0, MAX_BATCH_SIZE);
  const body = JSON.stringify({ events: batch });
  try {
    if (
      useSendBeacon &&
      typeof navigator !== "undefined" &&
      typeof navigator.sendBeacon === "function"
    ) {
      if (
        navigator.sendBeacon(TELEMETRY_ENDPOINT, new Blob([body], { type: "application/json" }))
      ) {
        return;
      }
    }
    void fetch(TELEMETRY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => undefined);
  } catch {
    // Never surface telemetry failures to the application.
  }
}

function reportErrorEvent(message: string, error: unknown): void {
  const stack = error instanceof Error && error.stack ? error.stack : "";
  const name = error instanceof Error ? error.name : "Error";
  enqueue({
    type: "error",
    name: truncate(name, MAX_NAME_LENGTH),
    timestamp: Date.now(),
    message: truncate(message || String(error ?? "unknown error"), MAX_MESSAGE_LENGTH),
    stack: stack ? truncate(stack, MAX_STACK_LENGTH) : undefined,
    page_url: truncate(window.location.href, MAX_PAGE_URL_LENGTH),
    session_id: sessionId(),
  });
}

function reportWebVital(metric: Metric): void {
  enqueue({
    type: "web_vital",
    name: truncate(metric.name, MAX_NAME_LENGTH),
    timestamp: Date.now(),
    value: metric.value,
    rating: metric.rating,
    page_url: truncate(window.location.href, MAX_PAGE_URL_LENGTH),
    session_id: sessionId(),
  });
}

/**
 * Install the global error listeners and Web Vitals observers. Idempotent:
 * repeated calls (e.g. Strict Mode double-mount) register nothing extra.
 */
export function initClientTelemetry(): void {
  if (initialized || !isClientTelemetryEnabled() || typeof window === "undefined") {
    return;
  }
  initialized = true;

  const onError = (event: ErrorEvent): void => {
    reportErrorEvent(event.message, event.error);
  };
  const onRejection = (event: Event): void => {
    const reason = (event as PromiseRejectionEvent).reason;
    reportErrorEvent(reason instanceof Error ? reason.message : String(reason), reason);
  };
  const onPageHide = (): void => flush(true);
  const onVisibility = (): void => {
    if (document.visibilityState === "hidden") {
      flush(true);
    }
  };
  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onRejection);
  window.addEventListener("pagehide", onPageHide);
  document.addEventListener("visibilitychange", onVisibility);
  activeTeardown = [
    () => window.removeEventListener("error", onError),
    () => window.removeEventListener("unhandledrejection", onRejection),
    () => window.removeEventListener("pagehide", onPageHide),
    () => document.removeEventListener("visibilitychange", onVisibility),
  ];

  onCLS(reportWebVital);
  onFCP(reportWebVital);
  onINP(reportWebVital);
  onLCP(reportWebVital);
  onTTFB(reportWebVital);
}

/** Test hook: reset module state between tests (removes global listeners). */
export function resetClientTelemetryForTests(): void {
  for (const remove of activeTeardown) {
    remove();
  }
  activeTeardown = [];
  initialized = false;
  queue = [];
  if (flushTimer !== null) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
}

/** Test hook: current queued event count. */
export function queuedEventCountForTests(): number {
  return queue.length;
}
