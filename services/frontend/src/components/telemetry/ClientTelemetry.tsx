"use client";

import { useEffect } from "react";

import { initClientTelemetry } from "@/lib/telemetry/telemetry";

/**
 * Mounts the client telemetry beacon (JS errors, unhandled rejections, Web
 * Vitals -> POST /v1/telemetry/client). Renders nothing; safe to place once
 * in the root layout. Disabled entirely with
 * NEXT_PUBLIC_CLIENT_TELEMETRY_ENABLED=0 (MONITORING.md section 8.3).
 */
export default function ClientTelemetry() {
  useEffect(() => {
    initClientTelemetry();
  }, []);
  return null;
}
