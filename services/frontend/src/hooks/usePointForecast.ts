"use client";

import { useEffect, useState } from "react";

import { getPointForecast, RequestAbortedError } from "@/lib/api/client";
import { toPointSpecifier } from "@/lib/forecast/selection";
import type { PointForecast, SelectedLocation } from "@/lib/api/types";

export type FetchStatus = "idle" | "loading" | "success" | "error";

export interface UsePointForecastResult {
  forecast: PointForecast | null;
  status: FetchStatus;
  error: string | null;
}

/**
 * Fetch the `/v1/points` forecast for the shared selected location.
 *
 * The request is cancelled when the selection changes, and stale responses are
 * guarded by a sequence token, so a slow response for a previous selection can
 * never render over a newer one. The deterministic model (default `gfs`) is
 * pinned: `/v1/points` serves a single model and rejects ensemble models
 * (API.md section 2.1).
 */
export function usePointForecast(
  location: SelectedLocation | null,
  options: { model?: string; units?: "metric" | "imperial" } = {}
): UsePointForecastResult {
  const { model = "gfs", units = "metric" } = options;
  const [forecast, setForecast] = useState<PointForecast | null>(null);
  const [status, setStatus] = useState<FetchStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (location === null) {
      setForecast(null);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;
    setStatus("loading");
    setError(null);

    getPointForecast({
      location: toPointSpecifier(location),
      model,
      units,
      signal: controller.signal,
    })
      .then((next) => {
        if (!active) return;
        setForecast(next);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (!active || err instanceof RequestAbortedError) return;
        setForecast(null);
        setError(err instanceof Error ? err.message : "Failed to load the forecast.");
        setStatus("error");
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [location, model, units]);

  return { forecast, status, error };
}
