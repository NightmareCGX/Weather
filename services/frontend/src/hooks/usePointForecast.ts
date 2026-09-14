"use client";

import { useEffect, useRef, useState } from "react";

import { getPointForecast, RequestAbortedError } from "@/lib/api/client";
import { toPointSpecifier } from "@/lib/forecast/selection";
import type { PointForecast, SelectedLocation } from "@/lib/api/types";

export type FetchStatus = "idle" | "loading" | "success" | "error";

export interface UsePointForecastResult {
  forecast: PointForecast | null;
  status: FetchStatus;
  error: string | null;
}

export interface UsePointForecastOptions {
  model: string | null;
  units?: "metric" | "imperial";
  variables?: string[];
}

/**
 * Fetch the `/v1/points` forecast for the shared selected location and optional variable filter.
 *
 * The request is cancelled when the selection changes, and stale responses are
 * guarded by active tokens, so a slow response for a previous selection can
 * never render over a newer one. Requesting only specific variables reduces backend
 * array slicing, bilinear interpolation, and serialization from O(N_variables * N_leads)
 * down to O(1 * N_leads).
 *
 * An in-memory cache preserves fetched variables per location/model/unit session so
 * navigating back and forth across variables produces instant (0ms) switches.
 */
export function usePointForecast(
  location: SelectedLocation | null,
  options: UsePointForecastOptions
): UsePointForecastResult {
  const { model, units = "metric", variables } = options;
  const [forecast, setForecast] = useState<PointForecast | null>(null);
  const [status, setStatus] = useState<FetchStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const variablesKey = variables && variables.length > 0 ? [...variables].sort().join(",") : "";
  const locationKey = location
    ? `${location.resolvedVia}:${location.id ?? `${location.latitude},${location.longitude}`}`
    : null;
  const scopeKey = locationKey && model ? `${locationKey}:${model}:${units}` : null;

  // Cache fetched forecasts per variable under the active location+model scope
  const cacheRef = useRef<Map<string, PointForecast>>(new Map());
  const activeScopeRef = useRef<string | null>(null);

  // Invalidate cache when location, model, or units change
  if (activeScopeRef.current !== scopeKey) {
    cacheRef.current.clear();
    activeScopeRef.current = scopeKey;
  }

  useEffect(() => {
    if (location === null || model === null) {
      setForecast(null);
      setStatus("idle");
      setError(null);
      return;
    }

    const cached = cacheRef.current.get(variablesKey);
    if (cached) {
      setForecast(cached);
      setStatus("success");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;
    setForecast(null);
    setStatus("loading");
    setError(null);

    const targetVariables = variablesKey ? variablesKey.split(",") : undefined;

    getPointForecast({
      location: toPointSpecifier(location),
      model,
      units,
      variables: targetVariables,
      signal: controller.signal,
    })
      .then((next) => {
        if (!active) return;
        cacheRef.current.set(variablesKey, next);
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
  }, [location, model, units, variablesKey]);

  return { forecast, status, error };
}
