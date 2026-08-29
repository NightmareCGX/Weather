"use client";

import { useEffect, useState } from "react";

import { getEnsembleStatistics, RequestAbortedError } from "@/lib/api/client";
import type { EnsembleStatisticsData, SelectedLocation } from "@/lib/api/types";

export type EnsembleStatus = "idle" | "loading" | "success" | "error";

export interface UseEnsembleResult {
  /** Ensemble statistics keyed by lead time, when at least one lead succeeded. */
  byLead: Map<number, EnsembleStatisticsData>;
  status: EnsembleStatus;
  error: string | null;
  /** The active model. */
  model: string | null;
}

/**
 * Fetch `/v1/ensembles` statistics for every lead time of the selected
 * location's point forecast.
 *
 * `/v1/ensembles` answers one `lead_time_hours` per request, so the hook fans
 * out one request per lead in the point forecast. ``model`` is the selected
 * model and must be an ensemble model (the dashboard only calls this hook for
 * an ensemble model; deterministic models have no member axis). When ``model``
 * is null the hook stays idle so the caller can render an ensemble empty state.
 * Individual lead failures are tolerated: the successfully fetched leads are
 * still surfaced, and a fully-failed fan-out surfaces a single aggregated
 * error so the ensemble panel can degrade independently of the core point
 * forecast.
 */
export function useEnsemble(
  location: SelectedLocation | null,
  leads: number[],
  variable: string,
  options: { model: string | null }
): UseEnsembleResult {
  const { model } = options;
  const [byLead, setByLead] = useState<Map<number, EnsembleStatisticsData>>(new Map());
  const [status, setStatus] = useState<EnsembleStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (location === null || model === null || leads.length === 0) {
      setByLead(new Map());
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;
    const results = new Map<number, EnsembleStatisticsData>();
    let settled = 0;
    let firstError: string | null = null;

    setStatus("loading");
    setError(null);

    for (const lead of leads) {
      getEnsembleStatistics({
        latitude: location.latitude,
        longitude: location.longitude,
        variable,
        model,
        leadTimeHours: lead,
        signal: controller.signal,
      })
        .then((data) => {
          if (!active) return;
          results.set(lead, data);
        })
        .catch((err: unknown) => {
          if (!active || err instanceof RequestAbortedError) return;
          if (firstError === null) {
            firstError = err instanceof Error ? err.message : "Failed to load ensemble statistics.";
          }
        })
        .finally(() => {
          settled += 1;
          if (active && settled === leads.length) {
            if (results.size > 0) {
              setByLead(new Map(results));
              setStatus("success");
            } else {
              setByLead(new Map());
              setError(firstError ?? "No ensemble statistics available.");
              setStatus("error");
            }
          }
        });
    }

    return () => {
      active = false;
      controller.abort();
    };
  }, [location, variable, model, leads.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  return { byLead, status, error, model };
}
