"use client";

import { useEffect, useState } from "react";

import { getEnsembleStatistics, RequestAbortedError } from "@/lib/api/client";
import type { EnsembleStatisticsData, SelectedLocation } from "@/lib/api/types";

export type DistributionStatus = "idle" | "loading" | "success" | "error";

export interface UseEnsembleDistributionResult {
  data: EnsembleStatisticsData | null;
  status: DistributionStatus;
  error: string | null;
}

/**
 * Fetch the raw ensemble-member distribution for a single selected lead.
 *
 * This is the focused request behind the Ensemble Distribution View: it
 * requests `/v1/ensembles` with `include_members=true` for exactly one
 * location / variable / model / lead time and receives the genuine raw member
 * values. It is deliberately separate from {@link useEnsemble}, which fans out
 * statistics-only requests (no `include_members`) across leads for the
 * percentile timeline. A stale selection is cancelled and ignored so a slow
 * response for a previous lead can never render over a newer one.
 */
export function useEnsembleDistribution(
  location: SelectedLocation | null,
  leadTimeHours: number,
  variable: string,
  options: { model?: string } = {}
): UseEnsembleDistributionResult {
  const { model = "gefs" } = options;
  const [data, setData] = useState<EnsembleStatisticsData | null>(null);
  const [status, setStatus] = useState<DistributionStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (location === null) {
      setData(null);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;
    setStatus("loading");
    setError(null);

    getEnsembleStatistics({
      latitude: location.latitude,
      longitude: location.longitude,
      variable,
      model,
      leadTimeHours,
      includeMembers: true,
      signal: controller.signal,
    })
      .then((next) => {
        if (!active) return;
        setData(next);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (!active || err instanceof RequestAbortedError) return;
        setData(null);
        setError(err instanceof Error ? err.message : "Failed to load the ensemble distribution.");
        setStatus("error");
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [location, leadTimeHours, variable, model]);

  return { data, status, error };
}
