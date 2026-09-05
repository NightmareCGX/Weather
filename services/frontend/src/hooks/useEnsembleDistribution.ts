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
 * Fetch the raw ensemble-member distribution for a single selected valid time or lead.
 *
 * Under Lifecycle V2: accepts either a string ISO 8601 `validTime` or a number `leadTimeHours`.
 * When a `validTime` is passed, the backend dynamically resolves the newest committed source cycle.
 */
export function useEnsembleDistribution(
  location: SelectedLocation | null,
  timeSpecifier: number | string | null,
  variable: string,
  options: { model: string | null }
): UseEnsembleDistributionResult {
  const { model } = options;
  const [data, setData] = useState<EnsembleStatisticsData | null>(null);
  const [status, setStatus] = useState<DistributionStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const isString = typeof timeSpecifier === "string";
  const validTime = isString ? timeSpecifier : undefined;
  const leadTimeHours = typeof timeSpecifier === "number" ? timeSpecifier : undefined;

  useEffect(() => {
    if (location === null || model === null || timeSpecifier === null) {
      setData(null);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;
    setData(null);
    setStatus("loading");
    setError(null);

    getEnsembleStatistics({
      latitude: location.latitude,
      longitude: location.longitude,
      variable,
      model,
      validTime,
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
  }, [location, timeSpecifier, variable, model, validTime, leadTimeHours]);

  return { data, status, error };
}
