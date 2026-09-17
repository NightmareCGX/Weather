"use client";

import { useEffect, useState } from "react";

import { getEnsembleStatisticsSeries, RequestAbortedError } from "@/lib/api/client";
import type { EnsembleStatisticsData, SelectedLocation } from "@/lib/api/types";

export type EnsembleStatus = "idle" | "loading" | "success" | "error";

/** Leads requested per call. */
export const ENSEMBLE_LEAD_BATCH_SIZE = 20;

/**
 * Delays before retrying a failed batch (attempt 1 waits 5s, attempt 2 waits 20s,
 * then the failure is final).
 *
 * The API finishes a request server-side even after the client gives up on it and
 * caches the result, so a retry is frequently served from cache almost immediately;
 * these delays exist to space out genuine transient failures rather than to wait for
 * a slow computation.
 */
export const ENSEMBLE_BATCH_RETRY_DELAYS_MS: readonly number[] = [5_000, 20_000];

export interface UseEnsembleOptions {
  /** The selected model, or null for a deterministic model (no member axis). */
  model: string | null;
  /** Leads per request. Overridable for tests. */
  batchSize?: number;
  /** Retry backoff ladder. Overridable for tests. */
  retryDelaysMs?: readonly number[];
}

export interface UseEnsembleResult {
  /** Ensemble statistics keyed by lead time, when at least one lead succeeded. */
  byLead: Map<number, EnsembleStatisticsData>;
  status: EnsembleStatus;
  error: string | null;
  /** The active model. */
  model: string | null;
}

/** Split `items` into consecutive batches of at most `size` entries. */
function chunk<T>(items: readonly T[], size: number): T[][] {
  const batches: T[][] = [];
  const step = Math.max(1, size);
  for (let index = 0; index < items.length; index += step) {
    batches.push(items.slice(index, index + step));
  }
  return batches;
}

/** Resolve after `ms`, or immediately when `signal` aborts. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => {
    const onAbort = () => {
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Fetch `/v1/ensembles` statistics for every lead time of the selected location's
 * point forecast.
 *
 * `/v1/ensembles` answers a batch of leads per request, and this hook splits that
 * batch rather than asking for the whole series at once. A full GEFS point series
 * spans ~75-80 leads whose cold computation reads every ensemble member shard for
 * every lead (~4.4k shard reads for wind, ~15k for 3-hour precipitation), which
 * overruns the edge gateway's response timeout and left the fan chart blank. Batching
 * keeps each request well inside that budget and lets the chart render the leading
 * days while the tail is still loading.
 *
 * ``model`` is the selected model and must be an ensemble model (the dashboard only
 * calls this hook for an ensemble model; deterministic models have no member axis).
 * When ``model`` is null the hook stays idle so the caller can render an ensemble
 * empty state. Individual lead failures are tolerated: the successfully fetched leads
 * are still surfaced, and a fully-failed fan-out surfaces a single aggregated error
 * so the ensemble panel can degrade independently of the core point forecast.
 *
 * A selection change (location, variable, or model) discards the previous selection's
 * statistics before the first new response lands, so the chart can never render one
 * variable's values against another's selection.
 */
export function useEnsemble(
  location: SelectedLocation | null,
  leads: number[],
  variable: string,
  options: UseEnsembleOptions
): UseEnsembleResult {
  const {
    model,
    batchSize = ENSEMBLE_LEAD_BATCH_SIZE,
    retryDelaysMs = ENSEMBLE_BATCH_RETRY_DELAYS_MS,
  } = options;
  const [byLead, setByLead] = useState<Map<number, EnsembleStatisticsData>>(new Map());
  const [status, setStatus] = useState<EnsembleStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const retryKey = retryDelaysMs.join(",");
  const leadsKey = leads.join(",");

  useEffect(() => {
    if (location === null || model === null || leads.length === 0) {
      setByLead(new Map());
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;

    setByLead(new Map());
    setStatus("loading");
    setError(null);

    const fetchBatch = async (batch: number[]): Promise<EnsembleStatisticsData[]> => {
      for (let attempt = 0; ; attempt += 1) {
        try {
          const payload = await getEnsembleStatisticsSeries({
            latitude: location.latitude,
            longitude: location.longitude,
            variable,
            model,
            leads: batch,
            signal: controller.signal,
          });
          return Array.isArray(payload) ? payload : [payload];
        } catch (err) {
          if (err instanceof RequestAbortedError) throw err;
          const delay = retryDelaysMs[attempt];
          if (delay === undefined) throw err;
          await sleep(delay, controller.signal);
          if (!active) throw new RequestAbortedError();
        }
      }
    };

    void (async () => {
      const results = new Map<number, EnsembleStatisticsData>();
      let lastError: unknown = null;

      for (const batch of chunk(leads, batchSize)) {
        if (!active) return;
        try {
          const series = await fetchBatch(batch);
          if (!active) return;
          for (const item of series) {
            if (item && item.lead_time_hours !== undefined) {
              results.set(item.lead_time_hours, item);
            }
          }
          // Publish each batch as it lands so the caller can render partial data
          // (status stays "loading" until the whole series has been attempted).
          if (results.size > 0) {
            setByLead(new Map(results));
          }
        } catch (err) {
          if (!active || err instanceof RequestAbortedError) return;
          lastError = err;
          // With nothing on screen there is no partial result to protect, so stop
          // rather than pile further batches onto a backend that is already failing.
          if (results.size === 0) break;
          // Otherwise keep the partial series and continue with the remaining batches.
        }
      }

      if (!active) return;
      if (results.size > 0) {
        setByLead(new Map(results));
        setError(null);
        setStatus("success");
        return;
      }
      setByLead(new Map());
      setError(
        lastError instanceof Error ? lastError.message : "No ensemble statistics available."
      );
      setStatus("error");
    })();

    return () => {
      active = false;
      controller.abort();
    };
    // `location` is a stable per-selection object; the joined keys stand in for the
    // array/option identities so a fresh array literal on each render cannot re-fire.
  }, [location, variable, model, leadsKey, batchSize, retryKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return { byLead, status, error, model };
}
