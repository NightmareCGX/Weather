"use client";

import { useEffect, useState } from "react";

import { getMapLayer, RequestAbortedError } from "@/lib/api/client";
import type { SpatialLayer } from "@/lib/api/types";
import { useForecastSelection } from "@/context/forecast-selection";

export interface UseMapLayerResult {
  layer: SpatialLayer | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetches `/v1/maps` metadata whenever the forecast selection (model /
 * variable / initial time / lead time) changes.
 *
 * The layer is only requested when a full, valid selection exists. A stale
 * response from a previous selection is cancelled and ignored, so changing the
 * model/variable/initial time/lead time always loads the metadata (and thus
 * the tile layer) for the *current* selection. `/v1/maps` now validates the
 * selection against the catalog, so a 404 means "no forecast for this
 * selection" and is surfaced as a user-facing message rather than a raw
 * technical error.
 */
export function useMapLayer(): UseMapLayerResult {
  const { selection } = useForecastSelection();
  const [layer, setLayer] = useState<SpatialLayer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selection === null) {
      setLayer(null);
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setError(null);

    getMapLayer({
      model: selection.model,
      variable: selection.variable,
      leadTimeHours: selection.leadTimeHours,
      initialTime: selection.initialTime,
      signal: controller.signal,
    })
      .then((next) => {
        setLayer(next);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (err instanceof RequestAbortedError) return;
        setLayer(null);
        setError(err instanceof Error ? err.message : "Unable to load forecast data.");
        setLoading(false);
      });

    return () => controller.abort();
  }, [selection]);

  return { layer, loading, error };
}
