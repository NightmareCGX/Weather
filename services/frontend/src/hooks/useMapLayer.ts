"use client";

import { useMemo } from "react";

import type { SpatialLayer } from "@/lib/api/types";
import { useForecastSelection } from "@/context/forecast-selection";
import { resolveSpatialLayer } from "@/lib/forecast/availability";

export interface UseMapLayerResult {
  layer: SpatialLayer | null;
  loading: boolean;
  error: string | null;
}

/**
 * Synchronously derives the authoritative MapLibre raster layer from the cached
 * forecast availability state whenever the forecast selection changes.
 *
 * By resolving the authoritative SpatialLayer directly from backend-supplied
 * availability metadata (tile URL template pattern, zoom bounds, and legend stops),
 * selector transitions (model, variable, initial time, lead time) are instantaneous
 * and eliminate the sequential `/v1/maps` metadata network roundtrip.
 *
 * As soon as selection B becomes authoritative in state, layer B becomes
 * authoritative immediately, allowing WeatherMap to replace the MapLibre
 * source without retaining stale layer A during an asynchronous transition.
 */
export function useMapLayer(): UseMapLayerResult {
  const { availability, selection, status, error: availabilityError } = useForecastSelection();

  const layer = useMemo(
    () => resolveSpatialLayer(availability, selection),
    [availability, selection]
  );

  const loading = status === "loading";
  const error = availabilityError;

  return { layer, loading, error };
}
