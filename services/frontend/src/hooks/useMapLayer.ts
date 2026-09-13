"use client";

import { useMemo, useRef } from "react";

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
 *
 * Identity is stabilized by a semantic fingerprint of the layer contents
 * (tile URL, valid time, zoom bounds, legend, vector field URL, and the
 * serving source cycle). The 60-second availability heartbeat refreshes the
 * availability object without changing any of those values, so the heartbeat
 * alone never re-identifies the layer and never re-applies the map source.
 * The fingerprint DOES change when a newer initial time (cycle) begins
 * serving the same valid time (via `source_cycle`), which is exactly when
 * WeatherMap should re-apply the source and refetch tiles.
 */
function layerFingerprint(layer: SpatialLayer): string {
  return JSON.stringify([
    layer.tile_url_template,
    layer.valid_time ?? null,
    layer.source_cycle ?? null,
    layer.lead_time_hours ?? null,
    layer.min_zoom,
    layer.max_zoom,
    layer.vector_field_url_template ?? null,
    layer.legend,
  ]);
}

export function useMapLayer(): UseMapLayerResult {
  const { availability, selection, status, error: availabilityError } = useForecastSelection();

  const resolved = useMemo(
    () => resolveSpatialLayer(availability, selection),
    [availability, selection]
  );

  // Single-entry memoization: reuse the previous layer object while the
  // semantic fingerprint is unchanged so downstream identity comparisons
  // (WeatherMap applied-layer check) stay stable across availability refreshes.
  const resolvedRef = useRef<SpatialLayer | null>(null);
  resolvedRef.current = resolved;
  const stableRef = useRef<{ fingerprint: string; layer: SpatialLayer | null }>({
    fingerprint: "",
    layer: null,
  });
  const fingerprint = resolved === null ? "" : layerFingerprint(resolved);

  const layer = useMemo(() => {
    if (stableRef.current.layer !== null && stableRef.current.fingerprint === fingerprint) {
      // Same semantic content, new object identity (e.g. heartbeat refresh):
      // keep the previously applied layer object.
      return stableRef.current.layer;
    }
    stableRef.current = { fingerprint, layer: resolvedRef.current };
    return resolvedRef.current;
  }, [fingerprint]);

  const loading = status === "loading";
  const error = availabilityError;

  return { layer, loading, error };
}
