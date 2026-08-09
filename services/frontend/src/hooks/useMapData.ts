"use client";

import { useEffect, useState } from "react";

import { getMapLayer, listModels } from "@/lib/api/client";
import type { Model, SpatialLayer } from "@/lib/api/types";
import { useMapConfig } from "@/context/map-config";

export interface UseMapDataResult {
  models: Model[];
  layer: SpatialLayer | null;
  loading: boolean;
  error: string | null;
}

/**
 * Loads the model catalog once and the weather layer metadata whenever the
 * map configuration (model/variable/lead time) changes.
 *
 * The layer is `/v1/maps` metadata only — the backend serves no tile
 * imagery — so a layer being present does not mean tiles render; the map
 * component treats missing tiles as graceful degradation.
 */
export function useMapData(): UseMapDataResult {
  const { model, variable, leadTimeHours } = useMapConfig();
  const [models, setModels] = useState<Model[]>([]);
  const [layer, setLayer] = useState<SpatialLayer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listModels()
      .then((result) => {
        if (!cancelled) setModels(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load models.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getMapLayer({ model, variable, leadTimeHours })
      .then((result) => {
        if (!cancelled) setLayer(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load layer.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model, variable, leadTimeHours]);

  return { models, layer, loading, error };
}
