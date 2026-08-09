"use client";

import { useEffect, useRef } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";

import type { SpatialLayer } from "@/lib/api/types";
import { buildBaseStyle } from "@/lib/map/baseStyle";
import { applyWeatherLayer } from "@/lib/map/layers";

interface WeatherMapProps {
  /** `/v1/maps` metadata for the weather layer, or null while loading/erroring. */
  layer: SpatialLayer | null;
}

/**
 * MapLibre GL JS map wrapper.
 *
 * The map is created once on mount and destroyed on unmount; the create
 * effect is idempotent under React 18 strict mode (mount -> cleanup ->
 * mount). The weather layer is configured from `/v1/maps` metadata whenever
 * that metadata changes. Because `/v1/maps` is metadata-only, the backend
 * serves no tile imagery: missing tiles 404 and MapLibre keeps rendering the
 * base map — accepted graceful degradation.
 */
export function WeatherMap({ layer }: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const layerRef = useRef<SpatialLayer | null>(layer);

  useEffect(() => {
    if (mapRef.current !== null || containerRef.current === null) {
      return;
    }

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: buildBaseStyle(),
      center: [-106.8, 39.2],
      zoom: 5,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.on("load", () => {
      if (layerRef.current !== null) {
        applyWeatherLayer(map, layerRef.current);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    layerRef.current = layer;
    const map = mapRef.current;
    if (map !== null && layer !== null && map.loaded()) {
      applyWeatherLayer(map, layer);
    }
  }, [layer]);

  return <div ref={containerRef} className="h-full w-full" data-testid="weather-map" />;
}
