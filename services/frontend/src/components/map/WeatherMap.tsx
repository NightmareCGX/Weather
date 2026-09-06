"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";

import type { SelectedLocation, SpatialLayer } from "@/lib/api/types";
import { buildBaseStyle } from "@/lib/map/baseStyle";
import { applyWeatherLayer, removeWeatherLayer } from "@/lib/map/layers";
import { coordinatesToSelectedLocation } from "@/lib/forecast/selection";
import { WindParticleAnimation } from "@/lib/map/windParticles";
import { useVectorField } from "@/hooks/useVectorField";

interface WeatherMapProps {
  /** `/v1/maps` metadata for the weather layer, or null while loading/erroring. */
  layer: SpatialLayer | null;
  /** The shared selected location; a marker is shown at its coordinates. */
  selectedLocation: SelectedLocation | null;
  /** The valid time of the current selection; drives layer keying. */
  validTime: string | null;
  /** Available forecast lead hours for prefetch (optional). */
  availableLeads?: number[];
  /** Fired with a coordinate location when the user clicks the map. */
  onSelect: (location: SelectedLocation) => void;
}

/**
 * MapLibre GL JS map wrapper with progressive wind particle animation overlay.
 *
 * The map is created once on mount and destroyed on unmount; the create
 * effect is idempotent under React 18 strict mode (mount -> cleanup ->
 * mount). The weather layer is configured from `/v1/maps` metadata whenever
 * that metadata changes.
 *
 * For wind layers (Phase 1B.3):
 * - Stage A: Immediately renders the scalar wind speed raster.
 * - Stage B: In parallel, requests the quantized Int16 vector field and starts
 *   smooth Canvas 2D particle animation overlay once vector data arrives.
 */
export function WeatherMap({
  layer,
  selectedLocation,
  validTime,
  availableLeads,
  onSelect,
}: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const isStyleReadyRef = useRef<boolean>(false);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const animRef = useRef<WindParticleAnimation | null>(null);
  const layerRef = useRef<SpatialLayer | null>(layer);
  const appliedLayerRef = useRef<SpatialLayer | null>(null);
  // Keep the click callback fresh without recreating the map (the create
  // effect must run once with an empty dependency array).
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Progressive vector field fetching and prefetching in parallel
  const { field } = useVectorField({
    vectorFieldUrl: layer?.vector_field_url_template ?? null,
    availableLeads,
    currentLead: layer?.lead_time_hours ?? undefined,
    enabled: layer !== null && Boolean(layer.vector_field_url_template),
  });

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

    const handleClick = (event: maplibregl.MapMouseEvent) => {
      const lngLat = event.lngLat;
      if (lngLat === undefined) {
        return;
      }
      onSelectRef.current(coordinatesToSelectedLocation(lngLat.lat, lngLat.lng));
    };

    const handleLoad = () => {
      if (isStyleReadyRef.current) {
        return;
      }
      isStyleReadyRef.current = true;
      map.on("click", handleClick);

      if (canvasRef.current !== null && animRef.current === null) {
        animRef.current = new WindParticleAnimation(canvasRef.current, map);
      }

      if (layerRef.current !== null) {
        appliedLayerRef.current = layerRef.current;
        applyWeatherLayer(map, layerRef.current);
      }
    };

    if (map.isStyleLoaded()) {
      handleLoad();
    } else {
      map.on("load", handleLoad);
    }

    return () => {
      isStyleReadyRef.current = false;
      appliedLayerRef.current = null;
      map.off("click", handleClick);
      map.off("load", handleLoad);
      if (animRef.current !== null) {
        animRef.current.destroy();
        animRef.current = null;
      }
      if (markerRef.current !== null) {
        markerRef.current.remove();
        markerRef.current = null;
      }
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update scalar raster layer
  useEffect(() => {
    layerRef.current = layer;
    const map = mapRef.current;
    if (map === null || !isStyleReadyRef.current) {
      return;
    }
    if (appliedLayerRef.current === layer) {
      return;
    }
    appliedLayerRef.current = layer;
    if (layer === null) {
      removeWeatherLayer(map);
      return;
    }
    applyWeatherLayer(map, layer);
  }, [layer]);

  // Synchronize vector field with particle animation engine
  useEffect(() => {
    if (animRef.current === null) {
      return;
    }
    if (layer === null || !layer.vector_field_url_template) {
      animRef.current.setField(null);
      return;
    }
    animRef.current.setField(field);
  }, [field, layer]);

  // Keep the selection marker in sync with the shared selected location.
  useEffect(() => {
    const map = mapRef.current;
    if (map === null) {
      return;
    }
    if (selectedLocation === null) {
      if (markerRef.current !== null) {
        markerRef.current.remove();
        markerRef.current = null;
      }
      return;
    }
    if (markerRef.current === null) {
      markerRef.current = new maplibregl.Marker({ color: "#1d4ed8" })
        .setLngLat([selectedLocation.longitude, selectedLocation.latitude])
        .addTo(map);
    } else {
      markerRef.current.setLngLat([selectedLocation.longitude, selectedLocation.latitude]);
    }
  }, [selectedLocation]);

  return (
    <div ref={containerRef} className="relative h-full w-full" data-testid="weather-map">
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
        data-testid="wind-particle-canvas"
      />
    </div>
  );
}
