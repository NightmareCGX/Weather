"use client";

import { useEffect, useRef } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";

import type { SelectedLocation, SpatialLayer } from "@/lib/api/types";
import { buildBaseStyle } from "@/lib/map/baseStyle";
import { applyWeatherLayer, removeWeatherLayer } from "@/lib/map/layers";
import { coordinatesToSelectedLocation } from "@/lib/forecast/selection";

interface WeatherMapProps {
  /** `/v1/maps` metadata for the weather layer, or null while loading/erroring. */
  layer: SpatialLayer | null;
  /** The shared selected location; a marker is shown at its coordinates. */
  selectedLocation: SelectedLocation | null;
  /** The valid time of the current selection; drives layer keying. */
  validTime: string | null;
  /** Fired with a coordinate location when the user clicks the map. */
  onSelect: (location: SelectedLocation) => void;
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
 *
 * Milestone 13 adds point selection: clicking the map fires `onSelect` with
 * the clicked coordinates, and the shared {@link selectedLocation} drives a
 * marker. The map component stays presentation-only (props + callbacks).
 */
export function WeatherMap({ layer, selectedLocation, validTime, onSelect }: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const isStyleReadyRef = useRef<boolean>(false);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const layerRef = useRef<SpatialLayer | null>(layer);
  const appliedLayerRef = useRef<SpatialLayer | null>(null);
  // Keep the click callback fresh without recreating the map (the create
  // effect must run once with an empty dependency array).
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

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
      // MapLibre can deliver a click event before the style/canvas has
      // resolved coordinates (e.g. while raster tiles are still loading).
      // Guard so a coordinate-less event never crashes the map or trips the
      // Next.js error boundary.
      const lngLat = event.lngLat;
      if (lngLat === undefined) {
        return;
      }
      onSelectRef.current(coordinatesToSelectedLocation(lngLat.lat, lngLat.lng));
    };
    // Track base style readiness: once the base map style has loaded, forecast
    // raster tile requests must never block or delay subsequent layer transitions.
    const handleLoad = () => {
      if (isStyleReadyRef.current) {
        return;
      }
      isStyleReadyRef.current = true;
      map.on("click", handleClick);
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
      if (markerRef.current !== null) {
        markerRef.current.remove();
        markerRef.current = null;
      }
      map.remove();
      mapRef.current = null;
    };
  }, []);

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
      // Set the marker's coordinates before adding it to the map: maplibre's
      // `Marker.addTo` reads `this._lngLat` during `_update`, so adding an
      // un-positioned marker throws (reading `.lng` of undefined).
      markerRef.current = new maplibregl.Marker({ color: "#1d4ed8" })
        .setLngLat([selectedLocation.longitude, selectedLocation.latitude])
        .addTo(map);
    } else {
      markerRef.current.setLngLat([selectedLocation.longitude, selectedLocation.latitude]);
    }
  }, [selectedLocation]);

  return <div ref={containerRef} className="h-full w-full" data-testid="weather-map" />;
}
