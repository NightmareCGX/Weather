"use client";

import { useEffect, useRef } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";

import type { SelectedLocation, SpatialLayer } from "@/lib/api/types";
import { buildBaseStyle } from "@/lib/map/baseStyle";
import { applyWeatherLayer } from "@/lib/map/layers";
import { coordinatesToSelectedLocation } from "@/lib/forecast/selection";

interface WeatherMapProps {
  /** `/v1/maps` metadata for the weather layer, or null while loading/erroring. */
  layer: SpatialLayer | null;
  /** The shared selected location; a marker is shown at its coordinates. */
  selectedLocation: SelectedLocation | null;
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
export function WeatherMap({ layer, selectedLocation, onSelect }: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const layerRef = useRef<SpatialLayer | null>(layer);
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
    // Only enable point selection after the style has loaded so the map has
    // resolved geographic coordinates for click events (avoids racing
    // maplibre's own coordinate resolution during tile load).
    map.on("load", () => {
      map.on("click", handleClick);
      if (layerRef.current !== null) {
        applyWeatherLayer(map, layerRef.current);
      }
    });

    return () => {
      map.off("click", handleClick);
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
    if (map !== null && layer !== null && map.loaded()) {
      applyWeatherLayer(map, layer);
    }
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
