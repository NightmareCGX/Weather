"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";

import type { ApproximateStartupLocation, SelectedLocation, SpatialLayer } from "@/lib/api/types";
import { buildBaseStyle } from "@/lib/map/baseStyle";
import { applyWeatherLayer, removeWeatherLayer } from "@/lib/map/layers";
import { canonicalizeLongitude, coordinatesToSelectedLocation } from "@/lib/forecast/selection";
import { LocateMeButton } from "@/components/map/LocateMeButton";
import { WindParticleAnimation } from "@/lib/map/windParticles";
import { useVectorField } from "@/hooks/useVectorField";

interface WeatherMapProps {
  /** `/v1/maps` metadata for the weather layer, or null while loading/erroring. */
  layer: SpatialLayer | null;
  /** The shared selected location; a marker is shown at its coordinates. */
  selectedLocation: SelectedLocation | null;
  /** Ambient coarse startup IP location context (null if unavailable or still resolving). */
  approximateLocation?: ApproximateStartupLocation | null;
  /** The valid time of the current selection; drives layer keying. */
  validTime: string | null;
  /** Available forecast lead hours for prefetch (optional). */
  availableLeads?: number[];
  /** Fired with a coordinate location when the user clicks the map. */
  onSelect: (location: SelectedLocation) => void;
  /** Optional callback fired when the map finishes moving (moveend) with canonical center coordinates. */
  onCenterChange?: (center: { latitude: number; longitude: number }) => void;
  /** Optional callback to trigger browser geolocation ("Locate Me"). */
  onLocate?: () => void;
  /** Whether browser geolocation is currently in-flight. */
  isLocating?: boolean;
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
  approximateLocation,
  validTime,
  availableLeads,
  onSelect,
  onCenterChange,
  onLocate,
  isLocating,
}: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const isStyleReadyRef = useRef<boolean>(false);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const animRef = useRef<WindParticleAnimation | null>(null);
  const layerRef = useRef<SpatialLayer | null>(layer);
  const appliedLayerRef = useRef<SpatialLayer | null>(null);
  const isMapClickSelectionRef = useRef<boolean>(false);
  const startupAutoCenterEligibleRef = useRef<boolean>(true);

  // If a location is selected or Locate Me is active, permanently disqualify startup auto-centering
  if (selectedLocation !== null || isLocating) {
    startupAutoCenterEligibleRef.current = false;
  }

  // Keep callbacks fresh without recreating the map
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const onCenterChangeRef = useRef(onCenterChange);
  onCenterChangeRef.current = onCenterChange;

  const handleLocate = useCallback(() => {
    startupAutoCenterEligibleRef.current = false;
    onLocate?.();
  }, [onLocate]);

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
    if (typeof window !== "undefined") {
      (window as unknown as { __weatherMap?: MapLibreMap }).__weatherMap = map;
    }
    map.addControl(new maplibregl.NavigationControl(), "top-right");

    const container = containerRef.current;

    const handleUserGesture = (event: Event) => {
      if (event.isTrusted) {
        startupAutoCenterEligibleRef.current = false;
      }
    };

    container.addEventListener("pointerdown", handleUserGesture);
    container.addEventListener("wheel", handleUserGesture, { passive: true });
    container.addEventListener("touchstart", handleUserGesture, { passive: true });

    const handleClick = (event: maplibregl.MapMouseEvent) => {
      startupAutoCenterEligibleRef.current = false;
      const lngLat = event.lngLat;
      if (lngLat === undefined) {
        return;
      }
      isMapClickSelectionRef.current = true;
      const wrapped = typeof lngLat.wrap === "function" ? lngLat.wrap() : lngLat;
      onSelectRef.current(coordinatesToSelectedLocation(wrapped.lat, wrapped.lng));
    };

    const handleManualInteraction = () => {
      startupAutoCenterEligibleRef.current = false;
    };

    const handleManualZoomStart = (event?: { originalEvent?: unknown }) => {
      // Only user-initiated zoom gestures have an original DOM event.
      // Programmatic easeTo/flyTo calls omit originalEvent.
      if (event?.originalEvent) {
        startupAutoCenterEligibleRef.current = false;
      }
    };

    const handleMoveEnd = () => {
      if (onCenterChangeRef.current && mapRef.current) {
        const center = mapRef.current.getCenter ? mapRef.current.getCenter() : null;
        if (center) {
          const canonicalLon = canonicalizeLongitude(center.lng);
          const clampedLat = Math.max(-90, Math.min(90, center.lat));
          onCenterChangeRef.current({ latitude: clampedLat, longitude: canonicalLon });
        }
      }
    };

    map.on("click", handleClick);
    map.on("dragstart", handleManualInteraction);
    map.on("rotatestart", handleManualInteraction);
    map.on("pitchstart", handleManualInteraction);
    map.on("zoomstart", handleManualZoomStart);
    map.on("moveend", handleMoveEnd);

    const handleLoad = () => {
      if (isStyleReadyRef.current) {
        return;
      }
      isStyleReadyRef.current = true;

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
      container.removeEventListener("pointerdown", handleUserGesture);
      container.removeEventListener("wheel", handleUserGesture);
      container.removeEventListener("touchstart", handleUserGesture);
      map.off("click", handleClick);
      map.off("dragstart", handleManualInteraction);
      map.off("rotatestart", handleManualInteraction);
      map.off("pitchstart", handleManualInteraction);
      map.off("zoomstart", handleManualZoomStart);
      map.off("moveend", handleMoveEnd);
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
      if (typeof window !== "undefined") {
        delete (window as unknown as { __weatherMap?: MapLibreMap }).__weatherMap;
      }
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

  // Best-effort startup coarse IP localization camera transition.
  // Applied at most once per session when eligible.
  useEffect(() => {
    const map = mapRef.current;
    if (map === null || !approximateLocation) {
      return;
    }

    // Canonical location or active locate attempt permanently disqualifies auto-centering
    if (selectedLocation !== null || isLocating) {
      startupAutoCenterEligibleRef.current = false;
      return;
    }

    if (!startupAutoCenterEligibleRef.current) {
      return;
    }

    // One-time opportunity: mark ineligible immediately
    startupAutoCenterEligibleRef.current = false;

    if (typeof map.easeTo === "function") {
      map.easeTo({
        center: [approximateLocation.longitude, approximateLocation.latitude],
        zoom: 6.5,
        duration: 800,
        essential: false,
      });
    }
  }, [approximateLocation, selectedLocation, isLocating]);

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

    // Fly to the location only if selection was NOT initiated by a direct map click.
    if (isMapClickSelectionRef.current) {
      isMapClickSelectionRef.current = false;
    } else if (typeof map.flyTo === "function") {
      const currentZoom = typeof map.getZoom === "function" ? map.getZoom() : 8;
      map.flyTo({
        center: [selectedLocation.longitude, selectedLocation.latitude],
        zoom: Math.max(currentZoom, 8),
        essential: true,
      });
    }
  }, [selectedLocation]);

  return (
    <div ref={containerRef} className="relative h-full w-full" data-testid="weather-map">
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 z-10 h-full w-full"
        data-testid="wind-particle-canvas"
      />
      {onLocate && (
        <div className="absolute right-2.5 top-28 z-20">
          <LocateMeButton onClick={handleLocate} isLocating={isLocating} />
        </div>
      )}
    </div>
  );
}
