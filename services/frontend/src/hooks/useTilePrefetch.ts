"use client";

import { useCallback, useEffect, useRef, type RefObject } from "react";
import type { Map as MapLibreMap } from "maplibre-gl";

import type { SpatialLayer, ValidTimeAvailability } from "@/lib/api/types";

/** Fetch budget per trigger; restricted to viewport core to prevent background bursts. */
const MAX_TILES_PER_TRIGGER = 24;
/** Concurrent prefetch requests (gentle background workers). */
const MAX_CONCURRENCY = 2;
/** Delay after a trigger so foreground viewport and charts settle before warming starts. */
const PREFETCH_DELAY_MS = 2000;
/** Move-end debounce: wait for the viewport to stop changing. */
const MOVE_END_DEBOUNCE_MS = 500;
/** Bounded dedup-set size; URLs are immutable so clearing only costs redundant fetches. */
const MAX_TRACKED_URLS = 4096;

function shouldSkipPrefetch(): boolean {
  if (typeof navigator === "undefined") return false;
  const nav = navigator as Navigator & {
    connection?: {
      saveData?: boolean;
      effectiveType?: string;
    };
  };
  if (nav.connection?.saveData === true) return true;
  if (nav.connection?.effectiveType === "2g" || nav.connection?.effectiveType === "slow-2g") {
    return true;
  }
  return false;
}

/**
 * Web-Mercator tile indices covering the map's current viewport at an integer
 * zoom, row-major from the top edge, longitude-wrapped and latitude-clamped.
 */
export function coveringTiles(map: MapLibreMap, zoom: number): Array<{ x: number; y: number }> {
  const n = 2 ** zoom;
  const bounds = map.getBounds();
  if (bounds === undefined) {
    return [];
  }
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();

  const lngToX = (lng: number) => Math.floor(((lng + 180) / 360) * n);
  const latToY = (lat: number) => {
    const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
    const rad = (clamped * Math.PI) / 180;
    const merc = Math.log(Math.tan(Math.PI / 4 + rad / 2));
    return Math.floor(((1 - merc / Math.PI) / 2) * n);
  };

  // Clamp the x range: a bound edge at longitude 180 maps to x == n, which
  // must fold onto the last column rather than wrap to x == 0.
  const minX = Math.max(0, Math.min(n - 1, lngToX(sw.lng)));
  const maxX = Math.max(0, Math.min(n - 1, lngToX(ne.lng)));
  const minY = Math.max(0, latToY(ne.lat));
  const maxY = Math.min(n - 1, latToY(sw.lat));
  if (maxX < minX || maxY < minY) {
    return [];
  }

  const tiles: Array<{ x: number; y: number }> = [];
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (tiles.length >= MAX_TILES_PER_TRIGGER) {
        return tiles;
      }
      tiles.push({ x, y });
    }
  }
  return tiles;
}

/**
 * Concrete prefetch URLs for one adjacent valid time: the layer's tile URL
 * template (which still carries ``{z}/{x}/{y}`` for MapLibre) with the
 * valid-time and pinned serving-cycle query parameters swapped to the target
 * entry. Dropping the cycle parameter when the target entry has none yields
 * the unpinned (revalidation-only) URL the server treats identically.
 */
export function adjacentTileUrls(
  layerTileUrlTemplate: string,
  target: ValidTimeAvailability,
  zoom: number,
  tiles: Array<{ x: number; y: number }>
): string[] {
  const queryIndex = layerTileUrlTemplate.indexOf("?");
  const path = queryIndex === -1 ? layerTileUrlTemplate : layerTileUrlTemplate.slice(0, queryIndex);
  const params = new URLSearchParams(
    queryIndex === -1 ? "" : layerTileUrlTemplate.slice(queryIndex + 1)
  );
  params.set("valid_time", target.valid_time);
  if (target.source_cycle) {
    params.set("initial_time", target.source_cycle);
  } else {
    params.delete("initial_time");
  }
  const template = `${path}?${params.toString()}`;
  return tiles.map((tile) =>
    template
      .replace("{z}", String(zoom))
      .replace("{x}", String(tile.x))
      .replace("{y}", String(tile.y))
  );
}

/**
 * Sequential-workers fetch: materializes each response body (a few KB per
 * tile) so the browser HTTP cache stores it, which is the entire point — the
 * pinned tile URLs are ``immutable``, so MapLibre's later requests for the
 * same URLs resolve from the local cache without touching the network.
 */
async function fetchAll(urls: string[], signal: AbortSignal, tracked: Set<string>): Promise<void> {
  let index = 0;
  const worker = async () => {
    while (index < urls.length && !signal.aborted) {
      const url = urls[index++];
      try {
        const response = await fetch(url, { signal, priority: "low" } as RequestInit);
        if (response.ok) {
          await response.arrayBuffer();
        }
      } catch {
        if (signal.aborted) {
          return;
        }
        // Network failure: leave untracked so a later trigger can retry.
        tracked.delete(url);
        continue;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(MAX_CONCURRENCY, urls.length) }, worker));
}

export interface UseAdjacentTilePrefetchOptions {
  /** Map created by WeatherMap; set before this hook's effects run. */
  mapRef: RefObject<MapLibreMap | null>;
  /** Authoritative weather layer (valid-time mode only). */
  layer: SpatialLayer | null;
  /** Per-valid-time availability entries with their serving cycles. */
  validTimes: ValidTimeAvailability[];
  enabled?: boolean;
}

/**
 * Warms the browser HTTP cache with the raster tiles of the adjacent forecast
 * valid times (next, then previous) for the current viewport, so stepping the
 * time slider renders instantly instead of waiting on per-tile roundtrips.
 *
 * Tile URLs are immutable (pinned to their serving cycle), so fetched tiles
 * stay valid; the dedup set makes repeated triggers cheap. Failures are
 * silently ignored and never affect the active layer, mirroring the
 * vector-field prefetch contract.
 */
export function useAdjacentTilePrefetch({
  mapRef,
  layer,
  validTimes,
  enabled = true,
}: UseAdjacentTilePrefetchOptions): void {
  const layerRef = useRef(layer);
  layerRef.current = layer;
  const validTimesRef = useRef(validTimes);
  validTimesRef.current = validTimes;
  const prefetchedRef = useRef<Set<string>>(new Set());
  const controllerRef = useRef<AbortController | null>(null);
  const delayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runPrefetch = useCallback(() => {
    const map = mapRef.current;
    const currentLayer = layerRef.current;
    if (!enabled || map === null || currentLayer === null) {
      return;
    }
    if (currentLayer.valid_time === undefined || currentLayer.valid_time === null) {
      // Legacy lead-time mode: adjacent-lead warming is handled by the
      // vector-field prefetch; raster tiles have no stable identity to pin.
      return;
    }
    if (typeof document !== "undefined" && document.hidden) {
      return;
    }
    if (shouldSkipPrefetch()) {
      return;
    }

    const entries = [...validTimesRef.current]
      .filter((entry) => entry.servable)
      .sort((a, b) => new Date(a.valid_time).getTime() - new Date(b.valid_time).getTime());
    if (entries.length < 2) {
      return;
    }
    const currentMs = new Date(currentLayer.valid_time).getTime();
    const currentIndex = entries.findIndex(
      (entry) => new Date(entry.valid_time).getTime() === currentMs
    );
    if (currentIndex === -1) {
      return;
    }
    const targets: ValidTimeAvailability[] = [];
    if (currentIndex + 1 < entries.length) {
      targets.push(entries[currentIndex + 1]);
    }
    if (currentIndex - 1 >= 0) {
      targets.push(entries[currentIndex - 1]);
    }
    if (targets.length === 0) {
      return;
    }

    const zoom = Math.min(
      currentLayer.max_zoom,
      Math.max(currentLayer.min_zoom, Math.round(map.getZoom()))
    );
    const tiles = coveringTiles(map, zoom);
    if (tiles.length === 0) {
      return;
    }

    const urls = targets.flatMap((target) =>
      adjacentTileUrls(currentLayer.tile_url_template, target, zoom, tiles)
    );
    const pending = urls.filter((url) => !prefetchedRef.current.has(url));
    if (pending.length === 0) {
      return;
    }
    for (const url of pending) {
      prefetchedRef.current.add(url);
    }
    if (prefetchedRef.current.size > MAX_TRACKED_URLS) {
      prefetchedRef.current.clear();
    }

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    void fetchAll(pending, controller.signal, prefetchedRef.current);
  }, [enabled, mapRef]);

  const schedulePrefetch = useCallback(() => {
    if (delayTimerRef.current !== null) {
      clearTimeout(delayTimerRef.current);
    }
    delayTimerRef.current = setTimeout(() => {
      delayTimerRef.current = null;
      runPrefetch();
    }, PREFETCH_DELAY_MS);
  }, [runPrefetch]);

  // Re-warm whenever the layer becomes authoritative or changes semantic
  // content (valid time, serving cycle, variable). Aborting the previous run
  // on cleanup keeps rapid time-stepping from piling up stale requests.
  useEffect(() => {
    schedulePrefetch();
    return () => {
      if (delayTimerRef.current !== null) {
        clearTimeout(delayTimerRef.current);
        delayTimerRef.current = null;
      }
      controllerRef.current?.abort();
    };
  }, [schedulePrefetch, layer]);

  // Re-warm after the viewport settles so the adjacent steps cover what the
  // user actually sees, not just where they started.
  useEffect(() => {
    const map = mapRef.current;
    if (map === null) {
      return;
    }
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    const handleMoveEnd = () => {
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
      }
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        schedulePrefetch();
      }, MOVE_END_DEBOUNCE_MS);
    };
    map.on("moveend", handleMoveEnd);
    return () => {
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
      }
      map.off("moveend", handleMoveEnd);
    };
  }, [mapRef, schedulePrefetch]);
}
