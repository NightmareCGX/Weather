"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type { ApproximateStartupLocation, SelectedLocation } from "@/lib/api/types";
import { getApproximateLocation } from "@/lib/api/client";
import { canonicalizeLongitude, isValidCoordinate } from "@/lib/forecast/selection";
import { getTimezoneForCoordinates } from "@/lib/forecast/timezone";

/**
 * Shared selected-location state for Milestone 13 & Location Discovery.
 *
 * Search autocomplete, map point selection, and the forecast dashboard all
 * reduce to a single {@link SelectedLocation}; this context is the single
 * source of truth. A new selection (from search or a map click) replaces the
 * previous one and drives the forecast fetch.
 *
 * Implements a monotonic selection generation token guard so that asynchronous
 * location acquisition (e.g. Locate Me) cannot overwrite newer explicit user
 * selections (such as a map click or search selection).
 *
 * Also manages ambient startup coarse IP localization context. This is NOT a
 * SelectedLocation and does not trigger point forecast loading or map selection
 * markers, but provides an approximate regional viewport and a fallback display
 * timezone before a canonical location is chosen.
 */

export interface SelectedLocationContextValue {
  selectedLocation: SelectedLocation | null;
  /** IANA timezone identifier for the selected location, or null when no location is selected or lookup fails. */
  selectedTimezone: string | null;
  /** Coarse startup geographic context derived from IP headers. Ambient context only; NOT a SelectedLocation. */
  startupLocation: ApproximateStartupLocation | null;
  /** IANA timezone identifier derived from startup coarse IP location, or null if unavailable. */
  startupTimezone: string | null;
  /**
   * Resolved presentation timezone following the precedence:
   * selectedTimezone > startupTimezone > null (UTC fallback).
   */
  displayTimezone: string | null;
  /** Current selection generation sequence number. */
  selectionGeneration: number;
  /** Commit a synchronous location selection immediately, invalidating pending async selections. */
  selectLocation: (location: SelectedLocation) => void;
  /** Clear the active location selection, invalidating pending async selections. */
  clearSelection: () => void;
  /** Begin an asynchronous selection flow (e.g. Locate Me) and return a generation token. */
  beginAsyncSelection: () => number;
  /**
   * Commit an asynchronous selection only if its token matches the active generation.
   * Returns true if committed, false if rejected as stale.
   */
  commitAsyncSelection: (location: SelectedLocation, token: number) => boolean;
  /** Request resolution of startup coarse IP location context if not already in flight or resolved. */
  requestStartupLocation: () => void;
}

let startupLocationPromise: Promise<ApproximateStartupLocation | null> | null = null;

/**
 * Fetch and cache coarse startup IP location.
 *
 * Safe under React Strict Mode: the underlying request is kicked off once and
 * cached in a module-level Promise. Multiple mounts/remounts subscribe to the
 * same in-flight or resolved Promise without duplicate requests or aborted fetches.
 */
export function getCachedStartupLocation(): Promise<ApproximateStartupLocation | null> {
  if (!startupLocationPromise) {
    startupLocationPromise = Promise.resolve()
      .then(() => getApproximateLocation())
      .then((resp): ApproximateStartupLocation | null => {
        if (!resp) return null;
        const canonicalLon = canonicalizeLongitude(resp.longitude);
        if (!isValidCoordinate(resp.latitude, canonicalLon)) return null;
        return {
          latitude: resp.latitude,
          longitude: canonicalLon,
          city: resp.city,
          region: resp.region,
          country: resp.country,
          source: "ip",
        };
      })
      .catch(() => null);
  }
  return startupLocationPromise;
}

/** Reset the cached startup location Promise (for testing environments only). */
export function _resetStartupLocationPromiseForTesting(): void {
  startupLocationPromise = null;
}

const SelectedLocationContext = createContext<SelectedLocationContextValue | null>(null);

export function SelectedLocationProvider({ children }: { children: ReactNode }) {
  const [selectedLocation, setSelectedLocation] = useState<SelectedLocation | null>(null);
  const [startupLocation, setStartupLocation] = useState<ApproximateStartupLocation | null>(null);
  const [generation, setGeneration] = useState(0);
  const generationRef = useRef(0);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const requestStartupLocation = useCallback(() => {
    getCachedStartupLocation().then((loc) => {
      if (isMountedRef.current && loc !== null) {
        setStartupLocation(loc);
      }
    });
  }, []);

  const selectLocation = useCallback((location: SelectedLocation) => {
    generationRef.current += 1;
    setGeneration(generationRef.current);
    setSelectedLocation(location);
  }, []);

  const clearSelection = useCallback(() => {
    generationRef.current += 1;
    setGeneration(generationRef.current);
    setSelectedLocation(null);
  }, []);

  const beginAsyncSelection = useCallback(() => {
    generationRef.current += 1;
    setGeneration(generationRef.current);
    return generationRef.current;
  }, []);

  const commitAsyncSelection = useCallback((location: SelectedLocation, token: number): boolean => {
    if (token === generationRef.current) {
      setSelectedLocation(location);
      return true;
    }
    return false;
  }, []);

  const selectedLat = selectedLocation?.latitude;
  const selectedLon = selectedLocation?.longitude;
  const selectedTimezone = useMemo(() => {
    if (selectedLat === undefined || selectedLon === undefined) {
      return null;
    }
    return getTimezoneForCoordinates(selectedLat, selectedLon);
  }, [selectedLat, selectedLon]);

  const startupLat = startupLocation?.latitude;
  const startupLon = startupLocation?.longitude;
  const startupTimezone = useMemo(() => {
    if (startupLat === undefined || startupLon === undefined) {
      return null;
    }
    return getTimezoneForCoordinates(startupLat, startupLon);
  }, [startupLat, startupLon]);

  const displayTimezone = useMemo(() => {
    return selectedTimezone ?? startupTimezone ?? null;
  }, [selectedTimezone, startupTimezone]);

  const value = useMemo<SelectedLocationContextValue>(
    () => ({
      selectedLocation,
      selectedTimezone,
      startupLocation,
      startupTimezone,
      displayTimezone,
      selectionGeneration: generation,
      selectLocation,
      clearSelection,
      beginAsyncSelection,
      commitAsyncSelection,
      requestStartupLocation,
    }),
    [
      selectedLocation,
      selectedTimezone,
      startupLocation,
      startupTimezone,
      displayTimezone,
      generation,
      selectLocation,
      clearSelection,
      beginAsyncSelection,
      commitAsyncSelection,
      requestStartupLocation,
    ]
  );

  return (
    <SelectedLocationContext.Provider value={value}>{children}</SelectedLocationContext.Provider>
  );
}

export function useSelectedLocation(): SelectedLocationContextValue {
  const context = useContext(SelectedLocationContext);
  if (context === null) {
    throw new Error("useSelectedLocation must be used within a SelectedLocationProvider");
  }
  return context;
}

/**
 * Convenience hook to access the resolved IANA timezone for the active selection.
 * Safely returns null if no selection exists or if called outside of the provider.
 */
export function useSelectedLocationTimezone(): string | null {
  const context = useContext(SelectedLocationContext);
  return context?.selectedTimezone ?? null;
}

/**
 * Convenience hook to access coarse startup IP location context (null if unavailable or resolving).
 * Subscribes to startup IP resolution on mount.
 * This is ambient context only — NOT a SelectedLocation.
 */
export function useStartupLocation(): ApproximateStartupLocation | null {
  const context = useContext(SelectedLocationContext);
  useEffect(() => {
    context?.requestStartupLocation();
  }, [context]);
  return context?.startupLocation ?? null;
}

/**
 * Convenience hook to access the resolved presentation timezone with precedence:
 * selectedLocation timezone > startup coarse IP timezone > null (UTC fallback).
 * Subscribes to startup IP resolution on mount.
 */
export function useDisplayTimezone(): string | null {
  const context = useContext(SelectedLocationContext);
  useEffect(() => {
    context?.requestStartupLocation();
  }, [context]);
  return context?.displayTimezone ?? null;
}
