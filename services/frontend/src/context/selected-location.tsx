"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type { SelectedLocation } from "@/lib/api/types";

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
 */

export interface SelectedLocationContextValue {
  selectedLocation: SelectedLocation | null;
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
}

const SelectedLocationContext = createContext<SelectedLocationContextValue | null>(null);

export function SelectedLocationProvider({ children }: { children: ReactNode }) {
  const [selectedLocation, setSelectedLocation] = useState<SelectedLocation | null>(null);
  const [generation, setGeneration] = useState(0);
  const generationRef = useRef(0);

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

  const value = useMemo<SelectedLocationContextValue>(
    () => ({
      selectedLocation,
      selectionGeneration: generation,
      selectLocation,
      clearSelection,
      beginAsyncSelection,
      commitAsyncSelection,
    }),
    [
      selectedLocation,
      generation,
      selectLocation,
      clearSelection,
      beginAsyncSelection,
      commitAsyncSelection,
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
