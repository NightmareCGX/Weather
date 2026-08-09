"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

import type { SelectedLocation } from "@/lib/api/types";

/**
 * Shared selected-location state for Milestone 13.
 *
 * Search autocomplete, map point selection, and the forecast dashboard all
 * reduce to a single {@link SelectedLocation}; this context is the single
 * source of truth. A new selection (from search or a map click) replaces the
 * previous one and drives the forecast fetch.
 */

interface SelectedLocationContextValue {
  selectedLocation: SelectedLocation | null;
  selectLocation: (location: SelectedLocation) => void;
  clearSelection: () => void;
}

const SelectedLocationContext = createContext<SelectedLocationContextValue | null>(null);

export function SelectedLocationProvider({ children }: { children: ReactNode }) {
  const [selectedLocation, setSelectedLocation] = useState<SelectedLocation | null>(null);

  const value = useMemo<SelectedLocationContextValue>(
    () => ({
      selectedLocation,
      selectLocation: setSelectedLocation,
      clearSelection: () => setSelectedLocation(null),
    }),
    [selectedLocation]
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
