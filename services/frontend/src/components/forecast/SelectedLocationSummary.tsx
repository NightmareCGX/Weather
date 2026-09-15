"use client";

import { locationTypeLabel } from "@/lib/forecast/selection";
import type { SelectedLocation } from "@/lib/api/types";
import type { ElevationFetchStatus } from "@/hooks/useElevation";

interface SelectedLocationSummaryProps {
  location: SelectedLocation;
  elevation_m?: number | null;
  elevationStatus?: ElevationFetchStatus;
  onClose?: () => void;
}

/**
 * Presentation-only summary of the selected location: type badge, name,
 * region/country, coordinates, and elevation (when the resolved record defines
 * one or when dynamically resolved).
 *
 * Includes an optional Close button to deselect the location.
 */
export function SelectedLocationSummary({
  location,
  elevation_m,
  elevationStatus,
  onClose,
}: SelectedLocationSummaryProps) {
  const displayElevation = elevation_m !== undefined ? elevation_m : location.elevation_m;
  const isElevationLoading = elevationStatus === "loading" && displayElevation === null;
  return (
    <>
      <div className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-slate-800 bg-slate-900 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="shrink-0 rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 text-xs font-mono text-cyan-400">
            {locationTypeLabel(location)}
          </span>
          <h2 className="truncate text-sm font-bold text-slate-100">{location.name}</h2>
        </div>
        {onClose && (
          <button
            type="button"
            aria-label="Close forecast panel"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded text-slate-400 hover:bg-slate-800 hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
          >
            <svg
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>

      <section aria-label="Selected location" className="border-b border-slate-800 px-4 py-3">
        {(location.region !== null || location.country !== null) && (
          <p className="text-xs text-slate-400">
            {[location.region, location.country].filter(Boolean).join(", ")}
          </p>
        )}
        <dl
          className={`${
            location.region !== null || location.country !== null ? "mt-1.5 " : ""
          }grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-slate-300`}
        >
          <div className="flex justify-between">
            <dt className="text-slate-400">Latitude</dt>
            <dd className="font-mono tabular-nums text-slate-200">
              {location.latitude.toFixed(4)}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-slate-400">Longitude</dt>
            <dd className="font-mono tabular-nums text-slate-200">
              {location.longitude.toFixed(4)}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-slate-400">Elevation</dt>
            <dd className="font-mono tabular-nums text-cyan-400">
              {isElevationLoading
                ? "loading…"
                : displayElevation !== null
                  ? `${Math.round(displayElevation).toLocaleString()} m`
                  : "unavailable"}
            </dd>
          </div>
        </dl>
      </section>
    </>
  );
}
